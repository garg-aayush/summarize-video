import argparse
import json
from pathlib import Path

import mlx_whisper

# Two presets:
#   turbo: distilled 4-decoder large-v3-turbo, fp16 (~1.6 GB safetensors).
#          Fast, English-tuned. Best when audio is English and we want speed.
#   v3:    full large-v3 (32 decoder layers), 8-bit quantized (~1.6 GB).
#          Slower but markedly better on non-English / accented / noisy audio.
MODEL_PRESETS = {
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "v3": "mlx-community/whisper-large-v3-mlx-8bit",
}
DEFAULT_PRESET = "turbo"


def resolve_model(model: str) -> str:
    return MODEL_PRESETS.get(model, model)


def transcribe(
    audio_path: Path,
    model: str = DEFAULT_PRESET,
    language: str | None = None,
    word_timestamps: bool = True,
    initial_prompt: str | None = None,
    beam_size: int | None = None,
    temperature: tuple[float, ...] | float | None = None,
    compression_ratio_threshold: float | None = None,
    hallucination_silence_threshold: float | None = None,
) -> dict:
    kwargs: dict = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if compression_ratio_threshold is not None:
        kwargs["compression_ratio_threshold"] = compression_ratio_threshold
    if hallucination_silence_threshold is not None:
        kwargs["hallucination_silence_threshold"] = hallucination_silence_threshold
    if beam_size is not None:
        # Passed through to the DecodingOptions via **decode_options.
        # NOTE: mlx-whisper 0.4.3 raises NotImplementedError if beam_size > 1.
        kwargs["beam_size"] = beam_size

    return mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=resolve_model(model),
        language=language,
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt,
        verbose=False,
        **kwargs,
    )


def write_outputs(audio_path: Path, result: dict) -> tuple[Path, Path]:
    json_path = audio_path.with_suffix(".json")
    txt_path = audio_path.with_suffix(".txt")
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    txt_path.write_text(result["text"].strip() + "\n")
    return json_path, txt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe an audio file with mlx-whisper.")
    parser.add_argument("audio", type=Path, help="Path to audio file (m4a, wav, mp3, ...)")
    parser.add_argument(
        "-m", "--model", default=DEFAULT_PRESET,
        help=(
            f"Model preset ({', '.join(MODEL_PRESETS)}) or a raw HF repo / local path. "
            f"Default: {DEFAULT_PRESET} -> {MODEL_PRESETS[DEFAULT_PRESET]}"
        ),
    )
    parser.add_argument(
        "-l", "--language", default=None,
        help="Force language code (e.g., 'en'). Default: auto-detect.",
    )
    parser.add_argument(
        "--no-word-timestamps", action="store_true",
        help="Skip per-word timestamps (faster, segment-only).",
    )
    parser.add_argument(
        "-p", "--initial-prompt", default=None,
        help=(
            "Text passed to Whisper before decoding the first chunk. "
            "Use Latin-script English words (e.g., 'LSD Mumbai Reliance School') "
            "to bias code-switched audio toward MacWhisper-style Hinglish output."
        ),
    )
    parser.add_argument(
        "--beam-size", type=int, default=None,
        help="Beam search width. Default greedy (None). Try 5 to escape repetition loops.",
    )
    parser.add_argument(
        "--compression-ratio-threshold", type=float, default=None,
        help=(
            "Trip wire for repetition loops. Default 2.4; lower (e.g., 2.0) catches "
            "loops earlier and triggers temperature-fallback re-decoding sooner."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, nargs="+", default=None,
        help=(
            "Sampling temperature(s). Pass one value (e.g., 0.0) or the full "
            "fallback ladder (e.g., 0.0 0.2 0.4 0.6 0.8 1.0)."
        ),
    )
    parser.add_argument(
        "--hallucination-silence-threshold", type=float, default=None,
        help=(
            "Skip text generated during silent stretches longer than N seconds. "
            "Requires word_timestamps. Try 2.0 to suppress silence-driven "
            "repetition loops in long monologues."
        ),
    )
    args = parser.parse_args()

    print(f"Transcribing {args.audio} with {args.model} ({resolve_model(args.model)})")
    temperature = (
        args.temperature[0] if args.temperature and len(args.temperature) == 1
        else (tuple(args.temperature) if args.temperature else None)
    )
    result = transcribe(
        args.audio,
        model=args.model,
        language=args.language,
        word_timestamps=not args.no_word_timestamps,
        initial_prompt=args.initial_prompt,
        beam_size=args.beam_size,
        temperature=temperature,
        compression_ratio_threshold=args.compression_ratio_threshold,
        hallucination_silence_threshold=args.hallucination_silence_threshold,
    )

    json_path, txt_path = write_outputs(args.audio, result)
    print(f"Detected language: {result.get('language')}")
    print(f"Segments: {len(result.get('segments', []))}")
    print(f"Wrote transcript: {txt_path}")
    print(f"Wrote JSON:       {json_path}")


if __name__ == "__main__":
    main()
