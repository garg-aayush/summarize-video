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
) -> dict:
    return mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=resolve_model(model),
        language=language,
        word_timestamps=word_timestamps,
        verbose=False,
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
    args = parser.parse_args()

    print(f"Transcribing {args.audio} with {args.model} ({resolve_model(args.model)})")
    result = transcribe(
        args.audio,
        model=args.model,
        language=args.language,
        word_timestamps=not args.no_word_timestamps,
    )

    json_path, txt_path = write_outputs(args.audio, result)
    print(f"Detected language: {result.get('language')}")
    print(f"Segments: {len(result.get('segments', []))}")
    print(f"Wrote transcript: {txt_path}")
    print(f"Wrote JSON:       {json_path}")


if __name__ == "__main__":
    main()
