import argparse
import json
import sys
from pathlib import Path

# Two backends, chosen by platform:
#   mlx    — mlx-whisper on Apple Silicon (Metal).
#   faster — faster-whisper on CUDA (CTranslate2).
#
# Preset names are stable across backends; each backend has its own
# concrete HF repo because MLX and CT2 use different model formats.
#   turbo: fast English-focused preset.
#   v3:    full large-v3, better on non-English / accented / noisy audio.
MODEL_PRESETS: dict[str, dict[str, str]] = {
    "mlx": {
        "turbo": "mlx-community/whisper-large-v3-turbo",
        "v3": "mlx-community/whisper-large-v3-mlx-8bit",
    },
    "faster": {
        # No 1:1 CT2 port of the distilled turbo exists from Systran; on a
        # 4090 the full v3 is already faster than mlx turbo on a Mac, so we
        # alias both presets to it. Swap in deepdml/faster-whisper-large-v3-turbo-ct2
        # if we ever want the true turbo distill.
        "turbo": "Systran/faster-whisper-large-v3",
        "v3": "Systran/faster-whisper-large-v3",
    },
}

DEFAULT_PRESET = "turbo"
DEFAULT_BACKEND = "faster" if sys.platform == "linux" else "mlx"


def resolve_backend(backend: str | None) -> str:
    return backend or DEFAULT_BACKEND


def resolve_model(model: str, backend: str | None = None) -> str:
    """Turn a preset name into a concrete repo; pass-through unknown names
    (raw HF repos, local paths)."""
    presets = MODEL_PRESETS[resolve_backend(backend)]
    return presets.get(model, model)


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
    backend: str | None = None,
    compute_type: str = "float16",
) -> dict:
    backend = resolve_backend(backend)
    if backend == "mlx":
        return _transcribe_mlx(
            audio_path,
            model=model,
            language=language,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            beam_size=beam_size,
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            hallucination_silence_threshold=hallucination_silence_threshold,
        )
    if backend == "faster":
        return _transcribe_faster(
            audio_path,
            model=model,
            language=language,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            beam_size=beam_size,
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            hallucination_silence_threshold=hallucination_silence_threshold,
            compute_type=compute_type,
        )
    raise ValueError(f"Unknown backend: {backend!r} (expected 'mlx' or 'faster')")


def _transcribe_mlx(
    audio_path: Path,
    model: str,
    language: str | None,
    word_timestamps: bool,
    initial_prompt: str | None,
    beam_size: int | None,
    temperature: tuple[float, ...] | float | None,
    compression_ratio_threshold: float | None,
    hallucination_silence_threshold: float | None,
) -> dict:
    import mlx_whisper

    kwargs: dict = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if compression_ratio_threshold is not None:
        kwargs["compression_ratio_threshold"] = compression_ratio_threshold
    if hallucination_silence_threshold is not None:
        kwargs["hallucination_silence_threshold"] = hallucination_silence_threshold
    if beam_size is not None:
        # NOTE: mlx-whisper 0.4.3 raises NotImplementedError if beam_size > 1.
        kwargs["beam_size"] = beam_size

    return mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=resolve_model(model, backend="mlx"),
        language=language,
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt,
        verbose=False,
        **kwargs,
    )


def _transcribe_faster(
    audio_path: Path,
    model: str,
    language: str | None,
    word_timestamps: bool,
    initial_prompt: str | None,
    beam_size: int | None,
    temperature: tuple[float, ...] | float | None,
    compression_ratio_threshold: float | None,
    hallucination_silence_threshold: float | None,
    compute_type: str,
) -> dict:
    from faster_whisper import WhisperModel

    whisper = WhisperModel(
        resolve_model(model, backend="faster"),
        device="cuda",
        compute_type=compute_type,
    )
    kwargs: dict = {}
    if temperature is not None:
        # faster-whisper accepts float or list[float] (fallback ladder).
        kwargs["temperature"] = list(temperature) if isinstance(temperature, tuple) else temperature
    if compression_ratio_threshold is not None:
        kwargs["compression_ratio_threshold"] = compression_ratio_threshold
    if hallucination_silence_threshold is not None:
        kwargs["hallucination_silence_threshold"] = hallucination_silence_threshold
    if initial_prompt is not None:
        kwargs["initial_prompt"] = initial_prompt
    # Beam search is faster-whisper's default; preserve greedy if user asked for it.
    kwargs["beam_size"] = 5 if beam_size is None else beam_size

    segments_iter, info = whisper.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=word_timestamps,
        **kwargs,
    )
    return _fw_to_dict(segments_iter, info)


def _fw_to_dict(segments_iter, info) -> dict:
    """Normalize faster-whisper output to the mlx-whisper schema that
    dedupe.py / merge.py / summarize_video.py consume.

    Key adapter job: re-add the leading space on word tokens. OpenAI
    Whisper / mlx-whisper emit words as ' hello', ' world' so that a plain
    concat reconstructs the text. faster-whisper strips that leading
    space — we put it back, otherwise dedupe.py:114 and merge.py:62
    produce run-together words (helloworld).
    """
    segments: list[dict] = []
    texts: list[str] = []
    for s in segments_iter:
        words = [
            {
                "word": " " + (w.word or "").lstrip(),
                "start": w.start,
                "end": w.end,
                "probability": w.probability,
            }
            for w in (s.words or [])
        ]
        segments.append({
            "id": s.id,
            "seek": s.seek,
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "words": words,
            "avg_logprob": s.avg_logprob,
            "compression_ratio": s.compression_ratio,
            "no_speech_prob": s.no_speech_prob,
            "temperature": s.temperature,
        })
        texts.append(s.text)
    return {
        "text": "".join(texts),
        "language": info.language,
        "segments": segments,
    }


def write_outputs(audio_path: Path, result: dict) -> tuple[Path, Path]:
    json_path = audio_path.with_suffix(".json")
    txt_path = audio_path.with_suffix(".txt")
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    txt_path.write_text(result["text"].strip() + "\n")
    return json_path, txt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe an audio file with Whisper (mlx on Mac, faster-whisper on Linux/CUDA).")
    parser.add_argument("audio", type=Path, help="Path to audio file (m4a, wav, mp3, ...)")
    parser.add_argument(
        "-b", "--backend", choices=sorted(MODEL_PRESETS.keys()), default=None,
        help=f"Transcription backend. Default: {DEFAULT_BACKEND} (platform default).",
    )
    parser.add_argument(
        "-m", "--model", default=DEFAULT_PRESET,
        help=(
            f"Model preset (turbo|v3) or a raw HF repo / local path. "
            f"Default: {DEFAULT_PRESET}."
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
        help="Beam search width. Default: mlx=greedy (None), faster=5.",
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
    parser.add_argument(
        "--compute-type", default="float16",
        help="CTranslate2 compute type (faster backend only). Default: float16. "
             "Try int8_float16 for lower VRAM.",
    )
    args = parser.parse_args()

    backend = resolve_backend(args.backend)
    print(f"Transcribing {args.audio} with backend={backend} model={args.model} ({resolve_model(args.model, backend=backend)})")
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
        backend=backend,
        compute_type=args.compute_type,
    )

    json_path, txt_path = write_outputs(args.audio, result)
    print(f"Detected language: {result.get('language')}")
    print(f"Segments: {len(result.get('segments', []))}")
    print(f"Wrote transcript: {txt_path}")
    print(f"Wrote JSON:       {json_path}")


if __name__ == "__main__":
    main()
