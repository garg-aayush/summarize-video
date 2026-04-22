"""End-to-end pipeline: URL -> diarized transcript.

Runs all five steps in order:
  1. download   (steps.download.download_audio)
  2. transcribe (steps.transcribe.transcribe)
  3. dedupe     (steps.dedupe.dedupe_transcript)
  4. diarize    (steps.diarize.diarize)
  5. merge      (steps.merge.merge)

Each step is skipped if its output file already exists, so re-running is cheap
and you can resume after a crash. Pass --force to ignore caches.
"""

import argparse
import json
import time
from pathlib import Path

from steps import download as download_step
from steps import transcribe as transcribe_step
from steps import dedupe as dedupe_step
from steps import diarize as diarize_step
from steps import merge as merge_step


def _step(name: str):
    print(f"\n=== {name} ===")
    return time.perf_counter()


def _done(t0: float) -> None:
    print(f"  ({time.perf_counter() - t0:.1f}s)")


def run(
    url: str,
    output_dir: Path,
    model: str,
    language: str | None,
    compression_ratio_threshold: float | None,
    hallucination_silence_threshold: float | None,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    force: bool,
) -> None:
    # --- 1. download -------------------------------------------------------
    t0 = _step(f"1/5 download  {url}")
    audio_path = download_step.download_audio(url, output_dir)
    print(f"  -> {audio_path}")
    _done(t0)

    transcript_path = audio_path.with_suffix(".json")
    diarization_path = audio_path.with_suffix(".diarization.json")
    diarized_path = audio_path.with_suffix(".diarized.txt")

    # --- 2. transcribe -----------------------------------------------------
    t0 = _step(f"2/5 transcribe  model={model} lang={language or 'auto'}")
    if transcript_path.exists() and not force:
        print(f"  cached: {transcript_path}")
    else:
        result = transcribe_step.transcribe(
            audio_path,
            model=model,
            language=language,
            word_timestamps=True,
            compression_ratio_threshold=compression_ratio_threshold,
            hallucination_silence_threshold=hallucination_silence_threshold,
        )
        transcribe_step.write_outputs(audio_path, result)
        print(f"  detected language: {result.get('language')}")
        print(f"  segments: {len(result.get('segments', []))}")
    _done(t0)

    # --- 3. dedupe ---------------------------------------------------------
    t0 = _step("3/5 dedupe")
    raw_backup = audio_path.with_suffix(".raw.json")
    if raw_backup.exists() and not force:
        print(f"  cached (backup exists): {raw_backup}")
    else:
        raw_text = transcript_path.read_text()
        transcript = json.loads(raw_text)
        n_before = sum(len(s.get("words") or []) for s in transcript.get("segments", []))
        cleaned = dedupe_step.dedupe_transcript(transcript)
        n_after = sum(len(s.get("words") or []) for s in cleaned.get("segments", []))
        if not raw_backup.exists():
            raw_backup.write_text(raw_text)
        transcript_path.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False))
        audio_path.with_suffix(".txt").write_text(cleaned["text"].strip() + "\n")
        print(f"  words: {n_before} -> {n_after} (collapsed {n_before - n_after})")
    _done(t0)

    # --- 4. diarize --------------------------------------------------------
    t0 = _step("4/5 diarize")
    if diarization_path.exists() and not force:
        print(f"  cached: {diarization_path}")
    else:
        pipeline = diarize_step.load_pipeline()
        result = diarize_step.diarize(
            audio_path, pipeline,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        diarize_step.write_outputs(audio_path, result)
        speakers = sorted({t["speaker"] for t in result["diarization"]})
        print(f"  speakers: {len(speakers)} ({', '.join(speakers)})")
    _done(t0)

    # --- 5. merge ----------------------------------------------------------
    t0 = _step("5/5 merge")
    transcript = json.loads(transcript_path.read_text())
    diarization = json.loads(diarization_path.read_text())
    utterances = merge_step.merge(transcript, diarization)
    merge_step.write_outputs(audio_path, utterances)
    print(f"  utterances: {len(utterances)}")
    print(f"  -> {diarized_path}")
    _done(t0)

    print(f"\nDone. Final transcript: {diarized_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", help="Source URL (e.g., a YouTube video)")
    parser.add_argument("-o", "--output-dir", type=Path, default=download_step.DOWNLOADS_DIR)
    parser.add_argument("-m", "--model", default=transcribe_step.DEFAULT_PRESET,
                        help=f"Whisper preset or HF repo (default: {transcribe_step.DEFAULT_PRESET})")
    parser.add_argument("-l", "--language", default=None,
                        help="Force language code (e.g., 'hi'). Default: auto-detect.")
    parser.add_argument("--compression-ratio-threshold", type=float, default=None)
    parser.add_argument("--hallucination-silence-threshold", type=float, default=None)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("-f", "--force", action="store_true",
                        help="Re-run every step even if cached outputs exist.")
    args = parser.parse_args()

    run(
        url=args.url,
        output_dir=args.output_dir,
        model=args.model,
        language=args.language,
        compression_ratio_threshold=args.compression_ratio_threshold,
        hallucination_silence_threshold=args.hallucination_silence_threshold,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        force=args.force,
    )


if __name__ == "__main__":
    main()
