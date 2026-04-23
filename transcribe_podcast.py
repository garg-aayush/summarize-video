"""End-to-end pipeline: URL -> diarized transcript.

Runs five steps in order:
  1. download   (steps.download.download_audio)
  2. transcribe (steps.transcribe.transcribe)
  3. dedupe     (steps.dedupe.dedupe_transcript)
  4. diarize    (steps.diarize.diarize)         [skipped with --no-diarize]
  5. merge      (steps.merge.merge)             [skipped with --no-diarize]

Intermediate files land in a per-URL system temp dir
(`/tmp/podcasts-<id>/...`), so re-running the same URL skips already-done
steps. Final transcripts are copied into the output directory (default:
current working directory).

Always-produced final outputs:
  <id>.txt         plain deduped text
  <id>.timed.txt   `[mm:ss - mm:ss] text` per segment

With diarization (default):
  <id>.diarized.txt   `[mm:ss - mm:ss] SPEAKER_xx: text`
"""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import yt_dlp

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


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


def _resolve_video_id(url: str) -> str:
    """Ask yt-dlp for the URL's id without downloading audio.
    Lets us pick a stable per-URL work dir before step 1 actually runs."""
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return info["id"]


def _write_timed_text(transcript: dict, path: Path) -> None:
    with path.open("w") as f:
        for s in transcript.get("segments", []):
            text = s.get("text", "").strip()
            if not text:
                continue
            f.write(f"[{_fmt_ts(s['start'])} - {_fmt_ts(s['end'])}] {text}\n")


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
    no_diarize: bool,
    force: bool,
) -> None:
    print(f"Resolving {url}")
    video_id = _resolve_video_id(url)
    work_dir = Path(tempfile.gettempdir()) / f"podcasts-{video_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Work dir: {work_dir}")

    audio_path = work_dir / f"{video_id}.m4a"
    transcript_path = audio_path.with_suffix(".json")
    text_path = audio_path.with_suffix(".txt")
    timed_path = audio_path.with_suffix(".timed.txt")
    raw_backup = audio_path.with_suffix(".raw.json")
    diarization_path = audio_path.with_suffix(".diarization.json")
    diarized_txt = audio_path.with_suffix(".diarized.txt")
    diarized_json = audio_path.with_suffix(".diarized.json")

    steps_run: list[str] = []
    steps_cached: list[str] = []

    # --- 1. download -------------------------------------------------------
    total_steps = 4 if no_diarize else 5
    t0 = _step(f"1/{total_steps} download")
    if audio_path.exists() and not force:
        steps_cached.append("download")
        print(f"  cached: {audio_path}")
    else:
        audio_path = download_step.download_audio(url, work_dir)
        steps_run.append("download")
        print(f"  -> {audio_path}")
    _done(t0)

    # --- 2. transcribe -----------------------------------------------------
    t0 = _step(f"2/{total_steps} transcribe  model={model} lang={language or 'auto'}")
    if transcript_path.exists() and not force:
        steps_cached.append("transcribe")
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
        steps_run.append("transcribe")
        print(f"  detected language: {result.get('language')}")
        print(f"  segments: {len(result.get('segments', []))}")
    _done(t0)

    # --- 3. dedupe ---------------------------------------------------------
    t0 = _step(f"3/{total_steps} dedupe")
    if raw_backup.exists() and not force:
        steps_cached.append("dedupe")
        print(f"  cached (backup exists): {raw_backup}")
    else:
        raw_text = transcript_path.read_text()
        transcript = json.loads(raw_text)
        n_before = sum(len(s.get("words") or []) for s in transcript.get("segments", []))
        cleaned = dedupe_step.dedupe_transcript(transcript)
        n_after = sum(len(s.get("words") or []) for s in cleaned.get("segments", []))
        raw_backup.write_text(raw_text)
        transcript_path.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False))
        text_path.write_text(cleaned["text"].strip() + "\n")
        steps_run.append("dedupe")
        print(f"  words: {n_before} -> {n_after} (collapsed {n_before - n_after})")
    _done(t0)

    # Always (re)write the timed text from the current (post-dedupe) transcript.
    transcript = json.loads(transcript_path.read_text())
    _write_timed_text(transcript, timed_path)

    if no_diarize:
        print("\nDiarization skipped (--no-diarize).")
        final_files = [text_path, timed_path]
    else:
        # --- 4. diarize ----------------------------------------------------
        t0 = _step(f"4/{total_steps} diarize")
        if diarization_path.exists() and not force:
            steps_cached.append("diarize")
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
            steps_run.append("diarize")
            print(f"  speakers: {len(speakers)} ({', '.join(speakers)})")
        _done(t0)

        # --- 5. merge ------------------------------------------------------
        t0 = _step(f"5/{total_steps} merge")
        diarization = json.loads(diarization_path.read_text())
        utterances = merge_step.merge(transcript, diarization)
        merge_step.write_outputs(audio_path, utterances)
        steps_run.append("merge")
        print(f"  utterances: {len(utterances)}")
        _done(t0)

        final_files = [text_path, timed_path, diarized_txt]

    # --- copy final outputs to user-facing directory -----------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in final_files:
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)

    # --- summary -----------------------------------------------------------
    n_words = sum(len(s.get("words") or []) for s in transcript.get("segments", []))
    dedupe_collapsed: int | None = None
    if raw_backup.exists():
        raw = json.loads(raw_backup.read_text())
        n_raw = sum(len(s.get("words") or []) for s in raw.get("segments", []))
        if n_raw != n_words:
            dedupe_collapsed = n_raw - n_words

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Source:        {url}")
    print(f"Work dir:      {work_dir}")
    print(f"Audio:         {audio_path}")
    print(f"Transcript:    {transcript_path}")
    if not no_diarize:
        print(f"Diarization:   {diarization_path}")
    print(f"Steps run:     {', '.join(steps_run) or '—'}")
    print(f"Cached:        {', '.join(steps_cached) or '—'}")
    print(f"Language:      {transcript.get('language', 'unknown')}")
    print(f"Segments:      {len(transcript.get('segments', []))}")
    if dedupe_collapsed:
        print(f"Words:         {n_words} (dedupe collapsed {dedupe_collapsed})")
    else:
        print(f"Words:         {n_words}")
    if not no_diarize and diarized_json.exists():
        utterances = json.loads(diarized_json.read_text())
        speakers = sorted({u["speaker"] for u in utterances})
        print(f"Speakers:      {len(speakers)} ({', '.join(speakers)})")
        print(f"Utterances:    {len(utterances)}")
    print()
    print(f"Final outputs (copied to {output_dir}):")
    for p in copied:
        print(f"  {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", help="Source URL (e.g., a YouTube video)")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path.cwd(),
        help="Where to drop final transcripts. Default: current working directory.",
    )
    parser.add_argument("-m", "--model", default=transcribe_step.DEFAULT_PRESET,
                        help=f"Whisper preset or HF repo (default: {transcribe_step.DEFAULT_PRESET})")
    parser.add_argument("-l", "--language", default=None,
                        help="Force language code (e.g., 'hi'). Default: auto-detect.")
    parser.add_argument("--compression-ratio-threshold", type=float, default=None)
    parser.add_argument("--hallucination-silence-threshold", type=float, default=None)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--no-diarize", action="store_true",
                        help="Skip diarization (steps 4 + 5). Output is plain + timed text only.")
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
        no_diarize=args.no_diarize,
        force=args.force,
    )


if __name__ == "__main__":
    main()
