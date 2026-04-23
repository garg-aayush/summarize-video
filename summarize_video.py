"""End-to-end pipeline: YouTube URL -> diarized transcript + Gemma summary.

Tuned for podcast-style videos (English or Hindi+English code-switched
discussions). Six steps:
  1. download   (steps.download.download_audio)
  2. transcribe (steps.transcribe.transcribe)
  3. dedupe     (steps.dedupe.dedupe_transcript)
  4. diarize    (steps.diarize.diarize)              [skipped with --no-diarize]
  5. merge      (steps.merge.merge)                  [skipped with --no-diarize]
  6. summarize  (steps.summarize.summarize)          [skipped with --no-summarize]

Step 6 requires a running llama-server. The orchestrator pre-flights the
server at startup and refuses to run if it's unreachable — start it
manually (see docs/summarize.md) or pass --no-summarize.

Intermediate files land in a per-URL system temp dir
(`/tmp/summarize-video-<id>/...`), so re-running the same URL skips
already-done steps. Final outputs are copied into the output directory
(default: current working directory).

Always produced:
  <id>.txt         plain deduped text
  <id>.timed.txt   `[mm:ss - mm:ss] text` per segment

With diarization:
  <id>.diarized.txt   `[mm:ss - mm:ss] SPEAKER_xx: text`

With summarization:
  <id>.diarized.summary.md  (or <id>.timed.summary.md under --no-diarize)
"""

import argparse
import gc
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import yt_dlp

from steps import download as download_step
from steps import transcribe as transcribe_step
from steps import dedupe as dedupe_step
from steps import diarize as diarize_step
from steps import merge as merge_step
from steps import summarize as summarize_step


def _step(name: str):
    print(f"\n=== {name} ===")
    return time.perf_counter()


def _done(t0: float) -> None:
    print(f"  ({time.perf_counter() - t0:.1f}s)")


def _free_gpu() -> None:
    """Release whisper/pyannote GPU references so llama-server has the
    full card to itself. Gemma 4 31B Q4_K_XL at 64K context sits within
    ~500 MB of the 24 GB limit on a 4090, so cached PyTorch blocks or
    lingering CT2 contexts will OOM the model load.

    Order matters: synchronize first (drain pending CUDA work), then two
    gc passes (first drops refs, second breaks cycles the first uncovered),
    then empty_cache to return freed blocks to the driver. Logs the
    before/after so failures are debuggable."""
    try:
        import torch
        if not torch.cuda.is_available():
            gc.collect()
            return
    except ImportError:
        gc.collect()
        return

    before_free, total = torch.cuda.mem_get_info()
    torch.cuda.synchronize()
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    after_free, _ = torch.cuda.mem_get_info()
    freed_mb = (after_free - before_free) / 1024 / 1024
    free_mb = after_free / 1024 / 1024
    total_mb = total / 1024 / 1024
    print(f"  freed {freed_mb:+.0f} MiB, {free_mb:.0f}/{total_mb:.0f} MiB now free")


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


def _resolve_video_id(
    url: str,
    cookies_from_browser: str | None = None,
    cookiefile: Path | None = None,
) -> str:
    """Ask yt-dlp for the URL's id without downloading audio.
    Lets us pick a stable per-URL work dir before step 1 actually runs.

    Uses the same extractor_args and UA as the download step so this
    resolve call doesn't trip YouTube's bot check with the default client
    (which happens even when cookies are supplied, because the default
    `web`/`mweb` clients are the ones currently gated)."""
    opts: dict = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "extractor_args": {"youtube": {"player_client": ["web_embedded", "android_vr"]}},
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        },
        "nocheckcertificate": True,
    }
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if cookiefile:
        opts["cookiefile"] = str(cookiefile)
    with yt_dlp.YoutubeDL(opts) as ydl:
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
    backend: str | None = None,
    compute_type: str = "float16",
    cookies_from_browser: str | None = None,
    cookiefile: Path | None = None,
    summarize: bool = True,
    summarize_server_url: str = summarize_step.DEFAULT_SERVER,
    summarize_model: Path = summarize_step.DEFAULT_MODEL,
    llama_server_bin: str = summarize_step.DEFAULT_SERVER_BIN,
    summarize_server_wait_timeout: int = 180,
) -> None:
    # Pre-flight: if we'll need to spawn llama-server later, fail now if the
    # binary or model doesn't exist — better than running 10 min of pipeline
    # then hitting it.
    if summarize and not summarize_step._server_alive(summarize_server_url):
        if not summarize_model.exists():
            print(
                f"ERROR: --summarize-model not found: {summarize_model}\n"
                f"Pass --summarize-model PATH or --no-summarize.",
                file=sys.stderr,
            )
            sys.exit(2)
        if shutil.which(llama_server_bin) is None and not Path(llama_server_bin).exists():
            print(
                f"ERROR: llama-server binary not found: {llama_server_bin}\n"
                f"Pass --llama-server-bin PATH or --no-summarize.\n"
                f"Build instructions: docs/summarize.md.",
                file=sys.stderr,
            )
            sys.exit(2)

    print(f"Resolving {url}")
    video_id = _resolve_video_id(url, cookies_from_browser, cookiefile)
    work_dir = Path(tempfile.gettempdir()) / f"summarize-video-{video_id}"
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
    total_steps = (4 if no_diarize else 5) + (1 if summarize else 0)
    t0 = _step(f"1/{total_steps} download")
    if audio_path.exists() and not force:
        steps_cached.append("download")
        print(f"  cached: {audio_path}")
    else:
        audio_path = download_step.download_audio(
            url, work_dir,
            cookies_from_browser=cookies_from_browser,
            cookiefile=cookiefile,
        )
        steps_run.append("download")
        print(f"  -> {audio_path}")
    _done(t0)

    # --- 2. transcribe -----------------------------------------------------
    resolved_backend = transcribe_step.resolve_backend(backend)
    t0 = _step(f"2/{total_steps} transcribe  backend={resolved_backend} model={model} lang={language or 'auto'}")
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
            backend=resolved_backend,
            compute_type=compute_type,
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
            # Drop the pipeline so its GPU memory is reclaimable before
            # llama-server tries to load 22 GB of weights at step 6.
            del pipeline
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

    # --- 6. summarize (optional) ------------------------------------------
    summary_path: Path | None = None
    if summarize:
        # Prefer the diarized transcript for summarization; fall back to
        # the timed text when --no-diarize.
        summary_input = diarized_txt if not no_diarize else timed_path
        summary_path = summary_input.with_suffix(".summary.md")
        t0 = _step(f"{total_steps}/{total_steps} summarize")
        if summary_path.exists() and not force:
            steps_cached.append("summarize")
            print(f"  cached: {summary_path}")
        else:
            # Free GPU before loading 22 GB of Gemma — whisper/pyannote
            # cached blocks would push us into OOM on a 24 GB card.
            _free_gpu()
            spawned_here = False
            try:
                if summarize_step._server_alive(summarize_server_url):
                    print(f"  reusing existing server at {summarize_server_url}")
                else:
                    print(f"  spawning llama-server (model load takes 30-90s)...")
                    summarize_step._ensure_server(
                        summarize_server_url,
                        summarize_model,
                        server_cmd=None,
                        wait_timeout=summarize_server_wait_timeout,
                        server_bin=llama_server_bin,
                    )
                    spawned_here = True
                summarize_step.summarize(
                    summary_input,
                    output=summary_path,
                    server_url=summarize_server_url,
                    auto_start=False,
                )
                steps_run.append("summarize")
                print(f"  -> {summary_path}")
            finally:
                # Only stop the server if we started it. If the user had
                # one running externally, leave it alone.
                if spawned_here:
                    print("  stopping llama-server...")
                    summarize_step.stop_server(quiet=True)
        _done(t0)
        final_files.append(summary_path)

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
    if summary_path is not None:
        print(f"Summary:       {summary_path}")
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
    parser.add_argument("-b", "--backend", choices=sorted(transcribe_step.MODEL_PRESETS.keys()), default=None,
                        help=f"Transcription backend. Default: {transcribe_step.DEFAULT_BACKEND} (platform default).")
    parser.add_argument("--compute-type", default="float16",
                        help="CTranslate2 compute type (faster backend only). Default: float16.")
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
    parser.add_argument(
        "--cookies-from-browser", default=None,
        help=("Pull cookies from a local browser (e.g. 'firefox', 'chrome', "
              "'brave', 'edge', 'safari') for yt-dlp. Use when YouTube asks "
              "to 'Sign in to confirm you're not a bot'."),
    )
    parser.add_argument(
        "--cookies", type=Path, default=None,
        help="Path to an exported Netscape-format cookies file for yt-dlp.",
    )
    parser.add_argument(
        "--no-summarize", action="store_true",
        help="Skip the summarize step (step 6). Runs transcription + "
             "diarization only; no llama-server needed.",
    )
    parser.add_argument(
        "--summarize-model", type=Path, default=summarize_step.DEFAULT_MODEL,
        help=f"GGUF model path that llama-server will load for summarization. "
             f"Default: {summarize_step.DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--llama-server-bin", default=summarize_step.DEFAULT_SERVER_BIN,
        help=f"llama-server binary (name on PATH or absolute path). "
             f"Default: {summarize_step.DEFAULT_SERVER_BIN}.",
    )
    parser.add_argument(
        "--summarize-server-url", default=summarize_step.DEFAULT_SERVER,
        help=f"llama-server URL (default: {summarize_step.DEFAULT_SERVER}). "
             f"If a server is already up at this URL, the orchestrator reuses "
             f"it (and leaves it running on completion); otherwise it spawns "
             f"one after step 5 and stops it after step 6.",
    )
    parser.add_argument(
        "--summarize-server-wait-timeout", type=int, default=180,
        help="Seconds to wait for spawned llama-server to be ready (default 180).",
    )
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
        backend=args.backend,
        compute_type=args.compute_type,
        cookies_from_browser=args.cookies_from_browser,
        cookiefile=args.cookies,
        summarize=not args.no_summarize,
        summarize_server_url=args.summarize_server_url,
        summarize_model=args.summarize_model,
        llama_server_bin=args.llama_server_bin,
        summarize_server_wait_timeout=args.summarize_server_wait_timeout,
    )


if __name__ == "__main__":
    main()
