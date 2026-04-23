# podcasts-transcribe

Local Mac pipeline that turns a podcast URL into a speaker-attributed
transcript. Uses `yt-dlp` for audio, `mlx-whisper` for transcription, and
`pyannote.audio` for diarization — all running on-device.

## Pipeline

```
                  ┌─► transcribe ─► dedupe ─┐
URL ─► download ──┤   (mlx-whisper)  (loops) ├─► merge ─► <id>.diarized.txt
   (yt-dlp)       │                          │
                  └─► diarize ───────────────┘
                       (pyannote)
```

Final output: `[mm:ss - mm:ss] SPEAKER_xx: utterance` lines.

## Setup

Requires `uv`, `ffmpeg`, and an `HF_TOKEN` for pyannote.

```bash
brew install ffmpeg
uv sync

export HF_TOKEN=hf_xxx...
# accept the gates on each pyannote model page (one-time):
#   https://huggingface.co/pyannote/speaker-diarization-3.1
#   https://huggingface.co/pyannote/segmentation-3.0
#   https://huggingface.co/pyannote/speaker-diarization-community-1
```

## Run

```bash
# English podcast — defaults are tuned for this; just force the language
# so Whisper doesn't waste a chunk auto-detecting.
uv run python transcribe_podcast.py "<URL>" -l en

# Hindi-English code-switched (best params we've found)
uv run python transcribe_podcast.py "<URL>" -m v3 -l hi \
  --compression-ratio-threshold 2.0 \
  --hallucination-silence-threshold 2.0

# Known speaker count helps diarization (works with any language)
uv run python transcribe_podcast.py "<URL>" -l en --num-speakers 2

# Skip diarization entirely (faster; plain + timed transcript only)
uv run python transcribe_podcast.py "<URL>" -l en --no-diarize

# Drop final transcripts in a specific folder
uv run python transcribe_podcast.py "<URL>" -l en -o ~/Documents/transcripts/
```

### Where outputs go

Intermediate files (audio, raw JSON, diarization, etc.) land in a per-URL
system temp dir (`/tmp/podcasts-<id>/`). Re-running the same URL skips any
step whose output is already in there, so re-runs are cheap and
crash-resumable. Pass `-f` to force a full re-run.

The **final transcripts** are copied into `--output-dir` (default: current
working directory):

| file | when | content |
|---|---|---|
| `<id>.txt` | always | plain deduped text |
| `<id>.timed.txt` | always | `[mm:ss - mm:ss] text` per segment |
| `<id>.diarized.txt` | unless `--no-diarize` | `[mm:ss - mm:ss] SPEAKER_xx: text` |

The pipeline prints a summary at the end with the work-dir path, steps run
vs cached, language, word/segment/speaker counts, and the final output
paths.

### Important params

| flag | when to use |
|---|---|
| `-m v3` | Non-English / accented / code-switched audio. Slower than `turbo` but multilingual quality is much better. |
| `-l <code>` | Force language (`hi`, `en`, …). Default auto-detects from the first chunk, which can mis-fire on code-switched audio. |
| `--no-diarize` | Skip steps 4 + 5. Useful when you don't care who said what (saves the pyannote download + ~minutes of compute). |
| `-o DIR` | Where final transcripts land. Default: current working directory. |
| `--compression-ratio-threshold 2.0` | Catch repetition loops earlier (default 2.4). Lower = more aggressive temperature-fallback re-decoding. |
| `--hallucination-silence-threshold 2.0` | Suppress text generated during silences > N seconds. |
| `--num-speakers N` / `--min-speakers` / `--max-speakers` | Hint the diarizer when you know the count. |

Reference test clip:
[`HeAGWTgi4sU`](https://www.youtube.com/watch?v=HeAGWTgi4sU) (~8 min,
Hindi-English).

## Run individual steps

If you want to re-run just one stage, each module is invocable directly:

```bash
uv run python -m steps.download   "<URL>"
uv run python -m steps.transcribe downloads/<id>.m4a -m v3 -l hi
uv run python -m steps.dedupe     downloads/<id>.json
uv run python -m steps.diarize    downloads/<id>.m4a
uv run python -m steps.merge      downloads/<id>.m4a
```

See [`docs/pipeline.md`](docs/pipeline.md) for what each step does, the
file formats it reads/writes, and the full flag list.

## Summarize a transcript (optional)

Once you have a `.diarized.txt` (or `.timed.txt`), generate a structured
summary with a local Gemma 4 31B running under `llama.cpp`:

```bash
# Easy: have the script start the server itself if it isn't running.
# Pays a one-time ~30-90s model load on the first call; subsequent calls
# are seconds. Server stays up between runs.
uv run python -m steps.summarize <id>.diarized.txt --auto-start
# -> <id>.diarized.summary.md

# When you're done:
uv run python -m steps.summarize --stop-server
```

Or start the server manually (full flag rationale in `docs/summarize.md`)
and call without `--auto-start`:

```bash
llama-server -m ~/models/gemma-4-31b/gemma-4-31B-it-UD-Q4_K_XL.gguf \
  -ngl 99 -c 65536 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 --batch-size 2048 --ubatch-size 1024 \
  --context-shift --metrics --jinja
uv run python -m steps.summarize <id>.diarized.txt
```

Output sections: TL;DR, Key points, Chapters (with timestamps), Main
takeaways, Important quotes, Resources. Full setup in
[`docs/summarize.md`](docs/summarize.md).

## Repo structure

```
transcribe_podcast.py     # orchestrator: URL -> diarized transcript
steps/
  download.py             # 1. yt-dlp -> downloads/<id>.m4a
  transcribe.py           # 2. mlx-whisper -> .json + .txt
  dedupe.py               # 3. collapse Whisper repetition loops
  diarize.py              # 4. pyannote -> .diarization.json + .rttm
  merge.py                # 5. words × turns -> .diarized.txt
  summarize.py            # (optional) local LLM summary via llama.cpp
docs/
  pipeline.md             # deep-dive on each step + flags
  definitions.md          # glossary (Whisper, MLX, DTW, diarization, …)
  experiments.md          # things we tried and what we learned
  summarize.md            # llama.cpp + Gemma 4 31B setup for steps/summarize.py
downloads/                # default sink for `python -m steps.*` (gitignored).
                          # The orchestrator uses /tmp/podcasts-<id>/ instead.
```

## TODO

- **`faster-whisper` backend** — beam search + `no_repeat_ngram_size` would
  prevent repetition loops at decode time instead of after the fact.
- **wav2vec2 forced alignment** — replace Whisper's DTW word timestamps
  with phoneme-level alignment to fix word-leakage at speaker turn changes.
- **Speaker name attribution** — map `SPEAKER_00` / `SPEAKER_01` / … to
  actual names.

Details for each in [`docs/pipeline.md`](docs/pipeline.md#todos).
