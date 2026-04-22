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
# English podcast
uv run python transcribe_podcast.py "<URL>"

# Hindi-English code-switched (best params we've found)
uv run python transcribe_podcast.py "<URL>" -m v3 -l hi \
  --compression-ratio-threshold 2.0 \
  --hallucination-silence-threshold 2.0

# Known speaker count helps diarization
uv run python transcribe_podcast.py "<URL>" --num-speakers 2
```

Each step is skipped if its output already exists, so re-runs are cheap and
crash-resumable. Pass `-f` to force a full re-run.

### Important params

| flag | when to use |
|---|---|
| `-m v3` | Non-English / accented / code-switched audio. Slower than `turbo` but multilingual quality is much better. |
| `-l <code>` | Force language (`hi`, `en`, …). Default auto-detects from the first chunk, which can mis-fire on code-switched audio. |
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

## Repo structure

```
transcribe_podcast.py     # orchestrator: URL -> diarized transcript
steps/
  download.py             # 1. yt-dlp -> downloads/<id>.m4a
  transcribe.py           # 2. mlx-whisper -> .json + .txt
  dedupe.py               # 3. collapse Whisper repetition loops
  diarize.py              # 4. pyannote -> .diarization.json + .rttm
  merge.py                # 5. words × turns -> .diarized.txt
docs/
  pipeline.md             # deep-dive on each step + flags
  definitions.md          # glossary (Whisper, MLX, DTW, diarization, …)
  experiments.md          # things we tried and what we learned
downloads/                # all artifacts land here (gitignored)
```

## TODO

- **`faster-whisper` backend** — beam search + `no_repeat_ngram_size` would
  prevent repetition loops at decode time instead of after the fact.
- **wav2vec2 forced alignment** — replace Whisper's DTW word timestamps
  with phoneme-level alignment to fix word-leakage at speaker turn changes.
- **Speaker name attribution** — map `SPEAKER_00` / `SPEAKER_01` / … to
  actual names.

Details for each in [`docs/pipeline.md`](docs/pipeline.md#todos).
