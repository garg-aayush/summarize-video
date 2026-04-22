# podcasts-transcribe

Local pipeline for downloading podcast audio and producing transcripts with
speaker diarization. Runs on a Mac.

## Setup

Requires `uv` and `ffmpeg` (for audio extraction).

```bash
brew install ffmpeg   # if not already installed
uv sync
```

## Step 1 — Download audio

Pulls the best-available audio from a URL via `yt-dlp` and writes an `.m4a`
into `downloads/`.

```bash
uv run python main.py "<URL>"
```

### Reference test clip

Use this short YouTube clip for end-to-end testing of the pipeline:

```
https://www.youtube.com/watch?v=HeAGWTgi4sU
```

Example:

```bash
uv run python main.py "https://www.youtube.com/watch?v=HeAGWTgi4sU"
# -> downloads/HeAGWTgi4sU.m4a (~8 min, stereo AAC ~129 kbps)
```

### YouTube extractor notes

YouTube has been pushing hard against `yt-dlp`. Two recent failure modes we
hit and the workaround we settled on:

- `ios` / `web` / `mweb` clients now require a GVS PO Token and skip all
  formats without one.
- `tv` client is currently flagged with a session-level DRM experiment
  (yt-dlp issue #12563) that marks all formats as DRM-protected.

We use `web_embedded` + `android_vr` clients, which still serve plain m4a
audio formats without PO tokens and are not affected by the DRM experiment.
If those break in the future, probe available clients with:

```bash
uv run yt-dlp --extractor-args "youtube:player_client=tv_simply,web_embedded,android_vr" -F "<URL>"
```

## Step 2 — Transcribe (mlx-whisper)

Runs Whisper locally via Apple MLX. Two presets:

| preset | model | size | when |
|---|---|---|---|
| `turbo` (default) | `mlx-community/whisper-large-v3-turbo` (fp16) | ~1.6 GB | English, fast |
| `v3` | `mlx-community/whisper-large-v3-mlx-8bit` | ~1.6 GB | multilingual / accented / noisy |

Turbo is the distilled 4-decoder large-v3-turbo and is English-tuned — its
multilingual quality is markedly worse than `v3`. Use `v3` for non-English
or code-switched audio (we observed turbo mis-detecting our Hindi-English
test clip as English; v3 detected Hindi).

```bash
uv run python transcribe.py downloads/<id>.m4a              # turbo
uv run python transcribe.py downloads/<id>.m4a -m v3        # full v3
uv run python transcribe.py downloads/<id>.m4a -l en        # force language
```

Writes:
- `downloads/<id>.txt` — plain text
- `downloads/<id>.json` — full result with segments and per-word timestamps

Per-word timestamps come from Whisper's cross-attention DTW
(`word_timestamps=True`); they're decent but a future step will tighten them
with wav2vec2 forced alignment.

## Playing a downloaded clip

```bash
open downloads/<id>.m4a       # QuickTime / Music
ffplay -nodisp -autoexit downloads/<id>.m4a   # terminal playback
```
