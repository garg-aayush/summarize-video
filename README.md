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

## Playing a downloaded clip

```bash
open downloads/<id>.m4a       # QuickTime / Music
ffplay -nodisp -autoexit downloads/<id>.m4a   # terminal playback
```
