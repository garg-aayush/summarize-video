# summarize-video

Local pipeline I built to turn a YouTube URL into a speaker-attributed transcript and a structured Markdown summary. Aimed at podcast-style discussions — English, or Hindi + English code-switched. Uses `yt-dlp` for audio, Whisper for transcription (`mlx-whisper` on Apple Silicon, `faster-whisper` on Linux/CUDA), `pyannote.audio` for diarization, and `llama.cpp` (Gemma 4 31B) for summarization — all running on-device.

## Pipeline

```
                  ┌─► transcribe ─► dedupe ─┐
URL ─► download ──┤   (mlx-whisper)  (loops) ├─► merge ─► <id>.diarized.txt
   (yt-dlp)       │                          │
                  └─► diarize ───────────────┘
                       (pyannote)
```

Final outputs: `[mm:ss - mm:ss] SPEAKER_xx: utterance` lines plus a `<id>.diarized.summary.md` (TL;DR, key points, chapters, takeaways, quotes, resources). The summarize step is opt-out via `--no-summarize`.

## Quickstart

### Linux + CUDA (RTX 4090 / similar)

Prereqs: NVIDIA driver R570+ (check with `nvidia-smi`), [`uv`](https://docs.astral.sh/uv/) installed.

```bash
# 1. System deps
sudo apt install ffmpeg git build-essential

# 2. Clone + install
git clone <repo-url> summarize-video && cd summarize-video
uv sync                      # pulls faster-whisper + torch+cu128 (~3-4 GB)

# 3. HF token for pyannote (one-time)
#    Create a read token at https://huggingface.co/settings/tokens, then
#    accept the gates on each of these pages while signed in:
#      https://huggingface.co/pyannote/speaker-diarization-3.1
#      https://huggingface.co/pyannote/segmentation-3.0
#      https://huggingface.co/pyannote/speaker-diarization-community-1
export HF_TOKEN=hf_xxx...

# 4. Build llama.cpp (CUDA) for the summarize step + download Gemma model
sudo apt install cmake libcurl4-openssl-dev
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build ~/llama.cpp/build --config Release -j$(nproc)

uv tool install huggingface_hub
mkdir -p ~/MODELS/unsloth/gemma-4-31B-it-GGUF
hf download unsloth/gemma-4-31B-it-GGUF gemma-4-31B-it-UD-Q4_K_XL.gguf \
  --local-dir ~/MODELS/unsloth/gemma-4-31B-it-GGUF

# 5. Transcribe + diarize + summarize a video
#    First run downloads ~2 GB of whisper/pyannote models.
#    Step 6 spawns llama-server, runs Gemma, and stops it (~60s overhead +
#    summarize time). Pass --no-summarize to skip if you only want the transcript.
LD_LIBRARY_PATH= uv run python summarize_video.py \
  "https://www.youtube.com/watch?v=HeAGWTgi4sU" -l en \
  --llama-server-bin ~/llama.cpp/build/bin/llama-server
```

The `LD_LIBRARY_PATH=` prefix is only needed if the system has an older cuDNN installed (common when a CUDA Toolkit is present) that shadows torch's bundled one — drop it otherwise.

### macOS (Apple Silicon)

```bash
brew install ffmpeg
git clone <repo-url> summarize-video && cd summarize-video
uv sync
export HF_TOKEN=hf_xxx...    # accept the 3 pyannote model gates as above
uv run python summarize_video.py \
  "https://www.youtube.com/watch?v=HeAGWTgi4sU" -l en
```

### Outputs (both platforms)

Three files land in the current working directory:

- `<id>.txt` — plain deduped text
- `<id>.timed.txt` — `[mm:ss - mm:ss] text`
- `<id>.diarized.txt` — `[mm:ss - mm:ss] SPEAKER_xx: text`

Intermediate artifacts (audio, JSON, 16 kHz wav, pyannote output) live in `/tmp/summarize-video-<id>/`, so re-running the same URL skips finished steps. Pass `-f` to force a full re-run.

## Linux / CUDA notes

`uv sync` on Linux pulls `faster-whisper` and a `torch+cu128` wheel (instead of `mlx-whisper`). This targets driver R570+ (CUDA 12.8). If you have R580+ you can drop to the default cu130 wheel by removing the `[tool.uv.sources]` block in `pyproject.toml`.

If the system has cuDNN installed (e.g. from the CUDA Toolkit) and it's older than torch's bundled one, pyannote will crash with `RuntimeError: cuDNN version incompatibility`. Clear `LD_LIBRARY_PATH` for the run so torch finds its own:

```bash
LD_LIBRARY_PATH= uv run python summarize_video.py "<URL>" -l en
```

## "Sign in to confirm you're not a bot"

YouTube now gates some videos behind a bot-check. If the download step fails with that message, hand yt-dlp a browser session's cookies:

```bash
# Use whichever browser you're signed in to YouTube on
uv run python summarize_video.py "<URL>" -l en --cookies-from-browser chrome
# alternatives: firefox, brave, edge, safari

# Or pass an exported Netscape-format cookie file:
uv run python summarize_video.py "<URL>" -l en --cookies ~/cookies.txt
```

On Linux, Chrome/Brave encrypt their cookie jar against the system keyring (gnome-keyring / kwallet); if yt-dlp can't unlock it, use Firefox (plaintext SQLite) or export a cookies file via a browser extension.

## More run examples

```bash
# Hindi-English code-switched (best params I've found)
uv run python summarize_video.py "<URL>" -m v3 -l hi \
  --compression-ratio-threshold 2.0 \
  --hallucination-silence-threshold 2.0

# Known speaker count helps diarization
uv run python summarize_video.py "<URL>" -l en --num-speakers 2

# Skip diarization entirely (faster; plain + timed transcript only)
uv run python summarize_video.py "<URL>" -l en --no-diarize

# Drop final transcripts in a specific folder
uv run python summarize_video.py "<URL>" -l en -o ~/Documents/transcripts/

# Force a full re-run, ignoring cached intermediates
uv run python summarize_video.py "<URL>" -l en -f
```

The pipeline prints a summary at the end with the work-dir path, steps run vs cached, language, word/segment/speaker counts, and the final output paths.

### Important params

| flag | when to use |
|---|---|
| `-m v3` | Non-English / accented / code-switched audio. Slower than `turbo` but multilingual quality is much better. |
| `-b {mlx,faster}` | Override the auto-selected backend. Default: `mlx` on Apple Silicon, `faster` on Linux. |
| `-l <code>` | Force language (`hi`, `en`, …). Default auto-detects from the first chunk, which can mis-fire on code-switched audio. |
| `--no-diarize` | Skip steps 4 + 5. Useful when you don't care who said what (saves the pyannote download + ~minutes of compute). |
| `-o DIR` | Where final transcripts land. Default: current working directory. |
| `--compression-ratio-threshold 2.0` | Catch repetition loops earlier (default 2.4). Lower = more aggressive temperature-fallback re-decoding. |
| `--hallucination-silence-threshold 2.0` | Suppress text generated during silences > N seconds. |
| `--num-speakers N` / `--min-speakers` / `--max-speakers` | Hint the diarizer when you know the count. |
| `--compute-type` | CTranslate2 compute type for the `faster` backend. Default `float16`; try `int8_float16` for lower VRAM. |
| `--no-summarize` | Skip step 6 (Gemma summary). No `llama-server` needed. |
| `--summarize-model PATH` | GGUF model used for summarization. Default: `~/MODELS/unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-UD-Q4_K_XL.gguf`. |
| `--llama-server-bin PATH` | `llama-server` binary. Default `llama-server` (PATH lookup); pass an absolute path otherwise. |
| `--summarize-server-url URL` | If a server is already running at this URL, reuse it instead of spawning. |

Reference test clip: [`HeAGWTgi4sU`](https://www.youtube.com/watch?v=HeAGWTgi4sU) (~8 min, Hindi-English).

## Benchmarks

Numbers from my box (RTX 4090 24 GB, i7-12700KF, 32 GB RAM, Ubuntu 24.04) via `./benchmark.sh all` — cold cache, `-f` forced, summarize on. Two canonical cases:

- **en** — 31-min English panel, `turbo` + 3 speakers
- **hi** — 8-min Hindi-English clip, `v3` + 2 speakers

| case | audio | transcribe | diarize | summarize | total | realtime |
|---|---|---|---|---|---|---|
| en (turbo) | 31m 11s | 104.8s | 40.7s | 75.1s | **3m 56s** | 7.9× |
| hi (v3)    | 8m 14s  | 99.5s  | 13.2s | 56.8s | **3m 4s**  | 2.7× |

Transcribe dominates on short clips; diarize scales with audio length; summarize scales with transcript length. `v3` on an 8-minute Hindi clip takes about as long as `turbo` on a 31-minute English panel — the deeper decoder (32 layers vs 4) is where the multilingual quality comes from.

Llama-server runs at `--ubatch-size 512 -c 49152` on this card (the orchestrator-aware default for 24 GB CUDA). The smaller ubatch doesn't noticeably slow the summarize step at these transcript sizes — prefill isn't the bottleneck vs server startup + decode — and buys 1.5× more context.

Re-run: `./benchmark.sh en`, `./benchmark.sh hi`, or `./benchmark.sh all`. Raw logs and per-step timings land in `benchmark/<video-id>-<platform>-<timestamp>/`.

## Run individual steps

If you want to re-run just one stage, I've made each module invocable directly:

```bash
uv run python -m steps.download   "<URL>"
uv run python -m steps.transcribe downloads/<id>.m4a -m v3 -l hi
uv run python -m steps.dedupe     downloads/<id>.json
uv run python -m steps.diarize    downloads/<id>.m4a
uv run python -m steps.merge      downloads/<id>.m4a
```

See [`docs/pipeline.md`](docs/pipeline.md) for what each step does, the file formats it reads/writes, and the full flag list.

## Summarize a transcript standalone

Step 6 runs as part of the orchestrator by default. If you already have a transcript and just want to (re-)summarize it without re-running download/transcribe/diarize, call the step directly:

```bash
# Auto-spawn llama-server (left running for re-use):
uv run python -m steps.summarize <id>.diarized.txt --auto-start \
  --server-bin ~/llama.cpp/build/bin/llama-server
# -> <id>.diarized.summary.md

# Stop the auto-started server:
uv run python -m steps.summarize --stop-server
```

Or start `llama-server` manually (recipe in [`docs/summarize.md`](docs/summarize.md)) and call without `--auto-start`. Output sections: TL;DR, Key points, Chapters (with timestamps), Main takeaways, Important quotes, Resources.

## Repo structure

```
summarize_video.py        # orchestrator: URL -> diarized transcript
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
  experiments.md          # things I tried and what I learned
  summarize.md            # llama.cpp + Gemma 4 31B setup for steps/summarize.py
downloads/                # default sink for `python -m steps.*` (gitignored).
                          # The orchestrator uses /tmp/summarize-video-<id>/ instead.
```

## TODO

- **wav2vec2 forced alignment** — replace Whisper's DTW word timestamps with phoneme-level alignment to fix word-leakage at speaker turn changes.
- **Speaker name attribution** — map `SPEAKER_00` / `SPEAKER_01` / … to actual names.
- **`no_repeat_ngram_size` on the faster backend** — CTranslate2 exposes it; wiring it through would prevent repetition loops at decode time rather than after the fact in `dedupe.py`.

Details for each in [`docs/pipeline.md`](docs/pipeline.md#todos).
