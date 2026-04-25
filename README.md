# YouTube Video Summarizer

This is a small pipeline I built to turn a YouTube URL into a speaker-attributed transcript and a structured Markdown summary. It runs entirely on my own machine with no API keys, no cloud calls and no proprietary models.

Every model in the pipeline is open-weight and swappable: 
- `yt-dlp` grabs the audio, 
- Whisper (large-v3/large-v3-turbo) does the transcription (`mlx-whisper` on Apple Silicon, `faster-whisper` on Linux/CUDA), 
- `pyannote.audio` handles speaker diarization
- `llama.cpp` runs Gemma 4 31B for the final summary

The main use case I have in mind is to summarize podcast-style discussions either in English or Hindi + English/Hinglish (code-switched) long-form conversations with a readable transcript with speaker labels and a quick structured summary.

## Pipeline

```
                         ┌─► transcribe ─► dedupe ─┐
URL ─► download ─────────┤   (whisper)    (loops)  ├─► merge ─► <id>.diarized.txt
 │     (yt-dlp)          │                         │   (align)            │
 │                       └─► diarize ──────────────┘                      │
 │                           (pyannote)                                   │  [transcript]
 ▼                                                                        │
<id>.description.txt                                                      │
 │                                                                        │
 ▼                                                                        │
extract episode ctx ──► <id>.episode_context.md ──┐                       │
(llama.cpp call 1:                                │  [grounding]          │
 Gemma 4 31B, reasoning off)                      ▼                       ▼
                                            ┌─────────────────────────────────┐
                                            │ summarize                       │
                                            │ (llama.cpp call 2:              │ ──► <id>.diarized.summary.md
                                            │  Gemma 4 31B, reasoning on)     │
                                            └─────────────────────────────────┘
```

Here is what each step is doing:

1. **Download**: `yt-dlp` pulls two things off YouTube: the audio track and the video description. I grab the description too because it is free context with the episode title, the host and guests names are almost always in there.

2. **Transcribe**: Whisper turns the audio into text with timestamps. This is the heavy step and does most of the actual "speech to text" work.

3. **Dedupe**: Whisper sometimes gets stuck in a loop and repeats the same phrase over and over, especially during silences or background music. This step spots those loops and collapses them so the transcript reads cleanly.

4. **Diarize**: While transcription is happening, `pyannote` looks at the same audio and works out *who* is speaking *when*. It just works out the speaker turns, not the words (e.g. speaker A from 0:00 to 0:12, speaker B from 0:12 to 0:18).

5. **Merge**: I line up the words from step 3 with the speaker turns from step 4 so every line of the transcript knows who said it. The result is a readable `[mm:ss - mm:ss] SPEAKER_xx: text` file, the main transcript output.

6. **Summarize**: This is where the local LLM (Gemma 4 31B, run through `llama.cpp`) comes in. I spin up `llama-server` once and make two back-to-back calls against the same loaded model:
   - **Episode context first**: I hand the raw YouTube description to the model and ask it to pull out the episode title, the host and guests names, and any other notable named entities into a small cheat-sheet (`<id>.episode_context.md`).
   - **Full summary next**: I prepend that episode context to the speaker-attributed transcript and ask the model to write a structured summary that includes TL;DR, key points, chapters with timestamps, takeaways, important quotes and resources. Note, having the episode context in front of it helps the model label speakers correctly and spell names right, especially useful for Hindi names that it would otherwise butcher. It also helps the model to ground the named entities in the transcript.

You end up with four files in your working directory:

- `<id>.txt` — plain deduped transcript text
- `<id>.timed.txt` — transcript with timestamps
- `<id>.diarized.txt` — speaker-attributed transcript
- `<id>.diarized.summary.md` — the structured summary

All the intermediate files live under `/tmp/summarize-video-<id>/`, keyed by the YouTube video id. Re-running the same URL picks up whatever is already on disk, so finished steps are skipped; pass `-f` to force a full re-run. 

> For the deeper notes — per-step flags, backend trade-offs, known quirks and why each choice was made — see [`docs/pipeline.md`](docs/pipeline.md).

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
hf download unsloth/gemma-4-31B-it-GGUF gemma-4-31B-it-UD-Q4_K_XL.gguf --local-dir ~/MODELS/unsloth/gemma-4-31B-it-GGUF

# 5. Transcribe + diarize + summarize a video
LD_LIBRARY_PATH= uv run python summarize_video.py "https://www.youtube.com/watch?v=HeAGWTgi4sU" -l en \
  --llama-server-bin ~/llama.cpp/build/bin/llama-server
```

The `LD_LIBRARY_PATH=` prefix is only needed if the system has an older cuDNN installed (common when a CUDA Toolkit is present), drop it otherwise.

### macOS (Apple Silicon)

_TBD — need to re-run the pipeline on a Mac and capture the exact steps before writing this up._

### Example commands

**English panel** (~31 min, 3 speakers):

```bash
LD_LIBRARY_PATH= uv run python summarize_video.py \
  "https://www.youtube.com/watch?v=02YLwsCKUww" \
  -l en --num-speakers 3 \
  --llama-server-bin ~/llama.cpp/build/bin/llama-server
```

- `-l en` tells Whisper the language up front so it doesn't mis-detect on the first audio chunk.
- `--num-speakers 3` hands `pyannote` the speaker count; the turns come out a lot cleaner than letting it guess.

**Hindi-English code-switched** (~8 min, 2 speakers):

```bash
LD_LIBRARY_PATH= uv run python summarize_video.py \
  "https://www.youtube.com/watch?v=HeAGWTgi4sU" \
  -m v3 -l hi --num-speakers 2 \
  --compression-ratio-threshold 2.0 \
  --hallucination-silence-threshold 2.0 \
  --llama-server-bin ~/llama.cpp/build/bin/llama-server
```

- `-m v3` switches Whisper to the full `large-v3` model. The default `turbo` is English-tuned and misdetects Hindi as English on this clip.
- `-l hi` forces Hindi, Whisper then writes Hindi words in Devanagari and English words in Latin script which is the "Hinglish" output I want.
- `--compression-ratio-threshold 2.0` and `--hallucination-silence-threshold 2.0` are tighter thresholds that help Whisper catch repetition loops and silence-filler hallucinations earlier. Code-switched audio is especially prone to both.
- `--num-speakers 2` for the two-speaker interview.

### Other things you might want to do

- **Run just one step**: every step is a standalone module you can call on its own, e.g. `uv run python -m steps.transcribe <file>.m4a`. It is handy when you want to iterate on one stage without re-running everything.
- **Skip the summary**: pass `--no-summarize` to just get the transcripts. `llama-server` won't even spawn, so no Gemma / llama.cpp setup is needed at all.
- **Skip the description cheat-sheet**: pass `--no-episode-context` to keep the summary but drop the grounding from the YouTube description.
- **Skip speaker diarization**: pass `--no-diarize` if you don't care who said what. The pipeline finishes a lot faster: no `pyannote` download, no 16 kHz resample, no merge step.
- **Force a full re-run**: pass `-f` to ignore the cached intermediates under `/tmp/summarize-video-<id>/` and redo every step from scratch.
- **Write outputs somewhere else**: pass `-o DIR` to land the final files in a different folder instead of the working directory.

> You can find the full flag list and per-step options in [`docs/pipeline.md#parameters`](docs/pipeline.md#parameters), backend trade-offs and the reasoning behind each default in [`docs/pipeline.md#design-rationale`](docs/pipeline.md#design-rationale).

## Benchmarks

**Linux + CUDA**: RTX 4090 box (24 GB, i7-12700KF, 32 GB RAM, Ubuntu 24.04)

| case | audio | transcribe | diarize | summarize | total wall | realtime |
|---|---|---|---|---|---|---|
| en (turbo, 3 speakers) | 31m 11s | 110.5s | 39.2s | 96.0s | **4m 22s** | 7.1× |
| hi (v3, 2 speakers)    | 8m 14s  | 87.8s  | 24.6s | 66.6s | **3m 15s** | 2.5× |

Mac (Apple Silicon) numbers TBD — need to re-run on a Mac.

Transcribe is usually the biggest step, summarize is close behind, diarize scales with audio length. Per-step wall times can swing 10–25% run-to-run on this box (GPU thermal state, background load) — treat as a single-run snapshot, not a tight average.

> Reasoning behind these defaults and how to re-run the canonical cases lives in [`docs/pipeline.md`](docs/pipeline.md#benchmarks).

## Repo structure

```
summarize_video.py        # orchestrator — URL to all outputs
steps/                    # one module per step, each runnable via `python -m steps.<name>`
  download.py             # 1. yt-dlp -> <id>.m4a + <id>.description.txt
  transcribe.py           # 2. whisper -> <id>.json + <id>.txt (per-word timestamps)
  dedupe.py               # 3. collapse whisper repetition loops
  diarize.py              # 4. pyannote -> <id>.diarization.json + <id>.rttm
  merge.py                # 5. align words × speaker turns -> <id>.diarized.txt
  summarize.py            # 6. llama.cpp + Gemma 4 31B -> <id>.diarized.summary.md
benchmark.sh              # canonical cold-cache timing runs (./benchmark.sh en | hi | all)
benchmark/                # per-run output + logs (gitignored)
docs/
  pipeline.md             # deep-dive: steps, flags, llama-server setup, design rationale, troubleshooting, benchmarks
  definitions.md          # glossary (Whisper, MLX, DTW, diarization, …)
  experiments.md          # things I tried and what I learned
pyproject.toml / uv.lock  # uv project (pins faster-whisper + torch+cu128 on Linux)
```

The orchestrator stages all intermediates in `/tmp/summarize-video-<id>/` (keyed by video id) and copies the four final files into the current working directory (or `-o DIR`). `python -m steps.<name>` invocations default to reading/writing in `downloads/` — useful when iterating on a single step.

## TODO

- [ ] **wav2vec2 forced alignment**: replace Whisper's DTW word timestamps with phoneme-level alignment to fix word-leakage at speaker turn changes.
- [ ] **`no_repeat_ngram_size` on the faster backend**: CTranslate2 exposes it; wiring it through would prevent repetition loops at decode time rather than after the fact in `dedupe.py`.
- [ ] **User notes as extra summary context**: while listening I often jot down my own notes and timestamps. Feeding them into step 6 alongside the episode-context cheat-sheet would let the summary lean on what I actually cared about.

> Details for each in [`docs/pipeline.md`](docs/pipeline.md#todos).
