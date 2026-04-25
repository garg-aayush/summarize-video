# Pipeline deep-dive

This is the detailed documentation of the pipeline. It is a companion to the README.

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

There are six steps, each a standalone module under `steps/`. The orchestrator `summarize_video.py` chains them and skips any step whose output already exists on disk. Note: **Every step is also runnable on its own** - useful when iterating on one stage without re-running everything.

### 1. Download (`steps/download.py`)

`yt-dlp` pulls two things off YouTube: *the audio track* and *the video description*. I grab the description because it is free context that includes the episode title, the host and guest names and named entities that the summarizer can use later to ground anonymous `SPEAKER_xx` labels into real names.

Outputs: `<id>.m4a` (stereo AAC) and `<id>.description.txt` (the video description).

### 2. Transcribe (`steps/transcribe.py`)

Whisper turns the audio into text with per-word timestamps. I have chosen two backends as I have a mac and RTX 4090 GPU: `mlx-whisper` on Apple Silicon (Metal), `faster-whisper` (CTranslate2, the optimised C++ runtime for transformer models) on Linux/CUDA. These two backends have two presets that share names across backends but resolve to different HF repos: `turbo` (default, English-tuned, fast) and `v3` (full multilingual large-v3, slower but much better on non-English like Hindi).

Outputs: `<id>.txt` (plain text) and `<id>.json` (segments + per-word timestamps).

Here, the per-word timestamps come from Whisper's cross-attention DTW (Dynamic Time Warping — a way of asking "which slice of audio was the model attending to when it wrote this word?"). This is cheap, since there's no extra model but the boundaries drift by tens of milliseconds around speaker changes. See the [wav2vec2 forced alignment TODO](#wav2vec2-forced-alignment) for the planned fix.

### 3. Dedupe (`steps/dedupe.py`)

Whisper's greedy decoder occasionally falls into "attractors" and emits a short n-gram many times in a row (`thank you thank you ...×30`, `सबसे सबसे ...×30`). This step collapses them in the transcript JSON. It has 2 passes: within each segment's word list (catches loops emitted as one chunk), then across segments after dropping zero-duration empties left by Whisper's silence guard. The collapser scores every n-gram length by total coverage (`n × repeats`) and picks the highest, so a 1-gram repeated 22 times beats a 4-gram repeated 5 times.

It backs up the original up to `<id>.raw.json` and overwrites `<id>.json` + `<id>.txt` with the cleaned version.

### 4. Diarize (`steps/diarize.py`)

`pyannote/speaker-diarization-3.1` (via `pyannote.audio` 4.x) splits the audio into speaker turns: `SPEAKER_00 from 0.0 to 4.2s, SPEAKER_01 from 4.2 to 7.8s, ...`. Here, the speakers are anonymous - `SPEAKER_00` is just an arbitrary cluster, not a person's name. Names come later from the episode-context cheat-sheet at step 6.

One-time setup: get an HF read token and accept the three model gates (linked in the README quickstart).

Outputs: `<id>.diarization.json` (both `diarization` which may overlap when speakers talk over each other, and `exclusive_diarization` the non-overlapping version used downstream), `<id>.rttm` (standard diarization format for evaluation), `<id>.16k.wav` (16 kHz mono cache that pyannote needs).

### 5. Merge (`steps/merge.py`)

Lines up Whisper's words with pyannote's speaker turns. Each word goes to the speaker with the largest interval overlap against `exclusive_diarization` (snap to nearest turn for words in silence gaps) then consecutive same-speaker words get grouped into utterances.

Outputs: `<id>.diarized.txt` (`[mm:ss - mm:ss] SPEAKER_xx: text`, the main transcript) and `<id>.diarized.json` (same data with full word lists).

### 6. Summarize (`steps/summarize.py`)

This step takes the diarized transcript and produces a structured Markdown summary using local Gemma 4 31B through `llama.cpp`. Two back-to-back calls against the same loaded model with extraction first and main summary second. You can find a deep-dive in [Summarize step](#summarize-step), one-time setup in [llama-server setup](#llama-server-setup).

Outputs: `<id>.episode_context.md` and `<id>.diarized.summary.md`.

## Parameters

You need to pass these parameters to the orchestrator (`summarize_video.py`). They are forwarded to the right step.

### Download (step 1)

| flag | what it does |
|---|---|
| `--cookies-from-browser NAME` | Forward signed-in YouTube cookies from a local browser (`firefox`, `chrome`, `brave`, `edge`, `safari`). Useful when YouTube nudges with "Sign in to confirm you're not a bot." |
| `--cookies PATH` | Same idea, but from an exported Netscape-format cookie file. |

### Transcribe (step 2)

| flag | what it does |
|---|---|
| `-m {turbo,v3}` | Whisper preset. Default `turbo`. Switch to `v3` for non-English / accented / code-switched audio (see [Why Whisper turbo vs v3](#why-whisper-turbo-vs-v3)). |
| `-l <code>` | Force language (`hi`, `en`, ...). Default auto-detects from the first chunk, which can mis-fire on code-switched audio. |
| `-b {mlx,faster}` | Override the auto-selected backend. Default: `mlx` on Apple Silicon, `faster` on Linux. |
| `--initial-prompt` | Bias the first chunk's vocabulary. Often hurts more than it helps — see [Whisper decoder pitfalls](#whisper-decoder-pitfalls). |
| `--compression-ratio-threshold 2.0` | Catch repetition loops earlier (default 2.4). Lower means more aggressive temperature-fallback re-decoding. |
| `--hallucination-silence-threshold 2.0` | Suppress text generated during silences longer than N seconds. |
| `--temperature` | One value (e.g. `0.0`) or the full fallback ladder. |
| `--beam-size` | `faster` backend: default 5. mlx-whisper 0.4.3 is greedy-only and raises `NotImplementedError`. |
| `--compute-type` | `faster` backend only. CTranslate2 compute type. Default `float16`; try `int8_float16` for lower VRAM. |

### Diarize (step 4)

| flag | what it does |
|---|---|
| `--no-diarize` | Skip steps 4 + 5. Saves the pyannote download and a few minutes of compute when I don't care who said what. |
| `--num-speakers N` | Exact speaker count. Big quality win when I know it. |
| `--min-speakers N` / `--max-speakers N` | Bound the count when I'm not sure of the exact number. |

### Summarize (step 6)

| flag | what it does |
|---|---|
| `--no-summarize` | Skip step 6 entirely. `llama-server` isn't spawned. |
| `--no-episode-context` | Keep step 6 but skip the extraction call. Useful when the description is a pure SEO keyword dump. |
| `--summarize-model PATH` | GGUF model used for summarization. Default: `~/MODELS/unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-UD-Q4_K_XL.gguf`. |
| `--llama-server-bin PATH` | `llama-server` binary. Default `llama-server` (PATH lookup); pass an absolute path otherwise. |
| `--summarize-server-url URL` | Reuse a llama-server already running at this URL instead of spawning one. |

### Orchestrator (overall)

| flag | what it does |
|---|---|
| `-o DIR` | Where final transcripts land. Default: current working directory. |
| `-f` | Force a full re-run, ignoring cached intermediates under `/tmp/summarize-video-<id>/`. |

## Running individual steps

Every step is also a module I can call directly. Useful when iterating or finetuning the Whisper knobs or any other parameters. 

```bash
uv run python -m steps.download "<URL>"                              # 1
uv run python -m steps.transcribe downloads/<id>.m4a                 # 2 (turbo)
uv run python -m steps.transcribe downloads/<id>.m4a -m v3 -l hi     # 2 (Hindi)
uv run python -m steps.dedupe downloads/<id>.json                    # 3
uv run python -m steps.diarize downloads/<id>.m4a                    # 4
uv run python -m steps.merge downloads/<id>.m4a                      # 5
uv run python -m steps.summarize <id>.diarized.txt --auto-start      # 6 (spawn server)
uv run python -m steps.summarize --stop-server                       # kill the auto-spawned server
```

`--auto-start` only spawns a server if one isn't already reachable on `--server-url`. The spawned PID is tracked in `/tmp/summarize-video-llama-server.pid` and the log streams to `/tmp/summarize-video-llama-server.log`.

## Summarize step

### Two-call flow

1. **Episode-context extraction.** This call gives the raw YouTube description to Gemma with a separate extraction prompt and asks for a structured `## Episode Context` block (Show, Title, Host, Guests, Event/venue, Themes, Language). Reasoning is **disabled** for this call — see [Episode context](#episode-context). Output: `<id>.episode_context.md`.
2. **Main summary.** This is the main summary call. It prepends the episode-context block to the diarized transcript inside an `<episode_context>` tag, then asks Gemma for the full summary. Reasoning stays **on** here — deliberation is what makes Gemma honor strict prompt constraints (verbatim quotes, language preservation, named entities). Output: `<id>.diarized.summary.md`.

The summary has seven sections: `Title`, `TL;DR`, `Key Points`, `Chapters with timestamps`, `Takeaways`, `Notable Quotes` (verbatim, attributed), `Resources Mentioned`. You can find the exact format in `steps/summarize.py:30` (`SYSTEM_PROMPT`).

Note: If parsing the model's `<summary>` XML fails, the raw response is saved as `<input>.summary.raw.txt` so I can debug the prompt or inspect the output myself.

We use the same sampling defaults (as recommended by Google) with 4096 max tokens for the summary.

### Episode context

Diarization gives me anonymous `SPEAKER_xx` labels. If the speakers don't introduce each other on mic (common on single-guest podcasts), the summarizer has nothing to anchor labels to and either falls back to generic roles ("host" / "actor") or fabricates names. On one Hindi benchmark the pre-context version confidently invented a show called *"The Andy Show Clips"* with context it correctly attributes lines to **Ranveer** and **Rajkummar Rao**.

The extraction call asks Gemma for one Markdown block:
```markdown
    ## Episode Context
    - **Show:** ...
    - **Title or topic:** ...
    - **Host:** ...
    - **Guests:** ...
    - **Event or venue:** ...
    - **Themes promised:** ...
    - **Language:** ...
```

Any field the description doesn't explicitly state is omitted (no "N/A", no guessing). The result is written to `<id>.episode_context.md` and prepended to the main summary call. Gemma is told to use it for grounding but defer to the transcript whenever the two conflict.

The first version of this feature shipped broken: Gemma's hidden reasoning consumed the entire `max_tokens=1024` budget on the small structured-output call and returned an empty string. The fix is one JSON kwarg per request:

```json
"chat_template_kwargs": {"enable_thinking": false}
```

`--jinja` makes llama-server forward this into Gemma 4's chat template which then skips the `<think>` block entirely. You can find the receipts in [experiments.md # episode context](experiments.md#episode-context-extraction-broken-then-fixed).

The orchestrator caches `<id>.episode_context.md` like every other intermediate — re-runs reuse it unless I pass `-f`. Standalone runs of `steps.summarize` accept `--context-file PATH` if I want to hand-write or reuse a context block.

## llama-server setup

The summarize step needs a running `llama-server`. The orchestrator spawns it after step 5 (so whisper/pyannote VRAM is freed first) and stops it on exit. Standalone runs use `--auto-start` / `--stop-server`. This section is the once-per-machine setup.

### Hardware target

Either:
- Apple Silicon Mac with a Max-class chip and **36 GB unified memory**, or
- Linux box with an NVIDIA GPU ≥ **24 GB VRAM** (e.g. RTX 4090).

### Install llama.cpp

**macOS (Metal).** Use the pre-built binary, Metal already enabled:

```bash
brew install llama.cpp
```

**Linux (CUDA).** There is no pre-built `llama-server` with CUDA in the standard apt repos and you need to build from source:

```bash
sudo apt install cmake build-essential libcurl4-openssl-dev
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
ln -sf "$PWD/build/bin/llama-server" ~/.local/bin/llama-server
```

It requires the CUDA toolkit (`nvcc`) matching the driver.

### Download the model

Unsloth's dynamic quant `UD-Q4_K_XL` is the best size/quality trade-off. It is ~500 MB larger than `Q4_K_M` but measurably higher quality because it keeps some layers at higher precision.

```bash
uv tool install huggingface_hub    # bundles hf_xet for parallel downloads
mkdir -p ~/MODELS/unsloth/gemma-4-31B-it-GGUF
hf download unsloth/gemma-4-31B-it-GGUF \
  gemma-4-31B-it-UD-Q4_K_XL.gguf \
  --local-dir ~/MODELS/unsloth/gemma-4-31B-it-GGUF
```

### Start the server

The orchestrator builds this command automatically; this is the canonical recipe to run by hand.

```bash
llama-server \
  -m ~/MODELS/unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-UD-Q4_K_XL.gguf \
  -ngl 99 \
  -c 65536 \
  -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --batch-size 2048 --ubatch-size 1024 \
  --context-shift \
  --metrics \
  --jinja \
  --host 127.0.0.1 --port 8081
```

| flag | what it does |
|---|---|
| `-ngl 99` | Offload all layers to the GPU (Metal on Mac, CUDA on Linux). |
| `-c 65536` | 64K context — plenty for a 2-hour transcript (~15K tokens). |
| `-fa on` | Flash attention. Required when using a quantized KV cache. |
| `--cache-type-k q8_0` / `--cache-type-v q8_0` | 8-bit KV cache. Halves its memory, near-lossless quality. |
| `--parallel 1` | One concurrent slot. I'm not multiplexing requests. |
| `--batch-size 2048` | Tokens per logical prefill batch. |
| `--ubatch-size 512` / `--ubatch-size 1024` | Tokens per GPU kernel call. **The main prefill-speed knob.** Orchestrator picks `512` on 24 GB CUDA cards (frees ~1 GB of activations so 64K KV cache fits) and `1024` on Macs / 48 GB+ cards. Logic in `steps/summarize.py:169` (`_pick_ubatch_and_ctx`). |
| `--context-shift` | Slide the window when input exceeds context, instead of failing. |
| `--metrics` | Expose Prometheus-style stats at `/metrics` for tuning. |
| `--jinja` | Use the model's embedded chat template. Required for Gemma 4. |
| `--host 127.0.0.1 --port 8081` | Bind to localhost only. |

The server exposes an OpenAI-compatible API at `http://127.0.0.1:8081/v1/chat/completions`. Quick sanity check: `curl http://127.0.0.1:8081/v1/models`.

### Why these flags help summarization specifically

Summarization is **prefill-bound**. A one-hour video can be 30K-60K input tokens but the summary is only ~1-2K output tokens, so most of the wall-clock time goes into digesting the input before generating anything.

The main lever is `--ubatch-size` (the chunk of tokens processed in a single GPU kernel call). The default 512 is conservative for memory-tight setups; bumping to 1024 roughly halves prefill at the cost of ~1 GB extra activation memory during prefill. 2048 halves it again on cards with another ~4 GB of headroom.

### VRAM math on 24 GB CUDA

- Don't raise `--ubatch-size` past 512 on 24 GB CUDA without re-running `./benchmark.sh en`. I observed OOMs with 64K at ubatch 1024 in this flow. 
- Note: TurboQuant (Google Research's 2025 ~3-bit KV-cache quantization) is **not in llama.cpp master yet** ([issue #20977](https://github.com/ggml-org/llama.cpp/issues/20977)) — only in community forks like `TheTom/llama-cpp-turboquant`. I am skipping it until it merges; `q8_0` KV already fits 64K-128K context on a 36 GB Mac.

## How the summarize prompt got here

The prompt is a few iterations old. The current version lives in `steps/summarize.py:30` (`SYSTEM_PROMPT`). Each XML block is fixing a specific failure mode I hit on real transcripts.

- **`<role>` + `<instructions>`**: Establishes that the output is a durable reference document, not a casual recap. Without this, Gemma defaults to a chat-style "here's what they discussed" intro.
- **`<input_format>`**: Explains the `[mm:ss - mm:ss] SPEAKER_xx: text` line shape and that `<episode_context>` may be absent. Stops the model from quoting timestamp brackets verbatim or wondering what to do when context is missing.
- **`<context_usage>`**: Tells Gemma to use the episode context for grounding `SPEAKER_xx → name` and entity spellings, but defer to the transcript on factual conflicts. Without this rule, the model occasionally hallucinated facts present only in the description.
- **`<depth_rules>`**: Seven rules with concrete anti-patterns: specifics over generics ("raised $40M in 2023" beats "raised significant funding"), argument over topic ("no 'talks about X'"), surface disagreements, no filler, attribute disagreements between speakers, keep technical terms technical, self-contained bullets. The pre-revision prompt produced output like `**AGI Timelines:** Dario Amodei predicts...` — exactly the topic-framing this section now kills.
- **`<output_format>`**: Section names and per-section length targets (Title, TL;DR, Key Points, Chapters, Takeaways, Notable Quotes, Resources Mentioned). The Notable Quotes block calls for verbatim quotes with `>` blockquote format and `(Speaker, [mm:ss])` attribution.
- **`<length_target>`**: "Medium depth, scale proportionally to transcript length." Stops Gemma from padding short transcripts or truncating long ones.
- **`<final_check>`**: A self-verification pass: every Key Point bullet makes a specific claim, TL;DR states positions, quotes are verbatim, chapter timestamps cover the full transcript, every bullet is understandable in isolation. Acts as a cheap last-mile filter on the output.

The XML scaffold matters too. Plain Markdown sections worked but were noisier — Gemma occasionally treated them as content and copied headers into the output. Tagged blocks read as instructions, not text, and Gemma respects the boundary.

### Using reasoning mode with Gemma 4

Gemma 4 has a hidden reasoning mode (`<think>` tokens) that fires before visible output when `--jinja` is on. I A/B tested disabling it on the main summarize call twice. Once with the loose v1 prompt, once with the strict current prompt. Both runs are ~2× faster on cold cache (45 s vs 100 s on the en case). Both regress quality, just on different axes:

- **Hindi quotes get translated to English** — "Never paraphrase" treated as a soft suggestion. The strict prompt fixes this on its own.
- **Key Points drift back to topic framing** even with `<depth_rules>` explicit. Reasoning catches the violation; greedy decoding doesn't.
- **Resources lists shrink** — Devanagari-named entities and entries that require reasoning over the transcript get dropped.
- **Token-mixing glitches.** One EN quote came back garbled with Japanese and Dutch tokens (`"...might be six to 12 months away from anいい mooi when..."`). Deliberation seems to catch these output-generation glitches; one-shot decoding doesn't.

So I keep reasoning on for the main call and pay the ~30-40 s. Reasoning stays off only for the episode-context extraction call, where the output is a plain Markdown skeleton with no strict-quote / language-preservation constraints. Receipts in [experiments.md: reasoning on the main summary call](experiments.md#reasoning-on-the-main-summary-call).

## Design rationale

### Why local-first / no API

- **Learning.** This is partly a project for understanding the primitives. Running Whisper, pyannote, and Gemma locally and tuning their knobs teaches me things that calling an API doesn't.
- **Fun**: It is fun to build something that works locally and is not dependent on an API. Overall, the quality is acceptable, which can further be improved by tuning the parameters and the prompt.

### Why Gemma 4 31B

I had the option to use either Gemma4 31B or Qwen3.6 27B. I chose Gemma4 31B because of this main reason:
- **Hindi handling.** Gemma 4 keeps Devanagari intact when the prompt asks for it. The hindi handling is better than the qwen model.
Note: I also tested the MOE versions of Gemma4 and Qwen3.6. However, the MOE versions were no where as good as the counterpart dense versions.

### Why Whisper turbo vs v3

Whisper large-v3 has 32 decoder layers; large-v3-turbo is distilled to 4 decoder layers. Roughly 5-8× faster, but most of the multilingual quality lives in those 28 layers turbo dropped.

- **turbo (default):** English-tuned. Fast. Use it for English-only audio.
- **v3:** Full multilingual heads. Use it for non-English / accented / code-switched audio.

For example, turbo auto-detected my Hindi-English test clip as English and produced garbled romanization. v3 detected Hindi and produced clean Devanagari + Latin Hinglish

### Why pyannote and the step split

Diarization and transcription are different model classes — one outputs speaker turns, the other wants 30-second log-Mel chunks and outputs words. They could run in parallel, and splitting them keeps each step's failure mode local.

- **`merge` as its own step.** Clean seam for a future wav2vec2 forced-alignment pass between transcribe and merge. It is phoneme-level word boundaries instead of Whisper's loose DTW timestamps.
- **`dedupe` exists because greedy decoding has attractors.** mlx-whisper 0.4.3 only supports greedy (`NotImplementedError` on `beam_size > 1`); without dedupe, `thank you ×22` loops slip into transcripts. The CUDA `faster` backend defaults to beam 5 and rarely needs dedupe but I still run it for consistency.
- **`--num-speakers` matters more than you'd think.** When I know the count, the turns come out a lot cleaner than letting pyannote guess.

## Troubleshooting


### YouTube extractor breakage

YouTube has been pushing hard against `yt-dlp`. Two recent failure modes I've hit:

- `ios` / `web` / `mweb` clients now require a GVS PO Token; they skip all formats served without one.
- `tv` client is currently flagged with a session-level DRM experiment ([yt-dlp #12563](https://github.com/yt-dlp/yt-dlp/issues/12563)) that marks all formats as DRM-protected.

```bash
uv run yt-dlp --extractor-args "youtube:player_client=tv_simply,web_embedded,android_vr" -F "<URL>"
```

**Bot-check on some videos.** YouTube sometimes nudges with "Sign in to confirm you're not a bot" on specific videos. Supplying a signed-in session's cookies clears it:

```bash
uv run python summarize_video.py "<URL>" -l en --cookies-from-browser chrome
uv run python summarize_video.py "<URL>" -l en --cookies ~/cookies.txt
```

### llama-server quirks

- **Empty response on a short structured call.** This happens when the model's reasoning mode consumes the entire `max_tokens` budget before any visible output starts. You can disable reasoning mode for such calls by passing `"chat_template_kwargs": {"enable_thinking": false}` in the request body.
- **OOM at ubatch 1024 on 24 GB CUDA + 64K context.** llama.cpp runs out of activation memory during prefill. The orchestrator picks `512 / 64K` on 24 GB cards via `_pick_ubatch_and_ctx` (`steps/summarize.py:169`). Don't override unless the card has more headroom.
- **`--jinja` is required for Gemma 4.** Without it, llama-server uses a generic chat template that doesn't match Gemma's tokens.

### Whisper decoder pitfalls

- **`--initial-prompt` over-anchors.** Passing an English seed (e.g. `"LSD Mumbai Reliance School"`) to bias Hinglish rendering of names locked the first chunk into `जितियों जितियों जितियों ...`. Removing the prompt restored normal output. Skip unless I have specific domain terms to anchor and have tested they don't trap the decoder.
- **Sampling-only temperature ladder.** Skipping `0.0` and starting the fallback ladder at `0.2` locked the model into `English English English ...` on a Hindi clip. Greedy is fine as the *first* try; sampling is for recovery. Stick with the default ladder `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`.
- **`beam_size > 1` on mlx-whisper 0.4.3.** Raises `NotImplementedError: Beam search decoder is not yet implemented`. Greedy + temperature fallback only. The `--beam-size` flag exists for forward compatibility; the CUDA `faster` backend supports beam search natively.

## Benchmarks

- Case 1: English panel 
  - URL: https://www.youtube.com/watch?v=02YLwsCKUww
  - **en**: 31-min English panel, `turbo` + 3 speakers
- Case 2: Hindi-English clip
  - URL: https://www.youtube.com/watch?v=HeAGWTgi4sU
  - **hi**: 8-min Hindi-English clip, `v3` + 2 speakers

You can run the benchmarks using the following command:
```bash
./benchmark.sh en
./benchmark.sh hi
./benchmark.sh all
```

### RTX 4090 24 GB, i7-12700KF, 32 GB RAM, Ubuntu 24.04
Numbers from my box (RTX 4090 24 GB, i7-12700KF, 32 GB RAM, Ubuntu 24.04) via `./benchmark.sh all` — cold cache, `-f` forced, summarize on. Two canonical cases:

| case | audio | transcribe | diarize | summarize | total | realtime |
|---|---|---|---|---|---|---|
| en (turbo) | 31m 11s | 110.5s | 39.2s | 96.0s | **4m 22s** | 7.1× |
| hi (v3)    | 8m 14s  | 87.8s  | 24.6s | 66.6s | **3m 15s** | 2.5× |

Transcribe is usually the biggest step, summarize is close behind, diarize scales roughly with audio length. `v3`'s deeper decoder (32 layers vs 4) costs ~3× more wall time per second of audio than `turbo`. This is where the multilingual quality comes from.

### Mac (Apple Silicon) numbers TBD — need to re-run on a Mac.


## TODOs

### `no_repeat_ngram_size` on the faster backend

The CUDA `faster` backend already gives me beam search by default which prevents most of the repetition loops `dedupe.py` was originally written to clean up. CTranslate2 also exposes `no_repeat_ngram_size` and `suppress_tokens`. If we wire them through, it would be the natural next step for eliminating loops at decode time instead of after the fact.

### wav2vec2 forced alignment

Replace Whisper's DTW word timestamps with phoneme-level forced alignment against the audio (the technique whisperx uses internally). Each word's start/end gets snapped to the actual acoustic boundary, eliminating the trailing-word-leaks-into-next-speaker artifact I see at turn changes.

### User notes as extra summary context

While listening to a podcast I often jot down my own notes — key claims, timestamps I want highlighted, topics the summary should emphasize. Step 6 already supports grounding context through the auto-extracted `<id>.episode_context.md`; extending it with a user-supplied notes file (e.g. `--notes-file PATH` on the orchestrator, prepended to the transcript in its own tag alongside `<episode_context>`) would let those notes bias the final summary toward the bits I actually cared about.