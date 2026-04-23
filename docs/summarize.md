# Summarize a transcript with local Gemma 4 31B (llama.cpp)

How to set up the local model server that powers `steps/summarize.py`.

## Hardware target

Apple Silicon Mac with a Max-class chip and **36 GB unified memory**. Less
RAM means a smaller quant or smaller model.

---

## 1. Install llama.cpp

```bash
brew install llama.cpp
```

Pre-built binary, Metal already enabled. From source if you want the
latest commits:

```bash
brew install cmake
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.logicalcpu)
```

## 2. Download the model

Unsloth's dynamic quant `UD-Q4_K_XL` is the best size/quality tradeoff —
~500 MB larger than `Q4_K_M` but measurably higher quality because it
keeps attention layers at higher precision.

```bash
# Install the HF CLI as a standalone uv tool (doesn't touch the project venv).
# This bundles `hf_xet`, which gives chunk-deduplicated, parallel downloads
# automatically — no extra flag needed.
uv tool install huggingface_hub

mkdir -p ~/models/gemma-4-31b
hf download unsloth/gemma-4-31B-it-GGUF \
  gemma-4-31B-it-UD-Q4_K_XL.gguf \
  --local-dir ~/models/gemma-4-31b
```

> **Note:** the old `huggingface-cli download` command is deprecated in favor
> of `hf download`. Likewise, `hf_transfer` (the old fast-download accelerator)
> is deprecated — `hf_xet` replaces it and ships in `huggingface_hub ≥ 0.32`.

Other quant options:

| file | size | notes |
|---|---|---|
| `gemma-4-31B-it-UD-Q4_K_XL.gguf` *(recommended)* | 18.8 GB | unsloth dynamic Q4 — best quality per GB |
| `gemma-4-31B-it-Q4_K_M.gguf` | 18.3 GB | standard fallback |
| `gemma-4-31B-it-Q5_K_M.gguf` | 21.7 GB | better quality, less room for context |
| `gemma-4-31B-it-Q6_K.gguf` | 25.2 GB | tight on 36 GB |
| `gemma-4-31B-it-Q8_0.gguf` | 32.6 GB | leaves no room for KV cache — skip |

Source: [unsloth/gemma-4-31B-it-GGUF](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF).

## 3. Start the server

```bash
llama-server \
  -m ~/models/gemma-4-31b/gemma-4-31B-it-UD-Q4_K_XL.gguf \
  -ngl 99 \
  -c 65536 \
  -fa on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --parallel 1 \
  --batch-size 2048 \
  --ubatch-size 1024 \
  --context-shift \
  --metrics \
  --jinja \
  --host 127.0.0.1 --port 8081
```

Flags:

| flag | what it does |
|---|---|
| `-ngl 99` | Offload all layers to Metal (GPU). |
| `-c 65536` | 64K context — enough for a ~2-hour podcast. Push higher (128K, 256K) if you need it. |
| `--flash-attn on` | Flash attention. Required when using a quantized KV cache. |
| `--cache-type-k q8_0` / `--cache-type-v q8_0` | 8-bit KV cache. Halves its memory, near-lossless quality. |
| `--parallel 1` | One concurrent slot. We're not multiplexing requests. |
| `--batch-size 2048` | Tokens per logical prefill batch. |
| `--ubatch-size 1024` | Tokens per Metal kernel call. **The main prefill-speed knob** — 2× the default 512. Push to 2048 if you've got memory headroom. |
| `--context-shift` | Slide the window when input exceeds context, instead of failing. |
| `--metrics` | Expose Prometheus-style stats at `/metrics` for tuning. |
| `--jinja` | Use the model's embedded chat template. Required for Gemma 4. |
| `--host 127.0.0.1 --port 8080` | Bind to localhost only. |

### Why these flags help summarization specifically

Summarization is **prefill-bound**: a one-hour podcast can be 30K–60K
input tokens but the summary is only ~1–2K output tokens, so we spend most
of the wall-clock time digesting the input before generating anything.

The main lever is `--ubatch-size` (the size of the chunk of tokens
processed in a single Metal kernel call). The default 512 is conservative
for memory-tight setups; bumping to **1024** roughly halves prefill time
on Apple Silicon at the cost of ~1 GB extra activation memory during
prefill. **2048** roughly halves it again if you have room (model 18.8 GB
+ KV ~3 GB + ~2 GB activations + ~7 GB OS still fits in 36 GB, but it's
tighter — watch Activity Monitor).

`--batch-size 2048` just makes sure the logical batch isn't smaller than
the micro-batch (otherwise it caps you at the smaller value).


The server exposes an **OpenAI-compatible** API at
`http://127.0.0.1:8080/v1/chat/completions`. Sanity check:

```bash
curl http://127.0.0.1:8080/v1/models
```

### TurboQuant — defer

[TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
is a 2025 Google Research KV-cache quantization that hits ~3 bits
near-losslessly. **Not in llama.cpp master yet**
([issue #20977](https://github.com/ggml-org/llama.cpp/issues/20977)) —
only in community forks (e.g. `TheTom/llama-cpp-turboquant` adds
`--cache-type-k turbo3`). Skip it until it merges upstream; `q8_0` KV
already fits 64K–128K context on a 36 GB Mac.

---

## 4. Run the summarizer

If you've already started the server manually (Step 3):

```bash
uv run python -m steps.summarize 02YLwsCKUww.diarized.txt
```

If you'd rather have the script start the server when needed:

```bash
# First call: spawns llama-server, waits ~30–90s for the model to load,
# then summarizes. Server is left running for next time.
uv run python -m steps.summarize 02YLwsCKUww.diarized.txt --auto-start

# Subsequent calls: server is already up, summary returns in seconds.
uv run python -m steps.summarize HeAGWTgi4sU.diarized.txt --auto-start

# When you're truly done for the day:
uv run python -m steps.summarize --stop-server
```

`--auto-start` only spawns a server if one isn't already reachable on
`--server-url`. If you started the server manually, it's reused as-is.
The auto-started PID is tracked in `/tmp/podcasts-llama-server.pid` and
its log streams to `/tmp/podcasts-llama-server.log`.

Output lands at `02YLwsCKUww.diarized.summary.md` next to the input. The
script prefers `.diarized.txt` (speaker-attributed) but works on plain
`.timed.txt` too.

### Flags

| flag | default | what it does |
|---|---|---|
| `-o FILE` | `<input>.summary.md` | Override the output file. |
| `--server-url URL` | `http://127.0.0.1:8080` | Point at a non-default server. |
| `--temperature` | `0.3` | Higher = more creative, lower = more deterministic. |
| `--max-tokens` | `2048` | Cap on the summary length. |
| `--auto-start` | off | Spawn the server (and leave it running) if not reachable. |
| `--stop-server` | off | Kill the auto-started server (read from PID file) and exit. |
| `--model PATH` | `~/models/gemma-4-31b/gemma-4-31B-it-UD-Q4_K_XL.gguf` | Model used when `--auto-start` spawns the server. |
| `--server-cmd CMD` | (built-in recipe) | Full custom command, e.g. `"llama-server -m foo.gguf -ngl 99 ..."`. Overrides `--model`. |
| `--server-wait-timeout` | `180` | Seconds to wait for the spawned server to become ready. |

### Why these defaults

Summarization is structural extraction, not reasoning, so we **don't enable
Gemma 4's thinking mode** — it would burn tokens planning output the prompt
already specifies. Sampling is tuned for faithful extraction:

| param | value | why |
|---|---|---|
| `temperature` | 0.3 | Deterministic but not greedy; greedy can lock into degenerate phrasings. |
| `top_p` | 0.95 | Google's recommended Gemma default. |
| `top_k` | 64 | Google's recommended Gemma default. |
| `min_p` | 0.05 | Trims tail noise that slips past top-p at low temperatures. |
| `repeat_penalty` | 1.0 (off) | Gemma is sensitive to penalties >1.0; the XML scaffold legitimately repeats tags. |

If a particularly dense or technical podcast comes back with shallow
chapters or missed arguments, **then** consider enabling thinking mode (by
prepending `<|think|>` in a custom system prompt). For everyday podcasts
the defaults above are sufficient.

If parsing the model's `<summary>` XML fails, the raw response is saved
as `<input>.summary.raw.txt` so you can debug the prompt or inspect
output yourself.

## What's in a summary

Six sections (see `steps/summarize.py` for the exact system prompt):

1. **TL;DR** — short summary of the podcast.
2. **Key points** — main points raised across the conversation.
3. **Chapters** — `[mm:ss] Short title`, in order, 5–10 entries.
4. **Main takeaways** — the load-bearing conclusions a listener should walk away with.
5. **Important quotes** — 1–4 verbatim lines, attributed if diarized.
6. **Resources** — people, books, papers, companies, tools, URLs, events mentioned.
