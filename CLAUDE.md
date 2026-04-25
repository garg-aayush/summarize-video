# Claude instructions for summarize-video

## Project

Local pipeline to summarize YouTube videos with a focus on podcast-style discussions (English, or Hindi + English code-switched): URL → speaker-attributed transcript (+ optional local-LLM summary). Components: `yt-dlp`, Whisper (`mlx-whisper` on Apple Silicon / `faster-whisper` on Linux+CUDA), `pyannote.audio`, and `llama.cpp` (Gemma 4 31B) — all on-device.

Docs: `pipeline.md` (deep-dive: steps, flags, llama-server setup, design rationale, troubleshooting, benchmarks), `definitions.md` (glossary), `experiments.md` (decisions log).

## Running things

Always use `uv run python …` — the project is uv-managed; never invoke the system Python or pip.

Intermediates land in `/tmp/summarize-video-<id>/` (cache-friendly across re-runs); finals are copied to `--output-dir` (default: CWD). **Don't reintroduce `downloads/`** as the orchestrator sink — that change was deliberate.

The summarize step needs `llama-server`; the orchestrator spawns it at step 6 (after freeing whisper/pyannote VRAM) and stops it on exit. Pass `--llama-server-bin PATH` if it's not on `$PATH`. Step 6 also runs a small extraction pass against the same loaded model to distill the YouTube description into `<id>.episode_context.md`, prepended to the main summary to ground speaker/entity names (skip with `--no-episode-context`; see `docs/pipeline.md#episode-context`). Run standalone: `uv run python -m steps.summarize <file>.diarized.txt --auto-start` (or `--stop-server`).

ubatch/ctx defaults are platform-aware (`_pick_ubatch_and_ctx` in `steps/summarize.py`): 24 GB CUDA → `512 / 64K`, Mac or 48 GB+ CUDA → `1024 / 64K`. Override via `--server-cmd`.

## Benchmarks

- `./benchmark.sh en|hi|all` runs cold-cache canonical benchmarks (`-f` forced); timings land in `benchmark/<id>-<platform>-<timestamp>/metadata.txt`.
- Per-step wall times vary ±10–15% run-to-run — don't chase small regressions without multiple runs.
- Re-validate `benchmark.sh en` after any change to transcribe/diarize/summarize defaults that could plausibly affect VRAM or wall time.

## Commit style

- **Never add `Co-Authored-By` trailers** (also not in code / docstrings). Persistent user preference.
- Short imperative subject; optional body explaining the *why*, not a line-by-line diff recap.
- No emojis in commits, code, or files unless explicitly requested.
- Match the style of recent commits (`git log --oneline`).

## Communication

- The user values understanding the primitives (this is partly a learning project). When they ask "why X over Y", explain the trade-off rather than just declaring the answer.

## Markdown style (README.md and docs/*.md)

This is a personal learning project — docs should read that way. Tone is grounded, first-person, functional; no hype, no superlatives, no emoji. Conventions to reuse across `docs/` and any future user-facing markdown:

- **No hard-wrapping of prose.** Write each paragraph, list item, or table row as one long line and let the editor soft-wrap. Fenced code blocks and ASCII diagrams keep their own line breaks.
- **First-person author voice** for explanatory prose: "I built", "my box", "the use case I have in mind", "the output I want". Keep tables, flag rows, and imperative setup steps ("Run X", "Install Y") in neutral voice.
- **Plain language over jargon.** When a technical term is unavoidable, define it inline with a concrete example — *"a spectrogram, basically a picture of the sound"*, *"a phoneme is the smallest sound unit (the `k` in `cat`)"*. Don't assume the reader knows domain-specific acronyms (MLX / CT2 / DTW / beam search / diarization / etc.).
- **Short, direct sentences.** Clear beats clever. Contractions are fine; casual word choices ("cheat-sheet", "butcher", "free context", "grab") are fine when they're unambiguous.
- **Definition-style bullets use a colon after the bold term**, not an em-dash: `- **Term**: explanation`. Reserve em-dashes for asides *within* a sentence.
- **Don't sell the project, describe it.** "This is a small pipeline I built to turn a YouTube URL into ..." reads better than "A powerful, flexible framework for ...". Skip marketing adjectives.
- **Trim aggressively.** If a paragraph, glossary entry, or TODO isn't load-bearing for the reader's current understanding, cut it — don't preserve archeology. Keep docs honest about what exists now, not what used to exist.
- **README is the short introduction; reference material lives in `docs/*.md`.** Full flag tables, platform quirks, per-step deep dives, benchmark commentary go into `docs/pipeline.md` (or a sibling doc). README links down, doesn't duplicate.

## Things to verify before recommending

- TurboQuant: not in llama.cpp master (only community forks). Check the upstream PR/issue before suggesting it as ready-to-use.
- HF CLI: it's `hf download` now (not `huggingface-cli download`). `hf_xet` ships by default; `hf_transfer` is deprecated.
- mlx-whisper: 0.4.3 is greedy-only (`NotImplementedError` on `beam_size > 1`).
- Linux/CUDA cuDNN: if the system has cuDNN on `LD_LIBRARY_PATH` and it's older than the one torch bundles, pyannote will crash with "cuDNN version incompatibility". Workaround: `LD_LIBRARY_PATH= uv run ...`.
- llama-server VRAM on 24 GB CUDA (orchestrator flow): ~1.2 GB of CUDA context (torch + CT2 + pyannote) survives `torch.cuda.empty_cache()`, leaving ~23 GB usable. Defaults (ubatch 512, `-c 65536`) fit with narrow headroom — llama.cpp's 1 GB safety-margin warning is expected, `-ngl 99` loads anyway. Don't raise `--ubatch-size` past 512 or drop `_free_gpu`'s cleanup without re-running `./benchmark.sh en`; 64K at ubatch 1024 definitively OOMs.
- llama-server `--jinja` with Gemma 4: reasoning is on by default (hundreds-to-thousands of hidden `<think>` tokens before visible output). Short structured calls (e.g. episode-context extraction) can eat the whole `max_tokens` budget and return empty — disable per-request via `"chat_template_kwargs": {"enable_thinking": false}`. **Don't** disable for the main summarize call — A/B in docs/experiments.md: reasoning-off translated Hindi quotes to English, dropped Devanagari-named resources, ignored half the `<episode_context>`. Deliberation is what makes Gemma honor strict prompt constraints (verbatim quotes, named entities, language preservation).
