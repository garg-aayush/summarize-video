# Claude instructions for summarize-video

## Project

Local pipeline to summarize YouTube videos with a focus on podcast-style discussions (English, or Hindi + English code-switched): URL → speaker-attributed transcript (+ optional local-LLM summary). Components: `yt-dlp`, Whisper (`mlx-whisper` on Apple Silicon / `faster-whisper` on Linux+CUDA), `pyannote.audio`, and `llama.cpp` (Gemma 4 31B) — all on-device.

## Layout

- `summarize_video.py` — orchestrator entry (download → transcribe → dedupe → diarize → merge → summarize).
- `steps/` — one module per step, each runnable as `python -m steps.<name>`. Includes the optional `steps/summarize.py`.
- `benchmark.sh` — canonical cold-cache timing runs (`en` / `hi` / `all` presets, or pass an ad-hoc URL). Results in `benchmark/<id>-<platform>-<timestamp>/` (gitignored).
- `docs/` — `pipeline.md` (per-step deep-dive), `definitions.md` (glossary), `experiments.md` (decisions log), `summarize.md` (llama.cpp setup).

## Running things

Always use `uv run python …` — the project is uv-managed; never invoke the system Python or pip.

Orchestrator examples:

    uv run python summarize_video.py "<URL>" -l en
    uv run python summarize_video.py "<URL>" -m v3 -l hi \
      --compression-ratio-threshold 2.0 --hallucination-silence-threshold 2.0
    uv run python summarize_video.py "<URL>" --no-diarize

Intermediates land in `/tmp/summarize-video-<id>/` (cache-friendly across re-runs); finals are copied to `--output-dir` (default: CWD). **Don't reintroduce `downloads/`** as the orchestrator sink — that change was deliberate.

The summarize step needs `llama-server`. The orchestrator spawns it automatically at step 6 (after freeing whisper/pyannote VRAM) and stops it on exit — pass `--llama-server-bin PATH` if the binary isn't on `$PATH`. `steps/summarize.py` can also be run standalone:

    uv run python -m steps.summarize <file>.diarized.txt --auto-start
    uv run python -m steps.summarize --stop-server

ubatch/ctx defaults for the spawned server are platform-aware (`_pick_ubatch_and_ctx` in `steps/summarize.py`): 24 GB CUDA → `512 / 64K`, Mac or 48 GB+ CUDA → `1024 / 64K`. Override via `--server-cmd` if you need to retune.

## Benchmarks

- `./benchmark.sh en|hi|all` runs cold-cache canonical benchmarks (`-f` forced) and writes per-step timings to `benchmark/<id>-<platform>-<timestamp>/metadata.txt`.
- Per-step wall times vary ±10–15% run-to-run on a given box — single-run numbers aren't tight averages. Don't chase small regressions without multiple runs.
- Re-validate `benchmark.sh en` after any change to transcribe/diarize/summarize defaults that could plausibly affect VRAM or wall time.

## Commit style

- **Never add `Co-Authored-By` trailers.** Persistent user preference.
- Short imperative subject; optional body explaining the *why*, not a line-by-line diff recap.
- No emojis in commits, code, or files unless explicitly requested.
- Match the style of recent commits (`git log --oneline`).

## Code style

- Default to no comments. Add one only when the *why* is non-obvious (a workaround, a hidden invariant, a counter-intuitive choice).
- Don't pre-build abstractions for hypothetical future requirements.
- Don't add error handling, fallbacks, or backwards-compat shims for scenarios that can't actually happen.
- No `Co-Authored-By` markers in code either (e.g., docstrings).

## Communication

- Terse responses. Skip trailing "summary of what I just did" blocks — the user reads the diff.
- For exploratory questions ("what should we do about X?"), give a recommendation + one-line tradeoff, not a full plan. Don't implement until the user agrees.
- The user values understanding the primitives (this is partly a learning project). When they ask "why X over Y", explain the trade-off rather than just declaring the answer.

## Markdown style

- Don't hard-wrap prose. Write each paragraph, list item, or table row as one long line and let the editor soft-wrap visually. Fenced code blocks and ASCII diagrams keep their own line breaks.
- In user-facing docs (README.md, docs/*.md), prefer first-person ("I", "my") for author-voice paragraphs — this is a personal learning project and the tone should read as such. Keep tables, flag descriptions, and imperative setup steps ("Run X", "Install Y") in neutral voice.

## Things to verify before recommending

- TurboQuant: not in llama.cpp master (only community forks). Check the upstream PR/issue before suggesting it as ready-to-use.
- HF CLI: it's `hf download` now (not `huggingface-cli download`). `hf_xet` ships by default; `hf_transfer` is deprecated.
- mlx-whisper: 0.4.3 is greedy-only (`NotImplementedError` on `beam_size > 1`).
- Linux/CUDA cuDNN: if the system has cuDNN on `LD_LIBRARY_PATH` and it's older than the one torch bundles, pyannote will crash with "cuDNN version incompatibility". Workaround: `LD_LIBRARY_PATH= uv run ...`.
- llama-server VRAM on 24 GB CUDA (orchestrator flow): only ~23 GB is actually usable — the orchestrator's Python process holds ~1.2 GB of CUDA context (torch + CT2 + pyannote shared libs) that `torch.cuda.empty_cache()` can't release. The current defaults (ubatch 512, `-c 65536`) fit with narrow headroom; llama.cpp warns about its 1 GB safety margin but loads anyway because `-ngl 99` is set. Don't raise `--ubatch-size` past 512 or drop the cleanup in `_free_gpu` without re-running `./benchmark.sh en` — 64K at ubatch 1024 definitively OOMs in this flow.
