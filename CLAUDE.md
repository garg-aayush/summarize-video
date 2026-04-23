# Claude instructions for summarize-video

## Project

Local pipeline to summarize YouTube videos with a focus on podcast-style
discussions (English, or Hindi + English code-switched): URL →
speaker-attributed transcript (+ optional local-LLM summary). Components:
`yt-dlp`, Whisper (`mlx-whisper` on Apple Silicon / `faster-whisper` on
Linux+CUDA), `pyannote.audio`, and `llama.cpp` (Gemma 4 31B) — all
on-device.

## Layout

- `summarize_video.py` — orchestrator entry (download → transcribe → dedupe → diarize → merge).
- `steps/` — one module per step, each runnable as `python -m steps.<name>`. Includes the optional `steps/summarize.py`.
- `docs/` — `pipeline.md` (per-step deep-dive), `definitions.md` (glossary), `experiments.md` (decisions log), `summarize.md` (llama.cpp setup).

## Running things

Always use `uv run python …` — the project is uv-managed; never invoke
the system Python or pip.

Orchestrator examples:

    uv run python summarize_video.py "<URL>" -l en
    uv run python summarize_video.py "<URL>" -m v3 -l hi \
      --compression-ratio-threshold 2.0 --hallucination-silence-threshold 2.0
    uv run python summarize_video.py "<URL>" --no-diarize

Intermediates land in `/tmp/summarize-video-<id>/` (cache-friendly across
re-runs); finals are copied to `--output-dir` (default: CWD). **Don't
reintroduce `downloads/`** as the orchestrator sink — that change was
deliberate.

The summarize step needs `llama-server` running. The script can spawn it:

    uv run python -m steps.summarize <file>.diarized.txt --auto-start
    uv run python -m steps.summarize --stop-server

## Commit style

- **Never add `Co-Authored-By` trailers.** Persistent user preference.
- Short imperative subject; optional body explaining the *why*, not a
  line-by-line diff recap.
- No emojis in commits, code, or files unless explicitly requested.
- Match the style of recent commits (`git log --oneline`).

## Code style

- Default to no comments. Add one only when the *why* is non-obvious
  (a workaround, a hidden invariant, a counter-intuitive choice).
- Don't pre-build abstractions for hypothetical future requirements.
- Don't add error handling, fallbacks, or backwards-compat shims for
  scenarios that can't actually happen.
- No `Co-Authored-By` markers in code either (e.g., docstrings).

## Communication

- Terse responses. Skip trailing "summary of what I just did" blocks —
  the user reads the diff.
- For exploratory questions ("what should we do about X?"), give a
  recommendation + one-line tradeoff, not a full plan. Don't implement
  until the user agrees.
- The user values understanding the primitives (this is partly a
  learning project). When they ask "why X over Y", explain the trade-off
  rather than just declaring the answer.

## Things to verify before recommending

- TurboQuant: not in llama.cpp master (only community forks). Check the
  upstream PR/issue before suggesting it as ready-to-use.
- HF CLI: it's `hf download` now (not `huggingface-cli download`).
  `hf_xet` ships by default; `hf_transfer` is deprecated.
- mlx-whisper: 0.4.3 is greedy-only (`NotImplementedError` on `beam_size > 1`).
- Linux/CUDA cuDNN: if the system has cuDNN on `LD_LIBRARY_PATH` and it's
  older than the one torch bundles, pyannote will crash with "cuDNN
  version incompatibility". Workaround: `LD_LIBRARY_PATH= uv run ...`.
