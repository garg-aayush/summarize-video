# Experiments — what I tried

A running log of decisions and dead-ends. Each entry: what I wanted, what I tried, what I learned. New entries go on top.

---

## Dedupe: largest-n-first kept partial loops

**Problem.** First version of `dedupe.py` walked left-to-right and at each position scanned `n` from `max_n` down to 1, taking the first `n` that hit `min_repeats`. On a `thank you ×22` loop it picked `n=4` (5 repeats of a 4-gram), kept the 4 items, advanced past 20, and left 2 stragglers behind.

**Fix.** Score *every* `n` in `[1..max_n]` by total coverage (`n × repeats`) and take the highest. Tie-break toward smaller `n`. Now `n=1, repeats=22` beats `n=4, repeats=5` (coverage 22 vs 20), and even on a tie a 1-gram collapses to 1 item rather than 4.

**Side bug.** Whisper sometimes emits the loop as 22 *separate* segments interleaved with zero-duration empty segments (silence-guard artifacts). Per-segment dedup couldn't see across them, and segment-level dedup saw non-consecutive matches. Fix: drop zero-duration empties before the cross-segment pass.

---

## `--initial-prompt` locked the first chunk into a loop

**Hypothesis.** Pass an English-word seed (e.g., `"LSD Mumbai Reliance School"`) to bias toward MacWhisper-style Hinglish rendering of names.

**Result.** First chunk decoded as `जितियों जितियों जितियों ...` — the model over-anchored to the prompt and got stuck. Removing the prompt restored normal output. The README knob table now flags this as "often hurts more than it helps."

---

## Sampling-only temperature ladder caused a new loop

**Hypothesis.** Skip the greedy step (`0.0`) and start the temperature fallback ladder at `0.2` to add randomness everywhere.

**Result.** Locked the model into `English English English ...` on a Hindi clip. Reverted to the default ladder (`0.0, 0.2, ..., 1.0`). Greedy is fine as the *first* try; sampling is for recovery.

---

## Beam search on mlx-whisper 0.4.3

**Hypothesis.** Pass `beam_size=5` to escape attractors (the standard fix for greedy loops).

**Result.** `NotImplementedError: Beam search decoder is not yet implemented`. mlx-whisper 0.4.3 only supports greedy + temperature fallback. Kept the `--beam-size` flag for forward-compat and opened the [`faster-whisper` TODO](pipeline.md#todos) since CT2 *does* implement it.

---

## turbo vs v3 on Hindi-English audio

**Hypothesis.** Use `large-v3-turbo` (faster, distilled) for everything since it's the new default.

**Result.** Turbo auto-detected my Hindi-English clip as English and produced garbled romanization. Full `v3` correctly detected Hindi and produced clean Devanagari + Latin Hinglish. Settled on:
- **turbo** — default, English-tuned audio
- **v3** — non-English / accented / code-switched

The 4-decoder distillation evidently dropped most of the multilingual quality.

---

## MacWhisper-style Hinglish output

**Question.** How does MacWhisper produce mixed Devanagari + Latin output on Hindi-English speech?

**Answer (after testing).** It's not a special mode. Force the multilingual Whisper model with `language="hi"` and it naturally:
- transcribes Hindi words in Devanagari
- emits English code-switched words in Latin script

No verifier loop, no second pass. The "self-correcting" appearance comes from Whisper's built-in temperature fallback re-decoding chunks that fail the compression-ratio check.

---

## quant variants on HF that don't exist or don't load

**Hypothesis.** `mlx-community/whisper-large-v3-turbo-q8` will give me a smaller turbo.

**Result.** That repo doesn't exist (404). The `-8bit` suffix variant exists (`mlx-community/whisper-large-v3-turbo-8bit`) but ships `model.safetensors` instead of `weights.safetensors` / `weights.npz`, which mlx-whisper doesn't pick up. Settled on the un-quantized `mlx-community/whisper-large-v3-turbo` for turbo (fp16, ~1.6 GB) and `mlx-community/whisper-large-v3-mlx-8bit` for v3 (8-bit, ~1.6 GB).

---

## yt-dlp player_client investigation

**Problem.** Out-of-the-box `yt-dlp` failed on YouTube with "requires a PO Token" or "all formats DRM-protected" depending on which client it chose.

**Findings.**
- `ios` / `web` / `mweb` — formats served without a PO Token are skipped.
- `tv` — caught by a session-level DRM experiment ([yt-dlp #12563](https://github.com/yt-dlp/yt-dlp/issues/12563)); every format gets the DRM flag.
- `web_embedded`, `android_vr` — still serve plain m4a audio without PO token and aren't in the DRM experiment.

**Fix.** Hard-code `extractor_args="youtube:player_client=web_embedded,android_vr"` in `steps/download.py`. This is fragile — if YouTube tightens those clients too, probe with:

```bash
uv run yt-dlp --extractor-args "youtube:player_client=tv_simply,web_embedded,android_vr" -F "<URL>"
```
