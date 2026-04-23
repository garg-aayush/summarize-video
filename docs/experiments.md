# Experiments — what I tried

A running log of decisions and dead-ends. Each entry: what I wanted, what I tried, what I learned. New entries go on top.

---

## Disabling reasoning on the main summary call — fast, but paraphrases Hindi quotes

**Hypothesis.** Disabling Gemma's hidden reasoning on the *extraction* call cut its eval from 25 s to 1.3 s without hurting output quality. The same `chat_template_kwargs.enable_thinking=false` flag should give a similar speed-up on the main summary call (which currently emits ~750–1500 hidden thinking tokens per run). If quality holds, we get a half-cost step 6.

**Speed result on cold cache (`./benchmark.sh all`):**

| case | step 6 — reasoning on (current default) | step 6 — reasoning off |
|---|---|---|
| `en` | 83.1 s | **45.1 s** (−46 %) |
| `hi` | 65.7 s | **35.1 s** (−47 %) |

Almost half the wall time, exactly as predicted.

**Quality result.** EN summaries (panel discussion) were broadly equivalent — both correctly named Hassabis / Amodei, both produced 8–10 reasonable chapters, the tldrs differed only in tone (reasoning-on slightly more polished). No clear winner.

**HI was a different story.** The system prompt says quotes should be verbatim ("Never paraphrase"). With reasoning on the model honored that:

> **Rajkummar Rao:** "Because कोई भी language आपको बोलने से आती है, आपका environment आपको वो language सिखाता है"

With reasoning off, on the same transcript, it silently translated the quote to English:

> **Rajkummar Rao:** "Because any language is learned by speaking, your environment teaches you that language..."

Same for the other Hindi quote. Without thinking, Gemma treats the "Never paraphrase" instruction as a soft suggestion and prefers the more uniform English-language output. With thinking, it actually deliberates over the constraint and keeps the original Devanagari + Latin code-switching intact.

Same pattern showed up in three other places on `hi`:

- **Resources section gutted**, 6 entries → 3. Reasoning-off dropped both Devanagari-named entities that are clearly in the transcript — *हांजल सर* (Hansal Mehta, director) and *अनीश* (Anish, friend who taught the guest English) — exactly the entries that would require copying non-English script. Same failure mode as the quote translation.
- **Episode-context grounding ignored.** Reasoning-off didn't credit Ranveer as host of The Ranveer Show in resources, even though that line was literally in the `<episode_context>` block we'd just prepended to the prompt. So the extraction cost was paid and then half-wasted.
- **Coverage shrank**: chapters 7 → 6 (lost "Learning from mentors and peers"), main takeaways 4 → 3 (lost "Practicality vs. Status").

Pattern: reasoning-off produces a more uniform, English-leaning, shorter output that drifts away from the system-prompt's specificity. Deliberation is what makes Gemma actually *use* the prepended context block and honor the named-entity / verbatim rules.

**Decision: reverted.** Keep reasoning on for the main summarize call. The 30–40 s saved isn't worth losing faithful Hinglish quotes on the project's primary use case (Hindi+English code-switched podcasts). Reasoning stays disabled only for the extraction call, where the output is a plain markdown skeleton and paraphrasing isn't a concern.

**Lesson.** Reasoning's value isn't just "smarter answers" — for tasks with strict-instruction constraints (verbatim quotes, exact format, language preservation), the deliberation step is what makes the model actually obey the system prompt. Cheap one-shot replies skip that check.

---

## Episode context — Gemma's hidden reasoning ate the extraction budget

**Hypothesis.** Add a small llama-server call at the top of step 6 that distills the YouTube description into an `## Episode Context` block, then prepend it to the main summarize call. Should ground `SPEAKER_xx` labels in real names and add only ~5s of wall time on a model that's already hot.

**Result on cold benchmarks (RTX 4090, `./benchmark.sh all`).**

| case | step 6 before | step 6 after | Δ | extraction output | summary names speakers? |
|---|---|---|---|---|---|
| `en` (02YLwsCKUww, 31 min) | 79.9 s | 103.2 s | +23.3 s | empty | yes — but only because Hassabis/Amodei introduce each other in the transcript |
| `hi` (HeAGWTgi4sU, 8 min)  | 55.2 s | 81.1 s  | +25.9 s | `##` (2 chars) | no — falls back to "host" / "actor" |

So both runs paid the wall-time cost but got nothing back. The orchestrator wrote a stub `<id>.episode_context.md` and the main summary lost the grounding it was supposed to gain.

**Root cause.** llama-server is launched with `--jinja`, which renders Gemma 4's chat template with reasoning **enabled by default**. The server log makes it explicit on every call:

    reasoning-budget: activated, budget=2147483647 tokens
    reasoning-budget: deactivated (natural end)

The extraction call's `max_tokens=1024` was consumed entirely by `<think>…</think>` tokens; the visible answer never started. From the server's per-call timings:

    eval time = 24998.53 ms / 1024 tokens   ← extraction, hit the cap, content empty
    eval time = 25073.61 ms / 1024 tokens   ← extraction, same story
    eval time = 22996.79 ms /  938 tokens   ← extraction (earlier ad-hoc test) — model
                                             chose to think less, emitted 222 chars

The 222-char success that made me ship the change was a coincidence of how much Gemma decided to think for that one prompt. The benchmarks just rolled the dice and lost.

**Side discovery.** The same overhead is hiding inside the existing main summary call. The visible summary is ~750 tokens, but each summarize eval emits ~1500–2200 tokens — meaning ~750–1500 of those are reasoning we never see, costing roughly 15–35 s per run.

**Fix shipped (extraction only).** Pass `"chat_template_kwargs": {"enable_thinking": false}` in the extraction call's JSON body — `--jinja` lets llama-server forward those kwargs into Gemma's chat template, which then skips the `<think>` block entirely. Scoped to extraction; the main summarize call still uses default template behavior (Gemma's reasoning may genuinely help summary quality — separate experiment).

**Cold-cache benchmarks after the fix (`./benchmark.sh all`, RTX 4090):**

| case | step 6 — no context | step 6 — context broken | step 6 — context fixed | extraction eval | extraction output |
|---|---|---|---|---|---|
| `en` (02YLwsCKUww) | 79.9 s | 103.2 s | **83.1 s** | ~1.3 s | full block (5 fields, 3 named guests) |
| `hi` (HeAGWTgi4sU) | 55.2 s | 81.1 s  | **65.7 s** | ~1.3 s | full block (Show + Guest + Themes) |

So the per-call extraction tax is now ~1–2 s instead of ~25 s, and the post-context step 6 lands within the run-to-run noise band (±10–15%) of the pre-context baseline.

**Quality difference, same transcript, broken vs. fixed extraction:**

`en` (panel where speakers introduce each other on-mic): both versions name *Hassabis* and *Amodei* correctly because the transcript contains their names. But only the fixed version surfaces `**Event or venue:** World Economic Forum 2026` (not stated verbatim in the dialogue) in the tldr and resources.

`hi` (single guest, no on-mic introductions) is where the difference is sharp:

| | broken (no context) | fixed (with context) |
|---|---|---|
| tldr | "A conversation between **a host and an actor** about…" | "**Ranveer and Rajkummar Rao** discuss their personal journeys…" |
| quote attribution | `**SPEAKER_01**:` / `**SPEAKER_00**:` | `**Rajkummar Rao**:` / `**Ranveer**:` |
| resources show entry | `**The Andy Show Clips** — The podcast/YouTube channel.` (hallucinated) | `**Ranveer:** Host of The Ranveer Show.` |

Without context, the model not only fails to attach names to `SPEAKER_xx` labels — it confidently fabricates a show name. The context block grounds it.

**Lesson.** Whenever a llama-server call comes back faster or shorter than expected with `--jinja`, check the server log for `reasoning-budget: activated` before trusting the output. A "successful" one-shot test isn't enough — Gemma's thinking is stochastic, so repeat the same prompt before declaring victory.

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
