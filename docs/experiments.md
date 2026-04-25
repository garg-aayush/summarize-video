# Experiments — what I tried

A running log of decisions and dead-ends, newest on top. `pipeline.md` cross-links into here for the receipts.

## reasoning on the main summary call

Disabling Gemma's hidden reasoning on the small extraction call cut its eval from 25 s to 1.3 s without hurting the output. So I figured the same `chat_template_kwargs.enable_thinking=false` flag should give me a near-half-cost step 6 on the main summary call too. The main call emits ~750-1500 hidden think tokens per run, roughly 30-40 s of overhead I never see. If output quality held up, that was a free win.

I ran this twice. First time was against the old loose prompt — one paragraph of instructions, no explicit format rules. Then I rewrote the prompt with explicit `<input_format>` / `<context_usage>` / `<depth_rules>` / `<output_format>` / `<final_check>` blocks (with concrete anti-patterns like "specifics over generics" and "argument over topic"), and figured the stricter prompt might do enough work on its own that reasoning-off would finally hold quality. Same flag, new prompt, second run.

Speed-wise, both runs landed almost exactly 2× faster on cold cache:

| case | step 6 — reasoning on (default) | step 6 — reasoning off |
|---|---|---|
| en (loose prompt)  | 83.1 s  | 45.1 s (−46%) |
| hi (loose prompt)  | 65.7 s  | 35.1 s (−47%) |
| en (strict prompt) | 101.9 s | 53.6 s (−47%) |
| hi (strict prompt) | 71.3 s  | 38.1 s (−47%) |

Note: the strict prompt added ~10 s of extra prefill but the speed-up ratio was identical.

Quality is where things came apart. The strict prompt did fix one of the headline regressions from the first run: Hindi quotes now stayed in Devanagari instead of getting silently translated to English. The "Never paraphrase or clean up grammar" rule in `<output_format>` was enough on its own:

> "Because कोई भी language आपको बोलने से आती है, आपका environment आपको वो language सिखाता है..."

For contrast, here is how badly the loose-prompt run had butchered the same quote when reasoning was off:

> Reasoning on: "Because कोई भी language आपको बोलने से आती है, आपका environment आपको वो language सिखाता है"
>
> Reasoning off: "Because any language is learned by speaking, your environment teaches you that language..."

Section headings, blockquote format, chapter timestamps, TL;DR-as-claims — all stayed intact in both reasoning-on and reasoning-off variants of the strict prompt.

What the strict prompt didn't fix:

- **Key Points drift back to topic framing.** Even with `<depth_rules>` explicitly saying "Argument over topic" and "Lead with the claim", reasoning-off reverted to `**AGI Timelines:** Dario Amodei predicts...` — exactly the anti-pattern the prompt was trying to kill. Reasoning-on produced clean claim-first bullets like "Anthropic's revenue grew exponentially from $100M in 2023 to $10B in 2025...".
- **Token-mixing glitches.** One EN quote in the reasoning-off output came back garbled with Japanese and Dutch tokens: `"...might be six to 12 months away from anいい mooi when the model..."`. Reasoning-on produced a clean verbatim quote from the same transcript. Deliberation seems to catch these output-generation glitches that greedy decoding doesn't.
- **Resources lists shrink.** On the hi case, reasoning-on surfaced 5-6 entities (Hansal Mehta, The Ranveer Show, Reliance School, LSD, Hindi/Urdu, Anish); reasoning-off dropped to 2-3, and the ones that disappeared were always the Devanagari-named entries (हांजल सर, अनीश).
- **Episode context gets ignored.** Reasoning-off didn't credit Ranveer as host of The Ranveer Show in resources either, even though that line was literally in the `<episode_context>` block I had just paid to extract. The extraction cost was paid and then half-wasted.
- **Coverage shrank on hi.** The loose-prompt run went from 7 chapters to 6 (lost "Learning from mentors and peers") and 4 takeaways to 3 (lost "Practicality vs. Status"). The strict-prompt run was less affected on this axis but didn't reverse it.

So I reverted both times. 30-50 s saved per run isn't worth losing faithful Hinglish quotes, complete resource lists, or claim-first framing — especially because Hindi+English code-switched podcasts are the project's primary use case.

What I take from running this twice is that prompt rigor and deliberation do different work. A stricter prompt closes some rule-adherence gaps (verbatim quotes, language preservation) but not others (argument-vs-topic framing, named-entity coverage, output-token coherence). Reasoning seems to do mechanism-level work at generation time — picking the right token for the verbatim-quote span, deciding which entity deserves a resource bullet — that prose instructions alone can't replicate. The reasoning cost on the main summary call is a tax on faithfulness, not on "smartness".

Note: reasoning stays disabled only on the extraction call, where the output is a plain Markdown skeleton and paraphrasing isn't a concern.

## Episode context extraction (broken, then fixed)

The plan for the episode-context feature was straightforward: add a small llama-server call at the top of step 6 that distills the YouTube description into an `## Episode Context` block, prepend it to the main summarize call, and let Gemma use it to ground `SPEAKER_xx` labels in real names. I expected maybe ~5 s of extra wall time on a model that is already hot.

Here is what actually happened on the first cold-cache benchmark run (RTX 4090, `./benchmark.sh all`):

| case | step 6 before | step 6 after | Δ | extraction output | summary names speakers? |
|---|---|---|---|---|---|
| en (02YLwsCKUww, 31 min) | 79.9 s | 103.2 s | +23.3 s | empty | yes — but only because Hassabis/Amodei introduce each other in the transcript |
| hi (HeAGWTgi4sU, 8 min)  | 55.2 s | 81.1 s  | +25.9 s | `##` (2 chars) | no — falls back to "host" / "actor" |

Both runs paid the wall-time cost and got nothing back. The orchestrator wrote a stub `<id>.episode_context.md` and the main summary lost the grounding it was supposed to gain.

The cause turned out to be `--jinja`. With that flag on, llama-server renders Gemma 4's chat template with reasoning enabled by default — and the server log makes it explicit on every call:

    reasoning-budget: activated, budget=2147483647 tokens
    reasoning-budget: deactivated (natural end)

The extraction call's `max_tokens=1024` was getting consumed entirely by `<think>...</think>` tokens. The visible answer never started. Per-call timings made this obvious in hindsight:

    eval time = 24998.53 ms / 1024 tokens   ← extraction, hit the cap, content empty
    eval time = 25073.61 ms / 1024 tokens   ← extraction, same story
    eval time = 22996.79 ms /  938 tokens   ← extraction (earlier ad-hoc test) — model
                                             chose to think less, emitted 222 chars

Note: the 222-char success that made me ship the change in the first place was a coincidence — Gemma had decided to think less on that one prompt, so the output happened to fit. The benchmarks just rolled the dice and lost.

While digging into the logs I also noticed the same overhead is hiding inside the existing main summary call. The visible summary is ~750 tokens per run, but each summarize eval emits ~1500-2200 tokens — so ~750-1500 of those are reasoning I never see, costing 15-35 s per run. That observation is what motivated the [reasoning A/B above](#reasoning-on-the-main-summary-call).

The fix for the extraction call is a single JSON kwarg per request: `"chat_template_kwargs": {"enable_thinking": false}`. With `--jinja` on, llama-server forwards those kwargs into Gemma's chat template, which then skips the `<think>` block entirely. Scoped to the extraction call only — the A/B above covers why the main summary call needs to keep deliberating.

After the fix, cold-cache benchmarks landed where I had expected the first time:

| case | step 6 — no context | step 6 — context broken | step 6 — context fixed | extraction eval | extraction output |
|---|---|---|---|---|---|
| en | 79.9 s | 103.2 s | **83.1 s** | ~1.3 s | full block (5 fields, 3 named guests) |
| hi | 55.2 s | 81.1 s  | **65.7 s** | ~1.3 s | full block (Show + Guest + Themes) |

Per-call extraction tax dropped from ~25 s to ~1-2 s, and the post-context step 6 landed within the run-to-run noise band (±10-15%) of the pre-context baseline.

The quality difference between the broken and fixed versions was sharpest on the hi case (single guest, no on-mic introductions):

| | broken (no context) | fixed (with context) |
|---|---|---|
| tldr | "A conversation between **a host and an actor** about..." | "**Ranveer and Rajkummar Rao** discuss their personal journeys..." |
| quote attribution | `**SPEAKER_01**:` / `**SPEAKER_00**:` | `**Rajkummar Rao**:` / `**Ranveer**:` |
| resources show entry | `**The Andy Show Clips** — The podcast/YouTube channel.` (hallucinated) | `**Ranveer:** Host of The Ranveer Show.` |

Without context, the model didn't just fail to attach names to `SPEAKER_xx` labels — it confidently fabricated a show name. The context block is what grounds it. The en case (panel where speakers introduce each other) was less dramatic: both versions named *Hassabis* and *Amodei* correctly because their names are already in the transcript, but only the fixed version surfaced `**Event or venue:** World Economic Forum 2026` (which isn't stated verbatim in the dialogue) in the tldr and resources.

The general thing I take from this is to never trust a llama-server call that comes back faster or shorter than expected when `--jinja` is on. Always check the server log for `reasoning-budget: activated` before trusting the output, and never declare victory on a single one-shot test — Gemma's thinking is stochastic, so the same prompt can fit one run and overflow the next.

## Dedupe: largest-n-first kept partial loops

The first version of `dedupe.py` walked left-to-right and at each position scanned `n` from `max_n` down to 1, taking the first `n` that hit `min_repeats`. On a `thank you ×22` loop it picked `n=4` (5 repeats of a 4-gram), kept the 4 items, advanced past 20, and left 2 stragglers behind in the output.

The fix was to score every `n` in `[1..max_n]` by total coverage (`n × repeats`) and take the highest, with ties broken toward smaller `n`. Now `n=1, repeats=22` beats `n=4, repeats=5` on coverage (22 vs 20), and even on a tie a 1-gram collapses to 1 item rather than 4.

Note: while testing this I also hit a side bug. Whisper sometimes emits the same loop as 22 separate segments interleaved with zero-duration empty segments (silence-guard artifacts). The per-segment dedup pass couldn't see across them, and the segment-level pass saw non-consecutive matches and bailed. Dropping zero-duration empties before the cross-segment pass fixed it.
