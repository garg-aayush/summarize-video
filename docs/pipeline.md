# Pipeline — deep dive

Five steps, each a standalone module under `steps/`. The orchestrator
`summarize_video.py` chains them and skips any step whose output already
exists on disk. Built for YouTube podcast-style discussions — English, or
Hindi + English code-switched.

```
                  ┌─► transcribe ─► dedupe ─┐
URL ─► download ──┤  (whisper)      (loops) ├─► merge ─► summarize ─► <id>.diarized.summary.md
   (yt-dlp)       │                          │             (Gemma 4)
                  └─► diarize ───────────────┘
                       (pyannote)
```

Transcribe uses `mlx-whisper` on Apple Silicon and `faster-whisper`
(CTranslate2) on Linux/CUDA; both are selected automatically by platform.
The summarize step is opt-out (`--no-summarize`); the orchestrator spawns
`llama-server` after step 5 (so whisper/pyannote VRAM is freed first) and
stops it after step 6 — pass `--llama-server-bin PATH` if the binary
isn't on PATH.

---

## Step 1 — Download (`steps/download.py`)

Pulls the best-available audio from a URL via `yt-dlp` and writes an `.m4a`
into `downloads/`.

```bash
uv run python -m steps.download "<URL>"
```

Output: `downloads/<id>.m4a` (stereo AAC, source bitrate preserved).

### YouTube extractor notes

YouTube has been pushing hard against `yt-dlp`. Two recent failure modes we
hit and the workaround we settled on:

- `ios` / `web` / `mweb` clients now require a GVS PO Token and skip all
  formats without one.
- `tv` client is currently flagged with a session-level DRM experiment
  (yt-dlp issue #12563) that marks all formats as DRM-protected.

We use `web_embedded` + `android_vr` clients, which still serve plain m4a
audio formats without PO tokens and are not affected by the DRM experiment.
If those break in the future, probe available clients with:

```bash
uv run yt-dlp --extractor-args "youtube:player_client=tv_simply,web_embedded,android_vr" -F "<URL>"
```

### Bot-check on some videos

YouTube sometimes nudges `yt-dlp` with "Sign in to confirm you're not a
bot" on specific videos. Supplying a signed-in session's cookies clears
it. Two flags, passed through from the orchestrator:

- `--cookies-from-browser <name>` — pulls cookies live from a local
  browser. Names: `firefox`, `chrome`, `brave`, `edge`, `safari`.
- `--cookies <path>` — Netscape-format cookie file (use a browser
  extension to export one).

Both flags are forwarded to the lightweight `_resolve_video_id` call at
the top of the orchestrator *and* to the full download, so the bot
check can't slip through on the resolve step.

---

## Step 2 — Transcribe (`steps/transcribe.py`)

Runs Whisper locally. Two backends, chosen by platform:

| backend | runtime | when |
|---|---|---|
| `mlx` (default on Apple Silicon) | `mlx-whisper` on Metal | Macs |
| `faster` (default on Linux) | `faster-whisper` / CTranslate2 on CUDA | Linux / NVIDIA |

Override with `-b {mlx,faster}`. Preset names are the same across
backends; the concrete HF repo differs because MLX and CT2 use different
model formats:

| preset | mlx repo | faster repo | size | when |
|---|---|---|---|---|
| `turbo` (default) | `mlx-community/whisper-large-v3-turbo` | `Systran/faster-whisper-large-v3` | ~1.6 GB / ~3 GB | English, fast |
| `v3` | `mlx-community/whisper-large-v3-mlx-8bit` | `Systran/faster-whisper-large-v3` | ~1.6 GB / ~3 GB | multilingual / accented / noisy |

Both `faster` presets currently point at the full v3 — there is no 1:1
CT2 port of the distilled turbo from Systran. On a 4090 the full v3 is
already faster than mlx turbo on a Mac, so the aliasing is fine for now.
Swap in `deepdml/faster-whisper-large-v3-turbo-ct2` if a true turbo
distill is wanted later.

Turbo is the distilled 4-decoder large-v3-turbo and is English-tuned — its
multilingual quality is markedly worse than `v3`. Use `v3` for non-English
or code-switched audio (we observed turbo mis-detecting our Hindi-English
test clip as English; v3 detected Hindi).

```bash
uv run python -m steps.transcribe downloads/<id>.m4a              # turbo
uv run python -m steps.transcribe downloads/<id>.m4a -m v3        # full v3
uv run python -m steps.transcribe downloads/<id>.m4a -l en        # force language
```

Writes:
- `downloads/<id>.txt` — plain text
- `downloads/<id>.json` — full result with segments and per-word timestamps

Per-word timestamps come from Whisper's cross-attention DTW
(`word_timestamps=True`); they're decent but loose around speaker turn
boundaries (see [TODO: wav2vec2 forced alignment](#todos)).

### Code-switched audio (Hindi + English)

For Hindi-English code-switched discussions, force `v3` and
`language=hi`. Whisper renders
Hindi in Devanagari and naturally drops English words in Latin script,
which is the MacWhisper-style "Hinglish" output.

```bash
uv run python -m steps.transcribe downloads/<id>.m4a -m v3 -l hi \
  --compression-ratio-threshold 2.0 \
  --hallucination-silence-threshold 2.0
```

### Useful decoder knobs

| flag | what it does |
|---|---|
| `--initial-prompt` | Bias the first chunk's vocabulary. **Often hurts more than it helps** — Whisper can over-anchor to the prompt and lock the first chunk into a loop. Skip unless you have specific domain terms to anchor. |
| `--compression-ratio-threshold` | Default 2.4. Lower (e.g. 2.0) catches repetition loops earlier and triggers temperature-fallback re-decoding sooner. |
| `--hallucination-silence-threshold` | Suppress text generated during silent stretches longer than N seconds. Helps with long monologues that drift into silence. |
| `--temperature` | One value (e.g. 0.0) or the full fallback ladder. |
| `--beam-size` | `faster` backend: default 5. `mlx` backend: raises `NotImplementedError` on mlx-whisper 0.4.3 (greedy only). |
| `--compute-type` | `faster` backend only. CTranslate2 compute type: `float16` (default, 4090), `int8_float16` (lower VRAM). |

---

## Step 3 — Dedupe Whisper repetition loops (`steps/dedupe.py`)

Whisper's greedy decoder occasionally falls into "attractors" and emits a
short n-gram many times in a row (`thank you thank you ...×30`,
`सबसे सबसे ...×30`). `dedupe.py` collapses these in a transcript JSON.

```bash
uv run python -m steps.dedupe downloads/<id>.json
```

Backs the original up to `<id>.raw.json`, overwrites the in-place JSON +
`.txt` with the cleaned version. Two passes:

1. Within each segment's `words[]` list — catches loops emitted as one chunk.
2. Across segments (after dropping zero-duration empties left by the
   silence guard) — catches loops Whisper split into many short segments.

The collapser scores every n-gram length and picks the one with the most
total coverage (ties → smaller n), so a 1-gram repeated 22 times collapses
to 1 even when a larger n-gram would also match.

---

## Step 4 — Diarize (`steps/diarize.py`)

Splits the audio into speaker turns using `pyannote/speaker-diarization-3.1`
(via pyannote.audio 4.x). Speakers are anonymous (`SPEAKER_00`, `SPEAKER_01`,
…) — names get attached later.

One-time setup: get an HF read token, accept the three model gates:

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-community-1

```bash
export HF_TOKEN=hf_xxx...
uv run python -m steps.diarize downloads/<id>.m4a
# optional speaker-count hints:
#   --num-speakers 2          (exact)
#   --min-speakers 2 --max-speakers 4
```

Writes:
- `downloads/<id>.diarization.json` — both `diarization` (may overlap) and
  `exclusive_diarization` (non-overlapping; use this when merging with words)
- `downloads/<id>.rttm` — standard diarization format for evaluation
- `downloads/<id>.16k.wav` — 16 kHz mono cache (pyannote-friendly)

---

## Step 5 — Merge transcript with diarization (`steps/merge.py`)

Assigns each Whisper word to a speaker (max interval overlap against
pyannote's `exclusive_diarization`, snap to nearest turn for words in
silence gaps), then groups consecutive same-speaker words into utterances.

```bash
uv run python -m steps.merge downloads/<id>.m4a
```

Reads `<id>.json` (transcript) and `<id>.diarization.json` (turns), writes:
- `downloads/<id>.diarized.txt` — `[mm:ss - mm:ss] SPEAKER_xx: text`
- `downloads/<id>.diarized.json` — same data with full word lists

Known limitation: Whisper's per-word timestamps come from cross-attention
DTW and tend to be loose around speaker turn boundaries (a trailing word
can leak into the next speaker's segment).

---

## Step 6 — Summarize (optional, `steps/summarize.py`)

Not part of the main orchestrator. Takes a `.diarized.txt` (or `.timed.txt`)
and produces a structured Markdown summary using a local Gemma 4 31B run
through `llama.cpp`.

```bash
uv run python -m steps.summarize <id>.diarized.txt --auto-start
```

Sections in the summary: TL;DR, Key points, Chapters with timestamps,
Main takeaways, Important quotes, Resources.

Full setup (model download, server flags, prefill tuning, sampling
defaults) lives in [`docs/summarize.md`](summarize.md).

---

## TODOs

### `no_repeat_ngram_size` on the faster backend

The CUDA `faster` backend already gives us beam search by default, which
prevents most of the repetition loops that `dedupe.py` was originally
written to clean up. CTranslate2 also exposes `no_repeat_ngram_size` and
`suppress_tokens`; wiring them through would be the natural next step for
eliminating loops at decode time instead of after the fact.

### wav2vec2 forced alignment

Replace Whisper's DTW word timestamps with phoneme-level forced alignment
against the audio (the technique whisperx uses internally). Each word's
start/end gets snapped to the actual acoustic boundary, eliminating the
trailing-word-leaks-into-next-speaker artifact we see at turn changes.
Models to consider: `facebook/wav2vec2-base-960h` (English) or a
Hindi/multilingual variant for code-switched audio. Adds an `align.py`
step between `steps/transcribe.py` and `steps/merge.py`.

### Speaker name attribution

Map `SPEAKER_00` / `SPEAKER_01` / … to actual names. Likely needs either a
small UI for manual labeling on a sample turn per speaker, or automatic
matching against speaker-embedding databases.

---

## Playing a downloaded clip

```bash
open downloads/<id>.m4a       # QuickTime / Music
ffplay -nodisp -autoexit downloads/<id>.m4a   # terminal playback
```
