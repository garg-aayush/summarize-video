# Definitions

Plain-English glossary for the jargon that shows up in this repo. Aimed at someone new to ASR / diarization / Whisper.

---

## ASR (Automatic Speech Recognition)

Turning audio into text. Whisper is an ASR model.

## Whisper

OpenAI's speech-to-text model. Two relevant properties for us:

- **Encoder–decoder transformer.** The encoder ingests a 30-second mel spectrogram, the decoder generates text tokens autoregressively.
- **Multilingual.** Trained on 99 languages and on translation tasks. You can force a language with `language="hi"` or let it auto-detect from the first chunk.

## large-v3 vs large-v3-turbo

Two flavors of the same family:

- **large-v3** — 32 decoder layers, the high-quality reference model.
- **large-v3-turbo** — distilled down to **4 decoder layers**, ~5–8× faster, but English-tuned. Multilingual quality drops noticeably (e.g., it mis-detects Hindi as English on our test clip).

## MLX / mlx-whisper

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework optimized for Apple Silicon (unified memory, Metal kernels, Neural Engine on supported ops). `mlx-whisper` is a port of Whisper that runs natively on M-series Macs without going through PyTorch + MPS.

## Quantization (fp16, 8-bit)

The model weights are originally float32 (4 bytes per number). Quantization stores them in fewer bits to save memory and speed up math:

- **fp16** — 16-bit floats, ~half the size, near-lossless quality.
- **8-bit** — integer quantization, ~quarter the size of fp32, small quality hit. Common for the full v3 model so it fits in ~1.6 GB.

## Greedy decoding vs beam search

How the decoder picks the next token at each step:

- **Greedy** — always pick the single most likely next token. Fast, deterministic, but can lock into bad paths (e.g., repetition loops).
- **Beam search** — keep the top-k candidate sequences alive at each step and pick the best overall. Slower but more robust. mlx-whisper 0.4.3 doesn't implement it.

## Temperature / temperature fallback

**Temperature** is a knob on token sampling: 0.0 = greedy, higher = more random. Whisper's standard "fallback ladder" tries `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` in order. If a chunk's output looks bad (see compression-ratio below), it bumps the temperature and re-decodes that chunk.

## compression_ratio_threshold

A heuristic for "is the output garbage?". After decoding a chunk, Whisper compresses the text with gzip and computes `len(text) / len(gzipped)`. Highly repetitive text (`thank you thank you ...`) compresses extremely well, so the ratio spikes. If it exceeds the threshold (default 2.4), Whisper triggers temperature fallback. Lowering to 2.0 catches loops sooner.

## hallucination_silence_threshold

Whisper sometimes "hallucinates" text during long silences — generating plausible-sounding sentences from background noise. This flag (in seconds) suppresses output during silent stretches longer than N. Requires word timestamps to know where silence is.

## initial_prompt

A string fed to the decoder before the first chunk. Originally meant to bias vocabulary (e.g., proper nouns, technical terms). In practice on our data it backfired — the decoder over-anchored to the prompt and locked the first chunk into a loop. We don't use it.

## DTW (Dynamic Time Warping) word timestamps

Whisper doesn't natively know where each word begins and ends — only chunk text and chunk start/end. For per-word times, it runs DTW on the decoder's **cross-attention weights** to align each token to a slice of the audio. Cheap (no extra model), but loose — words near speaker turn boundaries often drift by tens of milliseconds.

## Forced alignment / wav2vec2

A more accurate way to get word timestamps: take the already-known transcript and "force" it onto the audio using a phoneme-level acoustic model like wav2vec2. Each phoneme's location in the audio is found exactly, so word boundaries are tight. This is what [whisperx](https://github.com/m-bain/whisperX) does internally and what our `align.py` TODO would add.

## Attractor (repetition loop)

A degenerate decoding state where greedy sampling keeps re-emitting the same short n-gram (`thank you thank you ...×30`,  `सबसे सबसे ...×30`). Caused by the local maximum trap of greedy decoding. Beam search, `no_repeat_ngram_size`, and `suppress_tokens` are the textbook fixes — none available in mlx-whisper 0.4.3, which is why we have `dedupe.py`.

## Diarization

"Who spoke when?" Splits an audio file into speaker turns: `SPEAKER_00 from 0.0–4.2s, SPEAKER_01 from 4.2–7.8s, ...`. Note: the speakers are **anonymous** — `SPEAKER_00` is just an arbitrary cluster, not a person's name.

## pyannote / pyannote.audio

The leading open-source diarization toolkit. `pyannote/speaker-diarization-3.1` is the pretrained pipeline we use. It chains:

1. Voice activity detection (where is anyone speaking)
2. Speaker segmentation (where do speakers change)
3. Speaker embedding + clustering (which segments belong together)

## Exclusive diarization vs (overlapping) diarization

- **`diarization`** — raw output, may contain overlapping turns when two speakers talk over each other.
- **`exclusive_diarization`** — non-overlapping version (overlaps split or re-assigned). Easier to consume when assigning each word to one speaker.

## RTTM (Rich Transcription Time Marked)

Standard text format for diarization output. One line per turn:

```
SPEAKER <file_id> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
```

Used by evaluation tools like `pyannote.metrics` and `dscore`.

## Speaker turn / overlap

A **turn** is a continuous stretch where one speaker is speaking. **Overlap** is when two or more speakers talk simultaneously. `exclusive_diarization` flattens overlap into single-speaker spans.

## Code-switching / Hinglish

When a speaker mixes languages mid-sentence ("मैंने उसे call किया but he didn't answer"). Whisper's multilingual model handles this naturally if you force the dominant language — it transcribes Hindi in Devanagari and emits English words in Latin script, which is the MacWhisper-style "Hinglish" output.

## yt-dlp PO Token / DRM experiment / player_client

YouTube exposes its catalog through several internal "client" identities (web, ios, tv, …), each with different format menus. To slow scraping they've started:

- requiring a **PO Token** (proof of origin) on `web`/`ios`/`mweb` — formats served without it are skipped
- gating the `tv` client behind a **DRM experiment** that flags every format as DRM-protected

`yt-dlp` lets us pick which clients to ask via `extractor_args="youtube:player_client=..."`. We use `web_embedded` + `android_vr`, which still serve plain m4a without either restriction.

## HF gated models

Some Hugging Face models require you to accept terms before downloading (typically because the authors want to track usage or restrict commercial use). For pyannote you click through the gate on the model page once, then your `HF_TOKEN` environment variable carries the acceptance to your local `from_pretrained()` call.

## MPS

Metal Performance Shaders — Apple's GPU compute API. PyTorch exposes it as the `mps` device, equivalent to `cuda` on NVIDIA. pyannote uses MPS on Mac when available; mlx-whisper bypasses PyTorch entirely and uses MLX/Metal directly.
