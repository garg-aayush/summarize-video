# Definitions

## ASR (Automatic Speech Recognition)

ASR is just turning audio into text. Whisper is the ASR model I use.

## Whisper

OpenAI's speech-to-text model. Two properties of it matter for me:

- **Encoder-decoder transformer.** The encoder takes 30 seconds of audio at a time (converted into a spectrogram, basically a picture of the sound) and the decoder writes out the text one token at a time.
- **Multilingual.** It was trained on 99 languages and on translation tasks too. I can force a language with `language="hi"` or let it auto-detect from the first chunk.

### large-v3 vs large-v3-turbo

- **large-v3** has 32 decoder layers. It's the high-quality reference model.
- **large-v3-turbo** is distilled down to **4 decoder layers**. Roughly 5 to 8 times faster, but English-tuned. Multilingual quality drops noticeably (e.g. it mis-detects Hindi as English on my test clip).

## Greedy decoding vs beam search

How the decoder picks the next token at each step.

- **Greedy** always picks the single most likely next token. Fast and deterministic, but it can lock into bad paths (e.g. repetition loops).
- **Beam search** keeps the top-k candidate sequences alive at each step and picks the best overall. Slower but more robust. mlx-whisper 0.4.3 does not implement it.

## Temperature / temperature fallback

**Temperature** is a knob on token sampling: 0.0 is greedy, higher is more random. Whisper's standard "fallback ladder" tries `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` in order. If a chunk's output looks bad (see compression-ratio below), it bumps the temperature and re-decodes that chunk.

## compression_ratio_threshold

A heuristic for "is the output garbage?". After decoding a chunk, Whisper compresses the text with gzip and computes `len(text) / len(gzipped)`. Highly repetitive text (`thank you thank you ...`) compresses extremely well, so the ratio spikes. If it exceeds the threshold (default 2.4), Whisper triggers temperature fallback. I lower it to 2.0 to catch loops sooner.

## hallucination_silence_threshold

Whisper sometimes "hallucinates" text during long silences, generating plausible-sounding sentences out of background noise. This flag (in seconds) suppresses output during silent stretches longer than N seconds. It requires word timestamps so it knows where the silence is.

## DTW (Dynamic Time Warping) word timestamps

Whisper does not natively know where each word begins and ends. It only knows the chunk text and the chunk's start/end. To get per-word times, it runs DTW on the decoder's **cross-attention weights** to line each token up with a slice of the audio (basically, "which part of the audio was the model paying attention to when it wrote this word?"). Cheap, since there's no extra model, but loose. Words near speaker turn boundaries often drift by tens of milliseconds.

## Forced alignment / wav2vec2

A more accurate way to get word timestamps. I take the already-known transcript and "force" it onto the audio using a phoneme-level acoustic model like wav2vec2. A phoneme is the smallest sound unit (the `k` in `cat`), and wav2vec2 can point at exactly where each one sits in the audio, so word boundaries come out tight. This is what [whisperx](https://github.com/m-bain/whisperX) does internally.

## Diarization

"Who spoke when?". This splits an audio file into speaker turns: `SPEAKER_00 from 0.0 to 4.2s, SPEAKER_01 from 4.2 to 7.8s, ...`. Note that the speakers are **anonymous**. `SPEAKER_00` is just an arbitrary cluster, not a person's name.

## pyannote / pyannote.audio

The leading open-source diarization toolkit. `pyannote/speaker-diarization-3.1` is the pretrained pipeline I use. It chains:

1. Voice activity detection (where is anyone speaking)
2. Speaker segmentation (where do speakers change)
3. Speaker embedding + clustering (which segments belong to the same person)

## Exclusive diarization vs (overlapping) diarization

- **`diarization`** is the raw output. It may contain overlapping turns when two speakers talk over each other.
- **`exclusive_diarization`** is the non-overlapping version, where overlaps are split or re-assigned. Easier to consume when I am assigning each word to one speaker.