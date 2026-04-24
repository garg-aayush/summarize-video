"""Collapse Whisper repetition loops in a transcript JSON.

Whisper's greedy decoder can fall into "attractors" where it emits the same
token (or short n-gram) many times in a row, e.g.:

    सबसे सबसे सबसे सबसे ... (×30)
    thank you thank you thank you ... (×N)
    है हुआ है हुआ है हुआ ... (×N)

This module finds consecutive runs of an n-gram and keeps only its first
occurrence. Operates per-segment on the words[] list when present (preserves
timestamps for downstream merging) and falls back to whitespace tokens on
segments that lack word timestamps.
"""

import argparse
import json
import re
from pathlib import Path

# Strip everything except letters/digits and Devanagari (U+0900–U+097F) for
# match-equality. Keeps the original spelling intact in the output.
_NORM_RE = re.compile(r"[^\wऀ-ॿ]+")


def _norm(token: str) -> str:
    return _NORM_RE.sub("", token).lower()


def collapse_runs(
    words: list[dict],
    max_n: int = 6,
    min_repeats: int = 3,
) -> list[dict]:
    """Walk the word list left-to-right; at each position, look for the
    longest n-gram (n in [max_n..1]) that repeats >= min_repeats times in a
    row, and keep just its first occurrence."""
    if not words:
        return words

    cleaned: list[dict] = []
    i = 0
    n_words = len(words)
    while i < n_words:
        # Score every plausible n; prefer the n that covers the most words,
        # breaking ties toward smaller n. This avoids the "n=4 picks up 5
        # repeats of a 4-gram and keeps 4 items" failure mode when the
        # underlying repeat is really a 1-gram x 20.
        best: tuple[int, int] | None = None  # (n, repeats)
        max_check_n = min(max_n, (n_words - i) // min_repeats)
        for n in range(1, max_check_n + 1):
            ngram = tuple(_norm(words[i + k]["word"]) for k in range(n))
            if any(t == "" for t in ngram):
                continue
            repeats = 1
            j = i + n
            while j + n <= n_words:
                nxt = tuple(_norm(words[j + k]["word"]) for k in range(n))
                if nxt != ngram:
                    break
                repeats += 1
                j += n
            if repeats < min_repeats:
                continue
            coverage = n * repeats
            if best is None or coverage > best[0] * best[1]:
                best = (n, repeats)
        if best is not None:
            n, repeats = best
            cleaned.extend(words[i : i + n])
            i += n * repeats
        else:
            cleaned.append(words[i])
            i += 1
    return cleaned


def _collapse_text(text: str, max_n: int, min_repeats: int) -> str:
    tokens = text.split()
    if not tokens:
        return text
    fake = [{"word": t} for t in tokens]
    cleaned = collapse_runs(fake, max_n, min_repeats)
    return " ".join(w["word"] for w in cleaned)


def collapse_segment_runs(
    segments: list[dict],
    max_n: int = 4,
    min_repeats: int = 3,
) -> list[dict]:
    """Collapse runs of consecutive segments whose normalized text matches.
    Whisper sometimes emits a stuck loop as many *separate* segments
    (e.g., 20 consecutive single-segment "thank you"s), which per-segment
    dedup can't see — this catches that case."""
    if not segments:
        return segments
    keyed = [{"word": s.get("text", "").strip(), "_seg": s} for s in segments]
    cleaned = collapse_runs(keyed, max_n=max_n, min_repeats=min_repeats)
    return [k["_seg"] for k in cleaned]


def dedupe_transcript(transcript: dict, max_n: int = 6, min_repeats: int = 3) -> dict:
    # Pass 1: dedupe within each segment's word list.
    new_segments: list[dict] = []
    for seg in transcript.get("segments", []):
        new_seg = dict(seg)
        words = seg.get("words")
        if words:
            cleaned_words = collapse_runs(words, max_n, min_repeats)
            new_seg["words"] = cleaned_words
            # Rebuild text from cleaned words (Whisper words include leading
            # spaces, so a plain join reconstructs the segment text).
            new_seg["text"] = "".join(w["word"] for w in cleaned_words)
            new_seg["start"] = cleaned_words[0]["start"]
            new_seg["end"] = cleaned_words[-1]["end"]
        else:
            new_seg["text"] = _collapse_text(seg.get("text", ""), max_n, min_repeats)
        new_segments.append(new_seg)

    # Drop whitespace-only segments — Whisper's silence guard leaves these as
    # zero-duration artifacts, and they break the "strictly consecutive" check
    # in pass 2 by sitting between identical thank-you/etc. segments.
    new_segments = [s for s in new_segments if s.get("text", "").strip()]

    # Pass 2: dedupe at the segment level (catches loops split across segments).
    new_segments = collapse_segment_runs(new_segments, max_n=4, min_repeats=min_repeats)

    out = dict(transcript)
    out["segments"] = new_segments
    out["text"] = " ".join(s["text"].strip() for s in new_segments if s["text"].strip())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("transcript", type=Path, help="Transcript JSON to clean")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output path. Default: overwrite input (after backing up to .raw.json).",
    )
    parser.add_argument("--max-n", type=int, default=6,
                        help="Largest n-gram length to scan (default 6).")
    parser.add_argument("--min-repeats", type=int, default=3,
                        help="Collapse runs with this many consecutive repeats or more (default 3).")
    args = parser.parse_args()

    raw_text = args.transcript.read_text()
    transcript = json.loads(raw_text)

    n_words_before = sum(len(s.get("words") or []) for s in transcript.get("segments", []))
    n_chars_before = len(transcript.get("text", ""))

    cleaned = dedupe_transcript(transcript, args.max_n, args.min_repeats)

    n_words_after = sum(len(s.get("words") or []) for s in cleaned.get("segments", []))
    n_chars_after = len(cleaned.get("text", ""))

    print(f"Words: {n_words_before} -> {n_words_after}  (collapsed {n_words_before - n_words_after})")
    print(f"Chars: {n_chars_before} -> {n_chars_after}  (collapsed {n_chars_before - n_chars_after})")

    output = args.output or args.transcript
    if output == args.transcript:
        backup = args.transcript.with_suffix(".raw.json")
        if not backup.exists():
            backup.write_text(raw_text)
            print(f"Backed up original to {backup}")
    output.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False))
    txt_path = output.with_suffix(".txt")
    txt_path.write_text(cleaned["text"].strip() + "\n")
    print(f"Wrote: {output}")
    print(f"Wrote: {txt_path}")


if __name__ == "__main__":
    main()
