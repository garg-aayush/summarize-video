import argparse
import json
from bisect import bisect_left
from pathlib import Path

UNKNOWN = "UNKNOWN"


def assign_speaker(word_start: float, word_end: float, turns: list[dict],
                   turn_starts: list[float]) -> str:
    """Pick the speaker whose turn overlaps the word the most.
    Falls back to the nearest turn in time if there's no overlap (word in a gap).
    `turns` must be sorted by start; `turn_starts` is the parallel list of start times.
    """
    if not turns:
        return UNKNOWN

    # Candidate turns: any whose [start, end] could overlap [word_start, word_end].
    # Start scanning from the last turn that begins at or before word_end.
    hi = bisect_left(turn_starts, word_end)
    # And walk back until the turn ends before word_start.
    lo = hi
    while lo > 0 and turns[lo - 1]["end"] > word_start:
        lo -= 1

    best_speaker = None
    best_overlap = 0.0
    for t in turns[lo:hi]:
        overlap = min(word_end, t["end"]) - max(word_start, t["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = t["speaker"]

    if best_speaker is not None:
        return best_speaker

    # No overlap — word sits in a gap. Snap to nearest turn by time distance.
    word_mid = 0.5 * (word_start + word_end)
    nearest = min(
        turns,
        key=lambda t: 0 if t["start"] <= word_mid <= t["end"]
                        else min(abs(t["start"] - word_mid), abs(t["end"] - word_mid)),
    )
    return nearest["speaker"]


def iter_words(transcript: dict):
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            if w.get("start") is None or w.get("end") is None:
                continue
            yield w


def group_into_utterances(words: list[dict]) -> list[dict]:
    """Group consecutive same-speaker words into utterances."""
    utterances: list[dict] = []
    for w in words:
        if utterances and utterances[-1]["speaker"] == w["speaker"]:
            u = utterances[-1]
            u["end"] = w["end"]
            u["text"] += w["word"]  # whisper words come with leading spaces
            u["words"].append(w)
        else:
            utterances.append({
                "speaker": w["speaker"],
                "start": w["start"],
                "end": w["end"],
                "text": w["word"],
                "words": [w],
            })
    for u in utterances:
        u["text"] = u["text"].strip()
    return utterances


def merge(transcript: dict, diarization: dict) -> list[dict]:
    turns = sorted(diarization["exclusive_diarization"], key=lambda t: t["start"])
    turn_starts = [t["start"] for t in turns]

    annotated: list[dict] = []
    for w in iter_words(transcript):
        speaker = assign_speaker(w["start"], w["end"], turns, turn_starts)
        annotated.append({**w, "speaker": speaker})

    return group_into_utterances(annotated)


def fmt_ts(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


def write_outputs(audio_path: Path, utterances: list[dict]) -> tuple[Path, Path]:
    json_path = audio_path.with_suffix(".diarized.json")
    txt_path = audio_path.with_suffix(".diarized.txt")
    json_path.write_text(json.dumps(utterances, indent=2, ensure_ascii=False))
    with txt_path.open("w") as f:
        for u in utterances:
            f.write(f"[{fmt_ts(u['start'])} - {fmt_ts(u['end'])}] {u['speaker']}: {u['text']}\n")
    return json_path, txt_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Whisper words with pyannote turns into speaker-attributed utterances."
    )
    parser.add_argument("audio", type=Path, help="Audio path; expects sibling .json and .diarization.json")
    parser.add_argument("--transcript", type=Path, default=None,
                        help="Override transcript JSON path")
    parser.add_argument("--diarization", type=Path, default=None,
                        help="Override diarization JSON path")
    args = parser.parse_args()

    transcript_path = args.transcript or args.audio.with_suffix(".json")
    diarization_path = args.diarization or args.audio.with_suffix(".diarization.json")

    transcript = json.loads(transcript_path.read_text())
    diarization = json.loads(diarization_path.read_text())

    utterances = merge(transcript, diarization)
    json_path, txt_path = write_outputs(args.audio, utterances)

    speakers = sorted({u["speaker"] for u in utterances})
    print(f"Speakers: {len(speakers)} ({', '.join(speakers)})")
    print(f"Utterances: {len(utterances)}")
    print(f"Wrote: {txt_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
