import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
from pyannote.audio import Pipeline

DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"


def prepare_audio(audio_path: Path) -> Path:
    """pyannote wants 16 kHz mono. Convert with ffmpeg and cache next to source."""
    wav_path = audio_path.with_suffix(".16k.wav")
    if wav_path.exists():
        return wav_path
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(audio_path),
            "-ac", "1", "-ar", "16000",
            str(wav_path),
        ],
        check=True,
    )
    return wav_path


def load_pipeline(model: str = DEFAULT_MODEL) -> Pipeline:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set. Get a token at https://huggingface.co/settings/tokens "
            "and accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1 "
            "and https://huggingface.co/pyannote/segmentation-3.0."
        )
    pipeline = Pipeline.from_pretrained(model, token=token)
    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
    return pipeline


def diarize(
    audio_path: Path,
    pipeline: Pipeline,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict:
    """Returns the serialized DiarizeOutput dict:
        {"diarization": [...], "exclusive_diarization": [...]}
    `diarization` may contain overlapping turns; `exclusive_diarization` won't.
    """
    wav = prepare_audio(audio_path)
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers
    output = pipeline(str(wav), **kwargs)
    return output.serialize()


def write_outputs(audio_path: Path, result: dict) -> tuple[Path, Path]:
    json_path = audio_path.with_suffix(".diarization.json")
    rttm_path = audio_path.with_suffix(".rttm")
    json_path.write_text(json.dumps(result, indent=2))
    # RTTM uses the full (possibly overlapping) diarization.
    file_id = audio_path.stem
    with rttm_path.open("w") as f:
        for s in result["diarization"]:
            dur = s["end"] - s["start"]
            f.write(
                f"SPEAKER {file_id} 1 {s['start']:.3f} {dur:.3f} <NA> <NA> {s['speaker']} <NA> <NA>\n"
            )
    return json_path, rttm_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Diarize an audio file with pyannote.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading pipeline: {args.model}")
    pipeline = load_pipeline(args.model)

    print(f"Diarizing {args.audio}")
    result = diarize(
        args.audio, pipeline,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    turns = result["diarization"]
    speakers = sorted({s["speaker"] for s in turns})
    total = sum(s["end"] - s["start"] for s in turns)
    print(f"Speakers found: {len(speakers)} ({', '.join(speakers)})")
    print(f"Total speech: {total:.1f}s across {len(turns)} turns "
          f"({len(result['exclusive_diarization'])} non-overlapping)")

    json_path, rttm_path = write_outputs(args.audio, result)
    print(f"Wrote: {json_path}")
    print(f"Wrote: {rttm_path}")


if __name__ == "__main__":
    main()
