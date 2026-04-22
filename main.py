import argparse
from pathlib import Path

import yt_dlp

DOWNLOADS_DIR = Path("downloads")


def download_audio(url: str, output_dir: Path = DOWNLOADS_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(output_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "m4a/bestaudio/best",
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "m4a"},
        ],
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        # 403 mitigations: rotate player client, set a realistic UA,
        # bypass geo blocks, and retry transient failures.
        # `web_embedded` + `android_vr` currently serve audio formats without
        # PO tokens or the DRM experiment that affects the `tv` client.
        "extractor_args": {"youtube": {"player_client": ["web_embedded", "android_vr"]}},
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        },
        "geo_bypass": True,
        "retries": 10,
        "fragment_retries": 10,
        "nocheckcertificate": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    return (output_dir / f"{info['id']}.m4a").resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download audio (m4a) from a URL via yt-dlp.")
    parser.add_argument("url", help="Source URL (e.g., a YouTube video)")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=DOWNLOADS_DIR,
        help=f"Directory to write the audio file (default: {DOWNLOADS_DIR})",
    )
    args = parser.parse_args()

    print(f"Downloading audio from: {args.url}")
    audio_path = download_audio(args.url, args.output_dir)
    print(f"Saved audio to: {audio_path}")


if __name__ == "__main__":
    main()
