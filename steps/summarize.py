"""Summarize a transcript via a local llama.cpp server (Gemma 4 31B).

Talks to a llama-server OpenAI-compatible endpoint. By default expects the
server to already be running; pass --auto-start to spawn one if it isn't
(left running on success so subsequent calls are instant).

See docs/summarize.md for the full server setup.
"""

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_SERVER = "http://127.0.0.1:8081"
DEFAULT_MODEL = Path.home() / "models/gemma-4-31b/gemma-4-31B-it-UD-Q4_K_XL.gguf"
PID_FILE = Path("/tmp/podcasts-llama-server.pid")
LOG_FILE = Path("/tmp/podcasts-llama-server.log")

SYSTEM_PROMPT = """\
<role>
You are a podcast and youtube video summarizer.
</role>

<instructions>
You will be given a transcript of a podcast or youtube video. You will need to produce a tight, scannable summary of the transcript.
The transcript will be given in either of the following formats:
[mm:ss - mm:ss] text
[mm:ss - mm:ss] SPEAKER_xx: text

It will appears inside <transcript> tags.

The output should contain the following information:
- tldr: a short summary of the transcript
- key points: a list of key points from the transcript
- chapters: One line per thematic chapter, in order: `[mm:ss] Short title (≤8 words)`. Aim for 5–10 chapters covering the whole podcast.
- main takeaways: a list of main takeaways from the transcript
- important quotes: 1–4 verbatim quotes that best capture voice or argument. Attribute the speaker if labels are present. Never paraphrase.
- resources: Named entities mentioned: people, books, papers, companies, tools, URLs, events. One per bullet, with enough context to identify each.
</instructions>
"""


# --- server lifecycle --------------------------------------------------------

def _server_alive(server_url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"{server_url}/v1/models", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _print_loaded_models(server_url: str) -> None:
    try:
        with urllib.request.urlopen(f"{server_url}/v1/models", timeout=5) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError):
        return
    models = [m.get("id") for m in data.get("data", []) if m.get("id")]
    if models:
        print(f"Loaded model(s): {', '.join(models)}")


def _build_server_cmd(model: Path, host: str, port: int) -> list[str]:
    """Default llama-server command tuned for Gemma 4 31B on a 36 GB Mac.

    Summarization is prefill-bound (huge input, small output), so the main
    levers are `--ubatch-size` (Metal kernel batch — 2x default of 512) and
    `--batch-size` (logical prefill batch). KV-cache quantization halves the
    cache memory; flash-attn is required to use a quantized cache.

    Mirrors the recipe in docs/summarize.md.
    """
    return [
        "llama-server",
        "-m", str(model),
        "-ngl", "99",
        "-c", "65536",
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--parallel", "1",
        "--batch-size", "2048",
        "--ubatch-size", "1024",
        "--context-shift",
        "--metrics",
        "--jinja",
        "--host", host,
        "--port", str(port),
    ]


def _spawn_server(cmd: list[str]) -> int:
    """Start llama-server detached from this script and return its PID."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = LOG_FILE.open("a")
    log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} starting: {shlex.join(cmd)} ===\n")
    log.flush()
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT,
        # New session so the server keeps running after this script exits.
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))
    return proc.pid


def _wait_for_server(server_url: str, timeout: int = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _server_alive(server_url, timeout=2.0):
            return
        time.sleep(1.0)
    sys.exit(
        f"llama-server didn't become ready within {timeout}s. "
        f"Check the log: {LOG_FILE}"
    )


def _ensure_server(
    server_url: str,
    model: Path,
    server_cmd: str | None,
    wait_timeout: int,
) -> None:
    """If the server is already up, just return. Otherwise spawn one and wait."""
    if _server_alive(server_url):
        print(f"Server already up at {server_url}.")
        return

    parsed = urllib.parse.urlparse(server_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8080

    if server_cmd:
        cmd = shlex.split(server_cmd)
    else:
        if not model.exists():
            sys.exit(
                f"Model file not found: {model}\n"
                f"Download it (see docs/summarize.md) or pass --model PATH."
            )
        cmd = _build_server_cmd(model, host, port)

    print(f"Spawning llama-server: {shlex.join(cmd)}")
    pid = _spawn_server(cmd)
    print(f"  PID {pid}, log: {LOG_FILE}")
    print("  Waiting for model to load (this can take 30-90s)...")
    _wait_for_server(server_url, timeout=wait_timeout)
    print(f"Server ready at {server_url}.")


def _stop_server() -> None:
    if not PID_FILE.exists():
        sys.exit(
            f"No PID file at {PID_FILE}. Nothing to stop "
            "(or the server wasn't started by --auto-start)."
        )
    try:
        pid = int(PID_FILE.read_text().strip())
    except ValueError:
        PID_FILE.unlink(missing_ok=True)
        sys.exit(f"Bad PID in {PID_FILE}; removed it.")

    print(f"Stopping llama-server (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"  Process {pid} not running; cleaning up PID file.")
        PID_FILE.unlink(missing_ok=True)
        return

    # Up to 10s for graceful shutdown, then SIGKILL.
    for _ in range(20):
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
    else:
        print(f"  Force-killing {pid}...")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    PID_FILE.unlink(missing_ok=True)
    print("Stopped.")


# --- chat call ---------------------------------------------------------------

def _post_chat(
    server_url: str,
    transcript: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    # Sampling tuned for structured extraction, not reasoning. Top-p / top-k
    # are Google's recommended Gemma defaults; min_p trims tail noise that
    # slips past top-p at low temperatures; repeat_penalty stays at 1.0 since
    # the XML scaffold legitimately repeats tags.
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<transcript>\n{transcript}\n</transcript>"},
        ],
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 64,
        "min_p": 0.05,
        "repeat_penalty": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"]


def _clean_output(response: str) -> str:
    """Strip optional surrounding code fences the model sometimes wraps the
    response in (e.g. ```markdown ... ```)."""
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\s*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text


# --- entrypoint --------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("transcript", type=Path, nargs="?",
                        help="Transcript file (.diarized.txt preferred, .timed.txt also works). "
                             "Optional only when using --stop-server.")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path. Default: <input>.summary.md next to the input.")
    parser.add_argument("--server-url", default=DEFAULT_SERVER,
                        help=f"llama-server base URL (default: {DEFAULT_SERVER}).")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature. Lower = more deterministic. Default 0.3.")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Cap on summary length in tokens. Default 4096.")
    parser.add_argument("--timeout", type=int, default=900,
                        help="HTTP timeout in seconds. Default 900 (15 min).")

    server = parser.add_argument_group("server lifecycle")
    server.add_argument("--auto-start", action="store_true",
                        help="If the server isn't reachable, spawn one and wait for it. "
                             "Left running on success.")
    server.add_argument("--stop-server", action="store_true",
                        help=f"Stop the server we previously auto-started (via {PID_FILE}) and exit.")
    server.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help=f"GGUF model path used when --auto-start spawns the server. "
                             f"Default: {DEFAULT_MODEL}")
    server.add_argument("--server-cmd", default=None,
                        help="Full custom command for --auto-start (overrides --model + defaults). "
                             "Pass as a single quoted string.")
    server.add_argument("--server-wait-timeout", type=int, default=180,
                        help="Seconds to wait for --auto-start server to become ready. Default 180.")
    args = parser.parse_args()

    if args.stop_server:
        _stop_server()
        return

    if args.transcript is None:
        parser.error("transcript is required (unless using --stop-server)")

    transcript = args.transcript.read_text()
    print(f"Transcript: {args.transcript} ({len(transcript):,} chars)")

    if args.auto_start:
        _ensure_server(args.server_url, args.model, args.server_cmd, args.server_wait_timeout)
    elif not _server_alive(args.server_url):
        sys.exit(
            f"Cannot reach llama-server at {args.server_url}.\n"
            f"Start it manually (see docs/summarize.md), or re-run with --auto-start."
        )
    _print_loaded_models(args.server_url)

    print("Generating summary (this can take a few minutes)...")
    raw = _post_chat(args.server_url, transcript, args.temperature, args.max_tokens, args.timeout)

    output = args.output or args.transcript.with_suffix(".summary.md")
    output.write_text(_clean_output(raw) + "\n")
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
