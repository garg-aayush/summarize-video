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
DEFAULT_MODEL = Path.home() / "MODELS/unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-UD-Q4_K_XL.gguf"
DEFAULT_SERVER_BIN = "llama-server"
PID_FILE = Path("/tmp/summarize-video-llama-server.pid")
LOG_FILE = Path("/tmp/summarize-video-llama-server.log")

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


def _build_server_cmd(model: Path, host: str, port: int, server_bin: str = DEFAULT_SERVER_BIN) -> list[str]:
    """Default llama-server command tuned for Gemma 4 31B.

    Summarization is prefill-bound (huge input, small output), so the main
    levers are `--ubatch-size` (GPU kernel batch — 2x default of 512) and
    `--batch-size` (logical prefill batch). KV-cache quantization halves
    the cache memory; flash-attn is required to use a quantized cache.

    `--ubatch-size 1024` works on both 24 GB CUDA cards (RTX 4090) and
    36 GB Macs. Going higher (2048) buys ~2x prefill speed but pushes
    activation memory to ~4 GB which OOMs the 4090
    (19 GB weights + 3 GB q8 KV + 4 GB activations > 24 GB). On bigger
    cards (e.g. A6000 48 GB) push it via --server-cmd.

    `-c 32768` rather than 65536 because the orchestrator's own CUDA
    context (torch + CT2 + pyannote) keeps ~1.2 GB reserved even after
    empty_cache — that shadow leaves only ~23 GB visible to llama-server,
    which 65K projects over. 32K fits with ~1.5 GB of headroom, and is
    still plenty: a 2-hour transcript is ~15K tokens. Bigger cards or
    standalone llama-server runs can bump it via --server-cmd.

    Mirrors the recipe in docs/summarize.md.
    """
    return [
        server_bin,
        "-m", str(model),
        "-ngl", "99",
        "-c", "32768",
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
    server_bin: str = DEFAULT_SERVER_BIN,
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
            raise FileNotFoundError(
                f"Model file not found: {model}\n"
                f"Download it (see docs/summarize.md) or pass --summarize-model PATH."
            )
        cmd = _build_server_cmd(model, host, port, server_bin=server_bin)

    print(f"Spawning llama-server: {shlex.join(cmd)}")
    pid = _spawn_server(cmd)
    print(f"  PID {pid}, log: {LOG_FILE}")
    print("  Waiting for model to load (this can take 30-90s)...")
    _wait_for_server(server_url, timeout=wait_timeout)
    print(f"Server ready at {server_url}.")


def stop_server(quiet: bool = False) -> bool:
    """Stop the auto-started llama-server. Returns True if anything was
    actually killed, False if there was nothing to stop. Non-fatal — safe
    to call in a `finally` block."""
    if not PID_FILE.exists():
        if not quiet:
            print(f"No PID file at {PID_FILE}; nothing to stop.")
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
    except ValueError:
        PID_FILE.unlink(missing_ok=True)
        if not quiet:
            print(f"Bad PID in {PID_FILE}; removed it.")
        return False

    if not quiet:
        print(f"Stopping llama-server (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        if not quiet:
            print(f"  Process {pid} not running; cleaning up PID file.")
        PID_FILE.unlink(missing_ok=True)
        return False

    # Up to 10s for graceful shutdown, then SIGKILL.
    for _ in range(20):
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
    else:
        if not quiet:
            print(f"  Force-killing {pid}...")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    PID_FILE.unlink(missing_ok=True)
    if not quiet:
        print("Stopped.")
    return True


# CLI-facing wrapper kept for back-compat: prints + exits like before.
def _stop_server() -> None:
    if not stop_server():
        sys.exit("Nothing to stop (or wasn't started via --auto-start).")


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


# --- public API --------------------------------------------------------------

def summarize(
    transcript: Path,
    output: Path | None = None,
    server_url: str = DEFAULT_SERVER,
    model: Path = DEFAULT_MODEL,
    auto_start: bool = True,
    server_cmd: str | None = None,
    server_bin: str = DEFAULT_SERVER_BIN,
    server_wait_timeout: int = 180,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    timeout: int = 900,
) -> Path:
    """Summarize a transcript via llama-server; returns the output path.

    If `auto_start` is True (the default when called from the orchestrator),
    spawns `llama-server` if one isn't already reachable at `server_url`.
    The server is left running so subsequent calls are fast.
    """
    text = Path(transcript).read_text()
    if auto_start:
        _ensure_server(server_url, model, server_cmd, server_wait_timeout, server_bin)
    elif not _server_alive(server_url):
        raise RuntimeError(
            f"Cannot reach llama-server at {server_url}. "
            "Start it manually (see docs/summarize.md), or pass auto_start=True."
        )
    raw = _post_chat(server_url, text, temperature, max_tokens, timeout)
    out = output or Path(transcript).with_suffix(".summary.md")
    out.write_text(_clean_output(raw) + "\n")
    return out


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
    server.add_argument("--server-bin", default=DEFAULT_SERVER_BIN,
                        help=f"llama-server binary (name on PATH or absolute path). "
                             f"Default: {DEFAULT_SERVER_BIN}.")
    server.add_argument("--server-wait-timeout", type=int, default=180,
                        help="Seconds to wait for --auto-start server to become ready. Default 180.")
    args = parser.parse_args()

    if args.stop_server:
        _stop_server()
        return

    if args.transcript is None:
        parser.error("transcript is required (unless using --stop-server)")

    print(f"Transcript: {args.transcript} ({len(args.transcript.read_text()):,} chars)")
    print("Generating summary (this can take a few minutes)...")
    output = summarize(
        args.transcript,
        output=args.output,
        server_url=args.server_url,
        model=args.model,
        auto_start=args.auto_start,
        server_cmd=args.server_cmd,
        server_bin=args.server_bin,
        server_wait_timeout=args.server_wait_timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    _print_loaded_models(args.server_url)
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
