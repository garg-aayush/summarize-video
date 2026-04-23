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

You may also receive an `<episode_context>` block before the transcript. It contains metadata extracted from the video's description (show, host, guests with roles, themes). When present, use it to attribute SPEAKER_xx labels to real names where you can do so confidently from the transcript content, to identify the show, and to ground references to people, organizations, or events. Never let the context override what the transcript actually says — if they conflict, the transcript wins.

The output should contain the following information:
- tldr: a short summary of the transcript
- key points: a list of key points from the transcript
- chapters: One line per thematic chapter, in order: `[mm:ss] Short title (≤8 words)`. Aim for 5–10 chapters covering the whole podcast.
- main takeaways: a list of main takeaways from the transcript
- important quotes: 1–4 verbatim quotes that best capture voice or argument. Attribute the speaker if labels are present. Never paraphrase.
- resources: Named entities mentioned: people, books, papers, companies, tools, URLs, events. One per bullet, with enough context to identify each.
</instructions>
"""

EPISODE_CONTEXT_SYSTEM_PROMPT = """\
<role>
You are an information extraction agent that extracts video-specific metadata from a given noisy YouTube description. The output is used as grounding context for a downstream summarizer for the video.
</role>

<rules>
- Extract only what the description explicitly states. Never infer or use outside knowledge.
- Omit any field that is not present. No "N/A" or "unknown."
- Ignore: subscribe/follow links, social handles, promo and referral codes, sponsor pitches, other-channel cross-promotion, contact emails, generic SEO hashtags.
- Ignore the channel-level "About" block (usually at the bottom, describing the channel in general). Its themes are not episode themes.
- Ignore comma-separated SEO keyword dumps. Extract guests and themes only from prose sentences.
- Prefer full names with stated roles ("Demis Hassabis, CEO of Google DeepMind") over handles.
- Keep episode titles in their original language. Translate role and affiliation phrases to English.
- If nothing episode-specific is present, output the fallback.
</rules>

<output_format>
Output one Markdown block. Omit any line whose field is missing. No preamble, no code fences.

## Episode Context
- **Show:**
- **Title or topic:**
- **Host:**
- **Guests:** [one bullet per guest if multiple]
- **Event or venue:**
- **Themes promised:**
- **Language:**

Fallback when nothing episode-specific is extractable:

## Episode Context
_No episode-specific context available in description._
</output_format>
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


def _pick_ubatch_and_ctx() -> tuple[str, str]:
    """Pick (--ubatch-size, -c) based on available VRAM.

    24 GB CUDA cards (4090 / 3090) in the orchestrator flow: 512 / 64K.
      Going to ubatch 512 trades ~2x prefill speed for ~1 GB of freed
      activation memory, which we spend on keeping the full 64K. The
      fit is tight — llama.cpp warns about the 1 GB safety margin —
      but the actual model + KV + compute buffers land at ~22.3 GB,
      leaving ~650 MiB of real headroom after the ~1.2 GB CUDA-context
      shadow the orchestrator process can't release.

    Larger CUDA cards (A6000, H100) and Apple Silicon: 1024 / 64K.
      More memory headroom means we can keep the faster prefill at the
      same context. Macs see ~2x prefill speedup from 512 → 1024.
    """
    try:
        import torch
        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_gb < 32:
                return "512", "65536"
    except ImportError:
        pass
    return "1024", "65536"


def _build_server_cmd(model: Path, host: str, port: int, server_bin: str = DEFAULT_SERVER_BIN) -> list[str]:
    """Default llama-server command tuned for Gemma 4 31B.

    Summarization is prefill-bound (huge input, small output), so the main
    levers are `--ubatch-size` (GPU kernel batch) and `--batch-size`
    (logical prefill batch). KV-cache quantization halves the cache memory;
    flash-attn is required to use a quantized cache.

    ubatch / ctx defaults are picked by `_pick_ubatch_and_ctx()` based on
    detected VRAM. On 4090-class cards the orchestrator flow keeps ~1.2 GB
    of CUDA context reserved (torch + CT2 + pyannote) that `empty_cache`
    can't free, so we drop ubatch to 512 to claw back ~1 GB of activation
    headroom and keep the full 64K context. Larger cards and Macs get
    the faster 1024 / 64K recipe. Either can be overridden with
    --server-cmd.

    Mirrors the recipe in docs/summarize.md.
    """
    ubatch, ctx = _pick_ubatch_and_ctx()
    return [
        server_bin,
        "-m", str(model),
        "-ngl", "99",
        "-c", ctx,
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--parallel", "1",
        "--batch-size", "2048",
        "--ubatch-size", ubatch,
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

def _chat(
    server_url: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    chat_template_kwargs: dict | None = None,
) -> str:
    # Sampling tuned for structured extraction, not reasoning. Top-p / top-k
    # are Google's recommended Gemma defaults; min_p trims tail noise that
    # slips past top-p at low temperatures; repeat_penalty stays at 1.0 since
    # the XML scaffold legitimately repeats tags.
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 64,
        "min_p": 0.05,
        "repeat_penalty": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"]


def _build_user_message(transcript: str, context: str | None) -> str:
    if context and context.strip():
        return (
            f"<episode_context>\n{context.strip()}\n</episode_context>\n"
            f"<transcript>\n{transcript}\n</transcript>"
        )
    return f"<transcript>\n{transcript}\n</transcript>"


def extract_episode_context(
    description: str,
    server_url: str = DEFAULT_SERVER,
    *,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout: int = 300,
) -> str:
    """Distill a noisy YouTube description into a structured Episode Context block.

    Caller is responsible for ensuring llama-server is reachable at
    `server_url`; this function does not spawn one.

    Reasoning is disabled for this call: with `--jinja` Gemma 4's chat
    template enables thinking by default, and a small structured-output task
    like this can have its entire `max_tokens` budget consumed by hidden
    `<think>...</think>` tokens before any visible answer is emitted (see
    docs/experiments.md "Episode context — Gemma's hidden reasoning ate
    the extraction budget"). The main `summarize()` call still uses default
    template behavior.
    """
    raw = _chat(
        server_url,
        EPISODE_CONTEXT_SYSTEM_PROMPT,
        f"<description>\n{description}\n</description>",
        temperature,
        max_tokens,
        timeout,
        chat_template_kwargs={"enable_thinking": False},
    )
    return _clean_output(raw)


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
    context: str | None = None,
) -> Path:
    """Summarize a transcript via llama-server; returns the output path.

    If `auto_start` is True (the default when called from the orchestrator),
    spawns `llama-server` if one isn't already reachable at `server_url`.
    The server is left running so subsequent calls are fast.

    `context` is an optional pre-extracted Episode Context block (see
    `extract_episode_context`); when present it's prepended to the user
    message inside an `<episode_context>` tag so the model can ground
    speaker labels and named entities.
    """
    text = Path(transcript).read_text()
    if auto_start:
        _ensure_server(server_url, model, server_cmd, server_wait_timeout, server_bin)
    elif not _server_alive(server_url):
        raise RuntimeError(
            f"Cannot reach llama-server at {server_url}. "
            "Start it manually (see docs/summarize.md), or pass auto_start=True."
        )
    user_message = _build_user_message(text, context)
    raw = _chat(server_url, SYSTEM_PROMPT, user_message, temperature, max_tokens, timeout)
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
    parser.add_argument("--context-file", type=Path, default=None,
                        help="Optional Episode Context markdown file (see "
                             "extract_episode_context). Prepended to the "
                             "transcript so the model can ground speaker labels "
                             "and named entities.")

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
    context = args.context_file.read_text() if args.context_file else None
    if context:
        print(f"Context:    {args.context_file} ({len(context):,} chars)")
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
        context=context,
    )
    _print_loaded_models(args.server_url)
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
