#!/usr/bin/env bash
# Benchmark a summarize-video run on the current machine.
#
# Usage:
#   ./benchmark.sh <URL> [extra flags forwarded to summarize_video.py]
#
# Examples:
#   ./benchmark.sh "https://www.youtube.com/watch?v=HeAGWTgi4sU" -l en
#   ./benchmark.sh "https://www.youtube.com/watch?v=02YLwsCKUww" -l en \
#       --cookies-from-browser chrome
#   ./benchmark.sh "<URL>" -l en -f           # -f forces a cold cache re-run
#
# Saves to: benchmark/<video-id>-<platform>-<timestamp>/
#   run.log         — full orchestrator stdout/stderr
#   metadata.txt    — host, hardware, torch/python versions, timing summary
#   <id>.txt        — plain deduped transcript
#   <id>.timed.txt  — timestamped transcript
#   <id>.diarized.txt (unless --no-diarize)

set -euo pipefail

if [ $# -lt 1 ]; then
  sed -n '2,18p' "$0"
  exit 1
fi

URL="$1"
shift

# --- platform detection ------------------------------------------------------
OS="$(uname -s)"
case "$OS" in
  Darwin) PLATFORM="mac" ;;
  Linux)  PLATFORM="linux" ;;
  *)      PLATFORM="$(echo "$OS" | tr '[:upper:]' '[:lower:]')" ;;
esac

# --- extract video id from URL ----------------------------------------------
# Matches ?v=<id>, youtu.be/<id>, /shorts/<id>, /embed/<id>, /live/<id>.
if [[ "$URL" =~ [?\&]v=([A-Za-z0-9_-]{11}) ]]; then
  VIDEO_ID="${BASH_REMATCH[1]}"
elif [[ "$URL" =~ youtu\.be/([A-Za-z0-9_-]{11}) ]]; then
  VIDEO_ID="${BASH_REMATCH[1]}"
elif [[ "$URL" =~ /(shorts|embed|live)/([A-Za-z0-9_-]{11}) ]]; then
  VIDEO_ID="${BASH_REMATCH[2]}"
else
  VIDEO_ID="unknown"
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
BENCH_DIR="benchmark/${VIDEO_ID}-${PLATFORM}-${TIMESTAMP}"
mkdir -p "$BENCH_DIR"
LOG_FILE="$BENCH_DIR/run.log"
META_FILE="$BENCH_DIR/metadata.txt"

# --- capture env + hardware up-front ----------------------------------------
{
  echo "=== Run ==="
  echo "timestamp:   $TIMESTAMP"
  echo "url:         $URL"
  echo "video_id:    $VIDEO_ID"
  echo "platform:    $PLATFORM"
  echo "hostname:    $(hostname)"
  echo "uname:       $(uname -a)"
  echo "extra_args:  $*"
  echo

  echo "=== Hardware ==="
  if [ "$PLATFORM" = "linux" ]; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
      || echo "nvidia-smi unavailable"
    grep -m1 "model name" /proc/cpuinfo 2>/dev/null | sed 's/^[^:]*: //' | sed 's/^/cpu: /'
    awk '/MemTotal/ {printf "ram: %.1f GB\n", $2/1024/1024}' /proc/meminfo 2>/dev/null
  elif [ "$PLATFORM" = "mac" ]; then
    sysctl -n machdep.cpu.brand_string | sed 's/^/cpu: /'
    sysctl -n hw.memsize | awk '{printf "ram: %.1f GB\n", $1/1024/1024/1024}'
    system_profiler SPHardwareDataType 2>/dev/null \
      | grep -E "Chip|Model Name" | sed 's/^ *//'
  fi
  echo

  echo "=== Python / Torch ==="
  uv run python -c "
import sys
print('python:', sys.version.split()[0])
try:
    import torch
    print('torch:', torch.__version__)
    if torch.cuda.is_available():
        print('cuda:', True, torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        print('cuda:', False, 'mps: True')
    else:
        print('cuda:', False)
except Exception as e:
    print('torch import failed:', e)
" 2>&1 || true
  echo
} > "$META_FILE"

# --- build command ----------------------------------------------------------
# On Linux we clear LD_LIBRARY_PATH so torch finds its bundled cuDNN instead
# of a system cuDNN from the CUDA Toolkit.
CMD=(uv run python summarize_video.py "$URL" -o "$BENCH_DIR" "$@")

echo "Benchmark dir: $BENCH_DIR"
echo "Command:       ${CMD[*]}"
echo "Log:           $LOG_FILE"
echo

START_NS=$(python3 -c 'import time; print(time.time_ns())')

set +e
if [ "$PLATFORM" = "linux" ]; then
  LD_LIBRARY_PATH= "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
  STATUS=${PIPESTATUS[0]}
else
  "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
  STATUS=${PIPESTATUS[0]}
fi
set -e

END_NS=$(python3 -c 'import time; print(time.time_ns())')
TOTAL_S=$(python3 -c "print(f'{(${END_NS}-${START_NS})/1e9:.2f}')")

# --- append timing summary --------------------------------------------------
{
  echo "=== Timing ==="
  echo "total_wall_s: $TOTAL_S"
  echo "exit_status:  $STATUS"
  echo
  echo "=== Per-step (from orchestrator log) ==="
  # The orchestrator prints '=== N/M step ===' followed by a '  (N.Ns)'
  # line when the step finishes. Pull those pairs in order.
  awk '
    /^=== [0-9]+\/[0-9]+ / { step=$0; next }
    /^  \([0-9.]+s\)/      { print step " " $0; step="" }
  ' "$LOG_FILE" || true
} >> "$META_FILE"

echo
echo "=== Benchmark done ==="
echo "Total wall: ${TOTAL_S}s  (exit $STATUS)"
echo "Saved:"
ls -la "$BENCH_DIR"

exit "$STATUS"
