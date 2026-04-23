#!/usr/bin/env bash
# Benchmark a summarize-video run on the current machine.
#
# Two usage modes:
#
# 1) Named case (canonical benchmarks, always forced cold with -f):
#      ./benchmark.sh en       # English, 31-min panel, turbo, 3 speakers
#      ./benchmark.sh hi       # Hindi-English, 8-min clip, v3, 2 speakers
#      ./benchmark.sh all      # runs both sequentially
#
# 2) Ad-hoc URL + flags (forwarded to summarize_video.py):
#      ./benchmark.sh "https://www.youtube.com/watch?v=HeAGWTgi4sU" -l en
#      ./benchmark.sh "<URL>" -l en -f
#
# Saves to: benchmark/<video-id>-<platform>-<timestamp>/
#   run.log         full orchestrator stdout/stderr
#   metadata.txt    host, hardware, torch/python versions, timing summary
#   <id>.txt        plain deduped transcript
#   <id>.timed.txt  timestamped transcript
#   <id>.diarized.txt (unless --no-diarize)
#   <id>.diarized.summary.md (unless --no-summarize)

set -euo pipefail

if [ $# -lt 1 ]; then
  sed -n '2,21p' "$0"
  exit 1
fi

# --- named-case presets -----------------------------------------------------
# Cold-cache runs (-f) so per-step timings reflect real compute, not cache hits.
LLAMA_BIN_DEFAULT="$HOME/llama.cpp/build/bin/llama-server"

CASE_EN_URL="https://www.youtube.com/watch?v=02YLwsCKUww"
CASE_EN_ARGS=(-l en --cookies-from-browser chrome --num-speakers 3
              --llama-server-bin "$LLAMA_BIN_DEFAULT" -f)

CASE_HI_URL="https://www.youtube.com/watch?v=HeAGWTgi4sU"
CASE_HI_ARGS=(-m v3 -l hi --compression-ratio-threshold 2.0
              --hallucination-silence-threshold 2.0
              --cookies-from-browser chrome --num-speakers 2
              --llama-server-bin "$LLAMA_BIN_DEFAULT" -f)

run_one() {
  local url="$1"; shift
  local extra=("$@")

  # --- platform detection ---------------------------------------------------
  local os platform
  os="$(uname -s)"
  case "$os" in
    Darwin) platform="mac" ;;
    Linux)  platform="linux" ;;
    *)      platform="$(echo "$os" | tr '[:upper:]' '[:lower:]')" ;;
  esac

  # --- extract video id from URL -------------------------------------------
  local video_id
  if [[ "$url" =~ [?\&]v=([A-Za-z0-9_-]{11}) ]]; then
    video_id="${BASH_REMATCH[1]}"
  elif [[ "$url" =~ youtu\.be/([A-Za-z0-9_-]{11}) ]]; then
    video_id="${BASH_REMATCH[1]}"
  elif [[ "$url" =~ /(shorts|embed|live)/([A-Za-z0-9_-]{11}) ]]; then
    video_id="${BASH_REMATCH[2]}"
  else
    video_id="unknown"
  fi

  local timestamp bench_dir log_file meta_file
  timestamp="$(date +%Y%m%d-%H%M%S)"
  bench_dir="benchmark/${video_id}-${platform}-${timestamp}"
  mkdir -p "$bench_dir"
  log_file="$bench_dir/run.log"
  meta_file="$bench_dir/metadata.txt"

  # --- capture env + hardware up-front -------------------------------------
  {
    echo "=== Run ==="
    echo "timestamp:   $timestamp"
    echo "url:         $url"
    echo "video_id:    $video_id"
    echo "platform:    $platform"
    echo "hostname:    $(hostname)"
    echo "uname:       $(uname -a)"
    echo "extra_args:  ${extra[*]}"
    echo

    echo "=== Hardware ==="
    if [ "$platform" = "linux" ]; then
      nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
        || echo "nvidia-smi unavailable"
      grep -m1 "model name" /proc/cpuinfo 2>/dev/null | sed 's/^[^:]*: //' | sed 's/^/cpu: /'
      awk '/MemTotal/ {printf "ram: %.1f GB\n", $2/1024/1024}' /proc/meminfo 2>/dev/null
    elif [ "$platform" = "mac" ]; then
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
  } > "$meta_file"

  # --- run ------------------------------------------------------------------
  local cmd=(uv run python summarize_video.py "$url" -o "$bench_dir" "${extra[@]}")
  echo "Benchmark dir: $bench_dir"
  echo "Command:       ${cmd[*]}"
  echo "Log:           $log_file"
  echo

  local start_ns end_ns total_s status
  start_ns=$(python3 -c 'import time; print(time.time_ns())')

  set +e
  if [ "$platform" = "linux" ]; then
    LD_LIBRARY_PATH= "${cmd[@]}" 2>&1 | tee "$log_file"
    status=${PIPESTATUS[0]}
  else
    "${cmd[@]}" 2>&1 | tee "$log_file"
    status=${PIPESTATUS[0]}
  fi
  set -e

  end_ns=$(python3 -c 'import time; print(time.time_ns())')
  total_s=$(python3 -c "print(f'{(${end_ns}-${start_ns})/1e9:.2f}')")

  # --- append timing summary -----------------------------------------------
  {
    echo "=== Timing ==="
    echo "total_wall_s: $total_s"
    echo "exit_status:  $status"
    echo
    echo "=== Per-step (from orchestrator log) ==="
    awk '
      /^=== [0-9]+\/[0-9]+ / { step=$0; next }
      /^  \([0-9.]+s\)/      { print step " " $0; step="" }
    ' "$log_file" || true
  } >> "$meta_file"

  echo
  echo "=== Benchmark done ==="
  echo "Total wall: ${total_s}s  (exit $status)"
  echo "Saved:"
  ls -la "$bench_dir"

  return "$status"
}

# --- dispatch: named case vs URL --------------------------------------------
case "${1:-}" in
  en)
    run_one "$CASE_EN_URL" "${CASE_EN_ARGS[@]}"
    ;;
  hi)
    run_one "$CASE_HI_URL" "${CASE_HI_ARGS[@]}"
    ;;
  all)
    run_one "$CASE_EN_URL" "${CASE_EN_ARGS[@]}"
    echo
    run_one "$CASE_HI_URL" "${CASE_HI_ARGS[@]}"
    ;;
  *)
    url="$1"; shift
    run_one "$url" "$@"
    ;;
esac
