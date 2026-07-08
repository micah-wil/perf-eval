# vLLM server lifecycle. Source this from run.sh.
#
# Functions:
#   preflight_gpus
#   start_server <container> <port> <image> <model> <serve_args> <env> [runtime]
#   wait_healthy <port> [timeout_s=3600]
#   stop_server  <container>
#
# `env` is a newline-separated list of KEY=VALUE pairs. For Docker runtime,
# each value is injected into the container with -e. As a special case, HF_HOME
# is also bind-mounted at the same path inside the container so the model cache
# on the host is visible to vLLM. For native runtime, values are exported before
# starting `vllm serve` in the current job container.
#
# After start_server, vLLM logs are streamed to stdout (prefixed with `[vllm]`)
# so build output reflects server startup progress in real time. The streamer's
# PID is held in $VLLM_LOGS_PID; stop_server kills it.

DIR_SERVER_SH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_PROC_PATTERNS='VLLM::|vllm[[:space:]]+serve|vllm\.entrypoints\.openai'
# Seconds to wait for VRAM to drain before giving up.
GPU_FREE_TIMEOUT="${GPU_FREE_TIMEOUT:-180}"
# Seconds to wait for graceful shutdown (SIGTERM) before escalating to KILL.
SERVER_STOP_GRACE="${SERVER_STOP_GRACE:-60}"

# Kill every process in a process group (TERM, then KILL after a grace period)
_kill_pgroup() {
  local pgid=$1 grace=${2:-$SERVER_STOP_GRACE}
  [[ -z "$pgid" ]] && return 0
  kill -TERM -- "-${pgid}" 2>/dev/null || true
  local waited=0
  while (( waited < grace )); do
    if ! pgrep -g "$pgid" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    waited=$(( waited + 1 ))
  done
  kill -KILL -- "-${pgid}" 2>/dev/null || true
  sleep 1
}

_sweep_vllm_procs() {
  local grace=${1:-$SERVER_STOP_GRACE}
  local pids
  pids=$(pgrep -u "$(id -u)" -f "$VLLM_PROC_PATTERNS" 2>/dev/null || true)
  [[ -z "$pids" ]] && return 0
  echo "--- :broom: sweeping leftover vLLM processes: $(echo "$pids" | tr '\n' ' ')"
  # shellcheck disable=SC2086
  kill -TERM $pids 2>/dev/null || true
  local waited=0
  while (( waited < grace )); do
    pids=$(pgrep -u "$(id -u)" -f "$VLLM_PROC_PATTERNS" 2>/dev/null || true)
    [[ -z "$pids" ]] && break
    sleep 1
    waited=$(( waited + 1 ))
  done
  pids=$(pgrep -u "$(id -u)" -f "$VLLM_PROC_PATTERNS" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    # shellcheck disable=SC2086
    kill -KILL $pids 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}

_wait_gpus_free() {
  local timeout=${1:-$GPU_FREE_TIMEOUT}
  local start now
  start=$(date +%s)
  while :; do
    if python3 "$DIR_SERVER_SH/gpu_preflight.py" check >/dev/null 2>&1; then
      return 0
    fi
    now=$(date +%s)
    if (( now - start >= timeout )); then
      return 1
    fi
    sleep 3
  done
}

# Ensure GPUs are clean before we launch vLLM
preflight_gpus() {
  local runtime=${1:-native}
  [[ "$runtime" == "native" ]] || return 0

  if python3 "$DIR_SERVER_SH/gpu_preflight.py" check >/dev/null 2>&1; then
    echo "--- :white_check_mark: GPUs clean before startup"
    return 0
  fi

  echo "--- :warning: GPUs not clean before startup; cleaning up stale vLLM processes"
  _sweep_vllm_procs
  if _wait_gpus_free "$GPU_FREE_TIMEOUT"; then
    echo "GPUs clean after cleanup"
    return 0
  fi

  echo "+++ :x: GPUs still busy after ${GPU_FREE_TIMEOUT}s cleanup wait" >&2
  python3 "$DIR_SERVER_SH/gpu_preflight.py" check >&2 || true
  if [[ "${PERF_EVAL_REQUIRE_CLEAN_GPUS:-1}" =~ ^(0|false|no)$ ]]; then
    echo "PERF_EVAL_REQUIRE_CLEAN_GPUS unset/false; continuing despite dirty GPUs" >&2
    return 0
  fi
  echo "Refusing to start vLLM on dirty GPUs (likely a leaked process from a" \
       "previous job on this shared node). Set PERF_EVAL_REQUIRE_CLEAN_GPUS=0" \
       "to override." >&2
  return 1
}

start_server() {
  local container=$1 port=$2 image=$3 model=$4 serve_args=$5 env=$6 runtime=${7:-docker}
  echo "--- :rocket: starting vllm: $model"

  if [[ "$runtime" == "native" ]]; then
    while IFS= read -r kv; do
      [[ -z "$kv" ]] && continue
      export "$kv"
    done <<< "$env"
    local log_file="/tmp/${container}.log"
    VLLM_LOG_FILE="$log_file"

    # shellcheck disable=SC2086  # serve_args intentionally word-split
    setsid vllm serve "$model" --port "$port" $serve_args >"$log_file" 2>&1 &
    VLLM_SERVER_PID=$!
    VLLM_SERVER_PGID=$(ps -o pgid= -p "$VLLM_SERVER_PID" 2>/dev/null | tr -d ' ')
    [[ -z "$VLLM_SERVER_PGID" ]] && VLLM_SERVER_PGID=$VLLM_SERVER_PID
    echo "--- :memo: streaming vllm logs (pid=$VLLM_SERVER_PID pgid=$VLLM_SERVER_PGID)"
    ( tail -f "$log_file" 2>/dev/null | stdbuf -oL -eL sed 's/^/[vllm] /' ) &
    VLLM_LOGS_PID=$!
    return
  fi

  local docker_args=(--gpus all --ipc=host --ulimit nofile=65536:65536
                     -e VLLM_ENGINE_READY_TIMEOUT_S=3600
                     -p "${port}:${port}")
  local hf_home=""
  while IFS= read -r kv; do
    [[ -z "$kv" ]] && continue
    docker_args+=(-e "$kv")
    [[ "$kv" == HF_HOME=* ]] && hf_home="${kv#HF_HOME=}"
  done <<< "$env"
  if [[ -n "$hf_home" ]]; then
    docker_args+=(-v "${hf_home}:${hf_home}")
  fi

  # shellcheck disable=SC2086  # serve_args intentionally word-split
  # vllm/vllm-openai's entrypoint takes the model as the first positional
  # arg; do not prepend `vllm` or `serve`.
  docker run -d --rm --name "$container" "${docker_args[@]}" \
    "$image" \
    "$model" --port "$port" $serve_args

  # Install pytest to avoid cupy.testing import failure during torch.compile
  docker exec "$container" pip install -q pytest 2>/dev/null || true

  echo "--- :memo: streaming vllm logs"
  ( docker logs -f "$container" 2>&1 | stdbuf -oL -eL sed 's/^/[vllm] /' ) &
  VLLM_LOGS_PID=$!
}

wait_healthy() {
  local port=$1 timeout=${2:-3600}
  echo "+++ :hourglass: waiting for /health (timeout ${timeout}s)"
  local now start deadline next_status elapsed
  start=$(date +%s)
  deadline=$(( start + timeout ))
  next_status=$(( start + 60 ))
  while (( $(date +%s) < deadline )); do
    if curl -fs "http://localhost:${port}/health" >/dev/null 2>&1; then
      echo "server healthy"
      return 0
    fi
    if [[ -n "${VLLM_SERVER_PID:-}" ]] && ! kill -0 "$VLLM_SERVER_PID" 2>/dev/null; then
      echo "vLLM server exited before becoming healthy" >&2
      [[ -n "${VLLM_LOG_FILE:-}" ]] && tail -n 80 "$VLLM_LOG_FILE" >&2 || true
      return 1
    fi
    now=$(date +%s)
    if (( now >= next_status )); then
      elapsed=$(( now - start ))
      echo "still waiting for /health after ${elapsed}s"
      next_status=$(( now + 60 ))
    fi
    sleep 5
  done
  echo "server never came up" >&2
  return 1
}

stop_server() {
  local container=$1
  if [[ -n "${VLLM_LOGS_PID:-}" ]]; then
    kill "$VLLM_LOGS_PID" 2>/dev/null || true
    wait "$VLLM_LOGS_PID" 2>/dev/null || true
    VLLM_LOGS_PID=""
  fi

  if [[ -n "${VLLM_SERVER_PGID:-}" ]]; then
    echo "--- :stop_sign: stopping vLLM process group $VLLM_SERVER_PGID"
    _kill_pgroup "$VLLM_SERVER_PGID"
    _sweep_vllm_procs
    VLLM_SERVER_PGID=""
    VLLM_SERVER_PID=""
  elif [[ -n "${VLLM_SERVER_PID:-}" ]]; then
    kill "$VLLM_SERVER_PID" 2>/dev/null || true
    wait "$VLLM_SERVER_PID" 2>/dev/null || true
    VLLM_SERVER_PID=""
  fi

  docker rm -f "$container" >/dev/null 2>&1 || true
}
