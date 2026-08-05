# vLLM server lifecycle. Source this from run.sh.
#
# Functions:
#   start_server <container> <port> <image> <model> <serve_args> <env> [runtime]
#   wait_healthy <port> [timeout_s=1500]
#   stop_server  <container>
#
# `env` is a newline-separated list of KEY=VALUE pairs. For Docker runtime,
# each value is injected into the container with -e. As a special case, HF_HOME
# is also bind-mounted at the same path inside the container so the model cache
# on the host is visible to vLLM. For native runtime, values are exported before
# starting `vllm serve` in the current job container.
#
# GPU passthrough for the Docker runtime depends on the vendor, taken from
# WORKLOAD_GPU_VENDOR (default "nvidia"):
#   nvidia  -> --gpus all               (needs the NVIDIA Container Toolkit)
#   amd     -> --device /dev/kfd + DRI render nodes + the host `render` group
#             (the ROCm equivalent; there is no `--gpus all` for ROCm). This is
#             the path the bare-metal MI355X Buildkite agents use, mirroring
#             vLLM's own ROCm CI (`.buildkite/scripts/hardware_ci/run-amd-test.sh`).
# See DESIGN_mi355_bare_metal.md for how the bare-metal (docker-in-docker) agents
# select this runtime.
#
# After start_server, vLLM logs are streamed to stdout (prefixed with `[vllm]`)
# so build output reflects server startup progress in real time. The streamer's
# PID is held in $VLLM_LOGS_PID; stop_server kills it.

# Append vendor-specific GPU passthrough flags to the named docker-args array.
#   $1 = name of the docker_args array (nameref)
#   $2 = server port to publish (nvidia only; the amd path uses host networking)
# Vendor comes from WORKLOAD_GPU_VENDOR (default nvidia).
gpu_docker_args() {
  local -n _args=$1
  local port=$2
  local vendor="${WORKLOAD_GPU_VENDOR:-nvidia}"

  if [[ "$vendor" == "amd" || "$vendor" == "rocm" ]]; then
    # ROCm has no `--gpus all`; expose the kernel-fusion driver and the DRI
    # render nodes, and join the host `render` group that owns them. Host
    # networking keeps the served port reachable at localhost:$port on the
    # agent without a -p mapping (matches vLLM's ROCm CI).
    _args+=(--network=host --shm-size=16g
            --device=/dev/kfd --device=/dev/dri
            --security-opt seccomp=unconfined
            --cap-add=SYS_PTRACE --cap-add=IPC_LOCK)
    local render_gid video_gid
    render_gid=$(getent group render | cut -d: -f3)
    video_gid=$(getent group video | cut -d: -f3)
    [[ -n "$render_gid" ]] && _args+=(--group-add "$render_gid")
    [[ -n "$video_gid" ]] && _args+=(--group-add "$video_gid")
    return
  fi

  # NVIDIA (default): the container toolkit injects all visible GPUs.
  _args+=(--gpus all -p "${port}:${port}")
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
    vllm serve "$model" --port "$port" $serve_args >"$log_file" 2>&1 &
    VLLM_SERVER_PID=$!
    echo "--- :memo: streaming vllm logs"
    ( tail -f "$log_file" 2>/dev/null | stdbuf -oL -eL sed 's/^/[vllm] /' ) &
    VLLM_LOGS_PID=$!
    return
  fi

  local docker_args=(--ipc=host --ulimit nofile=65536:65536
                     -e VLLM_ENGINE_READY_TIMEOUT_S=3600)
  gpu_docker_args docker_args "$port"
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
  fi
  if [[ -n "${VLLM_SERVER_PID:-}" ]]; then
    kill "$VLLM_SERVER_PID" 2>/dev/null || true
    wait "$VLLM_SERVER_PID" 2>/dev/null || true
  fi
  docker rm -f "$container" >/dev/null 2>&1 || true
}
