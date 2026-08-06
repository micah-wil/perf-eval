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
# After start_server, vLLM logs are streamed to stdout (prefixed with `[vllm]`)
# so build output reflects server startup progress in real time. The streamer's
# PID is held in $VLLM_LOGS_PID; stop_server kills it.

# Resolve the digest of the image actually in use, exported as
# WORKLOAD_IMAGE_DIGEST. Docker runtime asks the local daemon, which is
# authoritative. Native runtime has no daemon — the job *is* the image — so fall
# back to the registry's current digest for the tag, which is only the same thing
# if the tag has not moved since the pull. Never fatal: a missing digest is
# recorded as NULL rather than failing the run.
resolve_image_digest() {
  local image=$1 runtime=$2 digest=""
  if [[ "$runtime" != "native" ]] && command -v docker >/dev/null 2>&1; then
    digest="$(docker inspect --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' \
              "$image" 2>/dev/null || true)"
  fi
  if [[ -z "$digest" ]]; then
    digest="$(python3 "$(dirname "${BASH_SOURCE[0]}")/image_digest.py" "$image" 2>/dev/null || true)"
  fi
  WORKLOAD_IMAGE_DIGEST="$digest"
  export WORKLOAD_IMAGE_DIGEST
  if [[ -n "$digest" ]]; then
    echo "  image digest: $digest"
  else
    echo "  image digest: unavailable (recorded as NULL)"
  fi
}

start_server() {
  local container=$1 port=$2 image=$3 model=$4 serve_args=$5 env=$6 runtime=${7:-docker}
  echo "--- :rocket: starting vllm: $model"

  # Capture the launch command for the reproduction record before running it.
  if [[ "$runtime" == "native" ]]; then
    WORKLOAD_SERVE_COMMAND="vllm serve $(printf '%q' "$model") --port ${port} ${serve_args}"
  else
    WORKLOAD_SERVE_COMMAND="docker run --rm --gpus all --ipc=host -p ${port}:${port} $(printf '%q' "$image") $(printf '%q' "$model") --port ${port} ${serve_args}"
  fi
  export WORKLOAD_SERVE_COMMAND
  echo "  serve command: $WORKLOAD_SERVE_COMMAND"

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
      # /version is the authoritative vLLM build actually serving; its
      # `+g<sha>` suffix carries the commit even when VLLM_COMMIT was not set.
      WORKLOAD_VLLM_VERSION="$(curl -fs "http://localhost:${port}/version" 2>/dev/null \
        | python3 -c 'import json,sys; print(json.load(sys.stdin).get("version",""))' 2>/dev/null || true)"
      export WORKLOAD_VLLM_VERSION
      [[ -n "$WORKLOAD_VLLM_VERSION" ]] && echo "  vllm version: $WORKLOAD_VLLM_VERSION"
      if [[ -z "${WORKLOAD_VLLM_COMMIT:-}" && "$WORKLOAD_VLLM_VERSION" =~ \+g([0-9a-f]{7,40}) ]]; then
        WORKLOAD_VLLM_COMMIT="${BASH_REMATCH[1]}"
        export WORKLOAD_VLLM_COMMIT
        echo "  vllm commit (from version): $WORKLOAD_VLLM_COMMIT"
      fi
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
