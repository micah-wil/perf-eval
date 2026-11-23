# Run a single `aiperf profile` config against the running vLLM container.
# Source this from run.sh *after* run_vllm_bench.sh, which provides the
# pip_install_quiet and append_bench_args helpers this file reuses.
#
# Usage:
#   run_aiperf <container> <port> <model> <name> <args_base64> <output_dir>
#
# aiperf is a client-side load generator that is not shipped in the
# vllm/vllm-openai images, so it is installed on first use as a uv tool on its
# own Python: aiperf requires Python <3.14 since 0.8, and on the 3.14 images a
# plain pip install silently falls back to 0.7, which lacks options such as
# --sweep-type. Native runtime installs it next to the job; Docker runtime
# installs and runs it inside the vLLM container. Artifacts land in
# "<output_dir>/aiperf-<name>/" and are picked up by the Buildkite artifact
# upload.

AIPERF_SPEC="aiperf>=0.12"
AIPERF_PYTHON=3.13

ensure_aiperf() {
  local runtime=$1 container=$2
  if [[ "$runtime" == "native" ]]; then
    command -v aiperf >/dev/null 2>&1 && return 0
    echo "--- :package: installing aiperf"
    pip_install_quiet uv
    python3 -m uv tool install --quiet --python "$AIPERF_PYTHON" "$AIPERF_SPEC"
    PATH="$(python3 -m uv tool dir --bin):$PATH"
  else
    docker exec "$container" bash -lc 'command -v aiperf >/dev/null 2>&1' && return 0
    echo "--- :package: installing aiperf in vLLM container"
    docker exec "$container" bash -lc \
      "PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --quiet uv \
        && UV_TOOL_BIN_DIR=/usr/local/bin python3 -m uv tool install --quiet \
             --python $AIPERF_PYTHON '$AIPERF_SPEC'"
  fi
}

run_aiperf() {
  local container=$1 port=$2 model=$3 name=$4 args_base64=$5 outdir=$6
  local runtime="${WORKLOAD_SERVER_RUNTIME:-docker}"
  local url="http://127.0.0.1:${port}"
  local host_dir="${outdir}/aiperf-${name}"
  local in_container_dir="/tmp/aiperf-${name}"

  echo "--- :chart_with_upwards_trend: aiperf profile ${name}"
  mkdir -p "$host_dir"
  ensure_aiperf "$runtime" "$container"

  local artifact_dir="$host_dir"
  [[ "$runtime" != "native" ]] && artifact_dir="$in_container_dir"

  local cmd=(
    aiperf profile
    --model "$model"
    --tokenizer "$model"
    --url "$url"
    --api-key EMPTY
    --output-artifact-dir "$artifact_dir"
  )
  [[ "$runtime" != "native" ]] && cmd=(docker exec "$container" "${cmd[@]}")

  append_bench_args "$args_base64" cmd

  "${cmd[@]}"

  if [[ "$runtime" != "native" ]]; then
    docker cp "${container}:${in_container_dir}/." "$host_dir/"
  fi
  echo "  saved aiperf artifacts to $host_dir"
}
