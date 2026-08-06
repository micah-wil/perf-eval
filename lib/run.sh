#!/usr/bin/env bash
# Orchestrate a workload: bring up vLLM, then dispatch each task to the
# helper script for its type.
#
# Usage: ./lib/run.sh workloads/qwen3_5_h200.yaml
set -euo pipefail

WORKLOAD="${1:?usage: $0 <workload.yaml>}"
[[ -f "$WORKLOAD" ]] || { echo "not found: $WORKLOAD" >&2; exit 2; }

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$DIR/server.sh"
# shellcheck disable=SC1091
source "$DIR/run_lm_eval.sh"
# shellcheck disable=SC1091
source "$DIR/run_vllm_bench.sh"
# shellcheck disable=SC1091
source "$DIR/sql_conn.sh"

# Resolve the ingestion destination once and hand it to every helper, so the
# detection in ingest_sink() (a configured TIGER_SQL_DB means SQL) happens here
# rather than separately in each ingest invocation. The selection is always
# logged: a run that silently used the endpoint because no credential was
# visible is otherwise indistinguishable from one that never tried.
echo "--- :floppy_disk: resolving ingest sink"
sql_debug_state          # before resolution, so INGEST_SINK shows as given
INGEST_SINK="$(ingest_sink)"
export INGEST_SINK
echo "  sql: ingest sink resolved to ${INGEST_SINK}"
if [[ "$INGEST_SINK" == "endpoint" ]]; then
  echo "  sql: no TIGER_SQL_DB visible, so results go to the public endpoint"
  echo "  sql: set INGEST_SINK=sql to fail the step instead of falling back"
fi

# Credentials are resolved and the connection probed once up front so a missing
# secret or an unreachable database fails before the GPU time is spent, rather
# than after every task has already run. The tables are expected to already
# exist: creating them is a one-time step, `lib/sql_upload.py --create-tables`.
if sql_sink_enabled; then
  if ! load_sql_conn; then
    echo "  sql: cannot resolve credentials; aborting before the run starts" >&2
    exit 1
  fi
  if ! python3 "$DIR/sql_upload.py" --check; then
    echo "  sql: cannot reach the database; aborting before the run starts" >&2
    exit 1
  fi
fi

WORKLOAD_EXPORTS="$(python3 "$DIR/parse_workload.py" "$WORKLOAD")"
eval "$WORKLOAD_EXPORTS"
export WORKLOAD_IMAGE WORKLOAD_VLLM_COMMIT WORKLOAD_SERVER_RUNTIME

PORT=8000
CONTAINER="perf-eval-${WORKLOAD_NAME}-$$"
RESULTS_DIR="results/${WORKLOAD_NAME}"
BASE_URL="http://localhost:${PORT}"
BENCH_TRUST_REMOTE_CODE=false
if [[ "$WORKLOAD_SERVE_ARGS" =~ (^|[[:space:]])--trust-remote-code([[:space:]]|$) ]] ||
   [[ "$WORKLOAD_SERVE_ARGS" =~ (^|[[:space:]])--trust-remote-code=(true|True|1|yes|Yes)([[:space:]]|$) ]]; then
  BENCH_TRUST_REMOTE_CODE=true
fi
mkdir -p "$RESULTS_DIR"

trap 'stop_server "$CONTAINER"' EXIT

start_server "$CONTAINER" "$PORT" "$WORKLOAD_IMAGE" "$WORKLOAD_MODEL" \
             "$WORKLOAD_SERVE_ARGS" "$WORKLOAD_ENV" "$WORKLOAD_SERVER_RUNTIME"
wait_healthy "$PORT"

# vllm bench serve runs first so we can validate perf flow without waiting
# on a full lm_eval pass. Each config's raw json lands in
# $RESULTS_DIR/bench-<name>.json and is then transformed and POSTed to the
# perf dashboard ingest endpoint.
while IFS=$'\t' read -r bname backend dataset isl osl nprompts conc speed_subset speed_category extra_args; do
  [[ -z "$bname" ]] && continue
  run_vllm_bench "$CONTAINER" "$PORT" "$WORKLOAD_MODEL" \
                 "$bname" "$backend" "$dataset" "$isl" "$osl" "$nprompts" \
                 "$conc" "$speed_subset" "$speed_category" "$extra_args" \
                 "$BENCH_TRUST_REMOTE_CODE" "$RESULTS_DIR"

  python3 "$DIR/ingest_perf.py" \
    --raw-result "${RESULTS_DIR}/bench-${bname}.json" \
    --device "$WORKLOAD_BENCH_DEVICE" \
    --tp "$WORKLOAD_BENCH_TP" \
    --precision "$WORKLOAD_BENCH_PRECISION" \
    --model "$WORKLOAD_MODEL" \
    --image "$WORKLOAD_IMAGE" \
    --workload "$WORKLOAD_NAME" \
    --bench-name "$bname" \
    --isl "$isl" --osl "$osl" --conc "$conc" || true
done <<< "$WORKLOAD_VLLM_BENCH_TSV"

if [[ "${BENCH_ONLY:-}" =~ ^([Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss])$ ]]; then
  echo "--- :stopwatch: BENCH_ONLY set; skipping lm_eval and bfcl tasks"
  exit 0
fi

while IFS=$'\t' read -r task fewshot model_args; do
  [[ -z "$task" ]] && continue
  run_lm_eval "$WORKLOAD_MODEL" "$BASE_URL" "$task" "$fewshot" \
              "$model_args" "$RESULTS_DIR"

  python3 "$DIR/ingest.py" \
    --results-dir "${RESULTS_DIR}/${task}" \
    --workload "$WORKLOAD_NAME" \
    --task "$task" \
    ${INGEST_NO_SAMPLES:+--no-samples} || true
done <<< "$WORKLOAD_LM_EVAL_TASKS_TSV"

# bfcl function-calling eval
while IFS=$'\t' read -r category num_threads temperature maximum_step_limit max_test_cases; do
  [[ -z "$category" ]] && continue
  echo "--- :phone: bfcl ${category}"
  python3 "$DIR/run_bfcl.py" "$WORKLOAD_MODEL" "$BASE_URL" \
    "$category" "$num_threads" "$temperature" "$RESULTS_DIR" \
    "$maximum_step_limit" "$max_test_cases"

  manifest="${RESULTS_DIR}/.bfcl_ingest/${category}.txt"
  if [[ -f "$manifest" ]]; then
    while IFS= read -r ingest_category; do
      [[ -z "$ingest_category" ]] && continue
      python3 "$DIR/ingest.py" \
        --results-dir "${RESULTS_DIR}/bfcl-${ingest_category}" \
        --workload "$WORKLOAD_NAME" \
        --task "bfcl_${ingest_category}" \
        --no-samples || true
    done < "$manifest"
  else
    python3 "$DIR/ingest.py" \
      --results-dir "${RESULTS_DIR}/bfcl-${category}" \
      --workload "$WORKLOAD_NAME" \
      --task "bfcl_${category}" \
      --no-samples || true
  fi
done <<< "$WORKLOAD_BFCL_TSV"
