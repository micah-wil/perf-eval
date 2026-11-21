# Run Kimi Vendor Verifier (KVV) jobs against an OpenAI-compatible endpoint.
# Source this from run.sh.
#
# Usage:
#   run_kvv <model> <base_url> <jobs> <output_dir>
#
# jobs is a space-separated list of KVV `run_eval.sh` job names, e.g.
# `verify_k3_features ocrbench_thinking_effort:max`. KVV is cloned at KVV_REF
# and manages its own Python env with uv. Its Git LFS test data is fetched over
# HTTPS, so the host needs no git-lfs. A failed job is logged and the remaining
# jobs still run; KVV never fails the workload.

KVV_REPO="MoonshotAI/Kimi-Vendor-Verifier"
KVV_REF="${KVV_REF:-7b938c38bbd89c2404a9ed5e7cb40da8079206b4}"

kvv_checkout() {
  local dir=$1 f
  if [[ ! -d "$dir/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --quiet "https://github.com/${KVV_REPO}.git" "$dir" || return 1
  fi
  GIT_LFS_SKIP_SMUDGE=1 git -C "$dir" checkout --quiet --force "$KVV_REF" || return 1
  while IFS= read -r f; do
    curl -fsSL --retry 3 -o "$dir/$f" \
      "https://media.githubusercontent.com/media/${KVV_REPO}/${KVV_REF}/${f}" || return 1
  done < <(cd "$dir" && grep -rlI --exclude-dir=.git --exclude-dir=.venv \
             '^version https://git-lfs.github.com/spec/v1' .)
  command -v uv >/dev/null || python3 -m pip install --quiet uv
}

run_kvv() {
  local model=$1 base_url=$2 jobs=$3 outdir=$4
  local dir="${KVV_DIR:-.kvv}" out job name rc
  local -a failed=()
  echo "--- :white_check_mark: KVV setup (${KVV_REPO}@${KVV_REF:0:12})"
  mkdir -p "$outdir/kvv"
  out="$(cd "$outdir/kvv" && pwd)"
  if ! kvv_checkout "$dir"; then
    echo "KVV checkout failed; skipping KVV" >&2
    return 0
  fi
  for job in $jobs; do
    name="${job//:/-}"
    echo "--- :white_check_mark: KVV ${job}"
    rc=0
    (
      cd "$dir" &&
      KIMI_BASE_URL="${base_url}/v1" KIMI_API_KEY=EMPTY MODEL_NAME="$model" \
      THINK_MODE=opensource ENABLE_THINKING_EFFORT=true \
      INSPECT_LOG_DIR="$out/$name" \
        ./run_eval.sh "$job"
    ) 2>&1 | tee "$out/$name.log" || rc=$?
    # pytest jobs write fixed-name reports into the KVV checkout.
    mv "$dir"/*-report.xml "$dir"/*-report.json "$out/" 2>/dev/null || true
    if (( rc != 0 )); then
      echo "KVV ${job} failed (exit ${rc})" >&2
      failed+=("$job")
    fi
  done
  if (( ${#failed[@]} )); then
    echo "KVV failed jobs: ${failed[*]}" >&2
  fi
}
