# Run a single lm-evaluation-harness task against an OpenAI-compatible endpoint.
# Source this from run.sh.
#
# Usage:
#   run_lm_eval <model> <base_url> <task> <num_fewshot> <model_args> <output_dir>
#
# model_args is the comma-separated key=value string lm_eval expects (without
# `model` or `base_url`; this helper injects those). Pass an empty string if
# there are no extra args.

run_lm_eval() {
  local model=$1 base_url=$2 task=$3 fewshot=$4 model_args=$5 outdir=$6
  echo "--- :microscope: lm_eval ${task} (${fewshot}-shot)"
  local cmd=(
    lm_eval --model local-completions
    --model_args "model=${model},base_url=${base_url}/v1/completions${model_args:+,$model_args}"
    --tasks "$task"
    --num_fewshot "$fewshot"
    --log_samples
    --output_path "${outdir}/${task}"
  )

  # Record the exact command so ingest can store it as the reproduction recipe.
  mkdir -p "${outdir}/${task}"
  local cmd_file="${outdir}/${task}.cmd"
  printf '%q ' "${cmd[@]}" > "$cmd_file"
  printf '\n' >> "$cmd_file"
  echo "  command: $(cat "$cmd_file")"

  "${cmd[@]}"
}
