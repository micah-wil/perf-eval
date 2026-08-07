# perf-eval

Run accuracy + perf workloads against vLLM, defined by small YAML recipes in `workloads/`.

Each recipe is one `(model, hardware, set of tasks)` combination. The Buildkite pipeline picks recipes up automatically — to ship a new run, you write a YAML file, push it, and trigger a build.

## Repo layout

```
workloads/        one YAML per (model, hardware) recipe
lib/              orchestrator (run.sh), helpers, GPU profiles
.buildkite/       pipeline bootstrap, step generator, and its tests
CLAUDE.md         agent conventions and detailed Buildkite workflow
```

## How to use this repo

### Add a new recipe

1. Copy an existing workload that targets the same GPU — e.g. `workloads/qwen3_5_h200.yaml` for H200 or `workloads/minimax_m3_b200.yaml` for B200.
2. Name the file `<model>_<hardware>.yaml`. Keep hardware variants in separate files.
3. Edit the fields to match your model and tasks. Set `nightly: true` if it should run in the nightly schedule; leave it off for opt-in recipes.
4. Open a PR. The pipeline auto-discovers `workloads/*.yaml` — no Buildkite YAML edits needed.

B200 workloads run in a single Kubernetes pod. `num_gpus` controls the pod's
GPU allocation; use at most 8 GPUs to keep the workload on one B200 node.

### Recipe schema

A recipe has top-level metadata plus up to three eval blocks:

- **`vllm:`** — *how the server runs.* Defines what model to serve and how (`model`, `serve_args`, optional image/env overrides). Required.
- **`lm_eval:`** — *what accuracy to measure.* Lists lm-evaluation-harness tasks to run against the live server (e.g. `gsm8k`, `aime25`). Each task's score is saved under `results/<name>/<task-name>/`. Optional.
- **`vllm_bench:`** — *what perf to measure.* Lists `vllm bench serve` configs (input/output lengths, concurrency, dataset). Raw JSON is saved and ingested into the perf dashboard. Optional.
- **`bfcl:`** — *function-calling eval.* Runs [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) test categories against the live server. Some models need `--enable-auto-tool-choice` and `--tool-call-parser` in `serve_args`. Results are transformed to lm_eval format and ingested as `bfcl_<category>` tasks. Optional.

Include one or more of `lm_eval:` / `vllm_bench:` / `bfcl:` depending on what you want out of this recipe.

```yaml
name: qwen3_5-h200       # used in container name and results/<name>/
gpu: H200                # picks queue/image/HF cache from lib/gpu_profiles.yaml
num_gpus: 8
nightly: true            # include in the nightly schedule (default: false)

vllm:                    # how the server is brought up
  model: Qwen/Qwen3.5-397B-A17B-FP8
  image: vllm/vllm-openai:nightly      # optional; only used when the build pins no image of its own
  env:                                  # optional; merged over the GPU profile's env
    SOME_VAR: value
  serve_args: >-                        # appended to `vllm serve <model>`; word-split
    -dp 8 --enable-expert-parallel
    --trust-remote-code

lm_eval:                 # accuracy tasks (optional)
  model_args:            # workload-level defaults, merged into every task
    tokenized_requests: false
    timeout: 6000
  tasks:
    - name: gsm8k                     # must match an lm-eval task name
      num_fewshot: 5
      model_args:                     # per-task overrides (merged on top of workload defaults)
        num_concurrent: 1024
        max_length: 40960
    - name: aime25
      num_fewshot: 0

bfcl:                    # function-calling eval (optional)
  test_categories:       # BFCL test categories to run
    - simple_python
    - multiple
    - parallel
  num_threads: 8         # optional, default 8
  temperature: 0.001     # optional, default 0.001
  maximum_step_limit: 40 # optional; multi-turn step cap (default 10). Overridden by BFCL_MAXIMUM_STEP_LIMIT env
  max_test_cases:        # optional; subsample categories (full suite if omitted)
    multi_turn: 100      # or set a single int to cap every category

vllm_bench:              # perf runs (optional) — fed to the perf dashboard
  configs:
    - name: 1k-in-1k-out-conc-256
      backend: openai                 # /v1/completions — exact ISL/OSL, no chat template
      dataset: random                 # synthetic fixed-length throughput dataset
      input_len: 1024
      output_len: 1024
      num_prompts: 500
      max_concurrency: 256
      args:                             # optional vllm bench serve arguments
        num_warmups: 16                 # becomes --num-warmups 16
        disable_tqdm: true              # becomes --disable-tqdm
```

A few things worth knowing:

- **`gpu`** must match a key in `lib/gpu_profiles.yaml`. The profile sets the Buildkite queue, the platform (`cuda` or `rocm` — which of the build's images this workload runs, see [Pinning the vLLM image](#pinning-the-vllm-image)), the HF cache path, and baseline env vars.
- **`nightly`** controls only the nightly schedule. Recipes with `nightly: false` (or omitted) are still triggerable explicitly via the `WORKLOADS` env var.
- **`lm_eval.tasks` is a list** because each entry runs as a separate `lm_eval` invocation — `--num_fewshot` is a single global flag, so different shot counts need separate runs. Each task's results land in `results/<name>/<task-name>/`.
- **`vllm_bench` runs first** if both blocks are present — that way perf-pipeline bugs surface quickly instead of waiting on a full lm-eval pass.
- **`vllm_bench` uses the `random` dataset with `--ignore-eos`** so every request prefills exactly `input_len` and decodes exactly `output_len` tokens — that's what makes the per-GPU decode throughput meaningful. Pair it with `backend: openai` (the `/v1/completions` endpoint) for exact token control. Avoid `dataset: speed_bench` for throughput numbers: it requires `--skip-tokenizer-init`, which makes `vllm bench serve` cap every request at a single output token, so output throughput reads as ~0.
- **`vllm_bench.configs[].args` forwards additional options to `vllm bench serve`.** Keys may use underscores, hyphens, or a leading `--`; they are normalized to `--kebab-case`. A `true` value emits a standalone flag, `false` and `null` omit it, scalar values emit a flag/value pair, and lists repeat the flag. Options managed by perf-eval itself, including the model, endpoint, dataset, request counts, lengths, concurrency, and result path, remain top-level config fields and cannot be overridden through `args`.
- **`bfcl` may need tool-call serve args.** Some models require `--enable-auto-tool-choice` and `--tool-call-parser` for function-calling; the parser warns if `--tool-call-parser` is absent. Each category runs as a separate generate + evaluate pass; scores appear on the eval dashboard as `bfcl_<category>` tasks.
- **`bfcl.maximum_step_limit`** caps how many inference steps BFCL allows per multi-turn turn (default 10 in perf-eval; BFCL upstream defaults to 20). Set it in the workload YAML, or override per-run with the `BFCL_MAXIMUM_STEP_LIMIT` env var (env wins over YAML). Useful for agentic / long multi-turn categories.
- **`bfcl.max_test_cases`** subsamples a category instead of running the full set — e.g. `multi_turn` (~800 cases) down to 300. For aggregate groups with multiple subcategories, the cap is split evenly across subcategories (by BFCL id order within each). Set a single integer to cap every category, or a map per category (`multi_turn: 240`). Override per-run with `BFCL_MAX_TEST_CASES`. Scores are partial-eval only and are not comparable to full BFCL leaderboard numbers.

For everything else (the full set of supported fields, defaults, validation rules), the existing files in `workloads/` are the working reference and `lib/parse_workload.py` is the source of truth.

### HF cache volume (Kubernetes profiles)

For profiles that run in-pod on Kubernetes (`server_runtime: native` with a `k8s_plugin`), the HuggingFace cache is a named `hf-cache` volume mounted at the profile's `hf_home`. **By default it is an `emptyDir`** — scoped to the benchmark pod, so the cache is reclaimed when the pod exits and can never accumulate on the node's disk.

A cluster with fast shared storage can keep a warm, cross-run cache by overriding the *volume source* (the mount path is unchanged either way — only cross-run persistence differs):

- **Per-cluster (recommended):** set a `{GPU}_HF_CACHE_VOLUME` env var on the Buildkite agent to a JSON volume source (everything except the `name`). This is per-cluster because storage backends differ per cluster — the same idiom as `{GPU}_QUEUE`. Example:

  ```
  MI300X_HF_CACHE_VOLUME='{"persistentVolumeClaim":{"claimName":"buildkite-hf-cache"}}'
  ```

- **Per-profile:** set `hf_cache_volume:` in the profile in `lib/gpu_profiles.yaml` (env override wins over this).

Do **not** set an `hf_home` under a node path like `/mnt/shared` unless that path is a real mount on every node in the queue — with the default `emptyDir` that only changes the in-pod path, but if you also point the volume at a `hostPath`, an unmounted path lands the cache on the node root disk with no reclamation.

Run the tests with `python3 .buildkite/test_generate_pipeline.py`, `python3 lib/test_images.py`, and `python3 lib/test_registry.py` (stdlib + pyyaml only; no GPU or network needed).

### Trigger a Buildkite build

The pipeline is [**`vllm/perf-eval`**](https://buildkite.com/vllm/perf-eval). With no extra config, a build runs every workload that has `nightly: true`.

**From the UI:** open the pipeline → New Build → pick branch and commit (must be pushed to GitHub) → optionally fill Environment Variables to scope the run → Create Build.

**Required env vars** — every build has to say which vLLM build it is testing:

- `VLLM_COMMIT` — vLLM commit SHA being tested. Used to tag results and track which vLLM version produced them.
- `VLLM_IMAGE` — full Docker image URI (e.g. `vllm/vllm-openai:nightly-abc1234`), or the per-platform `VLLM_IMAGE_CUDA` / `VLLM_IMAGE_ROCM` below when CUDA and ROCm are different artifacts. See [Pinning the vLLM image](#pinning-the-vllm-image) for how an image reaches AMD and NVIDIA workloads.

**Optional env vars:**

- `VLLM_IMAGE_CUDA` / `VLLM_IMAGE_ROCM` — per-platform images, for builds whose platform images can't be derived from one another.
- `WORKLOADS` — comma- or newline-separated list of workload paths or stems. Runs exactly those instead of the default `nightly: true` set.
- `NIGHTLY` — set to `1` to tag every ingested row with `nightly: true`. The dashboard's `/nightly` view filters on this to pair adjacent nightly builds; only the scheduled nightly cron should set it.

**Example — trigger a build from the Buildkite UI:**

1. Open the `vllm/perf-eval` pipeline → **New Build**.
2. Pick the branch and commit (must already be pushed to GitHub).
3. Set the environment variables:
   ```
   VLLM_COMMIT=abc1234def5678
   VLLM_IMAGE=vllm/vllm-openai:nightly-abc1234def5678
   WORKLOADS=qwen3_5_h200
   ```
4. Click **Create Build**.

This runs the `qwen3_5_h200` workload against the specified vLLM nightly image. Omit `WORKLOADS` to run all `nightly: true` workloads.

### Pinning the vLLM image

vLLM ships one image per platform, and their tags need not resemble each other — a release candidate might be `myrepo/vllm:v0.12.0rc2` on CUDA and `myrepo/amd-vllm:rc2-final` on ROCm. So a build pins an image *per platform*, and each workload runs the image for its GPU's `platform` (`cuda` for H200/B200, `rocm` for MI300X/MI355X — set in `lib/gpu_profiles.yaml`).

For one platform, in precedence order:

1. `VLLM_IMAGE_CUDA` / `VLLM_IMAGE_ROCM` — an explicit pin for that platform. Any registry, any tag.
2. `VLLM_IMAGE` — a single image, applied to the platform its name identifies (`rocm` anywhere in the ref means ROCm, otherwise CUDA). This is the common case: one nightly image covers everything.
3. **The same release build's tag for this platform.** The release pipeline publishes every platform of a build into one repo as `<sha>-<suffix>` — `…/vllm-release-repo:<sha>-x86_64` alongside `…:<sha>-rocm` — so pinning one platform locates the others. Suffixes live in `RELEASE_TAG_SUFFIX` in `lib/images.py`; a profile can override its own with `release_tag_suffix` (an ARM CUDA queue would want `aarch64`).
4. The nightly build of the pinned commit — `vllm/vllm-openai[-rocm]:nightly-<VLLM_COMMIT>` — when the build is testing nightlies anyway (every pinned image is `nightly`-tagged, or nothing is pinned). This is what keeps AMD covered on a nightly run that only passes a CUDA image.
5. The workload's own `vllm.image`, then `vllm/vllm-openai[-rocm]:nightly`. Only when the build pins nothing at all — typically a local run.

`VLLM_COMMIT` is the revision under test on every platform, and is what gets recorded with the results. If it isn't set, a commit embedded in the resolved image's tag is used.

Rule 3 means the usual release-candidate checkout needs no new variables — the `VLLM_COMMIT` and CUDA `VLLM_IMAGE` the release manager already passes cover AMD as well:

```
VLLM_COMMIT=e5949f10009c8b1803e2e37f5610b4dd047d432f
VLLM_IMAGE=public.ecr.aws/q9t5s3a7/vllm-release-repo:e5949f10009c8b1803e2e37f5610b4dd047d432f-x86_64
```

`VLLM_IMAGE_ROCM` is then only needed for an image that doesn't follow that convention — a one-off build, a private mirror, an image whose tag shares nothing with its CUDA counterpart.

**A derived ref (rule 3 or 4) is checked against the registry before it is scheduled.** Not every release build publishes every platform, so a guess can be wrong; when the tag isn't there the workload becomes a skipped step (`no ROCM image built for 7794b1e08bf5`) instead of queueing a GPU job that dies on `docker pull` an hour later. Only a registry answering *not found* rejects a ref: an unreachable registry, a rate limit, or one that needs credentials all count as present, so a flaky check can never quietly drop coverage. Images you pinned yourself are never second-guessed.

**When a platform has no image at all, its workloads are skipped, not silently retargeted.** If a build pins a custom image that nothing can be derived from, platforms without a pin have nothing representative to run: benchmarking last night's image and labelling it with the release candidate's commit is worse than not running. Those workloads become skipped steps whose reason says which variable to set (`needs a ROCM image: set VLLM_IMAGE_ROCM`). Skipped steps are hidden until you toggle *Skipped jobs* in the build view; the bootstrap step's log also lists the image every step resolved to.

**From an agent:** see `CLAUDE.md` for the Buildkite MCP and authenticated
`bk` workflows. Don't make raw Buildkite API calls with `curl`.

### Run a recipe end-to-end

A real run needs a GPU host with Docker, vLLM, and lm-eval available:

```bash
./lib/run.sh workloads/qwen3_5_h200.yaml
```

Locally, you can smoke-test recipe changes without a GPU — see `CLAUDE.md` for the parser stub and shell-syntax checks.

## Agents

`CLAUDE.md` has conventions for AI agents working in this repo: smoke-testing changes, launching Buildkite builds for a chosen branch/commit, and the AI-assistance disclosure rule for PRs and commits.
