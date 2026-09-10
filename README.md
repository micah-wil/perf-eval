# perf-eval

Run accuracy + perf workloads against vLLM, defined by small YAML recipes in `workloads/`.

Each recipe is one `(model, hardware, set of tasks)` combination. The Buildkite pipeline picks recipes up automatically — to ship a new run, you write a YAML file, push it, and trigger a build.

## Repo layout

```
workloads/        one YAML per (model, hardware) recipe
lib/              orchestrator (run.sh), helpers, GPU profiles, dashboard ingest,
                  cron_backfill.py
.buildkite/       pipeline bootstrap, step generator, and its tests
CLAUDE.md         agent conventions and detailed Buildkite workflow
```

## How to use this repo

### Add a new recipe

1. Copy an existing workload that targets the same GPU — e.g. `workloads/kimi_k2_6_mi355x.yaml` for MI355X or `workloads/deepseek_r1_0528_mi300x.yaml` for MI300X.
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
  image: vllm/vllm-openai:nightly      # optional; falls back to VLLM_IMAGE / VLLM_COMMIT / latest
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

- **`gpu`** must match a key in `lib/gpu_profiles.yaml`. The profile sets the Buildkite queue, default image, HF cache path, and baseline env vars.
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

Run the generator's tests with `python3 .buildkite/test_generate_pipeline.py` (stdlib + pyyaml only; no GPU needed).

### Where results go

Two things happen with every run's output, and they serve different purposes:

**Dashboards (live).** `lib/ingest.py` and `lib/ingest_perf.py` POST to the Cloud Run endpoints backing the vLLM eval and perf dashboards. This is best-effort — a failed upload is logged and never aborts a run.

**Buildkite artifacts (durable).** Every step uploads `results/**/*`. This is the record that outlives the agent, and it is deliberately complete enough to reconstruct a run without any database:

| Artifact | Contents |
| --- | --- |
| `run_metadata.json` | image and **image digest**, vLLM version and commit, serve command, server env vars, device/tp/precision, and the Buildkite build/job identifiers |
| `bench-<name>.json` | raw `vllm bench serve` output, one per config |
| `bench-<name>.cmd` | the exact `vllm bench serve` command line, as it ran |
| `<task>.cmd` | the exact `lm_eval` command line, as it ran |
| `<task>/**/results_*.json` | lm_eval scores |
| `<task>/**/samples_*.jsonl` | per-sample records |

`run_metadata.json` is written by `lib/write_run_metadata.py` once the server reports healthy, which is the earliest point the vLLM version is known. Anything it could not capture is recorded as `null` rather than omitted, so a reader can tell "not captured" from "not applicable". Values whose names look like credentials (`*TOKEN*`, `*SECRET*`, `*PASSWD*`, `*KEY*`) are redacted, since the file is a public artifact.

### Retrieving a build's results

```bash
bk artifacts download --build 46 --pipeline perf-eval            # whole build
bk artifacts download --build 46 --pipeline perf-eval --job-uuid <job>
```

The download reproduces the `results/<workload>/...` tree, so `run_metadata.json` sits beside the numbers it describes. To re-run exactly what a build ran, take `serve_command` and the `.cmd` files verbatim and pin the image by `image_digest` rather than its tag — a tag like `:nightly` moves, a digest does not.

### Cron: loading artifacts into SQL

The pipeline itself never writes to a database — the work servers cannot reach
one. `lib/cron_artifact_to_sql.py` is a **standalone** job that runs elsewhere:
it pulls each finished build's artifacts from Buildkite and loads them into the
SQL tables for processing downstream. It imports nothing from this repo and
shares no configuration with it, so it is deployed by copying the single file.

```cron
*/30 * * * * /usr/bin/python3 $HOME/perf-eval-sync/cron_artifact_to_sql.py \
    --since-days 2 --state $HOME/perf-eval-sync/state.json \
    >> $HOME/perf-eval-sync/sync.log 2>&1
```

Settings come from an env file — the intended route for cron, which supplies
almost no environment. It looks for `~/.perf-eval-sync.env` then `~/.env`, or
pass `--env-file`:

```
TIGER_SQL_HOST=db.example.com
TIGER_SQL_PORT=3307
TIGER_SQL_USER=someone
TIGER_SQL_PASSWD=...
TIGER_SQL_DB=tiger_buildkite
BUILDKITE_API_TOKEN=bkua_...
```

Keep it `chmod 600` — it holds a database password and an API token, and the job
warns if the file is readable by others.

**Every write is an upsert** keyed on a content hash, so running twice is
harmless: a build already loaded is updated in place, never duplicated. The
state file is therefore only an optimization — it avoids re-downloading
gigabytes of samples — and a flock stops two runs colliding. It never issues
DDL; `--check` verifies the tables and columns exist and `--print-schema` prints
the shape it expects.

| Flag | Effect |
| --- | --- |
| `--since-days N` | window of builds to examine (default 2) |
| `--build N` / `--range A-B` | specific builds, for a one-time catch-up |
| `--no-samples` | skip `samples_*.jsonl`, which dominate the volume |
| `--reload` | ignore the state file and load the selected builds again |
| `--check` | verify connection and schema, then exit |
| `--dry-run` | report what would be loaded, write nothing |

`image` and `vllm_commit` come from `run_metadata.json` when present (what
actually ran) and fall back to the build's `VLLM_IMAGE` / `VLLM_COMMIT` env.

**Builds predating `run_metadata.json` load eval data but not perf rows.** A
perf row needs the `isl`/`osl` the bench was asked for, and the raw bench JSON
records only `max_concurrency`. Those are reported per config and skipped.

### Checking connectivity to a database

`lib/sql_connectivity_check.py` is a leftover standalone diagnostic from when results were also written to a SQL database. Nothing in the pipeline uses it. It is kept because it is self-contained and useful for testing whether a host is reachable from a pod (DNS, TCP, MySQL handshake, and optionally credentials), and can be deleted if that is no longer wanted.

### Trigger a Buildkite build

The pipeline is [**`vllm/perf-eval`**](https://buildkite.com/vllm/perf-eval). With no extra config, a build runs every workload that has `nightly: true`.

**From the UI:** open the pipeline → New Build → pick branch and commit (must be pushed to GitHub) → optionally fill Environment Variables to scope the run → Create Build.

**Required env vars** — both must be set on every build:

- `VLLM_COMMIT` — vLLM commit SHA being tested. Used to tag results and track which vLLM version produced them.
- `VLLM_IMAGE` — full Docker image URI (e.g. `vllm/vllm-openai:nightly-abc1234`). This is the image that gets pulled and run.

**Optional env vars:**

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
