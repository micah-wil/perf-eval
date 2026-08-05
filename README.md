# perf-eval

Run accuracy + perf workloads against vLLM, defined by small YAML recipes in `workloads/`.

Each recipe is one `(model, hardware, set of tasks)` combination. The Buildkite pipeline picks recipes up automatically — to ship a new run, you write a YAML file, push it, and trigger a build.

## Repo layout

```
workloads/        one YAML per (model, hardware) recipe
lib/              orchestrator (run.sh), helpers, GPU profiles, ingestion sinks
.buildkite/       pipeline bootstrap, step generator, and its tests
.sqlconn          local-only SQL credentials (gitignored; see Ingestion destinations)
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

### Ingestion destinations

Results go to one of two destinations:

| `INGEST_SINK` | Destination |
| --- | --- |
| `endpoint` | POST to the public Cloud Run endpoints backing the Databricks tables — the vLLM eval and perf dashboards |
| `sql` | INSERT into a MySQL/MariaDB database |
| `both` | Write to both |

**You normally don't set this.** When `INGEST_SINK` is unset the destination is detected: if `TIGER_SQL_DB` is configured — in the environment, in Buildkite Secrets, or in `.sqlconn` — results go to SQL **instead of** the public endpoint. With no SQL settings present the endpoint remains the default, so nothing changes for runs that never configure a database.

Set `INGEST_SINK` explicitly to override the detection: `endpoint` to keep publishing to the dashboards even with a database configured, or `both` to write to each.

**SQL tables.** `lib/sql_upload.py` owns the schema. Creating it is a **one-time step per database**, run by hand — eval runs only ever `INSERT`:

```bash
python3 lib/sql_upload.py --create-tables
```

Every statement is `CREATE TABLE IF NOT EXISTS`, so re-running it is harmless. When the SQL sink is selected, `run.sh` probes the connection with `--check` before starting the server and aborts if a table is missing, so a forgotten bootstrap fails immediately instead of after an hour of GPU time. Four tables:

- **`eval_results`** — one row per lm_eval `results_*.json`, with the full JSON in a `data` column.
- **`eval_metrics`** — `eval_results` flattened to one row per `(subtask, metric)` with `value` and `stderr`, so the dashboard can query scores without parsing JSON.
- **`eval_samples`** — one row per line of `samples_*.jsonl`.
- **`perf_results`** — one row per `vllm bench serve` config, with the dashboard's per-GPU throughput and latency columns. Unrecognized fields land in an `extra` JSON column instead of being dropped.

Every table carries the workload, task, image, vLLM commit, `nightly` flag, and Buildkite build columns. Writes are idempotent: a `dedupe_hash` unique key means re-running a step updates rows instead of duplicating them.

Inspect or bootstrap the schema without running an eval:

```bash
python3 lib/sql_upload.py --print-schema     # dump the DDL, no connection needed
python3 lib/sql_upload.py --check            # verify credentials, connectivity, tables
python3 lib/sql_upload.py --create-tables    # one-time: create anything missing
```

The account used by eval runs only needs `INSERT`/`UPDATE`/`SELECT` on these tables. `CREATE` is needed just for the one-time bootstrap, so it can be done by a different (admin) account — pipe the DDL in directly if that is easier:

```bash
python3 lib/sql_upload.py --print-schema | mysql -h <host> -u <admin> -p <database>
```

**Credentials.** Five settings, one set of names used everywhere — in the Buildkite secrets, in the Kubernetes secret's keys, in the environment, and in `.sqlconn`:

```
TIGER_SQL_HOST  TIGER_SQL_PORT  TIGER_SQL_USER  TIGER_SQL_PASSWD  TIGER_SQL_DB
```

`TIGER_SQL_HOST`, `TIGER_SQL_USER`, and `TIGER_SQL_DB` are required; `TIGER_SQL_PORT` defaults to 3306 and `TIGER_SQL_PASSWD` may be empty. `lib/sql_conn.sh` resolves them, first hit per key wins:

1. **The environment** — a Buildkite step env var, a Kubernetes `secretKeyRef`, or a manual export.
2. **Buildkite Secrets** — `buildkite-agent secret get TIGER_SQL_PASSWD`, etc.
3. **A local `.sqlconn` file** — development only. It is gitignored and must stay that way.

Nothing in this repo logs a password: only key names and their source are printed, and the connection summary is rendered redacted.

In CI, store the five values as Buildkite Secrets under exactly those names. For steps that run in a Kubernetes pod (every B200 and AMD profile), whether `buildkite-agent secret get` resolves depends on the agent stack's configuration, so the generator also wires the settings in as `secretKeyRef` entries from a cluster secret — `perf-eval-sql` by default, overridable with `SQL_SECRET_NAME`:

```bash
kubectl create secret generic perf-eval-sql \
  --namespace <agent-namespace> \
  --from-literal=TIGER_SQL_HOST=... \
  --from-literal=TIGER_SQL_PORT=... \
  --from-literal=TIGER_SQL_USER=... \
  --from-literal=TIGER_SQL_PASSWD=... \
  --from-literal=TIGER_SQL_DB=...
```

Every ref is `optional: true`, so a missing secret yields the loader's `missing credentials` abort rather than a pod that won't schedule. Either source satisfies the loader, and credentials are never written into the generated pipeline YAML.

`SQL_HOST` may be given as a bare hostname or a URL; a `http://`/`https://`/`mysql://` scheme, trailing path, and embedded port are all normalized away before the driver sees it.

A MySQL driver is required: `pymysql` (preferred, and installed by the pipeline's setup step) or `mysql-connector-python`.

Locally:

```bash
cat > .sqlconn <<'EOF'
TIGER_SQL_HOST="db.example.com"
TIGER_SQL_PORT=3306
TIGER_SQL_USER="someone"
TIGER_SQL_PASSWD="..."
TIGER_SQL_DB="tiger_db"
EOF
python3 lib/sql_upload.py --create-tables   # once per database
python3 lib/sql_upload.py --check
./lib/run.sh workloads/qwen3_5_h200.yaml    # TIGER_SQL_DB is set, so results go to SQL
```

### Trigger a Buildkite build

The pipeline is [**`vllm/perf-eval`**](https://buildkite.com/vllm/perf-eval). With no extra config, a build runs every workload that has `nightly: true`.

**From the UI:** open the pipeline → New Build → pick branch and commit (must be pushed to GitHub) → optionally fill Environment Variables to scope the run → Create Build.

**Required env vars** — both must be set on every build:

- `VLLM_COMMIT` — vLLM commit SHA being tested. Used to tag results and track which vLLM version produced them.
- `VLLM_IMAGE` — full Docker image URI (e.g. `vllm/vllm-openai:nightly-abc1234`). This is the image that gets pulled and run.

**Optional env vars:**

- `WORKLOADS` — comma- or newline-separated list of workload paths or stems. Runs exactly those instead of the default `nightly: true` set.
- `NIGHTLY` — set to `1` to tag every ingested row with `nightly: true`. The dashboard's `/nightly` view filters on this to pair adjacent nightly builds; only the scheduled nightly cron should set it.
- `INGEST_SINK` — `endpoint`, `sql`, or `both`. Usually unnecessary: a configured `TIGER_SQL_DB` already routes results to SQL. See [Ingestion destinations](#ingestion-destinations). Don't put credentials in build env vars; they belong in Buildkite Secrets.
- `SQL_SECRET_NAME` — name of the Kubernetes secret holding the SQL settings for pod-based steps. Defaults to `perf-eval-sql`.

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
