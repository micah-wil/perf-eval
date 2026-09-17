#!/usr/bin/env python3
"""Write a run's reproduction context to <results-dir>/run_metadata.json.

Only files under `results/` are uploaded as Buildkite artifacts, so context that
exists solely as environment variables while run.sh executes -- which image
actually ran, which vLLM build, the serve command, the server env -- is lost
when the job ends. This lands it next to the results it describes.

Missing values are recorded as null rather than omitted, so a reader can tell
"not captured" from "not applicable".

Usage: write_run_metadata.py <results-dir>
"""

import base64
import json
import os
import sys
from datetime import datetime, timezone

USAGE = "usage: write_run_metadata.py <results-dir>"

# Build label recorded with every result; see ingest.py.
RUN_TYPE_ENV = "PERF_EVAL_RUN_TYPE"
DEFAULT_RUN_TYPE = "adhoc"

# environment variable -> field in the emitted JSON
WORKLOAD_FIELDS = (
    ("WORKLOAD_NAME", "workload"),
    ("WORKLOAD_MODEL", "model"),
    ("WORKLOAD_IMAGE", "image"),
    ("WORKLOAD_IMAGE_DIGEST", "image_digest"),
    ("WORKLOAD_VLLM_COMMIT", "vllm_commit"),
    ("WORKLOAD_VLLM_VERSION", "vllm_version"),
    ("WORKLOAD_SERVER_RUNTIME", "server_runtime"),
    ("WORKLOAD_SERVE_ARGS", "serve_args"),
    ("WORKLOAD_SERVE_COMMAND", "serve_command"),
    ("WORKLOAD_BENCH_DEVICE", "device"),
    ("WORKLOAD_BENCH_TP", "tp"),
    ("WORKLOAD_BENCH_PRECISION", "precision"),
)

BUILDKITE_FIELDS = (
    "BUILDKITE_BUILD_ID",
    "BUILDKITE_BUILD_NUMBER",
    "BUILDKITE_BUILD_URL",
    "BUILDKITE_BRANCH",
    "BUILDKITE_COMMIT",
    "BUILDKITE_PIPELINE_SLUG",
    "BUILDKITE_JOB_ID",
)

# The bench TSV columns parse_workload.py emits, in order.
BENCH_TSV_FIELDS = ("name", "backend", "dataset", "isl", "osl", "num_prompts",
                    "conc", "repetitions", "speed_bench_subset",
                    "speed_bench_category", "args")
BENCH_INT_FIELDS = ("isl", "osl", "num_prompts", "conc", "repetitions")

# The server env is merged from the GPU profile and the recipe, and this file is
# uploaded as a public artifact.
SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWD", "PASSWORD", "KEY", "CREDENTIAL")


def env_or_none(name):
    value = (os.environ.get(name) or "").strip()
    return value or None


def parse_env_block(raw):
    """WORKLOAD_ENV is a newline-separated KEY=VALUE list."""
    out = {}
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        if any(m in key.upper() for m in SECRET_MARKERS):
            out[key] = "<redacted>"
        else:
            out[key] = value.strip()
    return out


def parse_bench_tsv(raw):
    """WORKLOAD_VLLM_BENCH_TSV -> one dict per vllm_bench config.

    Recorded because the raw `vllm bench serve` JSON does not carry the input
    and output lengths it was asked for, so without this the per-GPU numbers
    cannot be interpreted from the artifacts alone.
    """
    out = []
    for line in (raw or "").splitlines():
        if not line.strip():
            continue
        cells = line.split("\t")
        cfg = {}
        for i, field in enumerate(BENCH_TSV_FIELDS):
            value = cells[i] if i < len(cells) else ""
            cfg[field] = None if value in ("", "-") else value
        for field in BENCH_INT_FIELDS:
            try:
                cfg[field] = int(cfg[field])
            except (TypeError, ValueError):
                pass
        if cfg["args"]:
            try:
                cfg["args"] = json.loads(base64.b64decode(cfg["args"]))
            except (ValueError, TypeError):
                cfg["args"] = None
        out.append(cfg)
    return out


def parse_task_tsv(raw):
    """WORKLOAD_LM_EVAL_TASKS_TSV -> one dict per lm_eval task."""
    out = []
    for line in (raw or "").splitlines():
        if not line.strip():
            continue
        cells = line.split("\t")
        out.append({"name": cells[0],
                    "num_fewshot": cells[1] if len(cells) > 1 else None,
                    "model_args": cells[2] if len(cells) > 2 else None})
    return out


def main():
    if len(sys.argv) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    results_dir = sys.argv[1]

    data = {"schema": 1,
            "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    for env_name, field in WORKLOAD_FIELDS:
        data[field] = env_or_none(env_name)
    data["env_vars"] = parse_env_block(os.environ.get("WORKLOAD_ENV"))
    data["bench_configs"] = parse_bench_tsv(os.environ.get("WORKLOAD_VLLM_BENCH_TSV"))
    data["lm_eval_tasks"] = parse_task_tsv(os.environ.get("WORKLOAD_LM_EVAL_TASKS_TSV"))
    data["run_type"] = (os.environ.get(RUN_TYPE_ENV) or "").strip() or DEFAULT_RUN_TYPE
    data["nightly"] = os.environ.get("NIGHTLY") == "1"
    data["buildkite"] = {k.lower(): env_or_none(k) for k in BUILDKITE_FIELDS}

    path = os.path.join(results_dir, "run_metadata.json")
    try:
        os.makedirs(results_dir, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
    except OSError as e:
        # Never fail a run over bookkeeping.
        print(f"  run metadata: could not write ({e})", file=sys.stderr)
        return 0

    print(f"  run metadata -> {path}")
    missing = [f for f in ("image", "image_digest", "vllm_version", "serve_command")
               if not data.get(f)]
    if missing:
        print(f"  run metadata: not captured: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
