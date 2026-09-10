#!/usr/bin/env python3
"""Write the run's reproduction context into the results tree.

Everything a run produces is uploaded as a Buildkite artifact from `results/`,
so anything not written there is lost when the job ends. The result JSONs cover
the numbers and the `.cmd` files cover the exact commands, but the surrounding
context -- which image actually ran, which vLLM build, what environment the
server had -- only exists as environment variables while run.sh is executing.
This lands it next to the results so a build stays reproducible from its
artifacts alone.

Usage: write_run_metadata.py <results-dir>

Reads the WORKLOAD_* and BUILDKITE_* variables run.sh already exports and writes
<results-dir>/run_metadata.json. Missing values are recorded as null rather than
omitted, so a reader can tell "not captured" from "not applicable".
"""

import json
import os
import sys
from datetime import datetime, timezone

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

# The bench TSV columns parse_workload emits, in order.
BENCH_TSV_FIELDS = ("name", "backend", "dataset", "isl", "osl", "num_prompts",
                    "conc", "speed_bench_subset", "speed_bench_category")

# Anything matching these is redacted: the workload env is merged from the GPU
# profile and the recipe, and this file is uploaded as a public artifact.
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

    Recorded because the raw `vllm bench serve` JSON does not carry the input and
    output lengths it was asked for, so without this a reader cannot reconstruct
    the per-GPU numbers from the artifacts alone.
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
        for field in ("isl", "osl", "num_prompts", "conc"):
            try:
                cfg[field] = int(cfg[field])
            except (TypeError, ValueError):
                pass
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
        print(__doc__.strip().splitlines()[8], file=sys.stderr)
        return 2
    results_dir = sys.argv[1]

    data = {"schema": 1,
            "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    for env_name, field in WORKLOAD_FIELDS:
        data[field] = env_or_none(env_name)
    data["env_vars"] = parse_env_block(os.environ.get("WORKLOAD_ENV"))
    data["bench_configs"] = parse_bench_tsv(os.environ.get("WORKLOAD_VLLM_BENCH_TSV"))
    data["lm_eval_tasks"] = parse_task_tsv(os.environ.get("WORKLOAD_LM_EVAL_TASKS_TSV"))
    data["nightly"] = os.environ.get("NIGHTLY") == "1"
    data["buildkite"] = {k.lower(): env_or_none(k) for k in BUILDKITE_FIELDS}

    try:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, "run_metadata.json")
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=False)
            fh.write("\n")
    except OSError as e:
        # Never fail a run over bookkeeping.
        print(f"  run metadata: could not write ({e})", file=sys.stderr)
        return 0

    missing = [f for f in ("image", "image_digest", "vllm_version", "serve_command")
               if not data.get(f)]
    print(f"  run metadata -> {path}")
    if missing:
        print(f"  run metadata: not captured: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
