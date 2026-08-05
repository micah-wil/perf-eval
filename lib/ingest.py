#!/usr/bin/env python3
"""Upload lm_eval JSON artifacts to the eval data ingestion destination.

Walks a per-task results dir produced by lm_eval and uploads:
  - results_*.json   -> one event per file
  - samples_*.jsonl  -> one event per line (per sample)

Each event is wrapped with workload/task/Buildkite metadata so rows in the
backing table can be filtered by run.

Two destinations are supported, selected by --sink (env: INGEST_SINK):
  endpoint  POST to the Cloud Run endpoint backing the Databricks table (default)
  sql       INSERT into the MySQL database described by lib/sql_upload.py
  both      write to both

Failures are logged but never fatal: ingestion is best-effort and must not
abort the lm_eval pipeline.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sql_upload  # noqa: E402  (same-dir helper; path set above)

DEFAULT_ENDPOINT = "https://vllm-eval-data-ingest-224810116257.us-central1.run.app/"
TIMEOUT = 30
# Databricks Zerobus rejects records larger than 10 MiB and closes the stream.
# Pack samples into batches that stay safely under that ceiling.
SAMPLES_BATCH_BYTES = 4 * 1024 * 1024
BK_ENV_VARS = (
    "BUILDKITE_BUILD_ID",
    "BUILDKITE_BUILD_NUMBER",
    "BUILDKITE_BUILD_URL",
    "BUILDKITE_BRANCH",
    "BUILDKITE_COMMIT",
    "BUILDKITE_PIPELINE_SLUG",
)
# Top-level fields the dashboard reads to show "image" and the vLLM commit.
# WORKLOAD_IMAGE is the resolved docker URI (set by parse_workload.py via the
# VLLM_IMAGE / VLLM_COMMIT override env vars or the workload yaml's vllm.image).
# WORKLOAD_VLLM_COMMIT is the commit used by that resolved image, when it can
# be determined from VLLM_COMMIT or a commit-bearing image tag.
VLLM_ENV_VARS = (
    ("WORKLOAD_IMAGE", "image"),
    ("WORKLOAD_VLLM_COMMIT", "vllm_commit"),
)
# Set NIGHTLY=1 in the build env to mark rows as part of the nightly schedule.
# The dashboard's /nightly view filters on this to pair adjacent nightlies.
NIGHTLY_ENV = "NIGHTLY"
# Destination selection; see the module docstring.
SINKS = ("endpoint", "sql", "both")


def post(endpoint: str, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"HTTP {resp.status}")


def metadata(workload: str, task: str) -> dict:
    md = {"workload": workload, "task": task}
    for k in BK_ENV_VARS:
        v = os.environ.get(k)
        if v:
            md[k.lower()] = v
    for env_key, field in VLLM_ENV_VARS:
        v = (os.environ.get(env_key) or "").strip()
        if v:
            md[field] = v
    if os.environ.get(NIGHTLY_ENV) == "1":
        md["nightly"] = True
    return md


def ingest_results(path: Path, md: dict, endpoint: str, conn=None) -> None:
    with path.open() as f:
        data = json.load(f)
    if endpoint:
        payload = {"kind": "results", "source_file": str(path), **md, "data": data}
        post(endpoint, payload)
    if conn is not None:
        n = sql_upload.write_results(conn, path, md, data)
        print(f"    sql: {path.name} + {n} metric row(s)")


def ingest_samples(path: Path, md: dict, endpoint: str, conn=None) -> int:
    sent = 0
    batch: list = []
    batch_bytes = 0
    batch_start = 0
    overhead = len(
        json.dumps({"kind": "samples", "source_file": str(path), **md, "samples": []})
    )

    def flush() -> None:
        nonlocal batch, batch_bytes, sent, batch_start
        if not batch:
            return
        if endpoint:
            payload = {
                "kind": "samples", "source_file": str(path), **md, "samples": batch,
            }
            post(endpoint, payload)
        if conn is not None:
            sql_upload.write_samples(conn, path, md, batch, start_index=batch_start)
        sent += len(batch)
        batch_start += len(batch)
        batch = []
        batch_bytes = 0

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"    skip malformed sample line: {e}", file=sys.stderr)
                continue
            sample_bytes = len(json.dumps(sample))
            # +1 for the array-element comma
            if batch and overhead + batch_bytes + sample_bytes + 1 > SAMPLES_BATCH_BYTES:
                flush()
            batch.append(sample)
            batch_bytes += sample_bytes + 1
    flush()
    return sent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True, help="Per-task results dir from lm_eval")
    p.add_argument("--workload", required=True, help="Workload (recipe) name")
    p.add_argument("--task", required=True, help="lm_eval task name")
    p.add_argument(
        "--endpoint",
        default=os.environ.get("INGEST_URL", DEFAULT_ENDPOINT),
        help="Ingestion endpoint (env: INGEST_URL)",
    )
    p.add_argument("--no-samples", action="store_true", help="Skip samples_*.jsonl uploads")
    p.add_argument(
        "--sink",
        choices=SINKS,
        default=sql_upload.default_sink(),
        help="Where to write: endpoint, sql, or both (env: INGEST_SINK;"
             " defaults to sql when TIGER_SQL_DB is configured)",
    )
    p.add_argument(
        "--sqlconn-file",
        default=None,
        help="Path to a .sqlconn file for --sink sql (env: SQLCONN_FILE)",
    )
    args = p.parse_args()

    root = Path(args.results_dir)
    if not root.is_dir():
        print(f"  ingest: results dir not found: {root}", file=sys.stderr)
        return 0

    results_files = sorted(root.glob("**/results_*.json"))
    samples_files = [] if args.no_samples else sorted(root.glob("**/samples_*.jsonl"))
    endpoint = args.endpoint if args.sink in ("endpoint", "both") else None

    conn = None
    if args.sink in ("sql", "both"):
        try:
            conn, config = sql_upload.open_sink(args.sqlconn_file)
            print(f"  ingest -> sql {sql_upload.describe(config)}")
        except sql_upload.SqlSinkError as e:
            print(f"  ingest: sql sink unavailable: {e}", file=sys.stderr)
        except Exception as e:
            print(f"  ingest: sql connect failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
    if endpoint:
        print(f"  ingest -> {endpoint}")
    if endpoint is None and conn is None:
        # Every selected destination failed to open. Say so plainly rather than
        # walking the files and reporting uploads that went nowhere.
        print(f"  ingest: no destination available for sink {args.sink!r};"
              f" {len(results_files)} results file(s) not uploaded", file=sys.stderr)
        return 0
    print(f"  ({len(results_files)} results, {len(samples_files)} sample file(s))")

    md = metadata(args.workload, args.task)

    try:
        for f in results_files:
            try:
                ingest_results(f, md, endpoint, conn)
                print(f"    uploaded {f.relative_to(root)}")
            except (urllib.error.URLError, RuntimeError, OSError) as e:
                print(f"    failed {f.relative_to(root)}: {e}", file=sys.stderr)
            except Exception as e:  # driver-specific write errors
                print(f"    failed {f.relative_to(root)}: {type(e).__name__}: {e}",
                      file=sys.stderr)

        for f in samples_files:
            try:
                n = ingest_samples(f, md, endpoint, conn)
                print(f"    uploaded {f.relative_to(root)} ({n} samples)")
            except OSError as e:
                print(f"    failed {f.relative_to(root)}: {e}", file=sys.stderr)
            except Exception as e:  # driver-specific write errors
                print(f"    failed {f.relative_to(root)}: {type(e).__name__}: {e}",
                      file=sys.stderr)
    finally:
        if conn is not None:
            conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
