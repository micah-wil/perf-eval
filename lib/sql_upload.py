#!/usr/bin/env python3
"""MySQL/MariaDB sink for perf-eval ingestion.

`ingest.py` and `ingest_perf.py` normally POST to the public Cloud Run
ingestion endpoints. This module is the alternative destination: a SQL database
whose connection details come from the environment (Buildkite secrets in CI) or
from a local `.sqlconn` file for development.

The five settings are `TIGER_SQL_HOST`, `TIGER_SQL_PORT`, `TIGER_SQL_USER`,
`TIGER_SQL_PASSWD`, and `TIGER_SQL_DB` — the same names in the Buildkite
secrets, the Kubernetes secret keys, the environment, and `.sqlconn`.

Nothing here ever prints a password: `describe()` renders a redacted summary and
that is the only connection string this repo logs.

Creating the tables is a one-time administrative step, run by hand against the
target database — the ingest scripts only ever INSERT:

    python3 lib/sql_upload.py --create-tables   # once, per database
    python3 lib/sql_upload.py --check           # verify credentials/connectivity

It reads the same credentials as the ingest scripts. Every statement is
`CREATE TABLE IF NOT EXISTS`, so re-running it on an existing database is safe.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Connection settings. These are the names used everywhere: the Buildkite
# secrets, the Kubernetes secret keys, the process environment, and `.sqlconn`.
# TIGER_SQL_PASSWD is deliberately never logged or echoed back.
CONN_KEYS = (
    "TIGER_SQL_HOST",
    "TIGER_SQL_PORT",
    "TIGER_SQL_USER",
    "TIGER_SQL_PASSWD",
    "TIGER_SQL_DB",
)
DEFAULT_PORT = 3306
# Local-dev fallback: repo-root .sqlconn (gitignored). Override with SQLCONN_FILE.
DEFAULT_CONN_FILE = Path(__file__).resolve().parent.parent / ".sqlconn"
CONNECT_TIMEOUT = 30
# Distinct CLI exit codes so run.sh can tell "the database is unreachable" (no
# point retrying anything) from "the connection is fine but the DDL failed"
# (likely just a user without CREATE, which is a warning, not a failure).
EXIT_STATEMENT = 1
EXIT_CONFIG = 2
EXIT_CONNECT = 3
# Rows per executemany() call when writing samples; keeps a single statement
# well under any server max_allowed_packet while still amortizing round trips.
INSERT_BATCH_ROWS = 200

TABLE_EVAL_RESULTS = "eval_results"
TABLE_EVAL_METRICS = "eval_metrics"
TABLE_EVAL_SAMPLES = "eval_samples"
TABLE_PERF_RESULTS = "perf_results"

# Buildkite provenance columns shared by every table, so any row can be traced
# back to the build that produced it.
_BK_COLUMNS = """  buildkite_build_id VARCHAR(64) NULL,
  buildkite_build_number VARCHAR(32) NULL,
  buildkite_build_url VARCHAR(512) NULL,
  buildkite_branch VARCHAR(255) NULL,
  buildkite_commit VARCHAR(64) NULL,
  buildkite_pipeline_slug VARCHAR(128) NULL,
  nightly TINYINT(1) NOT NULL DEFAULT 0,"""

BK_FIELDS = (
    "buildkite_build_id",
    "buildkite_build_number",
    "buildkite_build_url",
    "buildkite_branch",
    "buildkite_commit",
    "buildkite_pipeline_slug",
)

# `dedupe_hash` makes every insert idempotent: re-ingesting the same file (a
# retried Buildkite step, a re-run of run.sh) updates the existing row instead
# of appending a duplicate. It is hashed rather than indexed directly because
# source_file paths are longer than MySQL's index key limit.
SCHEMA = {
    TABLE_EVAL_RESULTS: f"""
CREATE TABLE IF NOT EXISTS {TABLE_EVAL_RESULTS} (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  workload VARCHAR(255) NOT NULL,
  task VARCHAR(255) NOT NULL,
  source_file VARCHAR(1024) NOT NULL,
  model VARCHAR(512) NULL,
  image VARCHAR(512) NULL,
  vllm_commit VARCHAR(64) NULL,
{_BK_COLUMNS}
  data JSON NULL,
  dedupe_hash CHAR(64) NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_eval_results_dedupe (dedupe_hash),
  KEY idx_eval_results_workload_task (workload, task),
  KEY idx_eval_results_build (buildkite_build_id),
  KEY idx_eval_results_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""",
    TABLE_EVAL_METRICS: f"""
CREATE TABLE IF NOT EXISTS {TABLE_EVAL_METRICS} (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  result_id BIGINT UNSIGNED NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  workload VARCHAR(255) NOT NULL,
  task VARCHAR(255) NOT NULL,
  subtask VARCHAR(255) NOT NULL,
  metric VARCHAR(255) NOT NULL,
  value DOUBLE NULL,
  stderr DOUBLE NULL,
  dedupe_hash CHAR(64) NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_eval_metrics_dedupe (dedupe_hash),
  KEY idx_eval_metrics_result (result_id),
  KEY idx_eval_metrics_lookup (task, metric)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""",
    TABLE_EVAL_SAMPLES: f"""
CREATE TABLE IF NOT EXISTS {TABLE_EVAL_SAMPLES} (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  workload VARCHAR(255) NOT NULL,
  task VARCHAR(255) NOT NULL,
  source_file VARCHAR(1024) NOT NULL,
  sample_index INT NOT NULL,
  doc_id VARCHAR(128) NULL,
  image VARCHAR(512) NULL,
  vllm_commit VARCHAR(64) NULL,
{_BK_COLUMNS}
  sample JSON NULL,
  dedupe_hash CHAR(64) NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_eval_samples_dedupe (dedupe_hash),
  KEY idx_eval_samples_workload_task (workload, task),
  KEY idx_eval_samples_build (buildkite_build_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""",
    TABLE_PERF_RESULTS: f"""
CREATE TABLE IF NOT EXISTS {TABLE_PERF_RESULTS} (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  run_date DATETIME NULL,
  workload VARCHAR(255) NULL,
  bench_name VARCHAR(255) NULL,
  device VARCHAR(64) NULL,
  model VARCHAR(512) NULL,
  image VARCHAR(512) NULL,
  vllm_commit VARCHAR(64) NULL,
  framework VARCHAR(64) NULL,
  precision_tag VARCHAR(64) NULL,
  spec_decoding VARCHAR(16) NULL,
  disagg VARCHAR(16) NULL,
  is_multinode VARCHAR(16) NULL,
  dp_attention VARCHAR(16) NULL,
  conc INT NULL,
  isl INT NULL,
  osl INT NULL,
  tp INT NULL,
  ep INT NULL,
  tput_per_gpu DOUBLE NULL,
  output_tput_per_gpu DOUBLE NULL,
  input_tput_per_gpu DOUBLE NULL,
  mean_ttft DOUBLE NULL,
  median_ttft DOUBLE NULL,
  std_ttft DOUBLE NULL,
  p99_ttft DOUBLE NULL,
  mean_tpot DOUBLE NULL,
  median_tpot DOUBLE NULL,
  std_tpot DOUBLE NULL,
  p99_tpot DOUBLE NULL,
  mean_itl DOUBLE NULL,
  median_itl DOUBLE NULL,
  std_itl DOUBLE NULL,
  p99_itl DOUBLE NULL,
  mean_e2el DOUBLE NULL,
  median_e2el DOUBLE NULL,
  std_e2el DOUBLE NULL,
  p99_e2el DOUBLE NULL,
  mean_intvty DOUBLE NULL,
  median_intvty DOUBLE NULL,
  std_intvty DOUBLE NULL,
  p99_intvty DOUBLE NULL,
{_BK_COLUMNS}
  extra JSON NULL,
  dedupe_hash CHAR(64) NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_perf_results_dedupe (dedupe_hash),
  KEY idx_perf_results_device_model (device, model),
  KEY idx_perf_results_build (buildkite_build_id),
  KEY idx_perf_results_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""",
}

# Columns perf_results stores natively; anything else the transform emits is
# folded into `extra` so a new dashboard field never drops data on the floor.
PERF_COLUMNS = (
    "run_date", "workload", "bench_name", "device", "model", "image",
    "vllm_commit", "framework", "precision_tag", "spec_decoding", "disagg",
    "is_multinode", "dp_attention", "conc", "isl", "osl", "tp", "ep",
    "tput_per_gpu", "output_tput_per_gpu", "input_tput_per_gpu",
    "mean_ttft", "median_ttft", "std_ttft", "p99_ttft",
    "mean_tpot", "median_tpot", "std_tpot", "p99_tpot",
    "mean_itl", "median_itl", "std_itl", "p99_itl",
    "mean_e2el", "median_e2el", "std_e2el", "p99_e2el",
    "mean_intvty", "median_intvty", "std_intvty", "p99_intvty",
)
# Payload key -> perf_results column, where the two names differ. `date` and
# `precision` are reserved-ish words, so they are stored under safer names.
PERF_RENAMES = {"date": "run_date", "precision": "precision_tag"}


class SqlSinkError(RuntimeError):
    """Raised for configuration and connection problems worth reporting."""


def default_sink(conn_file=None):
    """The ingestion destination to use when --sink/INGEST_SINK is not given.

    A configured TIGER_SQL_DB means the SQL database is the destination, so
    results go there rather than to the public endpoint. With no SQL settings
    present the endpoint remains the default.
    """
    explicit = (os.environ.get("INGEST_SINK") or "").strip().lower()
    if explicit:
        return explicit
    if (os.environ.get("TIGER_SQL_DB") or "").strip():
        return "sql"
    path = Path(conn_file or os.environ.get("SQLCONN_FILE") or DEFAULT_CONN_FILE)
    if path.is_file():
        try:
            if _parse_conn_file(path).get("TIGER_SQL_DB"):
                return "sql"
        except OSError:
            pass
    return "endpoint"


def print_debug_state(conn_file=None):
    """Print what the destination decision saw. Never prints a password."""
    path = Path(conn_file or os.environ.get("SQLCONN_FILE") or DEFAULT_CONN_FILE)
    print(f"  sql-debug: INGEST_SINK={os.environ.get('INGEST_SINK') or '<unset>'}")
    for key in CONN_KEYS:
        raw = os.environ.get(key)
        if key == "TIGER_SQL_PASSWD":
            shown = "<set>" if (raw or "").strip() else "<unset>"
        else:
            shown = raw if (raw or "").strip() else "<unset>"
        print(f"  sql-debug: {key}={shown}")
    print(f"  sql-debug: conn file {path}"
          f" {'exists' if path.is_file() else 'missing'}")
    if path.is_file():
        try:
            keys = sorted(_parse_conn_file(path))
        except OSError as e:
            print(f"  sql-debug: conn file unreadable: {e}")
        else:
            print(f"  sql-debug: conn file keys: {', '.join(keys) or '<none>'}")


def _parse_conn_file(path):
    """Read shell-style `KEY=value` / `KEY="value"` pairs from a .sqlconn file."""
    values = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if key not in CONN_KEYS:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def _normalize_host(host, port):
    """Strip a URL wrapper off the host and pull out an embedded port.

    `.sqlconn` may carry a dashboard-style URL (``http://host/``) even though
    the driver needs a bare hostname, so scheme, path, and trailing slash are
    removed here. An embedded ``host:1234`` supplies the port when none was set
    explicitly.
    """
    host = (host or "").strip()
    for scheme in ("http://", "https://", "mysql://", "tcp://"):
        if host.lower().startswith(scheme):
            host = host[len(scheme):]
            break
    host = host.split("/", 1)[0].strip()
    if "@" in host:  # user:pass@host — credentials come from their own keys
        host = host.rsplit("@", 1)[1]
    if host.count(":") == 1:
        host, _, embedded = host.partition(":")
        if embedded.isdigit() and not port:
            port = embedded
    return host, port


def load_config(conn_file=None):
    """Resolve connection settings from the environment, then a `.sqlconn` file.

    Environment wins so Buildkite secrets (exported by `lib/sql_conn.sh`) always
    take precedence over a stale local file.
    """
    values = {k: (os.environ.get(k) or "").strip() for k in CONN_KEYS}
    path = Path(conn_file or os.environ.get("SQLCONN_FILE") or DEFAULT_CONN_FILE)
    file_used = None
    if path.is_file():
        from_file = _parse_conn_file(path)
        for key, value in from_file.items():
            if not values.get(key) and value:
                values[key] = value
                file_used = str(path)

    port_raw = values["TIGER_SQL_PORT"]
    host, port_raw = _normalize_host(values["TIGER_SQL_HOST"], port_raw)
    required = ("TIGER_SQL_HOST", "TIGER_SQL_USER", "TIGER_SQL_DB")
    missing = [k for k in required if not values[k]]
    if not host and "TIGER_SQL_HOST" not in missing:
        missing.insert(0, "TIGER_SQL_HOST")
    if missing:
        raise SqlSinkError(
            "missing SQL connection settings: "
            + ", ".join(missing)
            + " — set them in the environment, or in .sqlconn for local runs"
        )
    try:
        port = int(port_raw) if port_raw else DEFAULT_PORT
    except ValueError:
        raise SqlSinkError(
            f"TIGER_SQL_PORT is not an integer: {port_raw!r}"
        ) from None

    return {
        "host": host,
        "port": port,
        "user": values["TIGER_SQL_USER"],
        "password": values["TIGER_SQL_PASSWD"],
        "database": values["TIGER_SQL_DB"],
        "conn_file": file_used,
    }


def describe(config):
    """Redacted connection summary — the only form safe to log."""
    return (
        f"{config['user']}@{config['host']}:{config['port']}/{config['database']}"
        " (password redacted)"
    )


def connect(config):
    """Open a connection using whichever MySQL driver is installed."""
    kwargs = {
        "host": config["host"],
        "port": config["port"],
        "user": config["user"],
        "password": config["password"],
        "database": config["database"],
    }
    try:
        import pymysql
    except ImportError:
        pass
    else:
        return pymysql.connect(
            charset="utf8mb4",
            connect_timeout=CONNECT_TIMEOUT,
            autocommit=False,
            **kwargs,
        )
    try:
        import mysql.connector
    except ImportError:
        raise SqlSinkError(
            "no MySQL driver available: pip install pymysql"
            " (or mysql-connector-python)"
        ) from None
    return mysql.connector.connect(
        charset="utf8mb4",
        connection_timeout=CONNECT_TIMEOUT,
        autocommit=False,
        **kwargs,
    )


def ensure_tables(conn, tables=None):
    """Create any missing tables. Idempotent; safe to call on every run."""
    names = tables or list(SCHEMA)
    with conn.cursor() as cur:
        for name in names:
            cur.execute(SCHEMA[name].strip())
    conn.commit()


def missing_tables(conn):
    """Names of expected tables that do not exist in the connected database."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables"
            " WHERE table_schema = DATABASE()"
        )
        present = {str(row[0]).lower() for row in cur.fetchall() or ()}
    return [t for t in SCHEMA if t.lower() not in present]


def _hash(*parts):
    joined = "\x00".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _upsert(conn, table, row, returning_id=False):
    """INSERT ... ON DUPLICATE KEY UPDATE for a single row dict.

    Returns the row's primary key when `returning_id` is set, resolving it via a
    `dedupe_hash` lookup on the duplicate path (`lastrowid` is unreliable there).
    """
    cols = list(row)
    placeholders = ", ".join(["%s"] * len(cols))
    updates = ", ".join(f"{c}=VALUES({c})" for c in cols if c != "dedupe_hash")
    sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
        f" ON DUPLICATE KEY UPDATE {updates}"
    )
    with conn.cursor() as cur:
        cur.execute(sql, [row[c] for c in cols])
        if not returning_id:
            return None
        cur.execute(
            f"SELECT id FROM {table} WHERE dedupe_hash = %s", (row["dedupe_hash"],)
        )
        found = cur.fetchone()
    return found[0] if found else None


def _insert_many(conn, table, rows):
    if not rows:
        return 0
    cols = list(rows[0])
    placeholders = ", ".join(["%s"] * len(cols))
    updates = ", ".join(f"{c}=VALUES({c})" for c in cols if c != "dedupe_hash")
    sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
        f" ON DUPLICATE KEY UPDATE {updates}"
    )
    written = 0
    with conn.cursor() as cur:
        for start in range(0, len(rows), INSERT_BATCH_ROWS):
            chunk = rows[start:start + INSERT_BATCH_ROWS]
            cur.executemany(sql, [[r[c] for c in cols] for r in chunk])
            written += len(chunk)
    return written


def _bk_row(md):
    row = {f: md.get(f) for f in BK_FIELDS}
    row["nightly"] = 1 if md.get("nightly") else 0
    return row


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _model_of(data):
    """Best-effort model name out of an lm_eval results JSON."""
    config = data.get("config") or {}
    model = config.get("model_name") or config.get("model")
    args = config.get("model_args")
    if isinstance(args, str):
        for part in args.split(","):
            key, sep, value = part.partition("=")
            if sep and key.strip() == "model":
                return value.strip()
    elif isinstance(args, dict) and args.get("model"):
        return str(args["model"])
    return str(model) if model else None


def _split_metric_key(key):
    """Split an lm_eval metric key into (name, filter, is_stderr).

    Keys are `<metric>,<filter>` — e.g. `exact_match,strict-match` — and the
    matching error bar is `<metric>_stderr,<filter>`. The `_stderr` marker sits
    on the metric name, *before* the filter, so it cannot be found by looking at
    the end of the whole key.
    """
    name, sep, filt = key.partition(",")
    is_stderr = name.endswith("_stderr")
    if is_stderr:
        name = name.removesuffix("_stderr")
    return (name + sep + filt), is_stderr


def _metric_rows(result_id, md, results):
    """Flatten lm_eval's `results` block into (subtask, metric, value, stderr)."""
    rows = []
    for subtask, metrics in (results or {}).items():
        if not isinstance(metrics, dict):
            continue
        stderrs = {}
        for k, v in metrics.items():
            base, is_stderr = _split_metric_key(k)
            if is_stderr:
                stderrs[base] = v
        for key, value in metrics.items():
            if key == "alias" or _split_metric_key(key)[1]:
                continue
            numeric = _as_float(value)
            if numeric is None:
                continue
            rows.append({
                "result_id": result_id,
                "workload": md["workload"],
                "task": md["task"],
                "subtask": str(subtask),
                "metric": str(key),
                "value": numeric,
                "stderr": _as_float(stderrs.get(key)),
                "dedupe_hash": _hash(result_id, subtask, key),
            })
    return rows


def write_results(conn, path, md, data):
    """Store one lm_eval `results_*.json` plus its flattened metrics."""
    row = {
        "workload": md["workload"],
        "task": md["task"],
        "source_file": str(path),
        "model": _model_of(data),
        "image": md.get("image"),
        "vllm_commit": md.get("vllm_commit"),
        **_bk_row(md),
        "data": json.dumps(data),
        "dedupe_hash": _hash(
            md.get("buildkite_build_id"), md["workload"], md["task"], str(path)
        ),
    }
    result_id = _upsert(conn, TABLE_EVAL_RESULTS, row, returning_id=True)
    metrics = _metric_rows(result_id, md, data.get("results"))
    _insert_many(conn, TABLE_EVAL_METRICS, metrics)
    conn.commit()
    return len(metrics)


def write_samples(conn, path, md, samples, start_index=0):
    """Store a batch of `samples_*.jsonl` records, one row each."""
    bk = _bk_row(md)
    rows = []
    for offset, sample in enumerate(samples):
        index = start_index + offset
        doc_id = sample.get("doc_id") if isinstance(sample, dict) else None
        rows.append({
            "workload": md["workload"],
            "task": md["task"],
            "source_file": str(path),
            "sample_index": index,
            "doc_id": None if doc_id is None else str(doc_id)[:128],
            "image": md.get("image"),
            "vllm_commit": md.get("vllm_commit"),
            **bk,
            "sample": json.dumps(sample),
            "dedupe_hash": _hash(
                md.get("buildkite_build_id"), md["workload"], md["task"],
                str(path), index,
            ),
        })
    written = _insert_many(conn, TABLE_EVAL_SAMPLES, rows)
    conn.commit()
    return written


def write_perf(conn, data, workload=None, bench_name=None):
    """Store one transformed `vllm bench serve` row."""
    row = {c: None for c in PERF_COLUMNS}
    extra = {}
    for key, value in data.items():
        if key == "nightly":
            continue
        column = PERF_RENAMES.get(key, key)
        if column in row:
            row[column] = value
        else:
            extra[key] = value
    if workload:
        row["workload"] = workload
    if bench_name:
        row["bench_name"] = bench_name
    row["vllm_commit"] = (os.environ.get("WORKLOAD_VLLM_COMMIT") or "").strip() or None

    md = {f: (os.environ.get(f.upper()) or "").strip() or None for f in BK_FIELDS}
    md["nightly"] = data.get("nightly") or os.environ.get("NIGHTLY") == "1"
    row.update(_bk_row(md))
    row["extra"] = json.dumps(extra) if extra else None
    # A build re-running the same bench config should update its row, not add
    # a second one; without a build id, fall back to the run timestamp.
    row["dedupe_hash"] = _hash(
        md.get("buildkite_build_id") or row["run_date"],
        row["workload"], row["bench_name"], row["device"], row["model"],
        row["isl"], row["osl"], row["conc"],
    )
    _upsert(conn, TABLE_PERF_RESULTS, row)
    conn.commit()


def open_sink(conn_file=None):
    """Resolve config and connect.

    The tables are *not* created here. Schema creation is a one-time
    administrative step (`python3 lib/sql_upload.py --create-tables`), not
    something an eval run should attempt on every task.
    """
    config = load_config(conn_file)
    return connect(config), config


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--create-tables", action="store_true",
                   help="Create any missing perf-eval tables, then exit")
    p.add_argument("--check", action="store_true",
                   help="Verify credentials and connectivity without writing")
    p.add_argument("--conn-file", default=None,
                   help="Path to a .sqlconn file (env: SQLCONN_FILE)")
    p.add_argument("--print-schema", action="store_true",
                   help="Print the DDL to stdout without connecting")
    args = p.parse_args()

    if args.print_schema:
        for name in SCHEMA:
            print(SCHEMA[name].strip() + ";\n")
        return 0
    if not (args.create_tables or args.check):
        p.error("nothing to do: pass --create-tables, --check, or --print-schema")

    try:
        config = load_config(args.conn_file)
    except SqlSinkError as e:
        print(f"sql: {e}", file=sys.stderr)
        return EXIT_CONFIG
    print(f"sql: connecting to {describe(config)}", flush=True)
    if config["conn_file"]:
        print(f"sql: some settings came from {config['conn_file']}", flush=True)

    try:
        conn = connect(config)
    except SqlSinkError as e:
        print(f"sql: {e}", file=sys.stderr)
        return EXIT_CONFIG
    except Exception as e:  # driver-specific connection errors
        print(f"sql: connection failed: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_CONNECT

    try:
        if args.create_tables:
            before = missing_tables(conn)
            ensure_tables(conn)
            still = missing_tables(conn)
            if still:
                print(f"sql: tables still missing after CREATE: {', '.join(still)}",
                      file=sys.stderr)
                return EXIT_STATEMENT
            created = ", ".join(before) if before else "none"
            print(f"sql: created: {created}")
            print("sql: tables present: " + ", ".join(SCHEMA))
        else:
            absent = missing_tables(conn)
            if absent:
                print(f"sql: connection ok, but missing table(s): {', '.join(absent)}",
                      file=sys.stderr)
                print("sql: create them once with:"
                      " python3 lib/sql_upload.py --create-tables", file=sys.stderr)
                return EXIT_STATEMENT
            print("sql: connection ok, all tables present")
    except Exception as e:
        print(f"sql: failed: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_STATEMENT
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
