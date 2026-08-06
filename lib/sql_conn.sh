# Resolve SQL sink credentials into the environment. Source this from run.sh.
#
# Usage:
#   load_sql_conn            # exports the five TIGER_SQL_* settings
#   sql_sink_enabled         # true when INGEST_SINK selects the SQL destination
#   endpoint_sink_enabled    # true when INGEST_SINK selects the HTTP endpoint
#
# Sources are tried in priority order and the first hit for each key wins:
#   1. the environment (Buildkite pipeline env, k8s secretKeyRef, manual export)
#   2. `buildkite-agent secret get TIGER_SQL_<...>` (Buildkite Secrets)
#   3. a local `.sqlconn` file (development only; gitignored)
#
# Values are never echoed. `buildkite-agent secret get` output is redacted by
# the agent in build logs, but this helper keeps it out of the logs regardless:
# only key names and the source they came from are printed.

# Keep in sync with CONN_KEYS in sql_upload.py and SQL_ENV_VARS in
# .buildkite/generate_pipeline.py.
SQL_CONN_KEYS=(TIGER_SQL_HOST TIGER_SQL_PORT TIGER_SQL_USER TIGER_SQL_PASSWD TIGER_SQL_DB)

# True when TIGER_SQL_DB can be found in any source. Its presence is what
# switches the default destination to SQL, so this probes the same places
# load_sql_conn does. Never prints a value.
sql_db_detected() {
  [[ -n "${TIGER_SQL_DB:-}" ]] && return 0
  local value
  value="$(_sql_secret_get TIGER_SQL_DB)" && [[ -n "$value" ]] && return 0
  local dir conn_file
  dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  conn_file="${SQLCONN_FILE:-${dir}/../.sqlconn}"
  [[ -f "$conn_file" ]] &&
    grep -qE '^[[:space:]]*(export[[:space:]]+)?TIGER_SQL_DB[[:space:]]*=[[:space:]]*[^[:space:]]' \
      "$conn_file" && return 0
  return 1
}

# INGEST_SINK: endpoint | sql | both. When unset, a detected TIGER_SQL_DB means
# SQL is the configured destination, so results go there instead of the public
# endpoint; with no SQL settings at all the endpoint stays the default. Set
# INGEST_SINK explicitly to override either way.
ingest_sink() {
  local sink="${INGEST_SINK:-}"
  if [[ -z "$sink" ]]; then
    if sql_db_detected; then sink=sql; else sink=endpoint; fi
  fi
  printf '%s' "${sink,,}"
}

sql_sink_enabled() {
  local sink; sink="$(ingest_sink)"
  [[ "$sink" == "sql" || "$sink" == "both" ]]
}

endpoint_sink_enabled() {
  local sink; sink="$(ingest_sink)"
  [[ "$sink" == "endpoint" || "$sink" == "both" ]]
}

# Fetch one key from Buildkite Secrets. Returns non-zero when unavailable so
# the caller can fall through to the next source.
_sql_secret_get() {
  local key=$1
  command -v buildkite-agent >/dev/null 2>&1 || return 1
  buildkite-agent secret get "$key" 2>/dev/null
}

# Debug dump of what the destination decision saw. Prints key names and where
# they came from, never a value.
sql_debug_state() {
  local dir conn_file key err
  dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  conn_file="${SQLCONN_FILE:-${dir}/../.sqlconn}"
  echo "  sql-debug: INGEST_SINK=${INGEST_SINK:-<unset>}"
  for key in "${SQL_CONN_KEYS[@]}"; do
    if [[ "$key" == TIGER_SQL_PASSWD ]]; then
      echo "  sql-debug: ${key}=$([[ -n "${!key:-}" ]] && echo '<set>' || echo '<unset>')"
    else
      echo "  sql-debug: ${key}=${!key:-<unset>}"
    fi
  done
  echo "  sql-debug: conn file ${conn_file} $([[ -f "$conn_file" ]] && echo exists || echo missing)"
  if command -v buildkite-agent >/dev/null 2>&1; then
    # Capture stderr only; a successful fetch's value goes to /dev/null.
    err="$(buildkite-agent secret get TIGER_SQL_DB 2>&1 >/dev/null)"
    if [[ -n "$err" ]]; then
      echo "  sql-debug: buildkite-agent secret get TIGER_SQL_DB failed: ${err}"
    else
      echo "  sql-debug: buildkite-agent secret get TIGER_SQL_DB succeeded"
    fi
  else
    echo "  sql-debug: buildkite-agent not on PATH"
  fi
}

load_sql_conn() {
  local dir key value from_secret=() from_env=() missing=()
  dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  for key in "${SQL_CONN_KEYS[@]}"; do
    if [[ -n "${!key:-}" ]]; then
      from_env+=("$key")
      continue
    fi
    value="$(_sql_secret_get "$key")" || value=""
    if [[ -n "$value" ]]; then
      export "$key=$value"
      from_secret+=("$key")
    fi
  done
  unset value

  # TIGER_SQL_PORT and TIGER_SQL_PASSWD are optional (default port,
  # passwordless auth); the rest are required.
  for key in TIGER_SQL_HOST TIGER_SQL_USER TIGER_SQL_DB; do
    [[ -n "${!key:-}" ]] || missing+=("$key")
  done

  local conn_file="${SQLCONN_FILE:-${dir}/../.sqlconn}"
  if [[ ${#missing[@]} -gt 0 && -f "$conn_file" ]]; then
    # sql_upload.py reads the file itself for whatever is still unset; just
    # report that it will be used so a CI run never silently depends on it.
    echo "  sql: falling back to ${conn_file} for: ${missing[*]}"
    missing=()
  fi

  [[ ${#from_env[@]} -gt 0 ]] && echo "  sql: from environment: ${from_env[*]}"
  [[ ${#from_secret[@]} -gt 0 ]] && echo "  sql: from buildkite secrets: ${from_secret[*]}"

  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "  sql: missing credentials: ${missing[*]}" >&2
    echo "  sql: set them as Buildkite secrets or step env, or create .sqlconn locally" >&2
    return 1
  fi
  return 0
}
