#!/usr/bin/env python3
"""Scrape finished Buildkite builds and push their results into SQL.

`lib/run.sh` uploads as it goes. When that upload could not happen — the
database was unreachable, credentials were missing, the build predates the SQL
sink — the results are still attached to the build as Buildkite artifacts. This
pulls them back down and ingests them after the fact.

    python3 lib/scrapper.py --build 46
    python3 lib/scrapper.py --range 30-46 --reconstruct-commands
    python3 lib/scrapper.py --build 46 --no-samples --dry-run

For each passed job in each build it: resolves the workload's settings with the
*build's own* `parse_workload.py` (so per-commit workload edits are honoured),
downloads that job's artifacts, and hands them to `ingest_perf.py` / `ingest.py`
with the build's provenance in the environment. Artifacts are deleted after each
job, so disk never holds more than one job's worth.

Everything is idempotent: writes are `dedupe_hash` upserts, so re-running a
build updates its rows instead of duplicating them. A per-run state file records
which jobs succeeded, so an interrupted scrape resumes where it stopped.

Requires:
  * the `bk` CLI on PATH (or --bk), authenticated via BUILDKITE_API_TOKEN
  * SQL settings resolvable the usual way (see lib/sql_upload.py)
  * git able to resolve each build's commit, to read the workloads as they were

Never uses raw `curl` against the Buildkite API — see CLAUDE.md.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import sql_upload  # noqa: E402  (same-dir helper; path set above)

DEFAULT_ORG = "amd-rocm"
DEFAULT_PIPELINE = "perf-eval"
# `vllm bench serve` writes this timestamp into every result JSON.
BENCH_DATE_RE = re.compile(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})")
# nightly-<full sha> is pinned to a commit and will not move; a bare tag like
# `:nightly` will, so its digest today may not be the one the build ran.
PINNED_TAG_RE = re.compile(r":[a-z]+-[0-9a-f]{12,40}$")


class ScrapeError(RuntimeError):
    pass


def run(cmd, **kw):
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    return subprocess.run(cmd, **kw)


# --- Buildkite ------------------------------------------------------------

def bk_json(args, bk, timeout=180):
    """Run a `bk` subcommand that prints JSON and parse it."""
    r = run([bk] + args, timeout=timeout)
    if r.returncode != 0:
        raise ScrapeError(f"bk {' '.join(args)}: {r.stderr.strip()[:300]}")
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise ScrapeError(f"bk {' '.join(args)}: unparseable output ({e})") from None


def passed_jobs(build, bk, pipeline):
    """The workload jobs of a build that finished successfully."""
    data = bk_json(["build", "view", str(build), "--pipeline", pipeline], bk)
    env = data.get("env") or {}
    jobs, seen = [], set()
    for j in data.get("jobs") or []:
        name = (j.get("name") or "").strip()
        if j.get("type") != "script" or j.get("state") != "passed":
            continue
        if "generate" in name or j["id"] in seen:
            continue
        seen.add(j["id"])
        jobs.append({
            "build": data["number"],
            "build_id": data["id"],
            "build_url": data.get("web_url", ""),
            "commit": data.get("commit", ""),
            "branch": data.get("branch", ""),
            "job": j["id"],
            # The label is "<emoji> <workload name>"; the workload name is what
            # names the results/ directory.
            "workload": name.split(" ", 1)[-1].strip() if " " in name else name,
            "vllm_image": env.get("VLLM_IMAGE", ""),
            "vllm_commit": env.get("VLLM_COMMIT", ""),
            "nightly": env.get("NIGHTLY", ""),
        })
    return jobs


def download_artifacts(job, dest, bk, pipeline):
    dest.mkdir(parents=True, exist_ok=True)
    r = run([bk, "artifacts", "download", "--build", str(job["build"]),
             "--pipeline", pipeline, "--job-uuid", job["job"]],
            cwd=str(dest), timeout=1800)
    if r.returncode != 0:
        raise ScrapeError(f"artifact download: {r.stderr.strip()[:300]}")
    roots = sorted(dest.glob("artifacts-build-*"))
    return roots[0] if roots else dest


# --- the build's own view of the workload ---------------------------------

def checkout(commit, cache):
    """Extract workloads/ and lib/ as of `commit` into a cache dir."""
    if not commit:
        return None
    out = cache / commit
    if (out / "lib" / "parse_workload.py").is_file():
        return out
    if run(["git", "-C", str(REPO), "cat-file", "-e", commit]).returncode != 0:
        # Not local yet; the build ran it, so the remote should still have it.
        run(["git", "-C", str(REPO), "fetch", "--quiet", "origin", commit], timeout=300)
    if run(["git", "-C", str(REPO), "cat-file", "-e", commit]).returncode != 0:
        return None
    out.mkdir(parents=True, exist_ok=True)
    tar = subprocess.Popen(["git", "-C", str(REPO), "archive", commit],
                           stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(out)], stdin=tar.stdout, check=True)
    tar.wait()
    return out


def workload_files(tree):
    """Map each workload's `name:` to its yaml filename."""
    import yaml
    out = {}
    for p in sorted((tree / "workloads").glob("*.yaml")):
        try:
            data = yaml.safe_load(p.read_text()) or {}
        except Exception:
            continue
        out[data.get("name") or p.stem] = p.name
    return out


def parse_workload(tree, yaml_file, job, tmp):
    """Run the build's parse_workload.py and return its WORKLOAD_* exports.

    lm_eval is not needed here and is usually absent, so BENCH_ONLY skips the
    task-name registry check. It changes nothing else that is emitted — every
    field, including LM_EVAL_TASKS_TSV, comes out the same.

    bash does the eval and hands back NUL-delimited pairs: the TSV values are
    multi-line, so parsing the `KEY=value` text directly would truncate them.
    """
    stub = tmp / "parse_stub.py"
    stub.write_text(
        "import sys, types\n"
        "m = types.ModuleType('lm_eval'); t = types.ModuleType('lm_eval.tasks')\n"
        "class TM: all_tasks = []\n"
        "t.TaskManager = TM\n"
        "sys.modules['lm_eval'] = m; sys.modules['lm_eval.tasks'] = t\n"
        f"sys.argv = ['parse_workload.py', {str(tree / 'workloads' / yaml_file)!r}]\n"
        f"exec(open({str(tree / 'lib' / 'parse_workload.py')!r}).read())\n"
    )
    env = dict(os.environ, BENCH_ONLY="1")
    for key, value in (("VLLM_IMAGE", job["vllm_image"]),
                       ("VLLM_COMMIT", job["vllm_commit"])):
        if value:
            env[key] = value
        else:
            env.pop(key, None)
    script = (
        "set -euo pipefail\n"
        f'exports="$({sys.executable} {stub})"\n'
        'eval "$exports"\n'
        "for v in ${!WORKLOAD_@}; do printf '%s\\0%s\\0' \"$v\" \"${!v}\"; done\n"
    )
    r = subprocess.run(["bash", "-c", script], cwd=str(tree), env=env,
                       capture_output=True, timeout=300)
    if r.returncode != 0:
        raise ScrapeError(f"parse_workload: {r.stderr.decode()[:300]}")
    parts = r.stdout.decode().split("\0")
    return {parts[i]: parts[i + 1] for i in range(0, len(parts) - 1, 2)}


def reconstruct_commands(root, wl, results_dir, tmp):
    """Re-derive the .cmd files for builds that predate the command capture.

    Drives the *current* run_* helpers with stubbed binaries, so the recorded
    line is whatever those helpers build today. That equals what the build ran
    only while the command construction is unchanged — verify with `git diff`
    over lib/run_vllm_bench.sh and lib/run_lm_eval.sh before trusting it, which
    is why this is opt-in.
    """
    stub = tmp / "bin"
    stub.mkdir(exist_ok=True)
    for name in ("vllm", "lm_eval"):
        p = stub / name
        p.write_text("#!/usr/bin/env bash\nexit 0\n")
        p.chmod(0o755)
    # The TSVs are tab- and newline-separated, so they travel through the
    # environment rather than being interpolated into the script: Python's repr
    # would escape the tabs, and bash does not unescape inside single quotes,
    # collapsing every row into a single field.
    env = dict(os.environ)
    env["PATH"] = f"{stub}:{env.get('PATH', '')}"
    env["RESULTS_DIR"] = str(results_dir)
    for key in ("WORKLOAD_MODEL", "WORKLOAD_SERVER_RUNTIME", "WORKLOAD_SERVE_ARGS",
                "WORKLOAD_VLLM_BENCH_TSV", "WORKLOAD_LM_EVAL_TASKS_TSV"):
        env[key] = wl.get(key, "")
    script = f"""
set -uo pipefail
source {str(REPO / 'lib' / 'run_vllm_bench.sh')!r}
source {str(REPO / 'lib' / 'run_lm_eval.sh')!r}
mkdir -p "$RESULTS_DIR"
# Mirrors run.sh: the bench inherits --trust-remote-code from the serve args.
TRC=false
if [[ "$WORKLOAD_SERVE_ARGS" =~ (^|[[:space:]])--trust-remote-code([[:space:]]|$) ]] ||
   [[ "$WORKLOAD_SERVE_ARGS" =~ (^|[[:space:]])--trust-remote-code=(true|True|1|yes|Yes)([[:space:]]|$) ]]; then
  TRC=true
fi
while IFS=$'\\t' read -r bname backend dataset isl osl nprompts conc ss sc extra; do
  [[ -z "$bname" ]] && continue
  run_vllm_bench c 8000 "$WORKLOAD_MODEL" "$bname" "$backend" "$dataset" \
    "$isl" "$osl" "$nprompts" "$conc" "$ss" "$sc" "$extra" "$TRC" "$RESULTS_DIR" \
    >/dev/null 2>&1 || true
done <<< "$WORKLOAD_VLLM_BENCH_TSV"
while IFS=$'\\t' read -r task fewshot margs; do
  [[ -z "$task" ]] && continue
  run_lm_eval "$WORKLOAD_MODEL" "http://localhost:8000" "$task" \
    "$fewshot" "$margs" "$RESULTS_DIR" >/dev/null 2>&1 || true
done <<< "$WORKLOAD_LM_EVAL_TASKS_TSV"
"""
    run(["bash", "-c", script], cwd=str(root), env=env, timeout=300)


def image_digest(image, cache):
    """Digest for a commit-pinned image tag, or "" when it would be a guess."""
    if not image or image in cache:
        return cache.get(image, "")
    digest = ""
    if PINNED_TAG_RE.search(image):
        r = run([sys.executable, str(HERE / "image_digest.py"), image], timeout=90)
        if r.returncode == 0:
            digest = r.stdout.strip()
    cache[image] = digest
    return digest


# --- ingest ---------------------------------------------------------------

def bench_run_date(path):
    try:
        m = BENCH_DATE_RE.match(str(json.loads(path.read_text()).get("date", "")))
    except Exception:
        return None
    return f"{m[1]}-{m[2]}-{m[3]} {m[4]}:{m[5]}:{m[6]}" if m else None


def ingest_env(job, wl, digest, sink):
    env = dict(os.environ)
    env.update({
        "INGEST_SINK": sink,
        "BUILDKITE_BUILD_ID": job["build_id"],
        "BUILDKITE_BUILD_NUMBER": str(job["build"]),
        "BUILDKITE_BUILD_URL": job["build_url"],
        "BUILDKITE_BRANCH": job["branch"],
        "BUILDKITE_COMMIT": job["commit"],
        "BUILDKITE_PIPELINE_SLUG": DEFAULT_PIPELINE,
        "WORKLOAD_IMAGE": wl.get("WORKLOAD_IMAGE", ""),
        "WORKLOAD_VLLM_COMMIT": wl.get("WORKLOAD_VLLM_COMMIT") or job["vllm_commit"],
        "WORKLOAD_ENV": wl.get("WORKLOAD_ENV", ""),
        "WORKLOAD_SERVE_COMMAND": (
            f"vllm serve {wl.get('WORKLOAD_MODEL', '')} --port 8000 "
            f"{wl.get('WORKLOAD_SERVE_ARGS', '')}".strip()
        ),
    })
    # Only tag rows nightly when the build itself did.
    if job["nightly"] == "1":
        env["NIGHTLY"] = "1"
    else:
        env.pop("NIGHTLY", None)
    if digest:
        env["WORKLOAD_IMAGE_DIGEST"] = digest
    else:
        env.pop("WORKLOAD_IMAGE_DIGEST", None)
    # Not captured retroactively; vllm_commit carries the same information.
    env.pop("WORKLOAD_VLLM_VERSION", None)
    return env


def scrape_job(job, args, caches):
    tree = checkout(job["commit"], caches["checkouts"])
    if tree is None:
        return {"skip": f"commit {job['commit'][:8]} unavailable"}
    names = caches["names"].setdefault(job["commit"], workload_files(tree))
    yaml_file = names.get(job["workload"])
    if not yaml_file:
        return {"skip": f"no workload yaml named {job['workload']!r}"}

    with tempfile.TemporaryDirectory(prefix="scrapper-") as td:
        tmp = Path(td)
        wl = parse_workload(tree, yaml_file, job, tmp)
        root = download_artifacts(job, tmp / "dl", args.bk, args.pipeline)
        results_dir = Path("results") / job["workload"]
        if args.reconstruct_commands and not list((root / results_dir).glob("*.cmd")):
            reconstruct_commands(root, wl, results_dir, tmp)

        digest = image_digest(job["vllm_image"], caches["digests"])
        env = ingest_env(job, wl, digest, args.sink)
        stats = {"perf": 0, "eval": 0, "errors": []}

        for line in (wl.get("WORKLOAD_VLLM_BENCH_TSV") or "").splitlines():
            if not line.strip():
                continue
            f = line.split("\t")
            bname, isl, osl, conc = f[0], f[3], f[4], f[6]
            raw = root / results_dir / f"bench-{bname}.json"
            if not raw.is_file():
                continue
            cmd = root / results_dir / f"bench-{bname}.cmd"
            call = [sys.executable, str(HERE / "ingest_perf.py"),
                    "--raw-result", str(raw.relative_to(root)),
                    "--device", wl["WORKLOAD_BENCH_DEVICE"],
                    "--tp", wl["WORKLOAD_BENCH_TP"],
                    "--precision", wl["WORKLOAD_BENCH_PRECISION"],
                    "--model", wl["WORKLOAD_MODEL"],
                    "--image", wl.get("WORKLOAD_IMAGE", ""),
                    "--workload", job["workload"], "--bench-name", bname,
                    "--isl", isl, "--osl", osl, "--conc", conc,
                    "--sink", args.sink]
            when = bench_run_date(raw)
            if when:
                call += ["--date", when]
            if cmd.is_file():
                call += ["--command-file", str(cmd.relative_to(root))]
            if args.dry_run:
                stats["perf"] += 1
                continue
            r = run(call, cwd=str(root), env=env, timeout=600)
            if "inserted into" in r.stdout:
                stats["perf"] += 1
            else:
                stats["errors"].append(f"perf {bname}: {(r.stdout + r.stderr)[-200:]}")

        for line in (wl.get("WORKLOAD_LM_EVAL_TASKS_TSV") or "").splitlines():
            if not line.strip():
                continue
            task = line.split("\t")[0]
            tdir = root / results_dir / task
            if not tdir.is_dir() or not list(tdir.glob("**/results_*.json")):
                continue
            cmd = root / results_dir / f"{task}.cmd"
            call = [sys.executable, str(HERE / "ingest.py"),
                    "--results-dir", str(tdir.relative_to(root)),
                    "--workload", job["workload"], "--task", task,
                    "--sink", args.sink]
            if cmd.is_file():
                call += ["--command-file", str(cmd.relative_to(root))]
            if args.no_samples:
                call.append("--no-samples")
            if args.dry_run:
                stats["eval"] += 1
                continue
            r = run(call, cwd=str(root), env=env, timeout=3600)
            # ingest.py uploads results and samples separately and stays exit-0
            # either way, so "uploaded" alone does not mean the whole task
            # landed — a failed samples file reports on stderr.
            failed = [ln for ln in r.stderr.splitlines() if ln.strip().startswith("failed ")]
            if "uploaded" in r.stdout and not failed:
                stats["eval"] += 1
            else:
                detail = failed[0] if failed else (r.stdout + r.stderr)[-200:]
                stats["errors"].append(f"eval {task}: {detail}")

    return stats


# --- driver ---------------------------------------------------------------

def build_numbers(args):
    if args.range:
        lo, _, hi = args.range.partition("-")
        return list(range(int(lo), int(hi or lo) + 1))
    return args.build


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--build", type=int, nargs="+", help="Build number(s) to scrape")
    sel.add_argument("--range", help="Inclusive build range, e.g. 30-46")
    p.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    p.add_argument("--org", default=os.environ.get("BUILDKITE_ORGANIZATION_SLUG", DEFAULT_ORG))
    p.add_argument("--bk", default=shutil.which("bk") or "bk", help="Path to the bk CLI")
    p.add_argument("--sink", default="sql", choices=("sql", "endpoint", "both"))
    p.add_argument("--no-samples", action="store_true",
                   help="Skip samples_*.jsonl (they are ~99%% of the bytes)")
    p.add_argument("--only-workload", action="append", default=[],
                   help="Restrict to these workload names; repeatable")
    p.add_argument("--reconstruct-commands", action="store_true",
                   help="Re-derive .cmd files for builds that predate the command "
                        "capture. Only correct while the run_* helpers build the "
                        "same command line; check `git diff` first.")
    p.add_argument("--state", default=None,
                   help="Resume file recording finished jobs (default: none)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be uploaded without writing")
    args = p.parse_args()

    os.environ.setdefault("BUILDKITE_ORGANIZATION_SLUG", args.org)
    if shutil.which(args.bk) is None and not Path(args.bk).is_file():
        sys.exit(f"scrapper: bk CLI not found at {args.bk!r}; install it or pass --bk")
    if args.sink in ("sql", "both") and not args.dry_run:
        try:
            cfg = sql_upload.load_config()
            print(f"scrapper: sql -> {sql_upload.describe(cfg)}")
        except sql_upload.SqlSinkError as e:
            sys.exit(f"scrapper: {e}")

    state_path = Path(args.state) if args.state else None
    done = json.loads(state_path.read_text()) if state_path and state_path.is_file() else {}
    caches = {"checkouts": Path(tempfile.gettempdir()) / "scrapper-checkouts",
              "names": {}, "digests": {}}
    caches["checkouts"].mkdir(parents=True, exist_ok=True)

    totals = {"perf": 0, "eval": 0, "jobs": 0, "failed": 0}
    for number in build_numbers(args):
        try:
            jobs = passed_jobs(number, args.bk, args.pipeline)
        except ScrapeError as e:
            print(f"build #{number}: {e}", file=sys.stderr)
            totals["failed"] += 1
            continue
        if args.only_workload:
            jobs = [j for j in jobs if j["workload"] in args.only_workload]
        print(f"build #{number}: {len(jobs)} passed job(s)")
        for job in jobs:
            if done.get(job["job"], {}).get("ok"):
                print(f"  skip (done)  {job['workload']}")
                continue
            t0 = time.time()
            try:
                s = scrape_job(job, args, caches)
            except Exception as e:                     # keep going on the rest
                s = {"skip": f"{type(e).__name__}: {e}"}
            if "skip" in s:
                print(f"  SKIP  {job['workload']:<44} {s['skip'][:90]}", file=sys.stderr)
                totals["failed"] += 1
                continue
            ok = not s["errors"]
            totals["perf"] += s["perf"]
            totals["eval"] += s["eval"]
            totals["jobs"] += 1
            totals["failed"] += 0 if ok else 1
            print(f"  {'ok  ' if ok else 'WARN'}  {job['workload']:<44} "
                  f"perf={s['perf']} eval={s['eval']} {time.time() - t0:.0f}s")
            for e in s["errors"][:3]:
                print(f"        ! {e}", file=sys.stderr)
            if state_path:
                done[job["job"]] = {"ok": ok, "perf": s["perf"], "eval": s["eval"]}
                state_path.write_text(json.dumps(done, indent=0))

    verb = "would upload" if args.dry_run else "uploaded"
    print(f"\nscrapper: {verb} {totals['perf']} perf row(s) and {totals['eval']} "
          f"eval task(s) from {totals['jobs']} job(s); {totals['failed']} problem(s)")
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
