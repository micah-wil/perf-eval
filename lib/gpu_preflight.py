#!/usr/bin/env python3
"""GPU state probe used by server.sh to guard vLLM startup.

vLLM aborts at init if a GPU has less free memory than
``--gpu-memory-utilization`` requires (e.g. "Free memory on device cuda:4
(262.47/287.98 GiB) on startup is less than desired GPU memory utilization
(0.92, 264.95 GiB)"). On the shared MI300X/MI355X CI cluster this happens when
a *previous* job leaked vLLM worker processes that are still holding VRAM.

This helper reports per-GPU memory so the orchestrator can:
  * wait a bounded time for transient allocations to drain (async free after a
    process exits can take ~10s), and
  * fail fast with an actionable message naming the busy GPUs instead of the
    cryptic vLLM crash, when a foreign process we cannot kill holds the memory.

It supports both AMD (``amd-smi``/``rocm-smi``) and NVIDIA (``nvidia-smi``).

Usage:
  python3 gpu_preflight.py check [--max-used-mib N] [--json]
    exit 0 if every visible GPU uses < N MiB, else exit 3 and print a report.

Only the standard library is used so it runs in the bare CI container before
any pip install.
"""

import argparse
import json
import shutil
import subprocess
import sys

# ~1 GiB: covers driver/context baseline (we measured ~0.3 GiB idle on MI355X)
# while still catching a leaked model shard (tens of GiB).
DEFAULT_MAX_USED_MIB = 1024


def _run(cmd):
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _from_nvidia_smi():
    if not shutil.which("nvidia-smi"):
        return None
    out = _run([
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return None
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        idx, used, total = parts[:3]
        gpus.append({"index": int(idx), "used_mib": int(float(used)),
                     "total_mib": int(float(total))})
    return gpus or None


def _from_amd_smi():
    if not shutil.which("amd-smi"):
        return None
    out = _run(["amd-smi", "metric", "--mem-usage", "--json"])
    if not out:
        return None
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None
    # amd-smi wraps the per-GPU list under "gpu_data"; older builds emit a
    # bare list. Values are objects like {"value": 283, "unit": "MB"} where
    # amd-smi's "MB" is really MiB (total 294896 == 309220868096 B / 1024^2).
    entries = data.get("gpu_data") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return None
    gpus = []
    for entry in entries:
        idx = entry.get("gpu")
        mem = entry.get("mem_usage", {})
        used = mem.get("used_vram", {})
        total = mem.get("total_vram", {})
        used_mib = used.get("value") if isinstance(used, dict) else used
        total_mib = total.get("value") if isinstance(total, dict) else total
        if used_mib is None or total_mib is None:
            return None
        gpus.append({"index": int(idx) if idx is not None else len(gpus),
                     "used_mib": int(used_mib), "total_mib": int(total_mib)})
    return gpus or None


def _from_rocm_smi():
    if not shutil.which("rocm-smi"):
        return None
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--json"])
    if not out:
        return None
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None
    gpus = []
    for key, val in sorted(data.items()):
        if not key.lower().startswith("card"):
            continue
        used = total = None
        for k, v in val.items():
            kl = k.lower()
            if "vram total used memory" in kl:
                used = int(v)
            elif "vram total memory" in kl:
                total = int(v)
        if used is None or total is None:
            continue
        # rocm-smi reports bytes here.
        digits = "".join(ch for ch in key if ch.isdigit())
        gpus.append({"index": int(digits) if digits else len(gpus),
                     "used_mib": used // (1024 * 1024),
                     "total_mib": total // (1024 * 1024)})
    return gpus or None


def read_gpus():
    """Return a list of {index, used_mib, total_mib} or None if no tool works."""
    for probe in (_from_nvidia_smi, _from_amd_smi, _from_rocm_smi):
        gpus = probe()
        if gpus:
            return gpus
    return None


def cmd_check(args):
    gpus = read_gpus()
    if gpus is None:
        # No probe available (or all failed). Don't block the run; vLLM will
        # still enforce its own memory check. Emit a note for the log.
        print("gpu_preflight: no GPU query tool available; skipping check",
              file=sys.stderr)
        return 0

    busy = [g for g in gpus if g["used_mib"] > args.max_used_mib]
    if args.json:
        print(json.dumps({"gpus": gpus, "busy": busy, "threshold_mib": args.max_used_mib}))

    if not busy:
        if not args.json:
            print(f"gpu_preflight: all {len(gpus)} GPU(s) below "
                  f"{args.max_used_mib} MiB used")
        return 0

    if not args.json:
        print("gpu_preflight: GPUs not clean before startup:", file=sys.stderr)
        for g in busy:
            print(f"  GPU {g['index']}: {g['used_mib']} MiB used "
                  f"/ {g['total_mib']} MiB total (threshold {args.max_used_mib})",
                  file=sys.stderr)
    return 3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="exit 3 if any GPU exceeds --max-used-mib")
    c.add_argument("--max-used-mib", type=int, default=DEFAULT_MAX_USED_MIB)
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_check)
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
