#!/usr/bin/env python3
"""finetrainers micro-benchmark + parity harness.

Device-aware (cuda / mps / cpu) timing, memory, and CPU-vs-device numeric
parity for finetrainers. Rides the same discipline as the repo's own
`tracker.timed("timing/*")` instrumentation, but works standalone and, unlike
the repo's `Timer` (CUDA-only event timing), correctly synchronizes MPS.

A "bench spec" is a plain Python file exposing:

    def setup():                 # build model/inputs once; return any ctx object
        return ctx
    def run(ctx):                # the ONE thing to time; called warmup+iters times
        ...

Optional in the spec:
    ITEMS_PER_ITER = 4           # samples/frames/tokens per run() -> throughput
    def parity_output(ctx, device):   # return a torch.Tensor computed ON `device`
        ...                            # harness runs it on cpu + target, compares

Usage:
    python finetrainers_bench.py run  specs/ltx_forward.py --device auto --iters 50 \
        --out baselines/ltx_forward.mps.json --label "ltx forward, mps, bf16"
    python finetrainers_bench.py run  specs/ltx_forward.py --baseline baselines/ltx_forward.cuda.json
    python finetrainers_bench.py compare a.json b.json
    python finetrainers_bench.py show  results.json

Exit code is non-zero when a regression threshold is exceeded or a parity
check fails, so it slots straight into CI / an agent's pass-fail logic.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional


# --------------------------------------------------------------------------- #
# device plumbing
# --------------------------------------------------------------------------- #
def _torch():
    import torch  # imported lazily so `compare`/`show` work without a GPU stack

    return torch


def resolve_device(name: str) -> str:
    torch = _torch()
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    """Block until queued device work is done. This is the part the repo's
    Timer gets wrong for MPS — without it, wall-clock timing on mps/cuda just
    measures kernel *launch*, not execution."""
    torch = _torch()
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    # cpu: nothing to sync


def reset_peak_memory(device: str) -> None:
    torch = _torch()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    elif device == "mps":
        # MPS has no peak-reset; we snapshot driver allocation as the baseline.
        torch.mps.empty_cache()


def memory_report(device: str, baseline_bytes: int) -> Dict[str, Any]:
    torch = _torch()
    mb = 1024 * 1024
    if device == "cuda":
        return {
            "device": "cuda",
            "peak_mb": round(torch.cuda.max_memory_allocated() / mb, 2),
            "current_mb": round(torch.cuda.memory_allocated() / mb, 2),
        }
    if device == "mps":
        driver = torch.mps.driver_allocated_memory()
        return {
            "device": "mps",
            # No true peak counter on MPS; report allocation growth over the run.
            "driver_allocated_mb": round(driver / mb, 2),
            "delta_mb": round((driver - baseline_bytes) / mb, 2),
            "note": "MPS has no peak-memory counter; delta_mb is driver-allocation growth.",
        }
    return {"device": "cpu", "note": "no device-memory tracking on cpu"}


def memory_baseline(device: str) -> int:
    torch = _torch()
    if device == "mps":
        return torch.mps.driver_allocated_memory()
    return 0


# --------------------------------------------------------------------------- #
# spec loading
# --------------------------------------------------------------------------- #
def load_spec(path: str):
    # Make the spec's own dir and this harness dir importable from the spec.
    for d in (os.path.dirname(os.path.abspath(path)), os.path.dirname(os.path.abspath(__file__))):
        if d not in sys.path:
            sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("bench_spec", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load bench spec: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run"):
        raise SystemExit(f"bench spec {path} must define run(ctx)")
    return mod


def _call_setup(mod, device: str):
    """setup() may take the resolved device or nothing."""
    if not hasattr(mod, "setup"):
        return None
    try:
        return mod.setup(device)
    except TypeError:
        return mod.setup()


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def time_iterations(fn: Callable[[], Any], device: str, warmup: int, iters: int) -> List[float]:
    # Warmup: kernel autotune, allocator warm, lazy compiles. Never measured.
    for _ in range(warmup):
        fn()
    sync(device)

    samples_ms: List[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        sync(device)  # measure execution, not just launch
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return samples_ms


def summarize(samples_ms: List[float], items_per_iter: float) -> Dict[str, float]:
    samples_ms = sorted(samples_ms)
    n = len(samples_ms)
    median = statistics.median(samples_ms)
    p90 = samples_ms[min(n - 1, int(0.9 * n))]
    mean = statistics.fmean(samples_ms)
    std = statistics.pstdev(samples_ms) if n > 1 else 0.0
    out = {
        "mean_ms": round(mean, 4),
        "median_ms": round(median, 4),
        "p90_ms": round(p90, 4),
        "std_ms": round(std, 4),
        "min_ms": round(samples_ms[0], 4),
        "max_ms": round(samples_ms[-1], 4),
        "cv": round(std / mean, 4) if mean else 0.0,  # coefficient of variation
    }
    if items_per_iter:
        out["throughput_per_s"] = round(items_per_iter / (median / 1000.0), 3)
    return out


# --------------------------------------------------------------------------- #
# parity: CPU as oracle (CT2 ops-parity discipline, deep-dive problem #6)
# --------------------------------------------------------------------------- #
def check_parity(mod, device: str, rtol: float, atol: float) -> Dict[str, Any]:
    torch = _torch()
    cpu_out = mod.parity_output("cpu").detach().to("cpu", torch.float32)
    dev_out = mod.parity_output(device).detach().to("cpu", torch.float32)
    if cpu_out.shape != dev_out.shape:
        return {"checked": True, "passed": False, "reason": f"shape {tuple(dev_out.shape)} != cpu {tuple(cpu_out.shape)}"}
    diff = (cpu_out - dev_out).abs()
    denom = cpu_out.abs().clamp_min(1e-8)
    max_abs = diff.max().item()
    max_rel = (diff / denom).max().item()
    passed = bool(torch.allclose(cpu_out, dev_out, rtol=rtol, atol=atol))
    return {
        "checked": True,
        "passed": passed,
        "rtol": rtol,
        "atol": atol,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "nan_on_device": bool(torch.isnan(dev_out).any().item()),
        "inf_on_device": bool(torch.isinf(dev_out).any().item()),
    }


# --------------------------------------------------------------------------- #
# environment metadata
# --------------------------------------------------------------------------- #
def git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def env_meta(device: str) -> Dict[str, Any]:
    torch = _torch()
    meta = {
        "device": device,
        "torch": torch.__version__,
        "git_sha": git_sha(),
        "dirty": _git_dirty(),
        "python": sys.version.split()[0],
        "timing_enabled_env": os.environ.get("FINETRAINERS_ENABLE_TIMING"),
    }
    if device == "cuda":
        meta["gpu"] = torch.cuda.get_device_name(0)
    elif device == "mps":
        meta["platform"] = "apple-silicon-mps"
    return meta


def _git_dirty() -> Optional[bool]:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        return bool(out.strip())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# regression comparison
# --------------------------------------------------------------------------- #
def compare_results(baseline: Dict[str, Any], current: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """Higher latency or lower throughput than baseline by >threshold is a regression."""
    b, c = baseline.get("stats", {}), current.get("stats", {})
    rows = []
    regressed = False
    # median latency: lower is better
    if "median_ms" in b and "median_ms" in c and b["median_ms"]:
        pct = (c["median_ms"] - b["median_ms"]) / b["median_ms"]
        hit = pct > threshold
        regressed |= hit
        rows.append(("median_ms", b["median_ms"], c["median_ms"], pct, hit, "lower-better"))
    # throughput: higher is better
    if "throughput_per_s" in b and "throughput_per_s" in c and b["throughput_per_s"]:
        pct = (c["throughput_per_s"] - b["throughput_per_s"]) / b["throughput_per_s"]
        hit = pct < -threshold
        regressed |= hit
        rows.append(("throughput_per_s", b["throughput_per_s"], c["throughput_per_s"], pct, hit, "higher-better"))
    return {"threshold": threshold, "regressed": regressed, "rows": rows}


def print_comparison(cmp: Dict[str, Any], b_label: str, c_label: str) -> None:
    print(f"\n  baseline: {b_label}")
    print(f"  current:  {c_label}")
    print(f"  {'metric':<20}{'baseline':>14}{'current':>14}{'change':>12}   verdict")
    print("  " + "-" * 74)
    for name, bv, cv, pct, hit, sense in cmp["rows"]:
        verdict = "REGRESSION" if hit else "ok"
        print(f"  {name:<20}{bv:>14.3f}{cv:>14.3f}{pct * 100:>11.1f}%   {verdict}  ({sense})")
    print()


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_run(args) -> int:
    torch = _torch()
    device = resolve_device(args.device)
    mod = load_spec(args.spec)
    items = float(getattr(mod, "ITEMS_PER_ITER", 0) or 0)

    torch.manual_seed(args.seed)
    ctx = _call_setup(mod, device)

    reset_peak_memory(device)
    mem_base = memory_baseline(device)

    samples = time_iterations(lambda: mod.run(ctx), device, args.warmup, args.iters)
    stats = summarize(samples, items)
    result: Dict[str, Any] = {
        "label": args.label or os.path.basename(args.spec),
        "spec": os.path.abspath(args.spec),
        "warmup": args.warmup,
        "iters": args.iters,
        "items_per_iter": items,
        "seed": args.seed,
        "stats": stats,
        "memory": memory_report(device, mem_base),
        "env": env_meta(device),
    }

    if hasattr(mod, "parity_output") and not args.no_parity:
        result["parity"] = check_parity(mod, device, args.rtol, args.atol)

    _print_result(result)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"  wrote {args.out}")

    exit_code = 0
    if "parity" in result and not result["parity"]["passed"]:
        print("  ✗ PARITY FAILED — device output diverges from CPU oracle")
        exit_code = 2

    if args.baseline:
        with open(args.baseline) as fh:
            baseline = json.load(fh)
        cmp = compare_results(baseline, result, args.threshold)
        print_comparison(cmp, baseline.get("label", args.baseline), result["label"])
        if cmp["regressed"]:
            print(f"  ✗ REGRESSION beyond {args.threshold:.0%} vs baseline")
            exit_code = max(exit_code, 1)
        else:
            print("  ✓ no regression vs baseline")
    return exit_code


def _print_result(r: Dict[str, Any]) -> None:
    s = r["stats"]
    print(f"\n  {r['label']}  [{r['env']['device']} · torch {r['env']['torch']} · {r['env'].get('git_sha')}]")
    print(f"  {r['iters']} iters (+{r['warmup']} warmup)")
    print(f"    median {s['median_ms']:.3f} ms   p90 {s['p90_ms']:.3f} ms   "
          f"mean {s['mean_ms']:.3f} ± {s['std_ms']:.3f} ms   cv {s['cv']:.2%}")
    if "throughput_per_s" in s:
        print(f"    throughput {s['throughput_per_s']:.2f} items/s")
    mem = r["memory"]
    memline = "    ".join(f"{k} {v}" for k, v in mem.items() if k not in ("device", "note"))
    if memline:
        print(f"    mem[{mem['device']}]: {memline}")
    if r.get("cv_warn"):
        print(f"    ⚠ noisy: {r['cv_warn']}")
    if "parity" in r:
        p = r["parity"]
        flag = "✓" if p["passed"] else "✗"
        print(f"    parity {flag}  max_abs {p.get('max_abs', '-'):.3e}  max_rel {p.get('max_rel', '-'):.3e}  "
              f"(rtol {p.get('rtol')}, atol {p.get('atol')})")


def cmd_compare(args) -> int:
    with open(args.baseline) as fh:
        b = json.load(fh)
    with open(args.current) as fh:
        c = json.load(fh)
    cmp = compare_results(b, c, args.threshold)
    print_comparison(cmp, b.get("label", args.baseline), c.get("label", args.current))
    if cmp["regressed"]:
        print(f"  ✗ REGRESSION beyond {args.threshold:.0%}")
        return 1
    print("  ✓ within threshold")
    return 0


def cmd_show(args) -> int:
    with open(args.result) as fh:
        _print_result(json.load(fh))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="finetrainers micro-benchmark + parity harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a bench spec")
    r.add_argument("spec")
    r.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    r.add_argument("--warmup", type=int, default=5)
    r.add_argument("--iters", type=int, default=50)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--out")
    r.add_argument("--label")
    r.add_argument("--baseline", help="compare against this baseline json")
    r.add_argument("--threshold", type=float, default=0.05, help="regression threshold fraction (default 5%%)")
    r.add_argument("--rtol", type=float, default=2e-2, help="parity rtol (bf16/fp16-friendly default)")
    r.add_argument("--atol", type=float, default=2e-2, help="parity atol")
    r.add_argument("--no-parity", action="store_true")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="compare two result jsons")
    c.add_argument("baseline")
    c.add_argument("current")
    c.add_argument("--threshold", type=float, default=0.05)
    c.set_defaults(func=cmd_compare)

    sh = sub.add_parser("show", help="pretty-print a result json")
    sh.add_argument("result")
    sh.set_defaults(func=cmd_show)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
