#!/usr/bin/env python3
"""End-to-end run benchmark: wrap a real finetrainers training launch, measure
throughput and (best-effort) resource use over a fixed, short step budget.

This is the coarse tier — it treats `train.py` as a black box and measures the
whole process. For per-phase timing (forward/backward/optimizer) or op-level
numbers, use the micro-bench tier (`finetrainers_bench.py`) instead.

It runs the launch command you pass, forces `FINETRAINERS_ENABLE_TIMING=1`,
tees output, then derives steps/sec from wall time over the measured window and
scrapes the tqdm postfix for `it/s` and the last logged loss.

Usage (quote the whole launch command):
    python run_e2e_benchmark.py \
        --steps 60 --warmup-steps 10 \
        --out baselines/ltx_lora.mps.e2e.json \
        --label "ltx lora, mps, bs1" \
        -- torchrun --nnodes=1 --nproc_per_node 1 train.py --parallel_backend ptd \
           --training_type lora --train_steps 60 ...

Everything after `--` is the launch command, run verbatim. Set --train_steps in
that command to match --steps so the run stops on its own.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional


ITPS_RE = re.compile(r"([0-9]+\.?[0-9]*)\s*(it/s|s/it)")
LOSS_RE = re.compile(r"global_avg_loss[=:]\s*([0-9]*\.?[0-9]+)")
STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")  # tqdm "  37/60"


def git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def scrape(lines: List[str]) -> Dict[str, Any]:
    last_itps = last_loss = None
    for ln in lines:
        m = ITPS_RE.search(ln)
        if m:
            val = float(m.group(1))
            last_itps = val if m.group(2) == "it/s" else (1.0 / val if val else None)
        m = LOSS_RE.search(ln)
        if m:
            last_loss = float(m.group(1))
    return {"tqdm_it_per_s": last_itps, "final_avg_loss": last_loss}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="finetrainers end-to-end run benchmark")
    p.add_argument("--steps", type=int, required=True, help="total train steps in the launch")
    p.add_argument("--warmup-steps", type=int, default=10, help="steps excluded from throughput (compile/alloc warmup)")
    p.add_argument("--out")
    p.add_argument("--label")
    p.add_argument("--baseline")
    p.add_argument("--threshold", type=float, default=0.10)
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="-- then the launch command")
    args = p.parse_args(argv)

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        p.error("provide the launch command after `--`")

    env = dict(os.environ, FINETRAINERS_ENABLE_TIMING="1")
    print(f"  launching: {' '.join(cmd)}\n")

    start = time.perf_counter()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    lines: List[str] = []
    for ln in proc.stdout:  # tee
        sys.stdout.write(ln)
        lines.append(ln)
    proc.wait()
    wall = time.perf_counter() - start

    if proc.returncode != 0:
        print(f"\n  ✗ launch exited {proc.returncode} — benchmark invalid")
        return proc.returncode

    scraped = scrape(lines)
    measured_steps = max(1, args.steps - args.warmup_steps)
    # Coarse: assumes uniform step cost; warmup share of wall time is small for short runs.
    steps_per_s = measured_steps / wall if wall else None

    result = {
        "label": args.label or "e2e",
        "tier": "end-to-end",
        "command": cmd,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "stats": {
            "wall_s": round(wall, 2),
            "steps_per_s_wall": round(steps_per_s, 4) if steps_per_s else None,
            "tqdm_it_per_s": scraped["tqdm_it_per_s"],
            "throughput_per_s": scraped["tqdm_it_per_s"],  # prefer tqdm's steady-state number
        },
        "final_avg_loss": scraped["final_avg_loss"],
        "env": {"git_sha": git_sha(), "timing_enabled": True},
    }

    s = result["stats"]
    print(f"\n  {result['label']}  [{git_sha()}]")
    print(f"    wall {s['wall_s']}s over {args.steps} steps")
    print(f"    steps/s (wall, warmup-excluded): {s['steps_per_s_wall']}")
    print(f"    tqdm steady-state it/s: {s['tqdm_it_per_s']}")
    print(f"    final avg loss: {result['final_avg_loss']}")
    print("    note: per-phase timing/* is captured by the run's tracker "
          "(enable a wandb/jsonl sink); process-level throughput is what this tier reports.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"    wrote {args.out}")

    if args.baseline:
        with open(args.baseline) as fh:
            base = json.load(fh)
        bt = base.get("stats", {}).get("throughput_per_s")
        ct = s.get("throughput_per_s")
        if bt and ct:
            pct = (ct - bt) / bt
            print(f"\n    throughput {ct:.3f} vs baseline {bt:.3f}  ({pct * 100:+.1f}%)")
            if pct < -args.threshold:
                print(f"    ✗ REGRESSION beyond {args.threshold:.0%}")
                return 1
            print("    ✓ no throughput regression")
    return 0


if __name__ == "__main__":
    sys.exit(main())
