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


def steady_state(step_times: Dict[int, float], warmup_steps: int) -> Optional[float]:
    """Median wall-clock seconds per step, from client-side timestamps of each step's
    first appearance, excluding warmup. Immune to tqdm's cumulative average (which the
    precompute-heavy first step dominates)."""
    steps = sorted(step_times)
    deltas = [
        step_times[s] - step_times[s - 1] for s in steps if s - 1 in step_times and s > warmup_steps
    ]
    # sub-50ms deltas are pipe-flush artifacts (several steps surfacing in one read), not steps
    deltas = [d for d in deltas if d > 0.05]
    if not deltas:
        return None
    deltas.sort()
    mid = len(deltas) // 2
    return deltas[mid] if len(deltas) % 2 else (deltas[mid - 1] + deltas[mid]) / 2


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

    # PYTHONUNBUFFERED: tqdm redraws with \r and no newline; a line-buffered child pipe
    # flushes them in bursts, giving identical client-side step timestamps (0s deltas)
    env = dict(os.environ, FINETRAINERS_ENABLE_TIMING="1", PYTHONUNBUFFERED="1")
    print(f"  launching: {' '.join(cmd)}\n")

    start = time.perf_counter()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines: List[str] = []
    step_times: Dict[int, float] = {}
    total_steps_seen = 0
    buf = ""
    fd = proc.stdout.fileno()
    # os.read returns whatever is available NOW; a text-mode read(N) would block until
    # N chars accumulate, batching minutes of tqdm redraws under one timestamp.
    # Split on \r as well as \n — tqdm redraws its bar with \r, so a line iterator
    # would only see it at the end. Timestamp each step's FIRST appearance client-side;
    # tqdm's own it/s is a cumulative average.
    while True:
        chunk_b = os.read(fd, 65536)
        if not chunk_b:
            break
        chunk = chunk_b.decode("utf-8", errors="replace")
        sys.stdout.write(chunk)
        buf += chunk
        parts = re.split(r"[\r\n]", buf)
        buf = parts.pop()
        now = time.perf_counter()
        for ln in parts:
            if not ln:
                continue
            lines.append(ln)
            m = STEP_RE.search(ln)
            if m and int(m.group(2)) == args.steps:
                step = int(m.group(1))
                total_steps_seen = max(total_steps_seen, step)
                step_times.setdefault(step, now)
    if buf:
        lines.append(buf)
    proc.wait()
    wall = time.perf_counter() - start

    if proc.returncode != 0:
        print(f"\n  ✗ launch exited {proc.returncode} — benchmark invalid")
        return proc.returncode

    scraped = scrape(lines)
    measured_steps = max(1, args.steps - args.warmup_steps)
    # Coarse: assumes uniform step cost; warmup share of wall time is small for short runs.
    steps_per_s = measured_steps / wall if wall else None

    ss_s_per_step = steady_state(step_times, args.warmup_steps)
    ss_steps_per_s = round(1.0 / ss_s_per_step, 4) if ss_s_per_step else None

    result = {
        "label": args.label or "e2e",
        "tier": "end-to-end",
        "command": cmd,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "stats": {
            "wall_s": round(wall, 2),
            "steady_state_s_per_step": round(ss_s_per_step, 3) if ss_s_per_step else None,
            "steady_state_steps_per_s": ss_steps_per_s,
            "steps_per_s_wall": round(steps_per_s, 4) if steps_per_s else None,
            "tqdm_it_per_s": scraped["tqdm_it_per_s"],
            # steady-state (client-side per-step wall deltas, warmup excluded) is the
            # honest throughput; tqdm's it/s is a cumulative average that the
            # precompute-heavy first step drags down
            "throughput_per_s": ss_steps_per_s or scraped["tqdm_it_per_s"],
        },
        "final_avg_loss": scraped["final_avg_loss"],
        "env": {"git_sha": git_sha(), "timing_enabled": True},
    }

    s = result["stats"]
    print(f"\n  {result['label']}  [{git_sha()}]")
    print(f"    wall {s['wall_s']}s over {args.steps} steps")
    print(f"    steady-state: {s['steady_state_s_per_step']}s/step ({s['steady_state_steps_per_s']} steps/s), warmup {args.warmup_steps} excluded")
    print(f"    steps/s (wall, warmup-excluded): {s['steps_per_s_wall']}")
    print(f"    tqdm cumulative it/s: {s['tqdm_it_per_s']}")
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
