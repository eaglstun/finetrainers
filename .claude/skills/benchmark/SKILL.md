---
name: benchmark
description: Write, run, and evaluate performance & correctness benchmarks for finetrainers — micro-benchmarks (op/module forward-backward timing, memory, CPU-vs-device numeric parity) and end-to-end run benchmarks (throughput over a fixed step budget), with device-aware timing (cuda/mps/cpu) and baseline regression tracking. Use when asked to benchmark, profile, measure throughput/step-time/memory, check for a perf regression, verify MPS/CPU numeric parity, or write a new benchmark. Rides finetrainers' own `tracker.timed("timing/*")` instrumentation.
---

# Benchmarking finetrainers

Two tiers. Pick by what the question is.

| Question                                                                       | Tier           | Tool                            | Needs a launcher?                          |
| ------------------------------------------------------------------------------ | -------------- | ------------------------------- | ------------------------------------------ |
| Is this op/module fast? Did my change regress the forward? Does MPS match CPU? | **micro**      | `scripts/finetrainers_bench.py` | no — runs single-process on the Mac        |
| What's real training throughput for model X on device Y?                       | **end-to-end** | `scripts/run_e2e_benchmark.py`  | yes — wraps `torchrun`/`accelerate launch` |

Both write a JSON result, both compare against a saved baseline and **exit non-zero on regression or parity failure**, so they drop straight into CI or an agent's pass/fail logic.

## What finetrainers already gives you

The trainer instruments phases with `self.tracker.timed("timing/forward")`, `timing/backward`, `timing/optimizer_step`, `timing/batch_preparation`, `timing/checkpoint` — gated by the `FINETRAINERS_ENABLE_TIMING` env var, logged via `parallel_backend.log(logs, step)` alongside `train/global_avg_loss`, `train/grad_norm`, `train/observed_data_samples`. **Caveat:** the repo's `Timer` (`finetrainers/utils/timing.py`) only does event timing for CUDA; for MPS it silently falls back to a CPU wall-clock **without a device sync**, so it measures kernel _launch_, not execution. The harness here syncs correctly per device — prefer it for MPS numbers.

## Micro-benchmark — the everyday tool

A **bench spec** is a plain Python file with up to three hooks:

```python
ITEMS_PER_ITER = 8              # optional: samples/frames per run() -> throughput
def setup(device):             # build model+inputs ONCE on `device`; returns ctx
    ...
def run(ctx):                  # the ONE thing timed (a forward, a full step)
    ...
def parity_output(device):     # optional: return a tensor computed ON `device`;
    ...                        # harness runs cpu + device and compares (CPU = oracle)
```

Run it:

```bash
cd .claude/skills/benchmark/scripts
python finetrainers_bench.py run specs/example_forward.py --device auto --iters 50 \
    --out ../baselines/example.mps.json --label "example fwd, mps"

# guard a change against a baseline (non-zero exit on >5% regression):
python finetrainers_bench.py run specs/my_spec.py --device mps --baseline ../baselines/my_spec.mps.json

# compare two saved results, or reprint one:
python finetrainers_bench.py compare baseline.json current.json --threshold 0.05
python finetrainers_bench.py show result.json
```

`specs/example_forward.py` is a runnable, self-contained template — copy it. To benchmark a real component, load a model spec's transformer in `setup()` and call its forward in `run()`. See `reference/authoring.md`.

## End-to-end run benchmark

Wrap a real, short training launch (set `--train_steps` small, e.g. 60). Everything after `--` runs verbatim with `FINETRAINERS_ENABLE_TIMING=1` forced on:

```bash
python run_e2e_benchmark.py --steps 60 --warmup-steps 10 \
    --out ../baselines/ltx_lora.mps.e2e.json --label "ltx lora mps bs1" -- \
    torchrun --nnodes=1 --nproc_per_node 1 train.py --parallel_backend ptd \
      --training_type lora --model_name ltx_video --train_steps 60 ...
```

Start from an `examples/**/train.sh` recipe for the launch flags — don't hand-assemble them. On the Apple Silicon branch this is single-process, `--parallel_backend accelerate`, `world_size=1` (per `docs/apple_silicon/PORT_PLAN.md`); do not try DTensor/FSDP2/NCCL on MPS.

## Evaluating results — what "good" means

- **Latency:** report **median**, not mean — it's robust to the OS-scheduler outliers a laptop throws. `p90` shows the tail.
- **Noise gate:** if `cv` (coefficient of variation) > ~15%, the run is too noisy to trust — raise `--iters`, close other apps, or pin the device. Don't call a <2× change on a high-cv run a regression.
- **Regression:** the harness flags median-latency _up_ or throughput _down_ by more than `--threshold` (default 5% micro, 10% e2e). Regenerate the baseline on the same machine — cross-machine baselines are meaningless.
- **Parity (correctness):** CPU is the oracle. `bf16`/`fp16` on MPS produces silently-wrong numbers, not crashes (port-plan problem #3) — a parity fail is a real bug, not just noise. Default tolerances (`rtol/atol 2e-2`) are bf16-friendly; tighten to `1e-4` for fp32. A `nan_on_device: true` is a hard stop.

See `reference/authoring.md` for writing real-model specs, memory caveats (MPS has no peak-memory counter), and baseline hygiene.

## The agent

`benchmark-runner` (`.claude/agents/`) does this end-to-end autonomously: picks the tier, writes the spec if one doesn't exist, runs on the right device with the right launcher, compares to baseline, and reports a verdict with regression/parity flags. Delegate to it for "benchmark X" / "did my change regress Y" / "check MPS parity for Z".
