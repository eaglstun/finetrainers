# Training on Apple Silicon (MPS)

finetrainers supports single-device training on Apple Silicon Macs via PyTorch's MPS backend.
This is a correctness-first port: one device, no distributed training, native attention, bf16.
Speed and memory optimizations are explicitly out of scope for now.

## Supported

- **Single-device training** with `--parallel_backend accelerate` and every parallel degree
  (`--pp_degree/--dp_degree/--dp_shards/--cp_degree/--tp_degree`) set to `1`, launched with plain
  `python train.py` (no `torchrun`, no `accelerate launch`).
- **LoRA training** (`--training_type lora`). Full finetune should work for models that fit in
  unified memory, but LoRA is the validated path.
- **Native attention** (`--attn_provider_* transformer:native`, PyTorch SDPA) — this is also the
  default when no provider is specified.
- **bf16 / fp16 / fp32** dtypes (`--transformer_dtype bf16` etc.). bf16 is the recommended
  low-precision dtype.
- **Precomputation** (`--enable_precomputation`), gradient checkpointing, checkpoint save/load.

## Unsupported (fails loudly at argument parsing)

| Feature                                                                                         | Why                                                                      | Use instead      |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------- |
| Multi-GPU / FSDP / HSDP / CP / TP / PP (`--*_degree > 1`)                                       | NCCL and DTensor/FSDP2 are CUDA-only; a Mac is one unified-memory device | All degrees `1`  |
| `flash`, `flash_varlen`, `flex`, `sage*`, `xformers`, `_native_cudnn/efficient/flash` attention | CUDA-only kernels                                                        | `native`         |
| fp8 layerwise upcasting (`--layerwise_upcasting_modules`)                                       | float8 dtypes have no MPS support                                        | bf16             |
| bitsandbytes optimizers (`--optimizer *-bnb*`)                                                  | bitsandbytes is CUDA-only                                                | `adamw` / `adam` |

## Environment variables

- `PYTORCH_ENABLE_MPS_FALLBACK=1` — **set this.** Operators without MPS kernels then fall back to
  CPU instead of raising `NotImplementedError`. finetrainers logs a warning at startup if it is
  unset. (Each fallback is a hidden CPU round-trip; fine for correctness, noted for a later
  performance pass.)
- `FINETRAINERS_DEVICE` — optional escape hatch to force the device (`mps`, `cuda`, or `cpu`),
  e.g. `FINETRAINERS_DEVICE=cpu` to run a CPU-only comparison on the same machine. Without it the
  device is auto-detected (MPS on Apple Silicon).

## Quickstart

```bash
# LTX-Video LoRA (2B) — the reference recipe
bash examples/training/sft/ltx_video/crush_smol_lora/train_mps.sh

# Wan T2V LoRA (1.3B)
bash examples/training/sft/wan/crush_smol_lora/train_mps.sh
```

Each script is the single-device mirror of `train.sh` in the same directory: Accelerate backend,
all parallel degrees 1, native attention, bf16, precomputation enabled, and a small step count for
a first smoke run. Raise `--train_steps` once you've confirmed loss goes down on your machine.

### Validated models

| Model           | Config                    | Step time (M-series 64 GB) | Notes                                      |
| --------------- | ------------------------- | -------------------------- | ------------------------------------------ |
| LTX-Video 2B    | LoRA bf16, 512×768×49     | ~7–9 s                     | reference recipe; parity + e2e benchmarked |
| Wan2.1 T2V 1.3B | LoRA bf16, **320×512×49** | ~32–36 s                   | see resolution limit below                 |

**Wan resolution limit (upstream bug):** at 480×832×49 (~20k tokens) the attention matmul takes
PyTorch's _tiled_ bmm path on MPS, which segfaults inside `MPSNDArray` encode
(`at::native::mps::tiled_bmm_out_mps_impl`, torch 2.12.1). 320×512×49 (~8k tokens) stays under
the tiling threshold and trains correctly. Re-test after torch upgrades; candidate for an
upstream PyTorch issue.

## Verifying correctness (CPU ↔ MPS parity)

MPS bugs usually manifest as silently-wrong numbers rather than crashes. The parity test runs the
same seeded LTX-Video transformer forward on CPU and MPS and asserts the outputs match within a
dtype-appropriate tolerance:

```bash
python -m pytest -s tests/mps/test_cpu_mps_parity.py
```

The test skips automatically on machines without MPS, so it is safe in CI.

## Performance notes (Batch 2 census, 2026-07-08)

**MPS fallback census: zero fallbacks.** A full LTX-Video LoRA run (T5-XXL text encoding, VAE
video encoding, transformer forward/backward, AdamW, checkpointing) on torch 2.12.1 emitted no
`aten::*` CPU-fallback warnings — the entire hot path runs natively on MPS. (Verified against a
known-missing op to confirm the detection works; re-run the census after any torch upgrade by
grepping a full training log for `not currently supported on the MPS backend`.)

**Gradient checkpointing is mandatory at 512×768×49, not a speed knob.** Without it, the backward
graph of the 2B transformer at latent sequence length 2688 allocates ~66 GB — it does not fit in
64 GB unified memory and thrashes swap (~80 s/iter measured). With checkpointing, a full training
step is ~5.5–7 s at this shape.

**Where a training step goes** (M-series 64 GB, torch 2.12.1, bf16, checkpointing on): a full
LoRA training step is **~7–9 s** (e2e baseline steady-state 7.8 s/step; ~6–7 s observed on a fully
idle machine). The raw transformer forward+backward is ~5.2 s of that (micro benchmark median
5162 ms, cv 3.9%); the remainder is LoRA adapter compute, batch preparation, and per-step
`.item()` syncs. The optimizer step is ~0.02 s. Speedups must come from the transformer compute
itself (attention kernel path, checkpointing granularity), not the data/optimizer path. Step 1
is minutes-long (precomputation + MPS shader compilation) — always exclude it from timing.

Benchmark baselines live in `.claude/skills/benchmark/baselines/` (micro:
`ltx_transformer_fwd_bwd`, end-to-end: the `train_mps.sh` config); see the `benchmark` skill for
running them against changes.

## Dependency notes for macOS

- **decord** has no macOS arm64 wheels. It is now an optional import: with `datasets >= 4.0.0`
  video decoding goes through **torchcodec** instead (`pip install torchcodec av`), which does ship
  arm64 wheels. Nothing to configure — the dataset code picks the right decoder for your installed
  `datasets` version.
- **bitsandbytes** is not installable/usable on macOS; don't include it, the default `adamw`
  optimizer path never touches it.
- Use Python 3.12 or earlier — several ML packages don't publish wheels for newer Pythons yet.

## Tested configuration (2026-07)

| Component   | Version                            |
| ----------- | ---------------------------------- |
| macOS       | Darwin 25.4 (Apple Silicon, 64 GB) |
| Python      | 3.12                               |
| torch       | 2.12.1 (MPS)                       |
| torchvision | 0.27.1                             |
| datasets    | 5.0.0                              |
| torchcodec  | 0.14.0                             |
| diffusers   | 0.39.0                             |
| accelerate  | 1.13.0                             |
| peft        | 0.19.1                             |

## Memory reality on 64 GB

LTX-Video LoRA at 512×768×49 with precomputation, gradient checkpointing, and bf16 fits
comfortably. Unified memory means the model, activations, and everything else share the same pool —
watch `Activity Monitor` memory pressure rather than expecting a CUDA-style OOM; macOS will swap
before it kills the process. Bigger models (Wan 14B, HunyuanVideo) are untested and likely need
the (future) offload work.
