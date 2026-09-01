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
- **bitsandbytes optimizers**, including 8-bit Adam and AdamW (`--optimizer adam-bnb-8bit` or
  `adamw-bnb-8bit`), when the Apple Silicon fork is installed as described below.
- **bf16 / fp16 / fp32** dtypes (`--transformer_dtype bf16` etc.). bf16 is the recommended
  low-precision dtype.
- **Precomputation** (`--enable_precomputation`), gradient checkpointing, checkpoint save/load.

## Unsupported (fails loudly at argument parsing)

| Feature                                                                                         | Why                                                                      | Use instead     |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | --------------- |
| Multi-GPU / FSDP / HSDP / CP / TP / PP (`--*_degree > 1`)                                       | NCCL and DTensor/FSDP2 are CUDA-only; a Mac is one unified-memory device | All degrees `1` |
| `flash`, `flash_varlen`, `flex`, `sage*`, `xformers`, `_native_cudnn/efficient/flash` attention | CUDA-only kernels                                                        | `native`        |
| fp8 layerwise upcasting (`--layerwise_upcasting_modules`)                                       | float8 dtypes have no MPS support                                        | bf16            |

## Environment variables

- `PYTORCH_ENABLE_MPS_FALLBACK=1` — **set this.** Operators without MPS kernels then fall back to
  CPU instead of raising `NotImplementedError`. finetrainers logs a warning at startup if it is
  unset. (Each fallback is a hidden CPU round-trip; fine for correctness, noted for a later
  performance pass.)
- `FINETRAINERS_DEVICE` — optional escape hatch to force the device (`mps`, `cuda`, or `cpu`),
  e.g. `FINETRAINERS_DEVICE=cpu` to run a CPU-only comparison on the same machine. Without it the
  device is auto-detected (MPS on Apple Silicon).

## Quickstart

The MPS recipes use 8-bit AdamW. Install the Apple optimizer branch into this repo's virtualenv
first (the branch has to be built in-source so its Metal shader archive lands beside the dylib):

```bash
git clone https://github.com/eaglstun/bitsandbytes.git
cd bitsandbytes
git switch feature/mps-8bit-optim
cmake -DCOMPUTE_BACKEND=mps -S . -B .
cmake --build . --config Release
BNB_SKIP_CMAKE=1 uv pip install --python /path/to/finetrainers/.venv/bin/python -e .
```

`BNB_SKIP_CMAKE=1` is intentional: the preceding in-source build creates the native artifacts;
letting the editable installer invoke CMake again currently uses an out-of-tree build directory
and cannot locate `csrc/mps_kernels.metal`.

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
`.item()` syncs. Step 1 is minutes-long (precomputation + MPS shader compilation) — always exclude
it from timing. A same-session 30-step comparison on the 2026-08-31 stack measured 14.381 s/step
with torch AdamW and 14.252 s/step with native bitsandbytes AdamW8bit (**+1.0% throughput**, inside
the 10% end-to-end noise threshold). The 8-bit optimizer is therefore not a demonstrated speedup;
use it for its reduced-state representation. These current-stack runs were slower overall than the
July baseline, so only the paired optimizer delta is meaningful.

Benchmark baselines live in `.claude/skills/benchmark/baselines/` (micro:
`ltx_transformer_fwd_bwd`, end-to-end: the `train_mps.sh` config); see the `benchmark` skill for
running them against changes.

## Dependency notes for macOS

- **decord** has no macOS arm64 wheels and is excluded there by the requirements marker. With
  `datasets >= 4.0.0`, video decoding goes through **TorchCodec 0.15** instead.
- Released TorchCodec 0.15 macOS wheels currently contain FFmpeg 4–8 loaders, while current
  Homebrew `ffmpeg` is 9. Install `brew install ffmpeg@7`; it is keg-only and can coexist with
  FFmpeg 9. The MPS recipes automatically add `/opt/homebrew/opt/ffmpeg@7/{bin,lib}` to `PATH` and
  `DYLD_FALLBACK_LIBRARY_PATH`. For parity tests, set those variables explicitly:

  ```bash
  DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@7/lib \
  PATH=/opt/homebrew/opt/ffmpeg@7/bin:$PATH \
    .venv/bin/python -m pytest -q tests/mps/test_cpu_mps_parity.py
  ```

- **bitsandbytes** must come from the Apple Silicon fork's `feature/mps-8bit-optim` branch for the
  MPS recipes. Its Adam/AdamW 8-bit update has a native Metal path and a pure-PyTorch MPS fallback.
  The plain `bitsandbytes` entry in `requirements.txt` remains portable for CUDA and other hosts;
  on Apple Silicon, replace it with the in-source fork install shown in Quickstart.
- Use Python 3.12 or earlier — several ML packages don't publish wheels for newer Pythons yet.

## Last validated configuration (2026-07)

This is the last stack that passed the finetrainers MPS parity and end-to-end training gates. Keep
it separate from the current environment audit below: an installed version is not considered
validated until those gates run successfully.

| Component   | Version                                    |
| ----------- | ------------------------------------------ |
| Hardware    | MacBook Pro, Apple M4 Max (16-core), 64 GB |
| macOS       | 26.4.1 (Darwin 25.4)                       |
| Python      | 3.12                                       |
| torch       | 2.12.1 (MPS)                               |
| torchvision | 0.27.1                                     |
| datasets    | 5.0.0                                      |
| torchcodec  | 0.14.0                                     |
| diffusers   | 0.39.0                                     |
| accelerate  | 1.13.0                                     |
| peft        | 0.19.1                                     |

### Current local environment audit (2026-08-31)

The current virtualenv has passed the focused CPU↔MPS parity, dataset decoding, bitsandbytes
native-kernel, and 10-step LTX trainer integration gates with TorchCodec using keg-only FFmpeg 7.
It has not replaced the July validated matrix because the full-size LTX and Wan reference recipes
have not been rerun on this stack.

| Component    | Installed version / status                             |
| ------------ | ------------------------------------------------------ |
| Hardware     | MacBook Pro, Apple M4 Max (16-core), 64 GB             |
| macOS        | 26.5.2 (build 25F84)                                   |
| Python       | 3.12.13                                                |
| torch        | 2.12.1; MPS built and available                        |
| torchvision  | 0.27.1                                                 |
| datasets     | 5.0.0                                                  |
| torchcodec   | 0.15.0; imports with keg-only FFmpeg 7                 |
| FFmpeg       | 9.0.1 system default; 7.1.5 keg-only for TorchCodec    |
| diffusers    | 0.39.0                                                 |
| accelerate   | 1.14.0                                                 |
| peft         | 0.19.1                                                 |
| bitsandbytes | 0.50.0.dev0; `feature/mps-8bit-optim` commit `212c745` |

The focused MPS suite passed **8 tests**, and the non-network dataset suite passed **19 tests**, on
2026-08-31. The latter covers lazy image/video decoding through the production preprocessing
wrapper. The bitsandbytes native optimizer suite passed **8 tests** with
`BNB_MPS_REQUIRE_NATIVE=1`, covering fused Adam/Lion updates for fp32, fp16, and bf16 plus 8-bit
optimizer end-to-end cases. With the fork installed in `.venv`, the LTX dummy-model trainer passed
**two 10-step runs** using `adamw-bnb-8bit` (precomputation off and on), including checkpoint saves
at steps 6 and 10. The small trainer model uses 32-bit optimizer state below bitsandbytes' minimum
8-bit tensor size, so the separate native suite is the proof that the fused Metal 8-bit path runs.

## Memory reality on 64 GB

LTX-Video LoRA at 512×768×49 with precomputation, gradient checkpointing, and bf16 fits
comfortably. Unified memory means the model, activations, and everything else share the same pool —
watch `Activity Monitor` memory pressure rather than expecting a CUDA-style OOM; macOS will swap
before it kills the process. Bigger models (Wan 14B, HunyuanVideo) are untested and likely need
the (future) offload work.
