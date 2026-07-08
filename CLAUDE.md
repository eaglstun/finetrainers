# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`finetrainers` is a library for training diffusion models (mostly video, some image) — LoRA and full-rank finetuning, plus conditional control training. It targets multi-backend distributed training on top of `diffusers`, `accelerate`, `peft`, and PyTorch's native distributed (DTensor/FSDP2). This is the `main` (development) branch; it is explicitly unstable — stable behavior lives on release tags (`v0.2.0`, etc.).

**This fork adds Apple Silicon (MPS) support** — a single-process, world_size=1 Accelerate lane (see "Apple Silicon" below and `docs/apple_silicon.md`). Local dev on this Mac uses the repo's `.venv` (Python 3.12, created with uv) — the pyenv-global Python is too new for several ML wheels. Use `.venv/bin/python` / `.venv/bin/ruff`.

Requires `diffusers` from `main` (`pip install git+https://github.com/huggingface/diffusers`), not just the pinned `>=0.32.1`. Use PyTorch 2.5.1+ — older versions produce black videos, OOM, or silent breakage.

## Commands

```bash
make style      # ruff format + ruff check --fix  (run before committing)
make quality    # ruff format --check + ruff check (CI gate; must pass)
pip install -e ".[dev]"   # installs pytest + ruff pinned versions
```

Training is launched via `train.py` under a distributed launcher — never `python train.py` directly for real runs (**exception: on Apple Silicon, plain `python train.py` is the only correct launch** — see the Apple Silicon section). The backend (`--parallel_backend`) decides the launcher:

```bash
# PTD backend (PyTorch native distributed — the default in current examples)
torchrun --nnodes=1 --nproc_per_node 2 train.py --parallel_backend ptd --training_type lora ...

# Accelerate backend
accelerate launch --config_file accelerate_configs/uncompiled_2.yaml train.py --parallel_backend accelerate ...
```

The `examples/training/**/train.sh` scripts are the canonical, copy-pasteable launch recipes (model args, dataset JSON, parallelism flags, precomputation). Start from one of those rather than assembling flags by hand. `--training_type` **must** appear in argv — `train.py` reads it out of `sys.argv` _before_ argparse to pick which config class to register.

### Tests

Fast trainer tests are parametrized by parallelism degree and backend, and must be run under the matching launcher — `pytest` alone won't set up the process group. See `tests/README.md` for the full matrix. Examples:

```bash
# Accelerate, single process
accelerate launch --config_file accelerate_configs/uncompiled_1.yaml -m pytest -s tests/trainer/test_sft_trainer.py -k "test___dp_degree_1___batch_size_1 and ___Accelerate"

# PTD, 2 processes (tests dp/fsdp/tp/cp degrees)
torchrun --nnodes=1 --nproc_per_node 2 -m pytest -s tests/trainer/test_sft_trainer.py -k "test___dp_degree_2___batch_size_1 and ___PTD"

# Context-parallel attention tests
torchrun --nnodes 1 --nproc_per_node 2 -m pytest -s tests/models/attention_dispatch.py::RingAttentionCP2Test
```

## Apple Silicon (MPS) — this fork's addition

Single-device lane only: `--parallel_backend accelerate`, every parallel degree 1, launched with
**plain `python train.py`** — no `torchrun`, no `accelerate launch` (launcher env vars make
Accelerate pick `gloo`/MULTI_CPU and the device silently becomes CPU). User doc:
`docs/apple_silicon.md`. Recipes: `examples/training/sft/{ltx_video,wan}/crush_smol_lora/train_mps.sh`.

Key seams (grep before editing — line numbers drift):

- Device chokepoint: `utils/torch.py::get_device_info()` (honors `FINETRAINERS_DEVICE` env; MPS→CUDA→CPU fallback).
- MPS guardrails: `args.py::_validate_device_args` — errors loudly on parallel degrees > 1, fp8 layerwise upcasting, bitsandbytes optimizers, non-native attention providers. Only fires when the resolved device is `mps`; programmatic `BaseArgs` (tests) bypasses it.
- Accelerate ws=1 trap: `parallel/accelerate.py` passes `InitProcessGroupKwargs` only when `LOCAL_RANK` is set.
- MPS-safe utilities: `utils/memory.py` (`reset_peak_memory_stats`, `free_memory`); grad-clipping `foreach` is device-aware in both trainers; checkpoint `states.pt` loads with `weights_only=False` (torch≥2.6).
- Video decode: decord is optional (no arm64 wheels); `data/dataset.py` dispatches by `datasets` version — torchcodec for `datasets>=4.0`.

Tests (plain pytest, skip without MPS): `tests/mps/test_cpu_mps_parity.py` (CPU-as-oracle forward
parity, per-model mixin) plus the dp_degree_1 Accelerate trainer tests. Set
`PYTORCH_ENABLE_MPS_FALLBACK=1` for training runs. Benchmarks: the `benchmark` skill
(`.claude/skills/benchmark/`), baselines in `.claude/skills/benchmark/baselines/`.

Known limits: gradient checkpointing is mandatory at LTX 512×768×49 (backward graph > 64 GB
without it); Wan trains at 320×512×49 but segfaults at 480×832×49 (upstream torch MPS tiled-bmm
bug, documented in `docs/apple_silicon.md`); don't set `HF_HUB_ENABLE_HF_TRANSFER=1` (silent
download hangs).

Tests use tiny dummy model specifications in `tests/models/<model>/` (not the real checkpoints), so they run on modest hardware.

## Architecture

The system is built around two orthogonal abstractions — **what model** (`ModelSpecification`) and **how to distribute** (`ParallelBackend`) — wired together by a **Trainer**. Adding a model or a backend should not require touching the other.

### The training flow (`train.py`)

1. `BaseArgs()` is constructed, then a training-type-specific config (`SFTLowRankConfig`, `SFTFullRankConfig`, `ControlLowRankConfig`, `ControlFullRankConfig`) is registered onto it and args are parsed. Args are composed from `ArgsConfigMixin` fragments — `BaseArgs` proxies attribute lookups across all registered mixins, so args like attention providers live in their own mixin class but read as `args.attn_provider_training`.
2. `_get_model_specifiction_cls(model_name, training_type)` looks up the concrete `ModelSpecification` subclass in `SUPPORTED_MODEL_CONFIGS` (`finetrainers/config.py`) — this dict is the registry of (model × training type) → spec class.
3. `SFTTrainer` or `ControlTrainer` is instantiated with the args + spec and `.run()` drives the phases: `_prepare_models → _prepare_trainable_parameters → _prepare_for_training → _prepare_dataset → _prepare_checkpointing → _train`.

### `ModelSpecification` (`finetrainers/models/modeling_utils.py`)

The interface every model implements. It is a _recipe_, not a model — it knows how to **load** components (`load_condition_models`, `load_latent_models`, `load_diffusion_models`, `load_pipeline`), how to **prepare** data (`prepare_conditions`, `prepare_latents`), and how to run the **forward/loss**. Each model lives in `finetrainers/models/<model>/`:

- `base_specification.py` — the standard T2V/T2I spec.
- `control_specification.py` — the control-conditioned variant (subclass of `ControlModelSpecification`), present only for models that support control training (currently `wan`, `cogview4`).

Condition (text encoder / tokenizer) and latent (VAE) preprocessing is expressed as **processors** (`finetrainers/processors/`, `ProcessorMixin`) — small callables (t5, llama, glm, clip, canny, …) that take/return dicts of named tensors. `prepare_conditions`/`prepare_latents` just run a chain of processors, with input/output name remapping handled by the mixin. This is what lets precomputation cache encoder outputs generically.

To add a model: create the dir + spec, register it in `SUPPORTED_MODEL_CONFIGS`, add a dummy spec under `tests/models/`. See `CONTRIBUTING.md` for the required training validation matrix.

### Parallel backends (`finetrainers/parallel/`)

`BaseParallelBackend` (`base.py`) defines the contract: `apply_ddp`, `apply_fsdp2`, `apply_context_parallel`, `prepare_model/dataset/dataloader/optimizer`, device-mesh access, checkpointing, tracker/logging routing, and process-rank properties. Two implementations:

- **`ptd.py`** (`PytorchDTensorParallelBackend`) — PyTorch-native DTensor/FSDP2; the backend the current examples default to. Supports the full parallelism cube via flags: `--dp_degree` (DDP replicas), `--dp_shards` (FSDP2 sharding; combine with dp_degree for HSDP), `--tp_degree` (tensor parallel), `--cp_degree` (context parallel), `--pp_degree`.
- **`accelerate.py`** (`AccelerateParallelBackend`) — HuggingFace Accelerate; more limited parallelism support.

`deepspeed.py` exists but is not wired into the enum. Backend is chosen by `--parallel_backend {ptd,accelerate}`; trainer code talks only to `BaseParallelBackend`, so it stays backend-agnostic.

### Attention dispatch (`finetrainers/models/attention_dispatch.py`)

Attention backend is pluggable and selectable _per module_ via `--attn_provider_training`/`--attn_provider_inference` as `"<component>:<provider>"` strings (e.g. `transformer:flash_varlen`). Providers include `flash`, `flash_varlen`, `flex`, `native` (+ specific SDPA kernels), `sage*` (inference only), and `xformers`. The `Trainer.attention_provider_ctx` context manager swaps the active provider for the wrapped forward (and holds it through backward for trained modules). Context-parallel (ring/ulysses) attention is implemented here too.

### Data (`finetrainers/data/`)

Datasets are `IterableDataset`s (`dataset.py`) auto-detected from format — file-pair, caption-file-list, folder, and WebDataset variants, each in image + video flavors. Multiple local/remote datasets are configured via a `--dataset_config` JSON (see `examples/**/training.json`) and can be chained; multi-resolution bucketing and combined image/video datasets are supported. `precomputation.py` caches condition/latent embeddings to disk (`--enable_precomputation`, `--precomputation_items`, `--precomputation_once`) so text-encoder/VAE passes don't rerun every epoch — important for fitting large models in limited VRAM.

### Patches (`finetrainers/patches/`)

Monkeypatches applied at trainer init via `patches.perform_patches_for_training(...)`. Two kinds: `dependencies/` (patch `diffusers`/`peft` behavior) and `models/` (per-model fixes for `ltx_video`, `wan`). When a model misbehaves in a way that traces back to upstream `diffusers`, check here before assuming the bug is in the spec.

## Conventions

- Line length 119, double quotes, ruff-formatted (config in `pyproject.toml`). `__init__.py` files are exempt from import-order/unused-import lint rules on purpose — re-exports live there.
- `examples/_legacy` is excluded from all lint/format targets; don't hold it to current style.
- Logging goes through `finetrainers.get_logger()`; the level is controlled by the `FINETRAINERS_LOG_LEVEL` env var (e.g. `DEBUG`).
- The codebase is early-stage and self-admittedly contains dead code and rough abstractions (grep for `TODO(aryan)`). Prefer opening an issue before large refactors, per `CONTRIBUTING.md`.
