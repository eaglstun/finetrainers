# finetrainers → Apple Silicon (MPS) Port Plan

**Status:** plan, ready to execute · **Branch:** `apple-silicon-mps` · **Executor:** Fable
**Author of plan:** Claude (Opus 4.8) · **Date:** 2026-07-08

This is an executable spec. It assumes the reader is comfortable in the finetrainers
codebase and has done Apple-Silicon/Metal work before (Fable shipped the CTranslate2 int8
Metal backend — see §0). Line numbers cite a snapshot and **drift** — re-grep the symbol
before editing, per the house convention in the `apple-silicon`/`ct2-internals` skills.

---

## Scoping decisions (locked with Eric)

| Decision        | Value                                                  | Consequence                                                                                                                                    |
| --------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Target hardware | M-series **64 GB+** (Max/Ultra)                        | Memory is not the binding constraint for phase 1; don't over-invest in offload yet.                                                            |
| First model     | **LTX-Video** (LoRA)                                   | ~5 GB LoRA footprint on CUDA — the smallest, fastest path to a real MPS training run. Build the single-device path around it, then generalize. |
| Goal            | **Correctness first**                                  | Get one model training _correctly_ on MPS end-to-end with CPU-parity verification. Speed/memory optimization is explicitly a later phase.      |
| Parallelism     | **Single-process, `world_size=1`, Accelerate backend** | Do **not** try to make DTensor/FSDP2/NCCL run on MPS. Carve out the single-device lane the library already half-supports.                      |

---

## §0 — Required reading & house context (reference these; do not re-derive)

**Eric's own MPS-porting playbook (authoritative):**
`https://ai.ericeaglstun.com/deep-dives/porting-ml-to-apple-silicon/` — the six-problems
framework this plan is organized around. Also saved as memory `apple-silicon-porting-deepdive`.

**House Apple-Silicon knowledge (skills):**

- **`apple-silicon` skill** (`~/.claude/skills/apple-silicon/`) — the Metal/MSL/MPS reference
  shelf built for the CTranslate2 Metal backend. **Relevance caveat below.**
- **`ct2-internals` skill** (`~/.claude/skills/ct2-internals/`) — engine-architecture patterns;
  most relevant here: the **op-parity test methodology** (`ops-test-suite-structure.md`,
  per-backend tolerances, CPU-as-oracle) and the **CPU-vs-device numeric discipline**.
- **`ai-dev` skill** — glossary; has the **MLX** entry. See MLX note below.

**Prior Apple-Silicon port to mine for lessons (not code):**

- CTranslate2 Metal backend: `/Users/eeaglstun/Documents/dev/CTranslate2/` (`METAL_BACKEND.md`
  at root). Fable's int8 variant: `/Users/eeaglstun/Documents/dev/CTranslate2-fable-int8/`.

### ⚠️ The load-bearing distinction — read this before reusing anything from CT2

**CTranslate2 is a hand-written C++/MSL Metal engine. finetrainers rides PyTorch's `torch.mps`
backend.** We are **not** writing Metal kernels, MPSMatrixMultiplication calls, or
autorelease-pool plumbing here. We let PyTorch's Metal backend do the GPU work.

So from the CT2 work we **reuse the discipline, not the kernels**:

- ✅ **Parity testing** — run the same forward on CPU and MPS, compare within tolerance
  (CT2's `ops-test-suite-structure.md` per-backend-tolerance model; deep-dive problem #6).
- ✅ **fp16/bf16 "confident garbage"** awareness — MPS produces silently-wrong numbers, not
  crashes (CT2's `fp16-numerics-on-gpu.md`, the Gemma2 tanh-NaN saga; deep-dive #3).
- ✅ **Unified-memory footprint** thinking (`memory-footprint-and-residency.md`) — later phase.
- ❌ **Do NOT** port MSL kernels, `MPSMatrixMultiplication`, the op-graduation playbook, or
  the autorelease-pool fix. None apply until/unless we later hand-write a custom Metal kernel
  for a hot op (a hypothetical _Phase 6_, explicitly out of scope now).

### MLX note

MLX is Apple's array framework. finetrainers is welded to **PyTorch + diffusers + peft + accelerate**;
an MLX rewrite is not on the table and MLX is **not** a drop-in backend here. Mention it only
as: the framework we'd reach for _if_ we ever wrote a standalone Apple-native optimization
outside the torch stack. Not this project.

---

## §1 — Architecture reality (why the plan is shaped this way)

finetrainers is built almost entirely around **distributed CUDA**: NCCL process groups,
DTensor, FSDP2, `torchrun`, and a full parallelism cube (`--dp_degree/--dp_shards/--tp_degree/
--cp_degree/--pp_degree`). Apple Silicon is a **single unified-memory device**. NCCL is
CUDA-only; DTensor/FSDP2 on MPS is effectively unsupported.

**Therefore the port = carve out a clean single-process MPS lane.** The good news: the code
already half-supports this and _someone started the MPS work and stopped_:

- `finetrainers/utils/torch.py::get_device_info()` already returns `"mps"` on Apple Silicon
  (via torch's `_get_available_device_type()`), and `synchronize_device()` already has an MPS branch.
- `finetrainers/parallel/accelerate.py` has an explicit `if world_size == 1:` branch that
  **skips the process group**, and even contains `if torch.backends.mps.is_available():
self._accelerator.native_amp = False`.
- With every parallel degree = 1, `sft_trainer/trainer.py` **skips `apply_fsdp2`/`apply_ddp`**
  and just calls `parallel_backend.prepare_model(...)` — **no DTensor, no FSDP touched**.
- All CUDA-only imports in `attention_dispatch.py` (`flash_attn`, `sageattention`, `xformers`)
  are `try/except → None` guarded, so the module **imports clean** on a Mac.

The one nailed-shut door: **`finetrainers/trainer/base.py::_init_distributed`** computes
`world_size = torch.cuda.device_count()` (→ **0** on a Mac) and hardcodes `backend="nccl"`.

---

## §2 — Blocker inventory (verified during survey; re-grep before editing)

| #   | Deep-dive problem | Location                                                                                             | Problem                                                                                                      | Fix                                                                                                                                   |
| --- | ----------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Device selection  | `trainer/base.py` `_init_distributed` (~L91)                                                         | `world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))` → 0 on Mac                       | Device-aware default: `1` when not CUDA.                                                                                              |
| 1   | Device selection  | `trainer/base.py` (~L104)                                                                            | `backend="nccl"` hardcoded into backend ctor                                                                 | `gloo` on MPS/CPU, `nccl` on CUDA.                                                                                                    |
| 1   | Device selection  | `parallel/ptd.py` (~L50 default, ~L81 use)                                                           | ctor default `backend="nccl"`; `_device_module.set_device(local_rank)` — **`torch.mps` has no `set_device`** | Guard `set_device`; device-aware backend default. (PTD is secondary — Accelerate is the phase-1 lane.)                                |
| 1   | Device selection  | `utils/torch.py::get_device_info` (~L224)                                                            | falls back to `"cuda"` when torch reports no device                                                          | Add `FINETRAINERS_DEVICE` env escape hatch (MPS→CUDA→CPU order) at this single chokepoint.                                            |
| 2   | Missing MPS ops   | env / launch                                                                                         | unsupported ops crash instead of falling back                                                                | Ensure `PYTORCH_ENABLE_MPS_FALLBACK=1` (set in `train_mps.sh`; warn at startup if unset on MPS).                                      |
| 3   | Float precision   | `args.py` fp8 (`float8_e4m3fn/e5m2`, L150–154, L393, L869) + `sft_trainer` `apply_layerwise_casting` | fp8 layerwise-upcasting unsupported on MPS                                                                   | If `--layerwise_upcasting_modules` set on MPS: **hard error** with a clear message (don't silently produce garbage). Force bf16 path. |
| 4   | CUDA-only deps    | `models/attention_dispatch.py` (`_check_device_cuda`, L479) + args providers                         | `flash/flash_varlen/sage*/xformers` are CUDA-only                                                            | On MPS: default `--attn_provider_*` to `native` (SDPA); **hard error** if a CUDA-only provider is explicitly requested.               |
| 4   | CUDA-only deps    | `optimizer.py` (bitsandbytes, ~L?? — grep `bitsandbytes`/`bnb`)                                      | bnb 8-bit optimizers are CUDA-only                                                                           | On MPS: if a bnb optimizer requested, error with a pointer to `adamw`/`adamw-torch` fallback.                                         |
| —   | (safe already)    | `trainer/base.py::_init_config_options`                                                              | TF32 path already guarded by `torch.cuda.is_available()`                                                     | No change — verify it stays a no-op on Mac.                                                                                           |

Also check (grep, may be no-ops): `torch.cuda.empty_cache`/`synchronize`/`reset_peak_memory_stats`
call sites — route through the existing `synchronize_device()` / device-module abstraction rather
than `torch.cuda.*` directly. `optimizer.py` L77 has a `# NCCL hang?` TODO — irrelevant at ws=1.

---

## §3 — Phased implementation

### Phase 1 — Unblock single-device MPS _(the only phase that gates a first run)_

1. **`utils/torch.py::get_device_info()`** — honor `FINETRAINERS_DEVICE` env var
   (`mps`/`cuda`/`cpu`); otherwise keep torch auto-detect but change the `None` fallback from
   `"cuda"` to a real resolution (MPS → CUDA → CPU). This is the single device chokepoint the
   deep-dive prescribes.
2. **`trainer/base.py::_init_distributed`** — replace `torch.cuda.device_count()` default with a
   device-aware `world_size` (1 when not CUDA), and pass a device-aware `backend`
   (`"gloo"` for mps/cpu, `"nccl"` for cuda). Add a small helper (e.g. `_default_comm_backend()`)
   rather than inlining string literals.
3. **Friendly guardrail** — on MPS, if any of `pp/dp_shards/cp/tp_degree > 1` (or `dp_degree > 1`),
   raise early with: "Apple Silicon supports single-device training only; use
   `--parallel_backend accelerate` with all parallel degrees = 1." Best home: extend the
   validation near `args.py` ~L1024 (the existing multi-GPU offload check) or a new
   `validate_args` hook.
4. **`parallel/ptd.py`** — guard `_device_module.set_device(...)` behind a `hasattr` / device-type
   check so importing/constructing the PTD backend on MPS doesn't hard-crash. (Low priority; PTD is
   not the phase-1 lane, but leave it non-crashing.)

**Exit criterion for P1:** `python train.py ... --parallel_backend accelerate` (all degrees 1)
constructs the trainer, resolves device `mps`, and reaches `_prepare_models` without a
CUDA/NCCL error.

### Phase 2 — Feature guards (deep-dive #2/#3/#4)

5. **Attention** — on MPS, default unset `attn_provider_training`/`attn_provider_inference` to
   `native`; raise a clear error if a CUDA-only provider is explicitly requested. (`native` SDPA
   is the only MPS-viable provider.)
6. **fp8 / layerwise upcasting** — on MPS, error if `--layerwise_upcasting_modules` is set (fp8
   dtypes unsupported). Document bf16 as the supported precision.
7. **Optimizer** — on MPS, error-with-guidance if a bitsandbytes optimizer is chosen; steer to
   `adamw`. (Default is already torch `adamw`, so the happy path is clean.)
8. **MPS fallback env** — at startup on MPS, if `PYTORCH_ENABLE_MPS_FALLBACK` is unset, log a
   prominent warning (and set it in the example script).

### Phase 3 — LTX-Video recipe + docs

9. **`examples/training/sft/ltx_video/crush_smol_lora/train_mps.sh`** — copy the existing
   `train.sh`, then: `--parallel_backend accelerate`, all degrees `1`, no `torchrun` (plain
   `accelerate launch` or direct `python`), `--attn_provider_* native`, bf16 dtypes, keep
   `--gradient_checkpointing`, drop any fp8/layerwise flags, `export PYTORCH_ENABLE_MPS_FALLBACK=1`,
   `export WANDB_MODE=offline`. Small `--train_steps` for a smoke run.
10. **`docs/apple_silicon.md`** — user-facing: supported (single-device LoRA, native attention,
    bf16), unsupported (multi-GPU/FSDP/CP/TP, flash/sage/xformers, fp8, bnb optimizers), env vars,
    the `train_mps.sh` walkthrough, and the memory reality on 64 GB.

### Phase 4 — Correctness verification (deep-dive #6 — non-negotiable)

11. **`tests/mps/test_cpu_mps_parity.py`** (or a `scripts/` verify script) — load the **LTX-Video
    dummy spec** used by the existing tests (`tests/models/ltx_video/base_specification.py`), run
    an identical transformer forward on `cpu` and `mps` with the same seeded inputs, assert
    allclose within a documented tolerance (bf16/fp16 tolerances per CT2's per-backend-tolerance
    convention). This is the gate that distinguishes "ran" from "correct." Wire it so it **skips**
    when MPS is unavailable (CI safety).
12. Manual acceptance: run `train_mps.sh` for N steps, confirm loss is finite and decreasing, and
    that a LoRA checkpoint saves and reloads.

---

## §4 — New house artifacts (follow existing patterns)

Only create these if it helps future work; follow the conventions exactly.

- **(Recommended) Project skill `finetrainers-mps`** — a reference skill in
  `~/.claude/skills/finetrainers-mps/SKILL.md` following the `apple-silicon` skill's shape:
  YAML frontmatter (`name`, `description` with explicit "Use when…" triggers), a short index body,
  and `references/` files (e.g. `single-device-lane.md`, `feature-guards.md`, `parity-testing.md`).
  Purpose: capture _this_ port's decisions so the next session doesn't re-survey. Cross-link to the
  `apple-silicon` and `ct2-internals` skills. Register it in the `switchboard` skill index if Eric
  wants it discoverable.
- **Do NOT** create a new _agent_ for this — the work is edits to one repo, which the general
  agent + this plan cover. (The `metal-renderer`/`swift-expert`-style agents are for the iOS app,
  a different codebase.) If Eric later wants a dedicated MPS agent, model its frontmatter on
  `~/.claude/agents/swift-expert.md`.

---

## §5 — Out of scope (do not do these now)

- ❌ Making DTensor / FSDP2 / NCCL / the parallelism cube work on MPS.
- ❌ Hand-written Metal / MSL kernels, MPS custom ops, op-graduation (the CT2 playbook). Only if a
  profiler later proves a specific hot op needs it — a future phase, separately scoped.
- ❌ fp8 / QAT on MPS.
- ❌ MLX rewrite or MLX backend.
- ❌ Models beyond LTX-Video. Generalize _after_ LTX trains correctly, reusing the same guards.
- ❌ Speed/throughput optimization (no flash-attn, no fp8, shaky `torch.compile` on MPS). Correctness first.

---

## §6 — Definition of done (Phase 1–4)

1. On a 64 GB Apple Silicon Mac, `bash examples/training/sft/ltx_video/crush_smol_lora/train_mps.sh`
   runs LTX-Video LoRA training on `mps` for the configured steps **without CUDA/NCCL/fp8 errors**.
2. Loss is finite and trends down; a LoRA checkpoint is written and reloads.
3. `test_cpu_mps_parity.py` passes (LTX transformer forward matches CPU within tolerance) and skips
   cleanly where MPS is absent.
4. Requesting an unsupported feature on MPS (flash attention, fp8, bnb optimizer, `dp_degree>1`)
   fails **loudly and early** with an actionable message — never silent garbage.
5. `docs/apple_silicon.md` documents the supported/unsupported matrix and the launch recipe.
6. `make quality` passes on all changed files.

---

## §7 — Risks & open questions for Fable to resolve during execution

- **`get_mesh()` at `world_size=1`** — the Accelerate backend still builds a device mesh via
  `init_device_mesh("mps", (1,))` in some call paths (`prepare_dataset`/`prepare_dataloader`).
  Verify this succeeds on MPS; if `init_device_mesh` chokes on the `mps` device type, add a
  ws=1 short-circuit that returns `None`/a trivial mesh and confirm downstream call sites tolerate it.
- **`gloo` + single process** — confirm a 1-process `gloo` group initializes cleanly (it should);
  if the ws=1 Accelerate path avoids the process group entirely, even simpler.
- **bf16 on MPS for the VAE/text-encoder** — watch for ops that silently fall back to CPU
  (the `PYTORCH_ENABLE_MPS_FALLBACK` transfers). Note any hot-path fallbacks for a later perf pass;
  do not fix them now.
- **Precomputation** — LTX example uses `--enable_precomputation`; confirm the precompute pass
  (text-encoder + VAE) runs on MPS. This is often where a missing-op fallback first bites.
- **PyTorch version** — README wants ≥2.5.1; confirm the installed torch has a healthy MPS backend
  (newer is better for bf16/SDPA coverage). Record the tested version in `docs/apple_silicon.md`.
