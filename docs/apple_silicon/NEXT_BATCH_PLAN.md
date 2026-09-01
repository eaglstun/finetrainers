# finetrainers MPS — Batch 2: Measure, Speed Up, Generalize

**Status:** 5A ✅ + 5C ✅ + 5D docs ✅ · 5B in progress (paired optimizer trial ✅ 2026-08-31; batch-size sweep ✅ 2026-09-01; 5B.6 precision audit ✅ / its timing follow-up blocked on an idle machine 2026-09-01) · upstream PRs (5D.13) awaiting Eric's call · **Branch:** `apple-silicon-mps-phase-5b` · **Executor:** Codex
**Author:** Claude (Fable 5) · **Date:** 2026-07-08 · **Predecessor:** `PORT_PLAN.md` (phases 1–4, ✅ complete)

Batch 1 delivered _correctness_: LTX-Video LoRA trains on MPS (plain `python train.py`,
Accelerate ws=1 lane), CPU↔MPS parity tests pass, guards fail loudly, checkpoint
save/resume verified. This batch delivers **evidence-based speed** and **breadth** —
in that order, because optimizing without a profile is astrology.

House context: the `finetrainers-mps` skill (port decisions + landmines), the
`benchmark` skill / `benchmark-runner` agent (the harness for all of Phase 5), and
memory `finetrainers-apple-silicon-port` (discovery log). Line numbers drift; re-grep.

---

## Baseline reality (from the Batch-1 acceptance run, 2026-07-08)

LTX-Video LoRA, 512×768×49, bf16, rank 32, gradient checkpointing ON, batch 1, M-series 64 GB:

- **~5.5–7 s/step** steady-state (first step ~3–8 min: MPS kernel compilation + precompute)
- Precomputation (T5-XXL + VAE encode, 25 items) dominates cold-start wall clock
- `PYTORCH_ENABLE_MPS_FALLBACK=1` was on — **we do not yet know which ops silently
  round-trip through CPU on the hot path**. That census is the first deliverable.

---

## Phase 5A — Measure (gate for everything else)

1. **Wire LTX-MPS into the benchmark harness** (`benchmark` skill conventions): an
   end-to-end run benchmark (fixed step budget, steps/sec + peak memory via
   `get_memory_statistics`) and a micro benchmark for the transformer fwd/bwd. Save a
   named baseline for the Batch-1 config so every 5B change reports a delta.
2. **MPS fallback census.** Run the smoke config with fallback _disabled_
   (`PYTORCH_ENABLE_MPS_FALLBACK` unset) and catalog every `NotImplementedError`; then
   with fallback enabled, profile (`torch.profiler`, or op-level timing at the
   `attention_dispatch`/processor seams) to rank fallbacks by hot-path cost. Output: a
   table in `docs/apple_silicon.md` — op → where it bites (precompute vs train step) →
   cost.
3. **Step-time breakdown**: transformer fwd vs bwd vs optimizer vs data, using the
   existing `tracker.timed("timing/*")` instrumentation (`FINETRAINERS_ENABLE_TIMING`).

**Exit:** a committed baseline + a ranked list of where the time actually goes.

## Phase 5B — Cheap wins (only what 5A justifies; each change = one benchmark delta)

4. ✅ **Gradient checkpointing OFF trial.** It trades compute for memory; at the reference
   shape the backward graph allocates ~66 GB, swaps, and slows to ~80 s/iteration. Keep
   checkpointing enabled at 512×768×49; this is a capacity requirement, not a speed knob.
5. ✅ **Batch size sweep** (1→2→4) at fixed resolution — no throughput gain. Keep batch
   size 1 at 512×768×49; larger batches fit but scale slightly worse than linearly.
6. ⚠️ **`torch.set_float32_matmul_precision` / SDPA path check** — audit answered
   (2026-09-01), and both suspicions confirmed: MPS SDPA runs attention in fp32 regardless
   of input dtype, and the LoRA matmuls are fp32 by upstream design. The follow-on timing
   question (is a bf16 attention decomposition actually faster on the real model?) is
   **blocked on a quiet machine** — see the experiment log.
7. **`torch.compile` on MPS — timeboxed probe only.** Known-shaky; one afternoon, keep
   iff it's a clean >10% win on the benchmark, otherwise document "not yet" and move on.
8. ❌ **No hand-written Metal kernels.** Still the hypothetical Phase 6, still gated on
   5A proving a specific op is the bottleneck AND torch upstream won't fix it.

### Phase 5B experiment log

- **SDPA / precision audit (2026-09-01): bf16 buys nothing inside attention.**
  `F.scaled_dot_product_attention` on torch 2.12.1 dispatches to
  `aten::_scaled_dot_product_attention_math_for_mps` for bf16, fp16 **and** fp32 — MPS has no
  backend menu to select from. That op computes attention entirely in fp32: its output is
  *bitwise identical* to an fp32-accumulated decomposition and differs from a bf16-accumulated
  one, for both bf16 and fp16 inputs. So at LTX's shape the 2688x2688 probability matrix is
  materialized in fp32 (~924 MB/attention vs ~462 MB in bf16). bf16 `matmul` on MPS already
  accumulates in fp32 in hardware, so a bf16 decomposition would lose precision only in storing
  the probabilities. `--float32_matmul_precision` is numerically inert on MPS (identical fp32
  GEMM error across highest/high/medium — the knob gates CUDA TF32). And the LoRA adapters are
  fp32 on purpose: `cast_training_params([transformer], torch.float32)` fires on the
  non-data-sharded path, i.e. the entire MPS lane, so every adapter GEMM is fp32 with
  bf16<->fp32 conversions around it (4 target modules x 28 blocks).
  **Open, and the reason this item is not closed:** an unverified synthetic probe suggested a
  bf16 math decomposition is ~2x faster than SDPA at the self-attention shape. That number is
  **not trustworthy** — it was taken at load average 100 with a pytorch clang-tidy/lint run
  saturating the machine, and the paired control on the real 2B model read 16.3 s/iter against
  a 5.16 s July baseline, i.e. 3x contamination. `specs/ltx_transformer_fwd_bwd_mathsdpa.py`
  (written, unmeasured) is the drop-in variant; re-run it paired against
  `specs/ltx_transformer_fwd_bwd.py` in one session on an idle machine before believing any
  delta. Check `sysctl -n vm.loadavg` first.

- **Batch size 1→2→4 sweep (2026-09-01): batch 1 wins.** Same-session 30-step LTX 2B
  torch AdamW runs at 512×768×49, excluding two warmup steps, measured 14.066, 29.859,
  and 60.407 s/step respectively. Useful throughput was 0.0711, 0.0670, and 0.0662
  samples/s, so batch 2 was **-5.8%** and batch 4 **-6.9%** versus batch 1. All runs
  completed with finite loss and saved step-30 checkpoints; unified memory had capacity,
  but MPS compute did not scale into a throughput win. The first batch>1 run exposed an
  LTX latent-statistics broadcasting bug (`[C]` was reshaped using batch size); the fixed
  `[1,C,1,1,1]` broadcast is covered by a regression test.
- Results: `ltx_lora_batch_sweep_bs{1,2,4}.mps.e2e.json` under the benchmark skill's
  `baselines/` directory.
- **torch AdamW vs native bitsandbytes AdamW8bit (2026-08-31): no speed win.** Paired
  30-step LTX 2B runs at the reference shape, on the same machine and stack, measured
  14.381 s/step for torch AdamW and 14.252 s/step for AdamW8bit. The bnb result is
  **+1.0% throughput**, inside the 10% end-to-end noise/regression threshold. Both runs
  completed with finite loss and saved step-30 checkpoints; the bnb run set
  `BNB_MPS_REQUIRE_NATIVE=1`. Treat bnb as a memory/capability option, not a speedup.
- The older July torch baseline was 7.839 s/step. Its apparent 45% advantage over the
  first bnb result disappeared in the same-session torch control, so it is retained as
  historical data rather than used for a cross-date regression verdict.
- Results: `ltx_lora_adamw.mps.e2e.json` and `ltx_lora_bnb8.mps.e2e.json` under the
  benchmark skill's `baselines/` directory. The paired runs reused the same precomputed
  condition/latent data; steady-state timing excludes two warmup steps.

## Phase 5C — Second model: Wan T2V LoRA

Why Wan: most-used model in the repo's examples after LTX, has a control variant
(exercises `ControlTrainer` on MPS later), and a dummy spec already exists
(`tests/models/wan/`).

9. **Parity test**: add a Wan forward-parity case to `tests/mps/test_cpu_mps_parity.py`
   mirroring the LTX one (same tolerances unless bf16 forces looser — investigate
   before widening; see `finetrainers-mps` skill, parity-testing.md).
10. **Recipe**: `examples/training/sft/wan/crush_smol_lora/train_mps.sh` (copy the LTX
    MPS recipe shape). Smoke: 10 steps, finite loss, checkpoint save/resume.
11. **Docs**: extend the supported-models table in `docs/apple_silicon.md`. The arg
    guards are already model-agnostic — expected new work is _model-specific fallback
    ops_, which 5A's census methodology will catch per-model.

## Phase 5D — Hygiene & upstream

12. **tests/README.md**: add the MPS section (plain-pytest lane — no launcher).
13. **Upstream candidates** (bugs fixed in Batch 1 that are NOT Mac-specific — PR to
    `a-r-r-o-w/finetrainers` if Eric wants): checkpoint-resume `weights_only` breakage
    on torch≥2.6; decord→torchcodec decode path for `datasets>=4.0`; torch 2.11
    `_AttentionOp` import guard; `get_memory_statistics` `round(None)` crash;
    grad-clipping `foreach` device-awareness.
14. **(Optional) CI**: a `macos-14` (arm64) GitHub Actions job running
    `tests/mps/test_cpu_mps_parity.py` + the two LTX dp=1 accelerate tests. Cheap
    (minutes), catches regressions; needs Eric's call on Actions billing for the fork.

---

## Out of scope (unchanged from Batch 1 §5)

Metal/MSL kernels · fp8/QAT on MPS · MLX · multi-device anything ·
models beyond Wan (next batch, same recipe) · offload work (64 GB isn't pressed yet).

## Definition of done

1. A saved benchmark baseline for LTX-MPS and a fallback-census table in
   `docs/apple_silicon.md`.
2. Every 5B change lands with a benchmark delta (or a documented "no win, reverted").
3. Wan T2V LoRA: parity test passes, `train_mps.sh` smoke run trains with finite loss
   and a resumable checkpoint.
4. `make quality` (`.venv/bin/ruff …`) passes; parity tests still green.
5. The upstream-candidate list is either PR'd or explicitly parked by Eric.

## Risks / open questions

- **Fallback census may implicate the VAE or T5** (precompute path) rather than the
  train step — fine; document, don't fix, precompute is once-per-dataset.
- **Wan dummy vs real divergence**: dummy-spec parity can pass while the real checkpoint
  hits an unimplemented op (bigger head dims, different norm). The real-model smoke run
  is the true gate, same as Batch 1.
- **Benchmark noise on shared-memory Macs**: pin config (close apps, plugged in);
  benchmark harness variance rules apply.
- **torch pace**: MPS coverage improves per release — re-run the census after any torch
  bump and record versions with every baseline.
