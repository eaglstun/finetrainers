# finetrainers MPS — Batch 2: Measure, Speed Up, Generalize

**Status:** plan, ready to execute · **Branch:** `apple-silicon-mps` · **Executor:** Fable
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

4. **Gradient checkpointing OFF trial.** It trades compute for memory; on 64 GB unified
   memory the trade may be backwards. If it fits, this is likely the single biggest
   free speedup. Document the memory/speed pair both ways.
5. **Batch size sweep** (1→2→4) at fixed resolution — unified memory may allow real
   throughput gains before pressure.
6. **`torch.set_float32_matmul_precision` / SDPA path check** — confirm bf16 SDPA hits
   the fast MPS kernel (not math fallback); confirm no fp32 upcasts sneak into the LoRA
   matmuls.
7. **`torch.compile` on MPS — timeboxed probe only.** Known-shaky; one afternoon, keep
   iff it's a clean >10% win on the benchmark, otherwise document "not yet" and move on.
8. ❌ **No hand-written Metal kernels.** Still the hypothetical Phase 6, still gated on
   5A proving a specific op is the bottleneck AND torch upstream won't fix it.

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
