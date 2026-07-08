# Authoring benchmarks

## The three spec hooks (micro tier)

```python
ITEMS_PER_ITER = 1                 # optional module-level int/float

def setup(device):                 # required-ish: build state ONCE, return it
    ...                            # (may also be defined as setup() with no arg)

def run(ctx):                      # required: the unit of work to time
    ...

def parity_output(device):         # optional: correctness check vs CPU oracle
    return some_tensor_on(device)
```

The harness (`finetrainers_bench.py`) does: `setup(device)` once → `warmup` untimed calls of `run(ctx)` → `iters` timed calls (each followed by a device sync) → stats + memory → optional parity. It puts the spec's own directory and the scripts directory on `sys.path`, so a spec can `import torch`, sibling modules, or `from finetrainers_bench import resolve_device`.

## Golden rules

1. **`run(ctx)` does one unit of work and nothing else.** No allocation, no `.item()`/`.cpu()` sync inside it (that serializes and distorts timing), no data loading. Build everything in `setup`. The harness inserts the sync.
2. **Generate tensors on CPU, then `.to(device)`.** `torch.randn(..., device="mps")` draws from the MPS RNG stream, which differs from CPU's under the same seed — it will silently break parity. (This bit the example spec during development; that's the whole point of the parity check.)
3. **Same seed, same weights across devices** for parity. Init on CPU under `torch.manual_seed(...)`, then move.
4. **`ITEMS_PER_ITER`** = however many samples/frames/tokens one `run()` processes. Only set it if items/s is meaningful; otherwise judge on median_ms.

## Timing a forward + backward (a real training-ish step)

```python
import torch
def run(ctx):
    net, x, target = ctx
    out = net(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    net.zero_grad(set_to_none=True)
```

Backward needs grad, so don't wrap in `no_grad`. Keep the optimizer out unless you're specifically benching `optimizer.step()`.

## Benchmarking a real finetrainers component

The models are loaded through a `ModelSpecification` (`finetrainers/models/<model>/base_specification.py`). Load the transformer via the spec's `load_diffusion_models`, or instantiate the diffusers module directly with a small config for a cheap, download-free bench:

```python
import torch
from diffusers import LTXVideoTransformer3DModel   # example

def setup(device):
    # tiny config -> no checkpoint download, fits on a laptop
    m = LTXVideoTransformer3DModel(num_layers=2, num_attention_heads=4,
                                   attention_head_dim=32, caption_channels=32)
    m = m.to(device=device, dtype=torch.bfloat16).eval()
    # build the exact input dict the module's forward expects (hidden_states,
    # encoder_hidden_states, timestep, ...) on CPU, then move
    inputs = _make_inputs()  # dict of tensors, CPU
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return m, inputs

def run(ctx):
    m, inputs = ctx
    with torch.no_grad():
        m(**inputs)
```

To bench the **actual** forward/loss path (attention provider, patches, etc.), instantiate the finetrainers `ModelSpecification` subclass and call `prepare_latents`/the forward the trainer uses — mirror what `tests/models/<model>/` does with the dummy specs, which are built exactly for cheap runs.

## Attention-provider sweeps

Attention backend is selectable per-module (`--attn_provider_training`). To compare providers, set the provider inside `setup` (via the same env/context the trainer uses) and run one spec per provider, then `finetrainers_bench.py compare flash.json flex.json`.

## Memory — read the caveats

- **CUDA:** true peak via `max_memory_allocated()` after `reset_peak_memory_stats()`. Trustworthy.
- **MPS:** _there is no peak-memory counter._ The harness reports `driver_allocated_mb` (absolute) and `delta_mb` (growth over the run). A non-zero, growing `delta_mb` across repeated runs is a leak signal; a one-shot peak it cannot give you. For a true high-water mark on MPS, watch `torch.mps.driver_allocated_memory()` yourself around the op, or use Instruments.
- **CPU:** not tracked here — use `psutil`/`/usr/bin/time -l` if you need RSS.

## Baseline hygiene

- **One baseline per (spec × device × machine).** Name them `spec.<device>.json` (e.g. `example.mps.json`, `example.cuda.json`). A CUDA baseline tells you nothing about an MPS run.
- **Regenerate when the machine or torch version changes.** The harness stamps `git_sha`, `dirty`, `torch`, and device into every result — check them before trusting a comparison. A `dirty: true` baseline is suspect.
- **Commit baselines** under `baselines/` so regressions are reviewable in a PR diff. Keep e2e baselines (`*.e2e.json`) separate from micro ones.
- **Cross-device comparison** (mps vs cuda) is legitimate for a _port_ — it answers "how far off Apple Silicon is" — but frame it as a gap measurement, never a regression.

## Interpreting output

```
median 0.392 ms   p90 0.490 ms   mean 0.407 ± 0.044 ms   cv 10.70%
```

- Trust **median**. `mean ± std` and `cv` tell you how noisy the measurement is.
- **cv > ~15%** → too noisy; raise `--iters`, quiet the machine, don't call small deltas.
- `p90 >> median` → bimodal (thermal throttling, GC, a background app). Investigate before trusting.
- Parity `max_rel` can look huge when outputs are near zero — that's why the verdict uses `allclose` (`atol` covers the near-zero case), not `max_rel` alone. Judge on `passed` + `max_abs` + the `nan/inf_on_device` flags.
