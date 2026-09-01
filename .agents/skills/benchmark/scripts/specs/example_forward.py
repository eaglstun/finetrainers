"""Example bench spec — copy this to author your own.

Self-contained (a small MLP) so it runs on any machine with torch, with no
model download. Demonstrates the three hooks the harness uses. To benchmark a
real finetrainers component, replace `setup()`/`run()` with a load of the
model spec's transformer and a forward pass — see reference/authoring.md.

Run it:
    python ../finetrainers_bench.py run example_forward.py --device auto --iters 50
    python ../finetrainers_bench.py run example_forward.py --device mps --rtol 2e-2 --atol 2e-2
"""
import torch
import torch.nn as nn

# samples processed per run() — turns median latency into items/s throughput.
ITEMS_PER_ITER = 8

_DTYPE = torch.bfloat16


def _build(device):
    # Init weights AND inputs on CPU under a fixed seed, THEN move to the device.
    # Generating directly on `device` would draw from that device's own RNG stream
    # (mps != cpu even with the same seed) and silently break parity — the exact
    # class of bug this harness is meant to surface.
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.GELU(),
        nn.Linear(4096, 1024),
    ).to(dtype=_DTYPE)
    x = torch.randn(ITEMS_PER_ITER, 1024, dtype=_DTYPE)
    return net.to(device), x.to(device)


def setup(device):
    """Build model + inputs once on `device`. Whatever you return is passed to run()."""
    return _build(device)


def run(ctx):
    """The ONE thing timed. Keep it to a single unit of work (a forward, a step)."""
    net, x = ctx
    with torch.no_grad():
        net(x)


def parity_output(device):
    """Optional. Return a tensor computed ON `device`; harness compares cpu vs device.
    Ties into the port's correctness-first goal — catch MPS 'confident garbage'."""
    net, x = _build(device)
    with torch.no_grad():
        return net(x)
