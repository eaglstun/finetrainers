"""LTX-Video transformer forward+backward at the Batch-1 training shape.

Loads the REAL checkpoint (a-r-r-o-w/LTX-Video-diffusers, ~4 GB, must already be
in the HF cache) in bf16 and times one fwd+bwd at the acceptance-run shape:
512x768x49 pixels -> latent 7 frames x 16 x 24 (VAE 8x temporal / 32x spatial)
-> sequence length 2688, in_channels 128, T5-XXL caption dim 4096, text len 128.

Gradient checkpointing is ON, matching real training. Measured 2026-07-08: with
checkpointing OFF the backward graph allocates ~66 GB at this shape — it does
not fit in 64 GB unified memory and swaps (median ~80 s/iter). Checkpointing is
mandatory at this resolution, not a tunable. No parity hook — CPU bf16 fwd on
the 2B model is minutes-slow; correctness is gated by
tests/mps/test_cpu_mps_parity.py (dummy spec, fast) instead.

Run:
    python ../finetrainers_bench.py run ltx_transformer_fwd_bwd.py --device mps \
        --warmup 3 --iters 15 --no-parity \
        --out ../../baselines/ltx_transformer_fwd_bwd.mps.json \
        --label "LTX 2B fwd+bwd bf16 seq2688 mps"
"""

import torch


_DTYPE = torch.bfloat16
_BATCH, _FRAMES, _HEIGHT, _WIDTH = 1, 7, 16, 24  # latent dims for 49x512x768
_SEQ = _FRAMES * _HEIGHT * _WIDTH
_TEXT_LEN, _CAPTION_CHANNELS = 128, 4096

# one video sample per run()
ITEMS_PER_ITER = 1


def setup(device):
    from diffusers import LTXVideoTransformer3DModel

    transformer = LTXVideoTransformer3DModel.from_pretrained(
        "a-r-r-o-w/LTX-Video-diffusers", subfolder="transformer", torch_dtype=_DTYPE
    )
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    transformer.train()  # diffusers only applies checkpointing in train mode
    transformer.to(device)

    torch.manual_seed(0)
    hidden_states = torch.randn(_BATCH, _SEQ, 128, dtype=_DTYPE)
    encoder_hidden_states = torch.randn(_BATCH, _TEXT_LEN, _CAPTION_CHANNELS, dtype=_DTYPE)
    timestep = torch.tensor([500], dtype=torch.long)
    encoder_attention_mask = torch.ones(_BATCH, _TEXT_LEN, dtype=torch.bool)

    inputs = {
        "hidden_states": hidden_states.to(device).requires_grad_(True),
        "encoder_hidden_states": encoder_hidden_states.to(device),
        "timestep": timestep.to(device),
        "encoder_attention_mask": encoder_attention_mask.to(device),
        "num_frames": _FRAMES,
        "height": _HEIGHT,
        "width": _WIDTH,
        "rope_interpolation_scale": (1.0, 1.0, 1.0),
        "return_dict": False,
    }
    return transformer, inputs


def run(ctx):
    transformer, inputs = ctx
    out = transformer(**inputs)[0]
    loss = out.float().mean()
    loss.backward()
    inputs["hidden_states"].grad = None
