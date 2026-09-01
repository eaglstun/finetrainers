"""LTX-Video transformer fwd+bwd with SDPA replaced by a bf16 math decomposition.

Phase 5B.6 probe. `ltx_transformer_fwd_bwd.py` measures the stock path, where
`F.scaled_dot_product_attention` on MPS dispatches to
`aten::_scaled_dot_product_attention_math_for_mps`, which upcasts q/k/v to fp32
internally (verified: its output matches an fp32-accumulated decomposition to
5 significant figures, while a bf16-accumulated one differs). This spec swaps in
the bf16 decomposition to price that upcast on the real 2B model.

Identical shape/config to `ltx_transformer_fwd_bwd.py` so the two results are
directly comparable; only the attention implementation differs.

Run:
    python ../finetrainers_bench.py run specs/ltx_transformer_fwd_bwd_mathsdpa.py --device mps \
        --warmup 3 --iters 15 --no-parity \
        --baseline ../baselines/ltx_transformer_fwd_bwd.mps.json
"""

import math
import pathlib
import sys

import torch
import torch.nn.functional as F


# The harness loads specs by file path without adding their directory to sys.path,
# so make the sibling stock spec importable.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ltx_transformer_fwd_bwd import ITEMS_PER_ITER, run  # noqa: E402,F401
from ltx_transformer_fwd_bwd import setup as _stock_setup  # noqa: E402


def _math_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """SDPA decomposed into matmul/softmax/matmul, accumulating in the input dtype."""
    if enable_gqa:
        raise NotImplementedError("gqa is not used by LTX-Video")
    scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
    attn = (query @ key.transpose(-2, -1)) * scale
    if is_causal:
        L, S = query.shape[-2], key.shape[-2]
        causal = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn = attn.masked_fill(~causal, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn = attn.masked_fill(~attn_mask, float("-inf"))
        else:
            attn = attn + attn_mask
    attn = attn.softmax(dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    return attn @ value


def setup(device):
    F.scaled_dot_product_attention = _math_sdpa
    torch.nn.functional.scaled_dot_product_attention = _math_sdpa
    return _stock_setup(device)
