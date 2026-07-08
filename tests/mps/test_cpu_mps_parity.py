"""
CPU <-> MPS numeric parity for the LTX-Video transformer forward.

MPS failures are usually silent (wrong numbers, not crashes), so "training ran" is not evidence of
correctness. This test runs an identical seeded forward on CPU and MPS and asserts the outputs
match within a dtype-appropriate tolerance:

  - fp32: atol/rtol 1e-4 — differences come only from GEMM accumulation-order reassociation.
  - bf16: atol/rtol 5e-2 — bf16 has ~2-3 significant decimal digits; per-kernel accumulation
    differences between backends compound across layers.

Run with: python -m pytest -s tests/mps/test_cpu_mps_parity.py
Skips cleanly on machines without an MPS device (e.g. CI on Linux).
"""

import copy
import unittest

import pytest
import torch

from ..models.ltx_video.base_specification import DummyLTXVideoModelSpecification


pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires an Apple Silicon MPS device")


def _make_transformer(dtype: torch.dtype) -> torch.nn.Module:
    spec = DummyLTXVideoModelSpecification(transformer_dtype=dtype)
    transformer = spec.load_diffusion_models()["transformer"]
    transformer.eval()
    return transformer


def _make_inputs(device: torch.device, dtype: torch.dtype) -> dict:
    # Shapes follow the dummy LTX transformer config: in_channels=8, caption_channels=32,
    # patch_size=1, so sequence length = num_frames * height * width in latent space.
    batch_size, num_frames, height, width = 1, 2, 4, 4
    caption_sequence_length, caption_channels = 8, 32

    generator = torch.Generator(device="cpu").manual_seed(42)
    hidden_states = torch.randn(batch_size, num_frames * height * width, 8, generator=generator, dtype=torch.float32)
    encoder_hidden_states = torch.randn(
        batch_size, caption_sequence_length, caption_channels, generator=generator, dtype=torch.float32
    )
    timestep = torch.tensor([500], dtype=torch.long)
    encoder_attention_mask = torch.ones(batch_size, caption_sequence_length, dtype=torch.bool)

    return {
        "hidden_states": hidden_states.to(device=device, dtype=dtype),
        "encoder_hidden_states": encoder_hidden_states.to(device=device, dtype=dtype),
        "timestep": timestep.to(device),
        "encoder_attention_mask": encoder_attention_mask.to(device),
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "rope_interpolation_scale": (1.0, 1.0, 1.0),
        "return_dict": False,
    }


class CPUMPSParityTest(unittest.TestCase):
    def _run_parity(self, dtype: torch.dtype, atol: float, rtol: float) -> None:
        transformer_cpu = _make_transformer(dtype)
        transformer_mps = copy.deepcopy(transformer_cpu).to("mps")

        with torch.no_grad():
            output_cpu = transformer_cpu(**_make_inputs(torch.device("cpu"), dtype))[0]
            output_mps = transformer_mps(**_make_inputs(torch.device("mps"), dtype))[0]

        output_cpu = output_cpu.float()
        output_mps = output_mps.float().cpu()

        self.assertTrue(torch.isfinite(output_cpu).all(), "CPU forward produced non-finite values")
        self.assertTrue(torch.isfinite(output_mps).all(), "MPS forward produced non-finite values")

        max_diff = (output_cpu - output_mps).abs().max().item()
        self.assertTrue(
            torch.allclose(output_cpu, output_mps, atol=atol, rtol=rtol),
            f"CPU and MPS forward outputs diverge for {dtype}: max abs diff {max_diff:.6f} (atol={atol}, rtol={rtol})",
        )

    def test_transformer_forward_parity_fp32(self):
        self._run_parity(torch.float32, atol=1e-4, rtol=1e-4)

    def test_transformer_forward_parity_bf16(self):
        self._run_parity(torch.bfloat16, atol=5e-2, rtol=5e-2)
