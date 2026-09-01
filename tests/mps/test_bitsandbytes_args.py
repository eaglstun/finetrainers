from types import SimpleNamespace

import pytest

from finetrainers import args as args_module


@pytest.mark.parametrize("optimizer", ["adam-bnb", "adamw-bnb", "adam-bnb-8bit", "adamw-bnb-8bit"])
def test_bitsandbytes_optimizers_are_allowed_on_mps(monkeypatch, optimizer):
    monkeypatch.setattr(args_module, "get_device_info", lambda: ("mps", 1))
    args = SimpleNamespace(
        pp_degree=1,
        dp_degree=1,
        dp_shards=1,
        cp_degree=1,
        tp_degree=1,
        layerwise_upcasting_modules=[],
        optimizer=optimizer,
        attn_provider_training=[],
        attn_provider_inference=[],
    )

    args_module._validate_device_args(args)
