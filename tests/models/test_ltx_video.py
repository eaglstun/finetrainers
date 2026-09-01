import torch

from finetrainers.models.ltx_video import LTXVideoModelSpecification


def test_normalize_latents_broadcasts_channel_statistics_across_batch():
    latents = torch.tensor(
        [
            [[[[1.0]]], [[[4.0]]]],
            [[[[3.0]]], [[[8.0]]]],
        ]
    )
    latents_mean = torch.tensor([1.0, 2.0])
    latents_std = torch.tensor([2.0, 2.0])

    normalized = LTXVideoModelSpecification._normalize_latents(latents, latents_mean, latents_std)

    expected = torch.tensor(
        [
            [[[[0.0]]], [[[1.0]]]],
            [[[[1.0]]], [[[3.0]]]],
        ]
    )
    torch.testing.assert_close(normalized, expected)
