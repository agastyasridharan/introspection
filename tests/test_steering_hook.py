from __future__ import annotations

import pytest
import torch
from torch import nn

from introspection.steer import BatchedLayerSteeringHook


class IdentityLayer(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class TupleLayer(nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        summary = hidden.sum(dim=-1)
        return hidden, summary


def register_single_sample_hook(
    layer: nn.Module,
    *,
    vector: torch.Tensor,
    strength: float,
    injection_index: int,
    layer_index: int = 0,
    debug_residual: bool = False,
) -> BatchedLayerSteeringHook:
    addend = (strength * vector).unsqueeze(0)
    hook = BatchedLayerSteeringHook(
        layer_module=layer,
        layer_index=layer_index,
        addend=addend,
        injection_index=injection_index,
        debug_residual=debug_residual,
    )
    hook.register()
    return hook


def test_single_sample_injection_matches_manual_adjustment() -> None:
    layer = IdentityLayer()
    vector = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float32)
    strength = 0.75
    injection_index = 2

    base_hidden = torch.arange(1, 1 + 4 * 3, dtype=torch.float32).view(1, 4, 3)
    expected = base_hidden.clone()
    expected[:, injection_index:, :] += strength * vector

    hook = register_single_sample_hook(
        layer,
        vector=vector,
        strength=strength,
        injection_index=injection_index,
    )
    try:
        output = layer(base_hidden.clone())
    finally:
        hook.remove()

    torch.testing.assert_close(output, expected)


def test_tuple_output_supported() -> None:
    layer = TupleLayer()
    vector = torch.tensor([-0.25, 0.75], dtype=torch.float32)
    strength = 1.5
    injection_index = 1

    base_hidden = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
        dtype=torch.float32,
    )
    expected_hidden = base_hidden.clone()
    expected_hidden[:, injection_index:, :] += strength * vector
    expected_summary = base_hidden.sum(dim=-1)

    hook = register_single_sample_hook(
        layer,
        vector=vector,
        strength=strength,
        injection_index=injection_index,
        layer_index=3,
    )
    try:
        hidden_out, summary_out = layer(base_hidden.clone())
    finally:
        hook.remove()

    torch.testing.assert_close(hidden_out, expected_hidden)
    torch.testing.assert_close(summary_out, expected_summary)


def test_hidden_dtype_respected() -> None:
    layer = IdentityLayer()
    vector = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float32)
    strength = -0.25
    injection_index = 0

    base_hidden = torch.tensor(
        [[[0.5, -1.5, 2.0, 4.0], [1.5, 0.0, -2.5, 3.5]]],
        dtype=torch.float16,
    )
    expected = base_hidden.clone()
    expected[:, injection_index:, :] += (strength * vector).to(expected.dtype)

    hook = register_single_sample_hook(
        layer,
        vector=vector,
        strength=strength,
        injection_index=injection_index,
        layer_index=7,
    )
    try:
        output = layer(base_hidden.clone())
    finally:
        hook.remove()

    torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)


def test_hook_persists_when_sequence_grows_no_cache() -> None:
    layer = IdentityLayer()
    d_model = 3
    vector = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float32)
    strength = 0.75
    injection_index = 2

    hook = register_single_sample_hook(
        layer,
        vector=vector,
        strength=strength,
        injection_index=injection_index,
    )
    try:
        seq_len = 5
        base_prefill = torch.arange(1, 1 + seq_len * d_model, dtype=torch.float32).view(
            1, seq_len, d_model
        )
        out_prefill = layer(base_prefill.clone())
        expected_prefill = base_prefill.clone()
        expected_prefill[:, injection_index:, :] += strength * vector
        torch.testing.assert_close(out_prefill, expected_prefill)

        base_extended = torch.arange(
            1, 1 + (seq_len + 1) * d_model, dtype=torch.float32
        ).view(1, seq_len + 1, d_model)
        out_extended = layer(base_extended.clone())
        expected_extended = base_extended.clone()
        expected_extended[:, injection_index:, :] += strength * vector
        torch.testing.assert_close(out_extended, expected_extended)
    finally:
        hook.remove()


def test_hook_injects_on_single_token_decode_step() -> None:
    layer = IdentityLayer()
    vector = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float32)
    strength = 1.0
    injection_index = 3

    hook = register_single_sample_hook(
        layer,
        vector=vector,
        strength=strength,
        injection_index=injection_index,
    )
    try:
        prefill = torch.zeros(1, 8, 3)
        _ = layer(prefill.clone())

        token = torch.randn(1, 1, 3)
        out = layer(token.clone())
        expected = token.clone()
        expected[:, 0:, :] += strength * vector
        torch.testing.assert_close(out, expected)
    finally:
        hook.remove()


def test_hook_remove_stops_and_resets() -> None:
    layer = IdentityLayer()
    vector = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    strength = 0.5
    injection_index = 1

    hook = register_single_sample_hook(
        layer,
        vector=vector,
        strength=strength,
        injection_index=injection_index,
    )
    out_with = layer(torch.zeros(1, 4, 3))
    hook.remove()

    out_without = layer(torch.zeros(1, 4, 3))
    expected_without = torch.zeros(1, 4, 3)
    expected_without[:, injection_index:, :] += strength * vector
    torch.testing.assert_close(out_with, expected_without)
    torch.testing.assert_close(out_without, torch.zeros_like(out_without))

    hook = register_single_sample_hook(
        layer,
        vector=vector,
        strength=strength,
        injection_index=injection_index,
    )
    try:
        out_again = layer(torch.zeros(1, 2, 3))
        expected_again = torch.zeros(1, 2, 3)
        expected_again[:, injection_index:, :] += strength * vector
        torch.testing.assert_close(out_again, expected_again)
    finally:
        hook.remove()


def test_debug_residual_prints_once(capsys: pytest.CaptureFixture[str]) -> None:
    layer = IdentityLayer()
    vector = torch.tensor([0.2, 0.1], dtype=torch.float32)
    strength = 2.0
    injection_index = 0
    hook = register_single_sample_hook(
        layer,
        vector=vector,
        strength=strength,
        injection_index=injection_index,
        layer_index=5,
        debug_residual=True,
    )
    try:
        tokens = torch.ones(1, 3, 2)
        _ = layer(tokens.clone())
        _ = layer(tokens.clone())
    finally:
        hook.remove()

    out = capsys.readouterr().out
    assert "residual injection max error" in out
    assert out.count("residual injection max error") == 1


def test_per_sample_addends_applied_independently() -> None:
    layer = IdentityLayer()
    addends = torch.stack(
        [
            torch.tensor([0.5, -1.0, 0.0], dtype=torch.float32),
            torch.tensor([-0.25, 0.0, 0.75], dtype=torch.float32),
        ],
        dim=0,
    )
    injection_index = 2

    hook = BatchedLayerSteeringHook(
        layer_module=layer,
        layer_index=4,
        addend=addends,
        injection_index=injection_index,
        debug_residual=False,
    )
    hook.register()
    try:
        tokens = torch.arange(1, 1 + 2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
        out = layer(tokens.clone())
        expected = tokens.clone()
        expected[:, injection_index:, :] += addends.unsqueeze(1)
        torch.testing.assert_close(out, expected)
    finally:
        hook.remove()
