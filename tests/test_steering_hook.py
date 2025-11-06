from __future__ import annotations

import pytest
import torch
from torch import nn

from introspection.steer import LayerSteeringHook


class IdentityLayer(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class TupleLayer(nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        summary = hidden.sum(dim=-1)
        return hidden, summary


def test_layer_steering_hook_matches_manual_adjustment() -> None:
    layer = IdentityLayer()
    vector = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float32)
    strength = 0.75
    injection_index = 2

    base_hidden = torch.arange(1, 1 + 4 * 3, dtype=torch.float32).view(1, 4, 3)
    expected = base_hidden.clone()
    expected[:, injection_index:, :] += strength * vector

    hook = LayerSteeringHook(
        layer_module=layer,
        layer_index=0,
        vector=vector,
        injection_index=injection_index,
        strength=strength,
        debug_residual=False,
    )
    hook.register()
    try:
        output = layer(base_hidden.clone())
    finally:
        hook.remove()

    torch.testing.assert_close(output, expected)


def test_layer_steering_hook_handles_tuple_outputs() -> None:
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

    hook = LayerSteeringHook(
        layer_module=layer,
        layer_index=3,
        vector=vector,
        injection_index=injection_index,
        strength=strength,
        debug_residual=False,
    )
    hook.register()
    try:
        hidden_out, summary_out = layer(base_hidden.clone())
    finally:
        hook.remove()

    torch.testing.assert_close(hidden_out, expected_hidden)
    torch.testing.assert_close(summary_out, expected_summary)


def test_layer_steering_hook_respects_hidden_dtype() -> None:
    layer = IdentityLayer()
    vector = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float32)
    strength = -0.25
    injection_index = 0

    base_hidden = torch.tensor(
        [[[0.5, -1.5, 2.0, 4.0], [1.5, 0.0, -2.5, 3.5]]],
        dtype=torch.float16,
    )
    expected = base_hidden.clone()
    expected[:, injection_index:, :] += strength * vector.to(expected.dtype)

    hook = LayerSteeringHook(
        layer_module=layer,
        layer_index=7,
        vector=vector,
        injection_index=injection_index,
        strength=strength,
        debug_residual=False,
    )
    hook.register()
    try:
        output = layer(base_hidden.clone())
    finally:
        hook.remove()

    torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)


def test_hook_persists_when_sequence_grows_no_cache():
    layer = IdentityLayer()
    d = 3
    vector = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float32)
    strength = 0.75
    injection_index = 2

    hook = LayerSteeringHook(
        layer_module=layer,
        layer_index=0,
        vector=vector,
        injection_index=injection_index,
        strength=strength,
        debug_residual=False,
    )
    hook.register()
    try:
        # Prefill (e.g., first forward of generation)
        L = 5
        base0 = torch.arange(1, 1 + L * d, dtype=torch.float32).view(1, L, d)
        out0 = layer(base0.clone())
        expected0 = base0.clone()
        expected0[:, injection_index:, :] += strength * vector
        torch.testing.assert_close(out0, expected0)

        # Next step with use_cache=False: whole sequence recomputed with one extra token
        base1 = torch.arange(1, 1 + (L + 1) * d, dtype=torch.float32).view(1, L + 1, d)
        out1 = layer(base1.clone())
        expected1 = base1.clone()
        expected1[:, injection_index:, :] += strength * vector
        torch.testing.assert_close(out1, expected1)  # <-- Fails on current code
    finally:
        hook.remove()


def test_hook_injects_on_single_token_decode_step():
    layer = IdentityLayer()
    vector = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float32)
    strength = 1.0
    injection_index = 3  # any index >= 0

    hook = LayerSteeringHook(
        layer_module=layer,
        layer_index=0,
        vector=vector,
        injection_index=injection_index,
        strength=strength,
        debug_residual=False,
    )
    hook.register()
    try:
        # Simulate prefill (long sequence once)
        prefill = torch.zeros(1, 8, 3)
        _ = layer(prefill.clone())

        # Now simulate a decode step with cache: the layer sees only the new token
        tok = torch.randn(1, 1, 3)
        out = layer(tok.clone())
        expected = tok.clone()
        expected[:, 0:, :] += strength * vector
        torch.testing.assert_close(out, expected)  # <-- Fails on current code
    finally:
        hook.remove()


def test_hook_remove_stops_and_resets():
    layer = IdentityLayer()
    vector = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    strength = 0.5
    injection_index = 1

    hook = LayerSteeringHook(
        layer_module=layer,
        layer_index=0,
        vector=vector,
        injection_index=injection_index,
        strength=strength,
        debug_residual=False,
    )
    hook.register()
    out_with = layer(torch.zeros(1, 4, 3))
    hook.remove()

    out_without = layer(torch.zeros(1, 4, 3))
    expected_without = torch.zeros(1, 4, 3)
    expected_without[:, injection_index:, :] += strength * vector
    torch.testing.assert_close(out_with, expected_without)
    torch.testing.assert_close(out_without, torch.zeros_like(out_without))

    # Re-register: state (last_seq_len) must be reset
    hook.register()
    try:
        out_again = layer(torch.zeros(1, 2, 3))
        expected = torch.zeros(1, 2, 3)
        expected[:, injection_index:, :] += strength * vector
        torch.testing.assert_close(out_again, expected)
    finally:
        hook.remove()


def test_debug_residual_prints_once(capsys: pytest.CaptureFixture[str]):
    layer = IdentityLayer()
    vector = torch.tensor([0.2, 0.1], dtype=torch.float32)
    strength = 2.0
    injection_index = 0
    hook = LayerSteeringHook(
        layer_module=layer,
        layer_index=5,
        vector=vector,
        injection_index=injection_index,
        strength=strength,
        debug_residual=True,
    )
    hook.register()
    try:
        x = torch.ones(1, 3, 2)
        _ = layer(x.clone())
        _ = layer(x.clone())  # should not print the debug line a second time
    finally:
        hook.remove()

    out = capsys.readouterr().out
    assert "residual injection max error" in out
    assert out.count("residual injection max error") == 1


def test_batched_injection_slice():
    layer = IdentityLayer()
    vector = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float32)
    strength = 0.25
    injection_index = 2

    hook = LayerSteeringHook(
        layer_module=layer,
        layer_index=0,
        vector=vector,
        injection_index=injection_index,
        strength=strength,
        debug_residual=False,
    )
    hook.register()
    try:
        x = torch.arange(1, 1 + 2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
        out = layer(x.clone())
        expected = x.clone()
        expected[:, injection_index:, :] += strength * vector
        torch.testing.assert_close(out, expected)
    finally:
        hook.remove()
