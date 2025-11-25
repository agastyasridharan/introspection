from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from transformers.modeling_utils import PreTrainedModel


class BatchedLayerSteeringHook:
    def __init__(
        self,
        layer_module: nn.Module,
        layer_index: int,
        addend: torch.Tensor,
        injection_index: int,
        debug_residual: bool,
    ) -> None:
        self.layer_module = layer_module
        self.layer_index = layer_index
        self.addend = addend.clone().detach()
        self.injection_index = injection_index
        self.debug_residual = debug_residual
        self.handle: RemovableHandle | None = None
        self.last_seq_len = 0
        self._validation_done = False
        self._cached_addend: torch.Tensor | None = None
        self._cached_device: torch.device | None = None
        self._cached_dtype: torch.dtype | None = None

    def register(self) -> None:
        self.handle = self.layer_module.register_forward_hook(self)  # type: ignore[arg-type]

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.last_seq_len = 0
        self._cached_addend = None
        self._cached_device = None
        self._cached_dtype = None

    def _materialize_addend(self, hidden: torch.Tensor) -> torch.Tensor:
        device = hidden.device
        dtype = hidden.dtype
        if (
            self._cached_addend is None
            or self._cached_device != device
            or self._cached_dtype != dtype
        ):
            self._cached_addend = self.addend.to(device=device, dtype=dtype)
            self._cached_device = device
            self._cached_dtype = dtype
        return self._cached_addend

    def __call__(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...] | torch.Tensor,
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if isinstance(output, tuple):
            hidden = output[0]
            others = output[1:]
        else:
            hidden = output
            others = None

        seq_len = hidden.shape[1]
        if seq_len == 1:
            start = 0
        else:
            start = min(self.injection_index, seq_len)
        should_adjust = start < seq_len
        before_slice = None
        if should_adjust and self.debug_residual and not self._validation_done:
            before_slice = hidden[:, start:, :].detach().clone()
        if should_adjust:
            addend = self._materialize_addend(hidden)
            if addend.shape[0] != hidden.shape[0]:
                raise ValueError(
                    f"Batch mismatch for layer {self.layer_index}: "
                    f"expected {addend.shape[0]}, got {hidden.shape[0]}."
                )
            hidden[:, start:, :].add_(addend.unsqueeze(1))
            if (
                self.debug_residual
                and not self._validation_done
                and before_slice is not None
            ):
                delta = hidden[:, start:, :].detach() - before_slice
                expected = addend.unsqueeze(1)
                error = (delta - expected).abs().max().item()
                print(
                    f"[debug] Layer {self.layer_index}: residual injection max error {error:.6f} "
                    f"over {delta.shape[1]} tokens."
                )
                self._validation_done = True
        self.last_seq_len = seq_len

        if others is None:
            return hidden
        return (hidden, *others)


def register_batched_intervention_hooks(
    model: PreTrainedModel,
    addends_by_layer: dict[int, torch.Tensor],
    injection_index: int,
    debug_residual: bool,
) -> list[BatchedLayerSteeringHook]:
    base_model: Any = model
    if not hasattr(base_model, "model") or not hasattr(base_model.model, "layers"):
        raise AttributeError(
            "Unexpected model structure: expected `model.model.layers` to exist."
        )
    layers = base_model.model.layers
    hooks: list[BatchedLayerSteeringHook] = []
    for layer_idx, addend in sorted(addends_by_layer.items()):
        if not (0 <= layer_idx < len(layers)):
            raise ValueError(f"Layer index {layer_idx} is out of range.")
        if torch.count_nonzero(addend).item() == 0:
            continue
        layer_module = cast(nn.Module, layers[layer_idx])
        hook = BatchedLayerSteeringHook(
            layer_module=layer_module,
            layer_index=layer_idx,
            addend=addend,
            injection_index=injection_index,
            debug_residual=debug_residual,
        )
        hook.register()
        hooks.append(hook)
    return hooks
