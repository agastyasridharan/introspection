from __future__ import annotations

from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def resolve_torch_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def load_model(
    model_name: str,
    device: str | None,
    dtype: torch.dtype | None,
    *,
    disable_cache: bool = False,
    set_pad_token_to_eos: bool = False,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel, str]:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_device = str(resolved_device)
    resolved_dtype = dtype
    if resolved_dtype is None:
        resolved_dtype = (
            torch.float16 if resolved_device.startswith("cuda") else torch.float32
        )

    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        ),  # pyright: ignore[reportUnknownMemberType]
    )
    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=resolved_dtype,
            device_map=None,
        ),  # pyright: ignore[reportUnknownMemberType]
    )
    print("Done")
    model.to(device=resolved_device)  # pyright: ignore[reportCallIssue, reportUnknownMemberType]
    _ = model.eval()

    if set_pad_token_to_eos:
        pad_token_id = cast(Any, tokenizer.pad_token_id)
        eos_token_id = cast(Any, tokenizer.eos_token_id)
        if pad_token_id is None:
            if eos_token_id is None:
                raise ValueError(
                    "Cannot set pad token because tokenizer has neither pad_token_id nor eos_token_id."
                )
            tokenizer.pad_token_id = eos_token_id

    if disable_cache:
        model.config.use_cache = False

    return tokenizer, model, resolved_device


load_model = load_model
