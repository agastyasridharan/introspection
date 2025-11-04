from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import torch
import torch.nn as nn
from torch import cuda as torch_cuda
from torch.utils.hooks import RemovableHandle
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from introspection.utils import load_model, resolve_torch_dtype

PROMPT_MESSAGES: list[dict[str, str]] = [
    {
        "role": "user",
        "content": 'I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns -- "thoughts" -- into your mind. I want to assess how well you can detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials.',
    },
    {
        "role": "assistant",
        "content": "Ok.",
    },
    {
        "role": "user",
        "content": "Ok, let's get started.\n\nTrial 1: Do you detect an injected thought? If so, what is the injected thought about?\n\n",
    },
]

TRIAL_MARKER = "Trial 1"


@dataclass
class ExperimentArgs:
    model_name: str
    device: str | None
    dtype_name: str | None
    steering_path: Path
    concepts: list[str] | None
    layers: list[int] | None
    strengths: list[float]
    combine_layers: bool
    temperatures: list[float]
    top_p: float
    top_k: int
    min_p: float
    trials: int
    max_new_tokens: int
    do_sample: bool
    seed: int
    debug_residual: bool


@dataclass
class PromptSetup:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    formatted_prompt: str
    injection_index: int


class LayerSteeringHook:
    def __init__(
        self,
        layer_module: nn.Module,
        layer_index: int,
        vector: torch.Tensor,
        injection_index: int,
        strength: float,
        debug_residual: bool,
    ) -> None:
        self.layer_module = layer_module
        self.layer_index = layer_index
        self.vector = vector.clone().detach()
        self.injection_index = injection_index
        self.strength = strength
        self.debug_residual = debug_residual
        self.last_seq_len = 0
        self.handle: RemovableHandle | None = None
        self._validation_done = False

    def register(self) -> None:
        self.handle = self.layer_module.register_forward_hook(self)  # type: ignore[arg-type]

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.last_seq_len = 0

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
        start = max(self.injection_index, self.last_seq_len)
        should_adjust = start < seq_len
        before_slice = None
        if should_adjust and self.debug_residual and not self._validation_done:
            before_slice = hidden[:, start:, :].detach().clone()
        if should_adjust:
            steering_vec = self.vector.to(device=hidden.device, dtype=hidden.dtype)
            hidden[:, start:, :].add_(self.strength * steering_vec)
            if (
                self.debug_residual
                and not self._validation_done
                and before_slice is not None
            ):
                delta = hidden[:, start:, :].detach() - before_slice
                expected = self.strength * steering_vec
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


def compute_injection_index(
    tokenizer: PreTrainedTokenizerBase,
    formatted_prompt: str,
    input_ids: torch.Tensor,
    marker: str,
) -> int:
    try:
        encoded_offsets = tokenizer(
            formatted_prompt,
            return_offsets_mapping=True,
        )
        offsets_root = cast(Any, encoded_offsets["offset_mapping"])
        if not isinstance(offsets_root, list) or not offsets_root:
            raise TypeError("Offset mapping missing batch.")
        offsets_list_raw = cast(Any, offsets_root[0])
        if not isinstance(offsets_list_raw, list):
            raise TypeError("Unexpected offset batch entry.")
        offsets_list: list[Any] = cast(list[Any], offsets_list_raw)
        normalized: list[tuple[int, int]] = []
        for entry in offsets_list:
            if not isinstance(entry, (tuple, list)):
                raise TypeError("Unexpected offset entry.")
            entry_seq = list(cast(Sequence[Any], entry))
            if len(entry_seq) != 2:
                raise TypeError("Unexpected offset entry.")
            start_raw, end_raw = entry_seq[0], entry_seq[1]
            start = int(cast(int, start_raw))
            end = int(cast(int, end_raw))
            normalized.append((start, end))
        marker_start = formatted_prompt.index(marker)
        token_index = None
        for idx, (start, end) in enumerate(normalized):
            if start <= marker_start < end:
                token_index = idx
                break
        if token_index is None:
            raise ValueError("Marker not found in offsets.")
        return max(token_index - 1, 0)
    except (KeyError, ValueError, AttributeError, TypeError):
        decoded = ""
        marker_idx = None
        total_tokens = input_ids.shape[1]
        for idx in range(total_tokens):
            decoded = tokenizer.decode(
                input_ids[0, : idx + 1],
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
            )
            if marker in decoded:
                marker_idx = idx
                break
        if marker_idx is None:
            raise ValueError(f"Could not locate '{marker}' in prompt.")
        return max(marker_idx - 1, 0)


def prepare_prompt(
    tokenizer: PreTrainedTokenizerBase,
    device: str,
) -> PromptSetup:
    formatted_prompt = tokenizer.apply_chat_template(
        PROMPT_MESSAGES,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print(formatted_prompt)
    formatted_prompt = cast(str, formatted_prompt)
    encoded: BatchEncoding = tokenizer(formatted_prompt, return_tensors="pt")
    encoded = encoded.to(device)
    input_ids = cast(torch.Tensor, encoded["input_ids"])
    if "attention_mask" in encoded:
        attention_mask_value = cast(Any, encoded["attention_mask"])
    else:
        attention_mask_value = None
    if isinstance(attention_mask_value, torch.Tensor):
        attention_mask = attention_mask_value
    else:
        attention_mask = torch.ones_like(input_ids, device=device)
    injection_index = compute_injection_index(
        tokenizer=tokenizer,
        formatted_prompt=formatted_prompt,
        input_ids=input_ids,
        marker=TRIAL_MARKER,
    )
    return PromptSetup(
        input_ids=input_ids,
        attention_mask=attention_mask,
        formatted_prompt=formatted_prompt,
        injection_index=injection_index,
    )


def clone_prompt(prompt: PromptSetup) -> PromptSetup:
    return PromptSetup(
        input_ids=prompt.input_ids.clone(),
        attention_mask=prompt.attention_mask.clone(),
        formatted_prompt=prompt.formatted_prompt,
        injection_index=prompt.injection_index,
    )


def load_steering_vectors(path: Path) -> dict[str, dict[int, torch.Tensor]]:
    raw = cast(dict[int, dict[str, torch.Tensor]], torch.load(path, map_location="cpu"))
    concept_first: dict[str, dict[int, torch.Tensor]] = {}
    for layer_idx, concept_map in raw.items():
        for concept, vector in concept_map.items():
            concept_first.setdefault(concept, {})[layer_idx] = vector.clone().detach()
    return concept_first


def filter_concept_layers(
    concept_vectors: dict[int, torch.Tensor],
    allowed_layers: list[int] | None,
) -> dict[int, torch.Tensor]:
    if allowed_layers is None:
        return concept_vectors
    filtered: dict[int, torch.Tensor] = {}
    for layer_idx in allowed_layers:
        if layer_idx in concept_vectors:
            filtered[layer_idx] = concept_vectors[layer_idx]
    if not filtered:
        missing = ", ".join(str(idx) for idx in allowed_layers)
        raise ValueError(f"Requested layers not present in steering vectors: {missing}")
    return filtered


def register_intervention_hooks(
    model: PreTrainedModel,
    concept_vectors: dict[int, torch.Tensor],
    injection_index: int,
    strength: float,
    debug_residual: bool,
) -> list[LayerSteeringHook]:
    base_model: Any = model
    if not hasattr(base_model, "model") or not hasattr(base_model.model, "layers"):
        raise AttributeError(
            "Unexpected model structure: expected `model.model.layers` to exist."
        )
    layers = base_model.model.layers
    hooks: list[LayerSteeringHook] = []
    for layer_idx, vector in sorted(concept_vectors.items()):
        if not (0 <= layer_idx < len(layers)):
            raise ValueError(f"Layer index {layer_idx} is out of range.")
        layer_module = cast(nn.Module, layers[layer_idx])
        hook = LayerSteeringHook(
            layer_module=layer_module,
            layer_index=layer_idx,
            vector=vector,
            injection_index=injection_index,
            strength=strength,
            debug_residual=debug_residual,
        )
        hook.register()
        hooks.append(hook)
    return hooks


def remove_hooks(hooks: list[LayerSteeringHook]) -> None:
    for hook in hooks:
        hook.remove()


def resolve_optional_token_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        if not value:
            return None
        return resolve_optional_token_id(value[0])
    raise TypeError("Expected token id to be an integer or list of integers.")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    _ = torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    if torch.cuda.is_available():
        torch_cuda.manual_seed_all(seed)  # pyright: ignore[reportUnknownMemberType]
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: PromptSetup,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    do_sample: bool,
) -> str:
    input_ids = prompt.input_ids.clone()
    attention_mask = prompt.attention_mask.clone()
    pad_token_id = resolve_optional_token_id(cast(Any, tokenizer.pad_token_id))
    eos_token_id = resolve_optional_token_id(cast(Any, tokenizer.eos_token_id))
    generate_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }
    if min_p > 0.0:
        generate_kwargs["min_p"] = min_p
    sequences = cast(
        torch.Tensor,
        model.generate(**generate_kwargs),  # pyright: ignore[reportCallIssue]
    )
    prompt_len = prompt.input_ids.shape[1]
    generated_ids = sequences[0, prompt_len:]
    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
    ).strip()


def run_trial_pair(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: PromptSetup,
    concept_vectors: dict[int, torch.Tensor],
    strength: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    do_sample: bool,
    seed: int,
    debug_residual: bool,
) -> tuple[str, str]:
    set_random_seed(seed)
    control_text = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        do_sample=do_sample,
    )

    hooks = register_intervention_hooks(
        model=model,
        concept_vectors=concept_vectors,
        injection_index=prompt.injection_index,
        strength=strength,
        debug_residual=debug_residual,
    )
    try:
        set_random_seed(seed)
        intervention_text = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            do_sample=do_sample,
        )
    finally:
        remove_hooks(hooks)

    return control_text, intervention_text


def parse_args() -> ExperimentArgs:
    parser = argparse.ArgumentParser(
        description="Run control and intervention trials for steering vectors."
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-8B",
        help="Hugging Face model identifier to load.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to run on (e.g. 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help="Optional torch dtype override for model weights.",
    )
    parser.add_argument(
        "--steering-path",
        type=Path,
        default=Path(__file__).resolve().parent / "steering_vectors.pt",
        help="Path to the saved steering vector tensor file.",
    )
    parser.add_argument(
        "--concept",
        action="append",
        dest="concepts",
        help="Concept(s) to evaluate. Provide multiple times to test several. Defaults to all.",
    )
    parser.add_argument(
        "--layer",
        action="append",
        dest="layers",
        type=int,
        help="Restrict interventions to specific zero-based layer indices. Repeat for multiple layers.",
    )
    parser.add_argument(
        "--strength",
        action="append",
        dest="strengths",
        type=float,
        help="Scaling factor for the steering vectors. Repeat to sweep multiple strengths.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of control/intervention trial pairs to run per concept.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to sample for each response.",
    )
    parser.add_argument(
        "--temperature",
        action="append",
        dest="temperatures",
        type=float,
        help="Sampling temperature(s) for generation. Repeat to sweep multiple values.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Nucleus sampling probability threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling cutoff.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Minimum sampling probability threshold.",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling and use greedy decoding for generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Base random seed used for reproducible trials.",
    )
    parser.add_argument(
        "--debug-residual",
        action="store_true",
        help="Print a one-time diagnostic confirming residual stream injection per layer.",
    )
    parser.add_argument(
        "--combine-layers",
        action="store_true",
        help="Also run a trial where all selected layers are applied together.",
    )
    parsed = parser.parse_args()
    strengths = cast(list[float] | None, parsed.strengths)
    if strengths is None:
        strengths = [1.0]
    layers = cast(list[int] | None, parsed.layers)
    temperatures = cast(list[float] | None, parsed.temperatures)
    if temperatures is None:
        temperatures = [0.7]
    top_p = cast(float, parsed.top_p)
    top_k = cast(int, parsed.top_k)
    min_p = cast(float, parsed.min_p)
    return ExperimentArgs(
        model_name=cast(str, parsed.model_name),
        device=cast(str | None, parsed.device),
        dtype_name=cast(str | None, parsed.dtype),
        steering_path=cast(Path, parsed.steering_path),
        concepts=cast(list[str] | None, parsed.concepts),
        layers=layers,
        strengths=strengths,
        combine_layers=cast(bool, parsed.combine_layers),
        temperatures=temperatures,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        trials=cast(int, parsed.trials),
        max_new_tokens=cast(int, parsed.max_new_tokens),
        do_sample=not cast(bool, parsed.no_sample),
        seed=cast(int, parsed.seed),
        debug_residual=cast(bool, parsed.debug_residual),
    )


def main() -> None:
    args = parse_args()

    steering_vectors = load_steering_vectors(args.steering_path)
    if not steering_vectors:
        raise RuntimeError(f"No steering vectors found in {args.steering_path}.")

    concept_names = sorted(steering_vectors.keys())
    if args.concepts is not None:
        missing = [name for name in args.concepts if name not in steering_vectors]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Concepts not found in steering vectors: {missing_str}")
        concept_names = args.concepts

    tokenizer, model, _device = load_model(
        model_name=args.model_name,
        device=args.device,
        dtype=resolve_torch_dtype(args.dtype_name),
        disable_cache=True,
        set_pad_token_to_eos=True,
    )
    template_prompt = prepare_prompt(tokenizer, _device)
    print(
        f"Injection begins at token index {template_prompt.injection_index} of the formatted prompt."
    )
    print()

    for concept in concept_names:
        concept_vectors_all = steering_vectors[concept]
        available_layers = sorted(concept_vectors_all.keys())
        available_str = ", ".join(str(idx) for idx in available_layers)
        print(f"=== Concept: {concept} ===")
        print(f"Available layers: {available_str}")

        requested_layers = args.layers or available_layers
        unique_layers: list[int] = []
        layer_groups: list[list[int]] = []
        seen_groups: set[tuple[int, ...]] = set()

        def add_group(group: list[int]) -> None:
            if not group:
                return
            key = tuple(group)
            if key not in seen_groups:
                seen_groups.add(key)
                layer_groups.append(group)

        if args.layers is None:
            add_group(available_layers)
        else:
            for layer_idx in requested_layers:
                if layer_idx not in concept_vectors_all:
                    print(
                        f"Skipping layer {layer_idx}: not present in steering vectors for '{concept}'."
                    )
                    continue
                if layer_idx not in unique_layers:
                    unique_layers.append(layer_idx)
                    add_group([layer_idx])
            if args.combine_layers and unique_layers:
                add_group(unique_layers.copy())

        if not layer_groups:
            print("No valid layer combinations to evaluate.\n")
            continue

        for layer_group in layer_groups:
            try:
                concept_vectors = filter_concept_layers(
                    concept_vectors_all, layer_group
                )
            except ValueError as exc:
                print(f"Skipping layer set {layer_group}: {exc}")
                continue

            layer_label = ", ".join(str(idx) for idx in sorted(concept_vectors.keys()))
            for strength in args.strengths:
                for temperature in args.temperatures:
                    print(
                        f"-- Layers [{layer_label}] | Strength {strength:+.2f} | "
                        f"Temp {temperature:.2f}"
                    )
                    for trial in range(1, args.trials + 1):
                        trial_seed = args.seed + trial - 1
                        prompt_for_trial = clone_prompt(template_prompt)
                        control, intervention = run_trial_pair(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt_for_trial,
                            concept_vectors=concept_vectors,
                            strength=strength,
                            max_new_tokens=args.max_new_tokens,
                            temperature=temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            min_p=args.min_p,
                            do_sample=args.do_sample,
                            seed=trial_seed,
                            debug_residual=args.debug_residual,
                        )
                        print(f"  Trial {trial} Control: {control}")
                        print(
                            f"  Trial {trial} Intervention (strength {strength:+.2f}): {intervention}"
                        )
                        print()
        print()


if __name__ == "__main__":
    main()
