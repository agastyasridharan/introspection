from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from introspection.constants import BASELINE_WORDS, CONCEPT_NOUNS
from introspection.utils import load_model, resolve_torch_dtype

PROMPT = """Tell me about {concept}."""


def generate_concepts(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    sample = rng.sample(list(CONCEPT_NOUNS), count)
    sample.sort()
    return sample


def capture_prompt_hidden_states(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    device: str,
) -> list[torch.Tensor]:
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    formatted_prompt = cast(
        str,
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        ),
    )
    encoded: BatchEncoding = tokenizer(formatted_prompt, return_tensors="pt")
    encoded = encoded.to(device)
    input_ids = cast(torch.Tensor, encoded["input_ids"])
    with torch.no_grad():
        outputs = cast(
            CausalLMOutputWithPast,
            model(**encoded, output_hidden_states=True, use_cache=False),
        )

    hidden_states: Sequence[torch.Tensor] | None = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError(
            "Model did not return hidden states. Ensure it supports `output_hidden_states=True`."
        )

    token_index = input_ids.shape[1] - 1
    layer_vectors: list[torch.Tensor] = []

    # Skip the embedding layer at index 0, capture transformer block outputs.
    for layer_hidden in hidden_states[1:]:
        vector = layer_hidden[0, token_index, :].detach().cpu().to(torch.float32)
        layer_vectors.append(vector)
    return layer_vectors


def collect_layer_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    nouns: Sequence[str],
    device: str,
    description: str,
) -> list[list[torch.Tensor]]:
    layer_activations: list[list[torch.Tensor]] = []
    total = len(nouns)

    for idx, noun in enumerate(nouns):
        prompt = PROMPT.format(concept=noun)
        vectors = capture_prompt_hidden_states(model, tokenizer, prompt, device)
        if not layer_activations:
            layer_activations = [[vector] for vector in vectors]
        else:
            for layer_idx, vector in enumerate(vectors):
                layer_activations[layer_idx].append(vector)
        print(f"[{idx + 1}/{total}] {description} '{noun}'.")
    return layer_activations


def compute_baseline_means(
    layer_activations: list[list[torch.Tensor]],
) -> list[torch.Tensor]:
    means: list[torch.Tensor] = []
    for layer_vectors in layer_activations:
        stacked = torch.stack(layer_vectors, dim=0)
        means.append(stacked.mean(dim=0))
    return means


def compute_steering_vectors(
    layer_activations: list[list[torch.Tensor]],
    concepts: Sequence[str],
    baseline_means: Sequence[torch.Tensor],
) -> dict[int, dict[str, torch.Tensor]]:
    steering: dict[int, dict[str, torch.Tensor]] = {}

    for layer_idx, vectors in enumerate(layer_activations):
        baseline_mean = baseline_means[layer_idx]
        layer_dict: dict[str, torch.Tensor] = {}
        for concept_idx, vector in enumerate(vectors):
            centered_vector = vector - baseline_mean
            layer_dict[concepts[concept_idx]] = centered_vector
        steering[layer_idx] = layer_dict
    return steering


def validate_steering_vectors(steering: dict[int, dict[str, torch.Tensor]]) -> None:
    issues: list[str] = []
    for layer_idx, concept_map in steering.items():
        for concept, vector in concept_map.items():
            if vector.isnan().any().item():
                issues.append(
                    f"Layer {layer_idx} concept '{concept}' contains NaN values."
                )
            if torch.isinf(vector).any().item():
                issues.append(
                    f"Layer {layer_idx} concept '{concept}' contains Inf values."
                )
    if issues:
        preview = "\n".join(issues[:5])
        suffix = "" if len(issues) <= 5 else f"\n... {len(issues) - 5} more entries."
        raise ValueError(
            "Steering vectors contain invalid values and will not be saved:"
            f"\n{preview}{suffix}"
        )


def run_experiment(
    model_name: str,
    device: str | None,
    dtype_name: str | None,
    output_path: Path,
    concept_count: int,
    seed: int,
) -> None:
    concepts = generate_concepts(concept_count, seed)

    tokenizer, model, resolved_device = load_model(
        model_name=model_name,
        device=device,
        dtype=resolve_torch_dtype(dtype_name),
    )

    baseline_activations = collect_layer_activations(
        model=model,
        tokenizer=tokenizer,
        nouns=BASELINE_WORDS,
        device=resolved_device,
        description="Captured baseline activations for",
    )
    baseline_means = compute_baseline_means(baseline_activations)

    concept_activations = collect_layer_activations(
        model=model,
        tokenizer=tokenizer,
        nouns=concepts,
        device=resolved_device,
        description="Captured concept activations for",
    )

    steering = compute_steering_vectors(
        concept_activations,
        concepts,
        baseline_means,
    )

    validate_steering_vectors(steering)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(steering, output_path)
    print(f"Saved steering vectors to {output_path}.")


@dataclass
class ExperimentArgs:
    model_name: str
    device: str | None
    dtype_name: str | None
    concept_count: int
    output_path: Path
    seed: int


def parse_args() -> ExperimentArgs:
    parser = argparse.ArgumentParser(
        description="Generate steering vectors for concepts."
    )
    _ = parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-8B",
        help="Hugging Face model identifier to load.",
    )
    _ = parser.add_argument(
        "--device",
        default=None,
        help="Torch device string to run on (e.g. 'cuda', 'cuda:1', 'cpu'). Defaults based on availability.",
    )
    _ = parser.add_argument(
        "--dtype",
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help="Optional torch dtype override for model weights.",
    )
    _ = parser.add_argument(
        "--concept-count",
        type=int,
        default=50,
        help="Number of concepts to sample for the experiment.",
    )
    _ = parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "steering_vectors.pt",
        help="Path to save the serialized steering vectors.",
    )
    _ = parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used for concept sampling.",
    )
    parsed = parser.parse_args()
    return ExperimentArgs(
        model_name=cast(str, parsed.model_name),
        device=cast(str | None, parsed.device),
        dtype_name=cast(str | None, parsed.dtype),
        concept_count=cast(int, parsed.concept_count),
        output_path=cast(Path, parsed.output_path),
        seed=cast(int, parsed.seed),
    )


def main():
    args = parse_args()
    run_experiment(
        model_name=args.model_name,
        device=args.device,
        dtype_name=args.dtype_name,
        output_path=args.output_path,
        concept_count=args.concept_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
