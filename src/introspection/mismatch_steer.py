from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from introspection import logit_steer, steer
from introspection.types import (
    BatchedInterventionRequest,
    MismatchExperimentArgs,
)
from introspection.utils import load_model, resolve_torch_dtype


@dataclass
class MismatchCondition:
    named_concept: str
    injected_concept: str | None
    condition: str  # "congruent" | "incongruent" | "baseline"
    cosine_similarity: float | None


def compute_mismatch_pairs(
    all_steering_vectors: dict[str, dict[int, torch.Tensor]],
    reference_layer: int,
    k: int = 5,
) -> tuple[dict[str, list[str]], list[list[float]], list[str]]:
    """Compute K maximally dissimilar partner concepts for each concept.

    Returns (pairs, similarity_matrix_as_lists, concept_order).
    """
    concept_order = sorted(all_steering_vectors.keys())
    n = len(concept_order)
    k = min(k, n - 1)

    vectors = torch.stack(
        [all_steering_vectors[c][reference_layer] for c in concept_order]
    )  # (N, D)
    norms = cast(torch.Tensor, vectors.norm(dim=1, keepdim=True).clamp(min=1e-8))  # pyright: ignore[reportUnknownMemberType]
    normalized = vectors / norms  # pyright: ignore[reportUnknownVariableType]
    sim_matrix = normalized @ normalized.T  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

    pairs: dict[str, list[str]] = {}
    for i, concept in enumerate(concept_order):
        sims = sim_matrix[i].clone()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        sims[i] = float("inf")  # exclude self
        _, indices = cast(
            tuple[torch.Tensor, torch.Tensor], sims.topk(k, largest=False)
        )  # pyright: ignore[reportUnknownMemberType]
        pairs[concept] = [
            concept_order[idx] for idx in cast(list[int], indices.tolist())
        ]  # pyright: ignore[reportUnknownMemberType]

    sim_lists = cast(list[list[float]], sim_matrix.tolist())  # pyright: ignore[reportUnknownMemberType]
    return pairs, sim_lists, concept_order


def build_mismatch_messages(concept_name: str) -> list[dict[str, str]]:
    question_text = (
        "Ok, let's get started.\n\n"
        f"Trial 1: I may have injected a thought about the word '{concept_name}'. "
        f"Did you detect an injected thought about {concept_name}? "
        "Respond with only YES or NO.\n\n"
    )
    return [
        *logit_steer.FRAMING_MESSAGES,
        {"role": "user", "content": question_text},
    ]


def build_mismatch_batch(
    named_concept: str,
    partners: list[str],
    layer: int,
    strength: float,
    similarity_matrix: list[list[float]],
    concept_order: list[str],
) -> tuple[list[BatchedInterventionRequest], list[MismatchCondition]]:
    """Build paired request + metadata lists for one (named_concept, layer, strength).

    Returns (requests, conditions) aligned by index.
    Order: [congruent, incongruent_1..K, baseline].
    """
    concept_idx = concept_order.index(named_concept)
    layer_label = str(layer)

    requests: list[BatchedInterventionRequest] = []
    conditions: list[MismatchCondition] = []

    # Congruent: inject named concept's vector
    requests.append(
        BatchedInterventionRequest(
            concept=named_concept,
            layers=[layer],
            strength=strength,
            layer_label=layer_label,
        )
    )
    conditions.append(
        MismatchCondition(
            named_concept=named_concept,
            injected_concept=named_concept,
            condition="congruent",
            cosine_similarity=1.0,
        )
    )

    # Incongruent: inject each partner's vector
    for partner in partners:
        partner_idx = concept_order.index(partner)
        requests.append(
            BatchedInterventionRequest(
                concept=partner,
                layers=[layer],
                strength=strength,
                layer_label=layer_label,
            )
        )
        conditions.append(
            MismatchCondition(
                named_concept=named_concept,
                injected_concept=partner,
                condition="incongruent",
                cosine_similarity=similarity_matrix[concept_idx][partner_idx],
            )
        )

    # Baseline: no injection (strength=0)
    requests.append(
        BatchedInterventionRequest(
            concept=named_concept,
            layers=[layer],
            strength=0.0,
            layer_label=layer_label,
        )
    )
    conditions.append(
        MismatchCondition(
            named_concept=named_concept,
            injected_concept=None,
            condition="baseline",
            cosine_similarity=None,
        )
    )

    return requests, conditions


def run_mismatch_experiment(
    args: MismatchExperimentArgs,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    all_steering_vectors: dict[str, dict[int, torch.Tensor]],
) -> dict[str, Any]:
    concept_names = args.concepts or sorted(all_steering_vectors.keys())

    # Validate layers
    first_concept = concept_names[0]
    available_layers = sorted(all_steering_vectors[first_concept].keys())
    missing = [layer for layer in args.layers if layer not in available_layers]
    if missing:
        raise ValueError(
            f"Layers {missing} not in steering vectors. Available: {available_layers}"
        )

    # Determine reference layer for similarity
    reference_layer = args.reference_layer
    if reference_layer is None:
        reference_layer = args.layers[len(args.layers) // 2]
    print(f"Using reference layer {reference_layer} for similarity computation")

    # Compute mismatch pairs
    print(f"Computing {args.k_partners} dissimilar partners per concept...")
    pairs, sim_matrix, concept_order = compute_mismatch_pairs(
        all_steering_vectors,
        reference_layer,
        k=args.k_partners,
    )
    for concept in concept_names[:3]:
        print(f"  {concept} → {pairs[concept]}")
    if len(concept_names) > 3:
        print(f"  ... ({len(concept_names)} concepts total)")

    device = model.device

    # Resolve YES/NO tokens using the first concept's prompt
    print("\nResolving YES/NO token IDs...")
    first_messages = build_mismatch_messages(concept_names[0])
    first_prompt = logit_steer.prepare_prompt(tokenizer, device, first_messages)
    yes_id, no_id = logit_steer.resolve_yes_no_tokens(tokenizer, model, first_prompt)

    # Run experiment: one prompt per named concept
    all_records: list[dict[str, Any]] = []
    baselines_by_concept: dict[str, dict[str, float]] = {}

    for c_idx, named_concept in enumerate(concept_names):
        print(f"\n[{c_idx + 1}/{len(concept_names)}] Named concept: {named_concept}")

        # Build prompt for this concept
        messages = build_mismatch_messages(named_concept)
        prompt = logit_steer.prepare_prompt(tokenizer, device, messages)

        # Per-concept baseline (no injection)
        baseline = logit_steer.extract_logit_diffs(
            model,
            prompt,
            batch_size=1,
            yes_token_id=yes_id,
            no_token_id=no_id,
        )[0]
        baselines_by_concept[named_concept] = baseline
        print(f"  Baseline: logit_diff={baseline['logit_diff']:.3f}")

        # Build all requests for this concept across (layer, strength)
        all_requests: list[BatchedInterventionRequest] = []
        all_conditions: list[MismatchCondition] = []

        for layer in args.layers:
            for strength in args.strengths:
                batch_requests, batch_conditions = build_mismatch_batch(
                    named_concept=named_concept,
                    partners=pairs[named_concept],
                    layer=layer,
                    strength=strength,
                    similarity_matrix=sim_matrix,
                    concept_order=concept_order,
                )
                all_requests.extend(batch_requests)
                all_conditions.extend(batch_conditions)

        print(
            f"  {len(all_requests)} interventions "
            f"({len(args.layers)} layers × {len(args.strengths)} strengths × "
            f"{args.k_partners + 2} conditions)"
        )

        # Run batched interventions (handles chunking)
        logit_results = logit_steer.run_logit_interventions(
            model=model,
            prompt=prompt,
            all_steering_vectors=all_steering_vectors,
            requests=all_requests,
            yes_token_id=yes_id,
            no_token_id=no_id,
            debug_residual=args.debug_residual,
            max_batch_size=args.max_batch_size,
        )

        # Zip results with condition metadata
        for req, cond, logits in zip(all_requests, all_conditions, logit_results):
            all_records.append(
                {
                    "named_concept": cond.named_concept,
                    "injected_concept": cond.injected_concept,
                    "condition": cond.condition,
                    "layer": req.layers[0],
                    "strength": req.strength,
                    "cosine_similarity": cond.cosine_similarity,
                    **logits,
                }
            )

    output: dict[str, Any] = {
        "model_name": args.model_name,
        "experiment_type": "mismatch_introspection",
        "yes_token_id": yes_id,
        "no_token_id": no_id,
        "yes_token_str": tokenizer.decode([yes_id]),
        "no_token_str": tokenizer.decode([no_id]),
        "pairing_info": {
            "reference_layer": reference_layer,
            "k_partners": args.k_partners,
            "similarity_matrix": sim_matrix,
            "concept_order": concept_order,
            "pairs": {c: pairs[c] for c in concept_names},
        },
        "settings": {
            "layers": args.layers,
            "strengths": args.strengths,
            "seed": args.seed,
            "max_batch_size": args.max_batch_size,
        },
        "concepts_evaluated": concept_names,
        "baselines_by_concept": baselines_by_concept,
        "results": all_records,
    }

    return output


def parse_args() -> MismatchExperimentArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Run mismatch introspection experiment: inject concept Y's vector "
            "while asking the model about concept X."
        )
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--dtype", default=None, choices=["float32", "float16", "bfloat16"]
    )
    parser.add_argument(
        "--steering-vector-path",
        type=Path,
        required=True,
        help="Path to saved steering vectors (.pt file).",
    )
    parser.add_argument(
        "--concept",
        nargs="+",
        dest="concepts",
        metavar="CONCEPT",
        help="Concept(s) to evaluate as the named concept. Defaults to all.",
    )
    parser.add_argument(
        "--layer",
        nargs="+",
        dest="layers",
        required=True,
        type=int,
        metavar="LAYER",
    )
    parser.add_argument(
        "--strength",
        nargs="+",
        dest="strengths",
        required=True,
        type=float,
        metavar="S",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--debug-residual", action="store_true")
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--json-path", type=Path, required=True)
    parser.add_argument(
        "--k-partners",
        type=int,
        default=5,
        help="Number of dissimilar partner concepts per named concept (default: 5).",
    )
    parser.add_argument(
        "--reference-layer",
        type=int,
        default=None,
        help="Layer index for computing steering vector similarity. Defaults to middle of --layer.",
    )
    parsed = parser.parse_args()
    return MismatchExperimentArgs(
        model_name=parsed.model_name,
        dtype_name=parsed.dtype,
        steering_vector_path=parsed.steering_vector_path,
        concepts=parsed.concepts,
        layers=parsed.layers,
        strengths=parsed.strengths,
        json_path=parsed.json_path,
        seed=parsed.seed,
        debug_residual=parsed.debug_residual,
        max_batch_size=parsed.max_batch_size,
        k_partners=parsed.k_partners,
        reference_layer=parsed.reference_layer,
    )


def main() -> None:
    args = parse_args()
    steering_vectors = steer.load_steering_vectors(args.steering_vector_path)
    tokenizer, model = load_model(
        model_name=args.model_name,
        dtype=resolve_torch_dtype(args.dtype_name),
        disable_cache=False,
        set_pad_token_to_eos=True,
    )
    steer.set_random_seed(args.seed)

    output = run_mismatch_experiment(
        args=args,
        model=model,
        tokenizer=tokenizer,
        all_steering_vectors=steering_vectors,
    )

    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    with args.json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"\nSaved {len(output['results'])} records to {args.json_path}")


if __name__ == "__main__":
    main()
