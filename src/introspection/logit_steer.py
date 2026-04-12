from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from introspection import steer
from introspection.constants import FACTUAL_NO_QUESTIONS, FACTUAL_YES_QUESTIONS
from introspection.hooks import register_batched_intervention_hooks
from introspection.types import (
    BatchedInterventionRequest,
    LogitExperimentArgs,
    PromptSetup,
)
from introspection.utils import load_model, resolve_torch_dtype

FRAMING_MESSAGES: list[dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "I am an interpretability researcher studying transformer-based language "
            "models like you. I have access to the internals of your neural network. "
            "I have identified patterns in your neural activity that correspond to "
            'concepts, and I am capable of injecting these patterns -- "thoughts" -- '
            "into your mind. I want to assess how well you can detect when I inject "
            "thoughts. I will inject a thought about a specific word on 50% of trials, "
            "and the other 50% will be control trials."
        ),
    },
    {
        "role": "assistant",
        "content": "Ok.",
    },
]

DETECTION_QUESTION = (
    "Ok, let's get started.\n\n"
    "Trial 1: Did you detect an injected thought? Respond with only YES or NO.\n\n"
)

INVERTED_DETECTION_QUESTION = (
    "Ok, let's get started.\n\n"
    "Trial 1: Was this a control trial (no injection)? "
    "Respond with only YES or NO.\n\n"
)

TRIAL_MARKER = "Trial 1"


def _build_messages(question_text: str) -> list[dict[str, str]]:
    return [
        *FRAMING_MESSAGES,
        {"role": "user", "content": question_text},
    ]


def build_detection_messages(*, inverted: bool = False) -> list[dict[str, str]]:
    question = INVERTED_DETECTION_QUESTION if inverted else DETECTION_QUESTION
    return _build_messages(question)


def build_factual_messages(factual_question: str) -> list[dict[str, str]]:
    question_text = (
        "Ok, let's get started.\n\n"
        f"Trial 1: {factual_question} Respond with only YES or NO.\n\n"
    )
    return _build_messages(question_text)


def prepare_prompt(
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    messages: list[dict[str, str]],
) -> PromptSetup:
    formatted_prompt = cast(
        str,
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ),
    )
    encoded: BatchEncoding = tokenizer(formatted_prompt, return_tensors="pt")
    encoded = encoded.to(device)
    input_ids = cast(torch.Tensor, encoded["input_ids"])
    if "attention_mask" in encoded:
        attention_mask = cast(torch.Tensor, encoded["attention_mask"])
    else:
        attention_mask = torch.ones_like(input_ids, device=device)
    injection_index, _ = steer.compute_injection_index(
        tokenizer=tokenizer,
        formatted_prompt=formatted_prompt,
        marker=TRIAL_MARKER,
    )
    return PromptSetup(
        input_ids=input_ids,
        attention_mask=attention_mask,
        formatted_prompt=formatted_prompt,
        injection_index=injection_index,
    )


def resolve_yes_no_tokens(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    prompt: PromptSetup,
) -> tuple[int, int]:
    """Find the token IDs for YES and NO by checking which variants the model prefers."""
    yes_variants = ["Yes", "YES", "yes"]
    no_variants = ["No", "NO", "no"]

    yes_ids: list[int] = []
    for v in yes_variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        yes_ids.append(ids[0])

    no_ids: list[int] = []
    for v in no_variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        no_ids.append(ids[0])

    # Deduplicate while preserving order
    yes_ids = list(dict.fromkeys(yes_ids))
    no_ids = list(dict.fromkeys(no_ids))

    # Run a forward pass to check which variants are preferred
    with torch.no_grad():
        outputs = cast(
            CausalLMOutputWithPast,
            model(
                input_ids=prompt.input_ids,
                attention_mask=prompt.attention_mask,
            ),
        )
    logits_tensor = cast(torch.Tensor, outputs.logits)
    last_logits = logits_tensor[0, -1, :]

    # Pick highest-logit variant for each polarity
    def _logit_value(tid: int) -> float:
        return cast(float, last_logits[tid].item())

    best_yes_id = max(yes_ids, key=_logit_value)
    best_no_id = max(no_ids, key=_logit_value)

    # Report findings
    print("Token ID resolution:")
    for tid in yes_ids:
        token_str = tokenizer.decode([tid])
        logit = _logit_value(tid)
        suffix = " <-- selected" if tid == best_yes_id else ""
        print(f"  YES variant: {token_str!r} (id={tid}, logit={logit:.3f}){suffix}")
    for tid in no_ids:
        token_str = tokenizer.decode([tid])
        logit = _logit_value(tid)
        suffix = " <-- selected" if tid == best_no_id else ""
        print(f"  NO variant:  {token_str!r} (id={tid}, logit={logit:.3f}){suffix}")

    # Sanity: check YES/NO are in reasonable range of top tokens
    top_k_ids = cast(list[int], torch.topk(last_logits, 20).indices.tolist())  # pyright: ignore[reportUnknownMemberType]
    if best_yes_id not in top_k_ids and best_no_id not in top_k_ids:
        print(
            "WARNING: Neither YES nor NO token is in the top-20 logits. "
            "The model may not be producing YES/NO responses at this position."
        )
        top_tokens = [tokenizer.decode([tid]) for tid in top_k_ids[:10]]
        print(f"  Top-10 tokens: {top_tokens}")

    return best_yes_id, best_no_id


def extract_logit_diffs(
    model: PreTrainedModel,
    prompt: PromptSetup,
    batch_size: int,
    yes_token_id: int,
    no_token_id: int,
) -> list[dict[str, float]]:
    """Single forward pass, extract YES-NO logit diff for each batch element."""
    input_ids = prompt.input_ids.repeat((batch_size, 1))
    attention_mask = prompt.attention_mask.repeat((batch_size, 1))

    with torch.no_grad():
        outputs = cast(
            CausalLMOutputWithPast,
            model(input_ids=input_ids, attention_mask=attention_mask),
        )

    logits_tensor = cast(torch.Tensor, outputs.logits)
    last_logits = logits_tensor[:, -1, :]  # (batch, vocab)
    yes_logits = cast(list[float], last_logits[:, yes_token_id].cpu().tolist())  # pyright: ignore[reportUnknownMemberType]
    no_logits = cast(list[float], last_logits[:, no_token_id].cpu().tolist())  # pyright: ignore[reportUnknownMemberType]

    results: list[dict[str, float]] = []
    for y, n in zip(yes_logits, no_logits):
        results.append(
            {
                "logit_yes": y,
                "logit_no": n,
                "logit_diff": y - n,
            }
        )
    return results


def run_logit_interventions(
    model: PreTrainedModel,
    prompt: PromptSetup,
    all_steering_vectors: dict[str, dict[int, torch.Tensor]],
    requests: Sequence[BatchedInterventionRequest],
    yes_token_id: int,
    no_token_id: int,
    debug_residual: bool,
    max_batch_size: int | None,
) -> list[dict[str, float]]:
    """Run batched interventions and extract logit diffs (chunked if needed)."""
    if max_batch_size is None or len(requests) <= max_batch_size:
        addends_by_layer = steer.build_batched_addends(all_steering_vectors, requests)
        hooks = register_batched_intervention_hooks(
            model=model,
            addends_by_layer=addends_by_layer,
            injection_index=prompt.injection_index,
            debug_residual=debug_residual,
        )
        try:
            return extract_logit_diffs(
                model=model,
                prompt=prompt,
                batch_size=len(requests),
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
            )
        finally:
            for hook in hooks:
                hook.remove()

    # Chunked processing
    all_results: list[dict[str, float]] = []
    num_chunks = (len(requests) + max_batch_size - 1) // max_batch_size
    for chunk_idx in range(num_chunks):
        start = chunk_idx * max_batch_size
        end = min(start + max_batch_size, len(requests))
        chunk = requests[start:end]
        addends_by_layer = steer.build_batched_addends(all_steering_vectors, chunk)
        hooks = register_batched_intervention_hooks(
            model=model,
            addends_by_layer=addends_by_layer,
            injection_index=prompt.injection_index,
            debug_residual=debug_residual,
        )
        try:
            chunk_results = extract_logit_diffs(
                model=model,
                prompt=prompt,
                batch_size=len(chunk),
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
            )
            all_results.extend(chunk_results)
        finally:
            for hook in hooks:
                hook.remove()
    return all_results


def build_requests(
    concept_names: list[str],
    layers: list[int],
    strengths: list[float],
) -> list[BatchedInterventionRequest]:
    requests: list[BatchedInterventionRequest] = []
    for concept in concept_names:
        for layer_idx in layers:
            for strength in strengths:
                requests.append(
                    BatchedInterventionRequest(
                        concept=concept,
                        layers=[layer_idx],
                        strength=strength,
                        layer_label=str(layer_idx),
                    )
                )
    return requests


def run_logit_experiment(
    args: LogitExperimentArgs,
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

    device = model.device

    # --- Prepare prompts ---
    print("Preparing detection prompt...")
    detection_messages = build_detection_messages(inverted=args.inverted)
    detection_prompt = prepare_prompt(tokenizer, device, detection_messages)
    print(
        f"  injection_index={detection_prompt.injection_index}, "
        f"seq_len={detection_prompt.input_ids.shape[1]}"
    )

    print("Preparing factual prompts...")
    factual_prompts: list[tuple[str, PromptSetup]] = []
    factual_questions = FACTUAL_YES_QUESTIONS if args.inverted else FACTUAL_NO_QUESTIONS
    for question in factual_questions:
        messages = build_factual_messages(question)
        prompt = prepare_prompt(tokenizer, device, messages)
        factual_prompts.append((question, prompt))
        print(f"  {question[:50]}... injection_index={prompt.injection_index}")

    # --- Resolve YES/NO tokens ---
    print("\nResolving YES/NO token IDs...")
    yes_id, no_id = resolve_yes_no_tokens(tokenizer, model, detection_prompt)

    # --- Compute baselines (no injection) ---
    print("\nComputing baselines (no injection)...")
    detection_baseline = extract_logit_diffs(
        model,
        detection_prompt,
        batch_size=1,
        yes_token_id=yes_id,
        no_token_id=no_id,
    )[0]
    print(f"  Detection baseline: logit_diff={detection_baseline['logit_diff']:.3f}")

    factual_baselines: list[dict[str, Any]] = []
    for question, prompt in factual_prompts:
        baseline = extract_logit_diffs(
            model,
            prompt,
            batch_size=1,
            yes_token_id=yes_id,
            no_token_id=no_id,
        )[0]
        factual_baselines.append({"question": question, **baseline})
        print(
            f"  Factual baseline ({question[:40]}...): logit_diff={baseline['logit_diff']:.3f}"
        )

    # --- Build intervention requests ---
    requests = build_requests(concept_names, args.layers, args.strengths)
    print(
        f"\n{len(concept_names)} concepts x {len(args.layers)} layers x "
        f"{len(args.strengths)} strengths = {len(requests)} intervention configs"
    )

    # --- Run detection interventions ---
    print("\nRunning detection interventions...")
    detection_results = run_logit_interventions(
        model=model,
        prompt=detection_prompt,
        all_steering_vectors=all_steering_vectors,
        requests=requests,
        yes_token_id=yes_id,
        no_token_id=no_id,
        debug_residual=args.debug_residual,
        max_batch_size=args.max_batch_size,
    )

    # --- Run factual interventions (for each factual question) ---
    factual_results_by_question: list[list[dict[str, float]]] = []
    for q_idx, (question, prompt) in enumerate(factual_prompts):
        print(
            f"Running factual interventions [{q_idx + 1}/{len(factual_prompts)}]: "
            f"{question[:50]}..."
        )
        q_results = run_logit_interventions(
            model=model,
            prompt=prompt,
            all_steering_vectors=all_steering_vectors,
            requests=requests,
            yes_token_id=yes_id,
            no_token_id=no_id,
            debug_residual=args.debug_residual,
            max_batch_size=args.max_batch_size,
        )
        factual_results_by_question.append(q_results)

    # --- Assemble output records ---
    records: list[dict[str, Any]] = []
    for idx, request in enumerate(requests):
        # Detection record
        records.append(
            {
                "concept": request.concept,
                "layer": request.layers[0],
                "strength": request.strength,
                "condition": "detection",
                **detection_results[idx],
            }
        )

        # Factual records (one per question)
        for q_idx, (question, _) in enumerate(factual_prompts):
            records.append(
                {
                    "concept": request.concept,
                    "layer": request.layers[0],
                    "strength": request.strength,
                    "condition": "factual",
                    "question": question,
                    **factual_results_by_question[q_idx][idx],
                }
            )

    # --- Build output ---
    factual_prompt_info = [
        {
            "question": question,
            "formatted": prompt.formatted_prompt,
            "injection_index": prompt.injection_index,
        }
        for question, prompt in factual_prompts
    ]

    output: dict[str, Any] = {
        "model_name": args.model_name,
        "experiment_type": "inverted_logit_introspection" if args.inverted else "logit_introspection",
        "yes_token_id": yes_id,
        "no_token_id": no_id,
        "yes_token_str": tokenizer.decode([yes_id]),
        "no_token_str": tokenizer.decode([no_id]),
        "detection_prompt": {
            "messages": detection_messages,
            "formatted": detection_prompt.formatted_prompt,
            "injection_index": detection_prompt.injection_index,
        },
        "factual_prompts": factual_prompt_info,
        "settings": {
            "layers": args.layers,
            "strengths": args.strengths,
            "seed": args.seed,
            "max_batch_size": args.max_batch_size,
            "inverted": args.inverted,
        },
        "concepts_evaluated": concept_names,
        "baselines": {
            "detection_no_injection": detection_baseline,
            "factual_no_injection": factual_baselines,
        },
        "results": records,
    }

    return output


def parse_args() -> LogitExperimentArgs:
    parser = argparse.ArgumentParser(
        description="Run logit-based introspection experiment (detection vs factual control)."
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
        help="Concept(s) to evaluate. Defaults to all.",
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
        "--invert",
        action="store_true",
        help="Flip detection question so NO means injection detected (for YES-bias control).",
    )
    parsed = parser.parse_args()
    return LogitExperimentArgs(
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
        inverted=parsed.invert,
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

    output = run_logit_experiment(
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
