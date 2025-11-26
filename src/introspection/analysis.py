from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from inspect_ai import log

from introspection.grader_prompts import GRADER_PROMPTS


def extract_model_scale(model_name: str) -> str:
    """Extract model scale identifier from model name.

    Examples:
        "Qwen/Qwen3-14B" -> "14B"
        "Qwen/Qwen3-32B" -> "32B"
        "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8" -> "235B"
    """
    match = re.search(r"(\d+B)", model_name)
    if match:
        return match.group(1)
    return model_name.split("/")[-1] if "/" in model_name else model_name


def map_scorer_to_prompt(scorer_key: str) -> str:
    """Map scorer key to grader prompt name.

    Scorer keys are like "model_graded_qa", "model_graded_qa1", "model_graded_qa2", etc.
    These correspond to the order of prompts in GRADER_PROMPTS.
    """
    if scorer_key == "model_graded_qa":
        return list(GRADER_PROMPTS.keys())[0]

    match = re.search(r"model_graded_qa(\d+)", scorer_key)
    if match:
        idx = int(match.group(1))
        prompt_names = list(GRADER_PROMPTS.keys())
        if 0 <= idx < len(prompt_names):
            return prompt_names[idx]

    return scorer_key


def load_eval_files(logs_dir: Path) -> list[dict[str, Any]]:
    """Load all .eval files and extract sample data.

    Uses Inspect AI's native log reading functions to load evaluation logs.
    Returns a list of sample dictionaries with scores and metadata.
    """
    samples: list[dict[str, Any]] = []

    eval_logs = log.list_eval_logs(str(logs_dir))
    for eval_log in eval_logs:
        for sample in log.read_eval_log_samples(eval_log):
            samples.append(sample.model_dump())

    return samples


def process_samples(samples: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert samples to DataFrame with proper structure.

    Asserts that each sample has exactly one layer.
    Extracts model scale from model_name.
    Maps scorer keys to grader prompt names.
    Converts YES/NO scores to numeric (1/0).
    """
    rows: list[dict[str, Any]] = []

    for sample in samples:
        metadata = sample.get("metadata", {})
        scores = sample.get("scores", {})

        if not scores:
            continue

        layers = metadata.get("layers", [])
        if len(layers) != 1:
            raise ValueError(
                f"Expected exactly one layer, got {len(layers)} layers: {layers} "
                f"in sample {sample.get('id', 'unknown')}"
            )

        layer_index = layers[0]
        model_name = metadata.get("model_name", "")
        model_scale = extract_model_scale(model_name)

        for scorer_key, score_data in scores.items():
            if not isinstance(score_data, dict) or "value" not in score_data:
                continue

            score_value = score_data["value"]
            score_numeric = 1 if score_value == "YES" else 0

            grader_prompt = map_scorer_to_prompt(scorer_key)

            rows.append(
                {
                    "grader_prompt": grader_prompt,
                    "layer_index": layer_index,
                    "model_scale": model_scale,
                    "strength": metadata.get("strength"),
                    "temperature": metadata.get("temperature"),
                    "concept": metadata.get("concept"),
                    "condition": metadata.get("condition"),
                    "score": score_numeric,
                    "model_name": model_name,
                    "trial": metadata.get("trial"),
                }
            )

    return pd.DataFrame(rows)


def load_and_process_data(logs_dir: str | Path = "logs/") -> pd.DataFrame:
    """Load evaluation files and process into DataFrame.

    Convenience function that combines load_eval_files and process_samples.
    """
    logs_path = Path(logs_dir)
    samples = load_eval_files(logs_path)
    return process_samples(samples)


def aggregate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scores by layer_index, strength, temperature, model_scale, grader_prompt, condition.

    Computes mean score (proportion of YES responses) for each combination.
    """
    agg_cols = [
        "layer_index",
        "strength",
        "temperature",
        "model_scale",
        "grader_prompt",
        "condition",
    ]
    aggregated = df.groupby(agg_cols, as_index=False)["score"].mean()
    aggregated.rename(columns={"score": "mean_score"}, inplace=True)
    return aggregated
