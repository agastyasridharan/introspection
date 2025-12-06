from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from inspect_ai import log

from introspection.constants import MODEL_LAYER_COUNTS

# Precompile regex for better performance
_MODEL_SCALE_PATTERN = re.compile(r"(\d+B)")


def _extract_model_scale(model_name: str) -> str:
    """Extract model scale (e.g., '8B', '14B') from model name."""
    match = _MODEL_SCALE_PATTERN.search(model_name)
    if match:
        return match.group(1)
    return model_name.split("/")[-1] if "/" in model_name else model_name


def load_and_process_data(logs_dir: str | Path = "logs/") -> pd.DataFrame:
    logs_path = Path(logs_dir)
    eval_logs = list(log.list_eval_logs(str(logs_path)))

    if not eval_logs:
        raise ValueError(f"No eval logs found in {logs_path}")

    rows: list[dict[str, Any]] = []

    for eval_log in eval_logs:
        for sample in log.read_eval_log_samples(eval_log):
            metadata = sample.metadata
            scores = sample.scores
            assert scores is not None

            layers = metadata["layers"]
            assert len(layers) == 1
            layer_index = layers[0]  # pyright: ignore[reportAttributeAccessIssue]

            model_name = metadata["model_name"]
            model_scale = _extract_model_scale(model_name)
            total_layers = MODEL_LAYER_COUNTS[model_name]
            layer_percentage = (layer_index / (total_layers - 1)) * 100.0  # pyright: ignore[reportAttributeAccessIssue]

            # Pre-extract common metadata once per sample
            strength = metadata["strength"]
            temperature = metadata["temperature"]
            concept = metadata["concept"]
            condition = metadata["condition"]
            trial = metadata["trial"]

            for scorer_key, score_data in scores.items():
                rows.append(
                    {
                        "grader_prompt": scorer_key,
                        "layer_index": layer_index,
                        "layer_percentage": layer_percentage,
                        "model_scale": model_scale,
                        "strength": strength,
                        "temperature": temperature,
                        "concept": concept,
                        "condition": condition,
                        "score": 1 if score_data.value == "YES" else 0,
                        "model_name": model_name,
                        "trial": trial,
                    }
                )

    if not rows:
        raise ValueError(f"No samples found in {logs_path}")

    return pd.DataFrame(rows)


def aggregate_scores(df: pd.DataFrame) -> pd.DataFrame:
    aggregated: pd.DataFrame = df.groupby(  # pyright: ignore[reportUnknownMemberType]
        [
            "layer_percentage",
            "strength",
            "temperature",
            "model_scale",
            "grader_prompt",
            "condition",
        ],
        as_index=False,
    ).agg({"score": "mean"})  # pyright: ignore[reportUnknownMemberType]
    assert isinstance(aggregated, pd.DataFrame)
    aggregated = aggregated.rename(columns={"score": "mean_score"})
    return aggregated
