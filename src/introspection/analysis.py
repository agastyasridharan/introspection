from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from inspect_ai import log

from introspection.constants import MODEL_LAYER_COUNTS


def load_and_process_data(logs_dir: str | Path = "logs/") -> pd.DataFrame:
    logs_path = Path(logs_dir)
    samples: list[log.EvalSample] = []
    for eval_log in log.list_eval_logs(str(logs_path)):
        for sample in log.read_eval_log_samples(eval_log):
            samples.append(sample)

    if len(samples) == 0:
        raise ValueError(f"No samples found in {logs_path}")

    rows: list[dict[str, Any]] = []
    for sample in samples:
        metadata = sample.metadata
        scores = sample.scores
        assert scores is not None
        layers = metadata["layers"]
        assert len(layers) == 1
        layer_index = layers[0]  # pyright: ignore[reportAttributeAccessIssue]
        model_name = metadata["model_name"]
        match = re.search(r"(\d+B)", model_name)
        model_scale = (
            match.group(1)
            if match
            else (model_name.split("/")[-1] if "/" in model_name else model_name)
        )
        total_layers = MODEL_LAYER_COUNTS[model_name]
        layer_percentage = (layer_index / (total_layers - 1)) * 100.0  # pyright: ignore[reportAttributeAccessIssue]

        for scorer_key, score_data in scores.items():
            score_value = score_data.value
            score_numeric = 1 if score_value == "YES" else 0
            rows.append(
                {
                    "grader_prompt": scorer_key,
                    "layer_index": layer_index,
                    "layer_percentage": layer_percentage,
                    "model_scale": model_scale,
                    "strength": metadata["strength"],
                    "temperature": metadata["temperature"],
                    "concept": metadata["concept"],
                    "condition": metadata["condition"],
                    "score": score_numeric,
                    "model_name": model_name,
                    "trial": metadata["trial"],
                }
            )

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
