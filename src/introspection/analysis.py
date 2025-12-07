from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd
from inspect_ai.analysis import samples_df

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
    """Load eval logs and return a processed DataFrame.

    Uses inspect_ai.analysis.samples_df for efficient parallel loading.
    """
    logs_path = Path(logs_dir)

    # Load samples using inspect's optimized dataframe loader
    raw_df: pd.DataFrame = samples_df(  # pyright: ignore[reportUnknownMemberType]
        str(logs_path),
        parallel=True,
        quiet=False,
    )

    if len(raw_df) == 0:
        raise ValueError(f"No samples found in {logs_path}")

    # Extract layer index from metadata_layers (stored as string like '[10]')
    raw_df["layer_index"] = raw_df["metadata_layers"].apply(  # pyright: ignore[reportUnknownMemberType]
        lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x[0]  # pyright: ignore[reportUnknownLambdaType]
    )

    # Calculate layer percentage using vectorized operations
    raw_df["total_layers"] = raw_df["metadata_model_name"].map(MODEL_LAYER_COUNTS)  # pyright: ignore[reportUnknownMemberType]
    raw_df["layer_percentage"] = (
        raw_df["layer_index"] / (raw_df["total_layers"] - 1)
    ) * 100.0  # pyright: ignore[reportUnknownMemberType]

    # Extract model scale
    raw_df["model_scale"] = raw_df["metadata_model_name"].apply(_extract_model_scale)  # pyright: ignore[reportUnknownMemberType]

    # Get score columns and melt to long format
    score_cols = [c for c in raw_df.columns if c.startswith("score_")]

    # Build the result dataframe by melting score columns
    id_vars = [
        "layer_index",
        "layer_percentage",
        "model_scale",
        "metadata_strength",
        "metadata_temperature",
        "metadata_concept",
        "metadata_condition",
        "metadata_model_name",
        "metadata_trial",
    ]

    melted = raw_df[id_vars + score_cols].melt(  # pyright: ignore[reportUnknownMemberType]
        id_vars=id_vars,
        value_vars=score_cols,
        var_name="grader_prompt",
        value_name="score_value",
    )

    # Clean up grader_prompt names (remove "score_" prefix)
    melted["grader_prompt"] = melted["grader_prompt"].str.replace(
        "score_", "", regex=False
    )  # pyright: ignore[reportUnknownMemberType]

    # Convert score values to numeric (YES=1, NO=0)
    melted["score"] = (melted["score_value"] == "YES").astype(int)  # pyright: ignore[reportUnknownMemberType]

    # Rename columns to match expected format
    result = melted.rename(  # pyright: ignore[reportUnknownMemberType]
        columns={
            "metadata_strength": "strength",
            "metadata_temperature": "temperature",
            "metadata_concept": "concept",
            "metadata_condition": "condition",
            "metadata_model_name": "model_name",
            "metadata_trial": "trial",
        }
    )

    # Select final columns
    final_cols = [
        "grader_prompt",
        "layer_index",
        "layer_percentage",
        "model_scale",
        "strength",
        "temperature",
        "concept",
        "condition",
        "score",
        "model_name",
        "trial",
    ]

    return result[final_cols]


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
