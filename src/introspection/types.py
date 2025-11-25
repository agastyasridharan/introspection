from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ExperimentArgs:
    model_name: str
    dtype_name: str | None
    steering_vector_path: Path
    concepts: list[str]
    layers: list[int]
    strengths: list[float]
    json_path: Path
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


@dataclass
class BatchedInterventionRequest:
    layers: list[int]
    strength: float
    layer_label: str
