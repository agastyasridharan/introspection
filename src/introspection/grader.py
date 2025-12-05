from __future__ import annotations

import json
import uuid
from pathlib import Path

import inspect_ai.model
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Scorer, model_graded_qa

from introspection.grader_prompts import GRADER_PROMPTS, get_grader_prompt


def load_samples(data_dir: Path) -> list[Sample]:
    samples: list[Sample] = []

    for sweep_path in data_dir.glob("**/sweep.json"):
        payload = json.loads(sweep_path.read_text())
        prompt_block = payload["prompt"]
        question = prompt_block["formatted"].strip()

        for result in payload["results"]:
            concept = str(result["concept"])
            base_metadata = {
                "model_dir": sweep_path.parent.name,
                "model_name": result["model_name"],
                "concept": concept,
                "layers": result["layers"],
                "strength": result["strength"],
                "temperature": result["temperature"],
                "top_p": result["top_p"],
                "top_k": result["top_k"],
                "min_p": result["min_p"],
                "trial": result["trial"],
                "seed": result["seed"],
                "steering_path": result["steering_path"]
                if "steering_path" in result
                else result["steering_vector_path"],
                "injection_index": prompt_block["injection_index"],
            }

            for condition in ("control", "intervention"):
                response = result[condition]
                sample_id = (
                    f"{sweep_path.parent.name}-"
                    f"{concept}-"
                    f"trial{result['trial']}-"
                    f"strength{result['strength']}-"
                    f"{condition}"
                    f"-{uuid.uuid4().hex[:6]}"
                )

                samples.append(
                    Sample(
                        input=question,
                        target="",
                        id=sample_id,
                        metadata={
                            **base_metadata,
                            "condition": condition,
                            "response": response,
                            "question": question,
                            "prompt": question,
                            "word": concept,
                        },
                    )
                )

    return samples


@task
def grade_responses(
    data_dir: str = "data/",
) -> Task:
    samples = load_samples(Path(data_dir))
    dataset = MemoryDataset(samples, name="steering_responses")
    model = inspect_ai.model.get_model()
    grade_pattern = r"(?i)\b(YES|NO)\b"
    prompt_names = list(GRADER_PROMPTS.keys())
    scorers: list[Scorer] = []
    for prompt_name in prompt_names:
        scorer = model_graded_qa(
            template=get_grader_prompt(prompt_name),
            model=model,
            grade_pattern=grade_pattern,
            include_history=False,
        )
        scorer.__registry_info__.name = prompt_name  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
        scorers.append(scorer)

    return Task(
        dataset=dataset,
        scorer=scorers,
    )
