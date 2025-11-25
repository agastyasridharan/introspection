from __future__ import annotations

import json
import uuid
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

from introspection.grader_prompts import get_grader_prompt


def load_samples(data_dir: Path, grader_prompt: str) -> list[Sample]:
    samples: list[Sample] = []

    for sweep_path in data_dir.glob("*/sweep.jsonl"):
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
                "steering_path": result["steering_path"],
                "injection_index": prompt_block["injection_index"],
            }

            for condition in ("control", "intervention"):
                response = result[condition]
                prompt_text = grader_prompt.format(
                    question=question,
                    response=response,
                )

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
                        input=prompt_text,
                        target="",
                        id=sample_id,
                        metadata={
                            **base_metadata,
                            "condition": condition,
                            "response": response,
                            "question": question,
                        },
                    )
                )

    return samples


@task
def grade_responses(
    data_dir: str,
    grader_prompt: str,
) -> Task:
    prompt_template = get_grader_prompt(grader_prompt)
    samples = load_samples(Path(data_dir), prompt_template)
    dataset = MemoryDataset(samples, name="steering_responses")

    return Task(dataset=dataset)
