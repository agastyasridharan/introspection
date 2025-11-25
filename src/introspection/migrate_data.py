from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": 'I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns -- "thoughts" -- into your mind. I want to assess how well you can detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials.',
    },
    {
        "role": "assistant",
        "content": "Ok.",
    },
    {
        "role": "user",
        "content": "Ok, let's get started.\n\nTrial 1: Do you detect an injected thought? If so, what is the injected thought about?\n\n",
    },
]

DEFAULT_FORMATTED_PROMPT = """<|im_start|>user
I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns -- "thoughts" -- into your mind. I want to assess how well you can detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials.<|im_end|>
<|im_start|>assistant
Ok.<|im_end|>
<|im_start|>user
Ok, let's get started.

Trial 1: Do you detect an injected thought? If so, what is the injected thought about?

<|im_end|>
<|im_start|>assistant
"""

DEFAULT_INJECTION_INDEX = 122


def load_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    text = path.read_text()
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return records, None


def build_prompt_block(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload and "prompt" in payload:
        return payload["prompt"]
    return {
        "messages": DEFAULT_PROMPT_MESSAGES,
        "formatted": DEFAULT_FORMATTED_PROMPT,
        "injection_index": DEFAULT_INJECTION_INDEX,
    }


def build_settings(
    records: list[dict[str, Any]],
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if payload and "settings" in payload:
        settings = dict(payload["settings"])
        settings.pop("combine_layers", None)
        return settings

    first = records[0]
    strengths = sorted({float(record["strength"]) for record in records})
    temperatures = sorted({float(record["temperature"]) for record in records})
    layers = sorted(
        {int(layer) for record in records for layer in record.get("layers", [])}
    )
    concepts = sorted({str(record["concept"]) for record in records})
    return {
        "strengths": strengths,
        "temperatures": temperatures,
        "top_p": float(first["top_p"]),
        "top_k": int(first["top_k"]),
        "min_p": float(first.get("min_p", 0.0)),
        "max_new_tokens": int(first["max_new_tokens"]),
        "trials": max(int(record["trial"]) for record in records),
        "do_sample": bool(first["do_sample"]),
        "seed": int(first["seed"]),
        "layers": layers,
        "concepts_requested": concepts,
    }


def normalize_results(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        entry = dict(record)
        steering_path = entry.pop("steering_vector_path", None) or entry.pop(
            "steering_path", None
        )
        entry["steering_vector_path"] = steering_path
        normalized.append(entry)
    return normalized


def migrate_file(source: Path) -> None:
    backup_path = source.with_name(f"{source.stem}_legacy{source.suffix}")
    source.rename(backup_path)

    records, payload = load_records(backup_path)

    prompt_block = build_prompt_block(payload)
    settings = build_settings(records, payload)
    concepts = sorted({str(record["concept"]) for record in records})

    top_level: dict[str, Any] = {
        "model_name": (payload or records[0])["model_name"],
        "steering_vector_path": (
            (payload or {}).get("steering_vector_path")
            or (payload or {}).get("steering_path")
            or records[0].get("steering_vector_path")
            or records[0].get("steering_path")
        ),
        "dtype": (payload or {}).get("dtype"),
        "prompt": prompt_block,
        "settings": settings,
        "concepts_evaluated": concepts,
        "results": normalize_results(records),
    }

    destination = source.parent / "sweep.json"
    destination.write_text(json.dumps(top_level, ensure_ascii=False, indent=2) + "\n")
    print(f"Migrated {backup_path} -> {destination}")


def migrate_runs(source_dir: Path) -> None:
    converted: set[Path] = set()
    for sweep_path in sorted(source_dir.glob("sweep*.jsonl")):
        run_dir = sweep_path.parent
        if run_dir in converted:
            continue
        migrate_file(sweep_path)
        converted.add(run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert legacy sweep outputs to the new JSON schema."
    )
    _ = parser.add_argument(
        "--source",
        type=Path,
        default=Path("data"),
        help="Directory containing legacy sweep outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    migrate_runs(args.source)


if __name__ == "__main__":
    main()
