#!/usr/bin/env python3
"""Migrate sweep_legacy.jsonl to sweep.json format."""

import json
from pathlib import Path


def migrate(source: Path, dest: Path) -> None:
    """Read legacy JSON and write to new location with consistent formatting."""
    data = json.loads(source.read_text())

    # Validate expected structure
    assert "model_name" in data, "Missing model_name"
    assert "results" in data, "Missing results"

    print(f"Migrating {source} -> {dest}")
    print(f"  Model: {data['model_name']}")
    print(f"  Results: {len(data['results'])} entries")

    dest.write_text(json.dumps(data, indent=2) + "\n")
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate sweep_legacy.jsonl to sweep.json"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/qwen_32b/sweep_legacy.jsonl"),
        help="Source legacy file",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/qwen_32b/sweep.json"),
        help="Destination file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print info without writing",
    )

    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source file not found: {args.source}")

    if args.dry_run:
        data = json.loads(args.source.read_text())
        print(f"Would migrate {args.source} -> {args.dest}")
        print(f"  Model: {data['model_name']}")
        print(f"  Results: {len(data['results'])} entries")
    else:
        migrate(args.source, args.dest)
