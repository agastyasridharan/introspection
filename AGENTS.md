# Repository Guidelines

## Project Structure & Module Organization

- package `introspection`
- `pyproject.toml`: dependency and metadata configuration (Python ≥3.13, torch, transformers).
- `steering_vectors.pt`: generated artifacts from `generate_steering_vectors.py`.

## Build, Test, and Development Commands

- `uv sync` (or `uv add`): install dependencies defined in `pyproject.toml`.
- `uv run python -m src.introspection.steer --prompt "Tell me about bridges." --layer-index 0`: quick activation inspection.
- `uv run python -m src.introspection.generate_steering_vectors`: run the 50-concept steering vector experiment; customize via flags like `--device cuda`.

* All python commands should be prefix with `uv run`. NEVER USE `python` or `python3` or any other non-uv invocation!

## Coding Style & Naming Conventions

- Python 3.13+ with type hints; follow standard PEP 8 (4-space indentation, snake_case names).
- Prefer explicit helper functions for data processing (see `compute_steering_vectors`).
- Keep docstrings concise and informative; add comments only for non-obvious logic.

* No excessive error handling or try/catching
* No excessive casting
* No excessive input validation and checking. Please fail loudly if something is unexpected!! Be concise, reduce branching where possible.
* Always run `uv run ruff check .` and `uv run basedpyright .` and fix all the errors after making code changes. If the issues tend to be with library code, it's okay to disable the pyright warnings if they're hard to fix
* Always format at the end of a change by running `uv run ruff format .`
* Never use Optional or Union or Tuple or List etc. for type hints, always prefer `| None` or `|` or `tuple` or `list` etc.
* Try to import modules and use it like `from package.module_a import module_b; module_b.foo()`.

## Testing Guidelines

- No formal test suite yet; verify changes by running `python main.py` or `python experiment.py` with representative parameters.
- When introducing modules, add targeted unit tests under a future `tests/` directory and use `pytest`.
- Ensure deterministic behavior by respecting seeds in experiments.

## Commit & Pull Request Guidelines

- Write concise commit subjects in imperative mood (e.g., “Add steering vector experiment”).
- Include brief body lines when context helps reviewers; wrap at ~72 characters.
- For pull requests, summarize behavioral changes, list testing commands run, and reference related issues or experiment IDs.
