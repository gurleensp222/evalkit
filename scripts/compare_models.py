"""Run one dataset on two models, save both runs, and compare them.

Example:
    python scripts/compare_models.py --a openai --b anthropic

OpenAI and Anthropic need their API keys. Ollama needs a local server.
Pass --judge when the dataset uses the llm_judge scorer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import uuid4

from evalkit.analysis import compare, load_runs
from evalkit.providers import Provider, ProviderError, create_provider
from evalkit.runner import load_tasks, run_eval
from evalkit.storage import save_results


def main(argv: list[str] | None = None) -> int:
    """Save a run for each model, then print compare()."""
    args = _parser().parse_args(argv)
    dataset = Path(args.dataset)
    if not dataset.is_file():
        print(f"dataset not found: {dataset}", file=sys.stderr)
        return 1

    try:
        tasks = load_tasks(dataset)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not tasks:
        print(f"dataset is empty: {dataset}", file=sys.stderr)
        return 1

    needs_judge = any(task.scorer == "llm_judge" for task in tasks)
    if needs_judge and not args.judge:
        print("dataset uses llm_judge; pass --judge", file=sys.stderr)
        return 1

    try:
        provider_a = _labeled(args.a, args.model_a)
        provider_b = _labeled(args.b, args.model_b)
        judge = create_provider(args.judge) if args.judge else None
    except ProviderError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if provider_a.name == provider_b.name:
        print(
            "the two models share a name; pass --model-a and --model-b",
            file=sys.stderr,
        )
        return 1

    directory = Path(args.runs_dir)
    token = uuid4().hex[:8]
    run_a = f"{token}-a"
    run_b = f"{token}-b"
    results_a = run_eval(tasks, provider_a, run_a, judge=judge)
    results_b = run_eval(tasks, provider_b, run_b, judge=judge)
    path_a = save_results(run_a, results_a, directory=directory)
    path_b = save_results(run_b, results_b, directory=directory)
    print(f"saved {path_a}")
    print(f"saved {path_b}")

    frame = load_runs([run_a, run_b], directory=directory)
    try:
        result = compare(frame, provider_a.name, provider_b.name)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        f"{result['model_a']} pass_rate {result['pass_rate_a']:.2f}  "
        f"{result['model_b']} pass_rate {result['pass_rate_b']:.2f}  "
        f"difference {result['difference']:.2f}  "
        f"ci [{result['ci_low']:.2f}, {result['ci_high']:.2f}]  "
        f"excludes_zero {result['excludes_zero']}"
    )
    return 0


def _labeled(name: str, model: str | None) -> Provider:
    """Build a provider and record ``model`` as its name when one is given."""
    provider = create_provider(name, model=model)
    chosen = model.strip() if isinstance(model, str) else ""
    if chosen:
        provider.name = chosen
    return provider


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="datasets/sample_tasks.jsonl",
        help="JSONL file of tasks. Both models see this file.",
    )
    parser.add_argument("--a", default="openai", help="First provider.")
    parser.add_argument("--b", default="anthropic", help="Second provider.")
    parser.add_argument("--model-a", default=None, help="Model id for --a.")
    parser.add_argument("--model-b", default=None, help="Model id for --b.")
    parser.add_argument(
        "--judge",
        default=None,
        help="Provider that grades llm_judge tasks.",
    )
    parser.add_argument("--runs-dir", default="runs", help="Directory for the JSONL runs.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
