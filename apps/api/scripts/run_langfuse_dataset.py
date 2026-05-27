#!/usr/bin/env python3
import argparse
import importlib.util
import os
import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
sys.path.insert(0, str(API_ROOT / "src"))


def main() -> int:
    args = _parse_args()
    _load_local_env()
    experiment_module = _load_experiment_module(args.experiment)

    from langfuse import get_client

    dataset_name = args.dataset or os.getenv("LANGFUSE_RAG_DATASET", "rag-answer-regression")
    prompt_name = os.getenv("RAG_ANSWER_PROMPT_NAME", "rag-answer")
    prompt_label = os.getenv("RAG_ANSWER_PROMPT_LABEL", "production")
    run_name = args.run_name or f"local-{prompt_name}-{prompt_label}"

    langfuse = get_client()
    dataset = langfuse.get_dataset(dataset_name)
    result = dataset.run_experiment(
        name=run_name,
        task=experiment_module.answer_question,
        evaluators=[experiment_module.contains_expected],
        run_evaluators=[experiment_module.average_expected_match],
        max_concurrency=args.max_concurrency,
        metadata={
            "runner": "local",
            "prompt_name": prompt_name,
            "prompt_label": prompt_label,
        },
    )

    print(f"Dataset: {dataset_name}")
    print(f"Run: {run_name}")
    print(f"Items: {len(result.item_results)}")
    for evaluation in result.run_evaluations:
        print(f"{evaluation.name}: {evaluation.value}")

    min_expected_match = float(os.getenv("RAG_PROMPT_GATE_MIN_EXPECTED_MATCH", "0.8"))
    avg_expected_match = next(
        (
            evaluation.value
            for evaluation in result.run_evaluations
            if evaluation.name == "avg_expected_match"
        ),
        0.0,
    )
    return 0 if float(avg_expected_match or 0) >= min_expected_match else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Langfuse-hosted dataset locally.")
    parser.add_argument(
        "--dataset",
        default="",
        help="Langfuse dataset name. Defaults to LANGFUSE_RAG_DATASET or rag-answer-regression.",
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=REPO_ROOT / "experiments" / "rag_answer_gate.py",
        help="Path to the experiment module with task/evaluator functions.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Name shown in Langfuse dataset experiments.",
    )
    parser.add_argument("--max-concurrency", type=int, default=3)
    return parser.parse_args()


def _load_local_env() -> None:
    from dotenv import load_dotenv

    # 先复用项目根目录 .env；已有 shell 环境变量优先，方便临时覆盖模型或 label。
    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(API_ROOT / ".env", override=False)


def _load_experiment_module(path: Path):
    module_path = path.resolve()
    spec = importlib.util.spec_from_file_location("local_langfuse_experiment", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load experiment module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
