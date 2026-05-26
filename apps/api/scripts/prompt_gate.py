#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT / "src"))

from rag_platform.services.prompt_gate import load_prompt_manifest, run_prompt_gate  # noqa: E402


def main() -> int:
    args = _parse_args()
    manifest = load_prompt_manifest(args.manifest)
    event_payload = _load_event_payload(args)
    report = run_prompt_gate(manifest, event_payload=event_payload)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.passed else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Langfuse prompt changes before release.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("prompts/manifest.json"),
        help="Path to prompts manifest JSON.",
    )
    parser.add_argument(
        "--github-event",
        type=Path,
        default=None,
        help="Path to GitHub event JSON. For repository_dispatch, client_payload is inspected.",
    )
    parser.add_argument(
        "--payload",
        type=Path,
        default=None,
        help="Path to a raw Langfuse webhook payload JSON.",
    )
    return parser.parse_args()


def _load_event_payload(args: argparse.Namespace) -> Dict[str, Any]:
    if args.payload:
        return _read_json(args.payload)
    if args.github_event and args.github_event.exists():
        return _read_json(args.github_event)
    return {}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
