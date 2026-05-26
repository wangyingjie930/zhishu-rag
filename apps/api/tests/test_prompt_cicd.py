import hashlib
import hmac
import json
import time

from rag_platform.services.prompt_cicd import verify_langfuse_signature
from rag_platform.services.prompt_gate import (
    extract_prompt_variables,
    run_prompt_gate,
    validate_langfuse_prompt_payload,
)


def test_verify_langfuse_signature_accepts_valid_header() -> None:
    raw_body = b'{"type":"prompt-version","prompt":{"name":"rag-answer"}}'
    secret = "test-secret"
    timestamp = int(time.time())
    signature = hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}.{raw_body.decode('utf-8')}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    assert verify_langfuse_signature(
        raw_body=raw_body,
        signature_header=f"t={timestamp},s={signature}",
        secret=secret,
        now=timestamp,
    )


def test_verify_langfuse_signature_rejects_expired_header() -> None:
    raw_body = b'{"type":"prompt-version","prompt":{"name":"rag-answer"}}'
    secret = "test-secret"
    timestamp = 1000
    signature = hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}.{raw_body.decode('utf-8')}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    assert not verify_langfuse_signature(
        raw_body=raw_body,
        signature_header=f"t={timestamp},s={signature}",
        secret=secret,
        tolerance_seconds=300,
        now=timestamp + 301,
    )


def test_extract_prompt_variables_from_chat_prompt() -> None:
    variables = extract_prompt_variables(
        [
            {"role": "system", "content": "Use {{contexts}} only."},
            {"role": "user", "content": "Question: {{ question }}"},
        ]
    )

    assert variables == {"contexts", "question"}


def test_prompt_gate_accepts_valid_langfuse_payload() -> None:
    manifest = _manifest()
    payload = {
        "name": "rag-answer",
        "type": "chat",
        "labels": ["candidate"],
        "prompt": [
            {"role": "system", "content": "Use {{contexts}} only."},
            {"role": "user", "content": "Question: {{question}}"},
        ],
        "config": {"temperature": 0.2, "max_tokens": 512},
    }

    report = validate_langfuse_prompt_payload(manifest, payload)

    assert report.passed
    assert report.checked_prompts == ["rag-answer"]


def test_prompt_gate_blocks_missing_required_variable() -> None:
    report = validate_langfuse_prompt_payload(
        _manifest(),
        {
            "name": "rag-answer",
            "labels": ["candidate"],
            "prompt": [{"role": "user", "content": "Question: {{question}}"}],
            "config": {"temperature": 0.2},
        },
    )

    assert not report.passed
    assert "contexts" in json.dumps(report.to_dict(), ensure_ascii=False)


def test_prompt_gate_blocks_unknown_config_key() -> None:
    report = validate_langfuse_prompt_payload(
        _manifest(),
        {
            "name": "rag-answer",
            "labels": ["candidate"],
            "prompt": [{"role": "user", "content": "{{question}} {{contexts}}"}],
            "config": {"temperature": 0.2, "api_key": "should-not-be-here"},
        },
    )

    assert not report.passed
    assert "api_key" in report.errors[0]


def test_run_prompt_gate_reads_repository_dispatch_payload() -> None:
    report = run_prompt_gate(
        _manifest(),
        event_payload={
            "client_payload": {
                "type": "prompt-version",
                "prompt": {
                    "name": "rag-answer",
                    "labels": ["production"],
                    "prompt": [{"role": "user", "content": "{{question}} {{contexts}}"}],
                    "config": {"temperature": 0.2},
                },
            }
        },
    )

    assert report.passed


def _manifest() -> dict:
    return {
        "version": 1,
        "prompts": [
            {
                "name": "rag-answer",
                "type": "chat",
                "required_variables": ["question", "contexts"],
                "release_labels": ["candidate", "production"],
                "fallback": [{"role": "user", "content": "{{question}} {{contexts}}"}],
                "config_schema": {
                    "allowed_keys": ["temperature", "max_tokens", "model"],
                },
                "quality_gates": {
                    "dataset": "rag-answer-regression",
                    "metrics": {"avg_expected_match": {"min": 0.8}},
                },
            }
        ],
    }
