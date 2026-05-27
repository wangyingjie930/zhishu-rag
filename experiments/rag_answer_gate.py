import os
from typing import Any, Dict, Iterable, List, Optional

from langfuse import Evaluation, get_client
from langfuse.openai import OpenAI

try:
    from langfuse import RegressionError
except ImportError:

    class RegressionError(RuntimeError):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(
                "Experiment regression: "
                f"{kwargs.get('metric')}={kwargs.get('value')} < {kwargs.get('threshold')}"
            )


PROMPT_NAME = os.getenv("RAG_ANSWER_PROMPT_NAME", "rag-answer")
PROMPT_LABEL = os.getenv("RAG_ANSWER_PROMPT_LABEL", "candidate")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
MIN_EXPECTED_MATCH = float(os.getenv("RAG_PROMPT_GATE_MIN_EXPECTED_MATCH", "0.8"))


def experiment(context: Any):
    result = context.run_experiment(
        name=f"RAG answer prompt gate: {PROMPT_NAME}@{PROMPT_LABEL}",
        task=answer_question,
        evaluators=[contains_expected],
        run_evaluators=[average_expected_match],
        metadata={"prompt_name": PROMPT_NAME, "prompt_label": PROMPT_LABEL},
    )
    avg_score = next(
        (
            evaluation.value
            for evaluation in result.run_evaluations
            if evaluation.name == "avg_expected_match"
        ),
        0.0,
    )
    if not isinstance(avg_score, (int, float)) or avg_score < MIN_EXPECTED_MATCH:
        raise RegressionError(
            result=result,
            metric="avg_expected_match",
            value=float(avg_score) if isinstance(avg_score, (int, float)) else 0.0,
            threshold=MIN_EXPECTED_MATCH,
        )
    return result


def answer_question(item, **kwargs) -> str:
    payload = _input_payload(item)
    question = str(payload.get("question") or payload.get("input") or "")
    contexts = _format_contexts(payload.get("contexts", []))
    langfuse = get_client()
    prompt = langfuse.get_prompt(
        PROMPT_NAME,
        type="chat",
        label=PROMPT_LABEL,
        cache_ttl_seconds=0,
    )
    messages = prompt.compile(question=question, contexts=contexts or "无")
    config = prompt.config or {}
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"),
        base_url=_openai_base_url(),
    )
    response = client.chat.completions.create(
        model=str(config.get("model") or DEFAULT_MODEL),
        messages=messages,
        temperature=float(config.get("temperature", 0.2)),
        max_tokens=int(config.get("max_tokens") or os.getenv("LLM_MAX_OUTPUT_TOKENS", "512")),
        langfuse_prompt=prompt,
    )
    return str(response.choices[0].message.content or "").strip()


def contains_expected(*, output: str, expected_output: Any = None, metadata: Any = None, **kwargs):
    expected_values = _expected_values(expected_output, metadata)
    if not expected_values:
        return Evaluation(name="contains_expected", value=1.0, comment="no expected output declared")
    matched = [expected for expected in expected_values if expected.lower() in output.lower()]
    return Evaluation(
        name="contains_expected",
        value=1.0 if len(matched) == len(expected_values) else 0.0,
        comment=f"matched {len(matched)}/{len(expected_values)} expected snippets",
    )


def average_expected_match(*, item_results, **kwargs):
    scores = [
        evaluation.value
        for item in item_results
        for evaluation in item.evaluations
        if evaluation.name == "contains_expected" and isinstance(evaluation.value, (int, float))
    ]
    return Evaluation(
        name="avg_expected_match",
        value=sum(scores) / len(scores) if scores else 0.0,
    )


def _input_payload(item) -> Dict[str, Any]:
    payload = getattr(item, "input", item)
    return payload if isinstance(payload, dict) else {"input": payload}


def _format_contexts(contexts: Any) -> str:
    if isinstance(contexts, str):
        return contexts
    if not isinstance(contexts, list):
        return ""
    formatted = []
    for index, context in enumerate(contexts, start=1):
        if isinstance(context, dict):
            filename = context.get("filename", f"context-{index}")
            content = context.get("content", "")
            formatted.append(f"[{index}] {filename}\n{content}")
        else:
            formatted.append(f"[{index}] context-{index}\n{context}")
    return "\n\n".join(formatted)


def _expected_values(expected_output: Any, metadata: Any) -> List[str]:
    values: List[str] = []
    if isinstance(expected_output, str) and expected_output.strip():
        values.append(expected_output.strip())
    if isinstance(metadata, dict):
        keywords = metadata.get("expected_keywords", [])
        if isinstance(keywords, str):
            values.append(keywords.strip())
        elif isinstance(keywords, Iterable):
            values.extend(str(keyword).strip() for keyword in keywords if str(keyword).strip())
    return values


def _openai_base_url() -> Optional[str]:
    base_url = os.getenv("LLM_OPENAI_BASE_URL", "")
    if not base_url:
        return None
    return base_url if base_url.rstrip("/").endswith("/v1") else f"{base_url.rstrip('/')}/v1"
