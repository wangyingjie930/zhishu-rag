import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from rag_platform.core.config import settings
from rag_platform.services.observability import get_langfuse_observability
from rag_platform.services.retrieval.hybrid import RetrievedChunk


logger = logging.getLogger(__name__)

DEFAULT_RAG_SYSTEM_PROMPT = (
    "你是企业知识库问答助手。只根据给定的知识库片段回答；"
    "如果片段不足以回答，就说明没有检索到足够依据。"
)


@dataclass(frozen=True)
class ResolvedPrompt:
    messages: List[Dict[str, str]]
    config: Dict[str, Any] = field(default_factory=dict)
    langfuse_prompt: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RagAnswerPromptManager:
    def __init__(self) -> None:
        self.observability = get_langfuse_observability()

    def resolve(self, question: str, contexts: Sequence[RetrievedChunk]) -> ResolvedPrompt:
        context_text = self._format_contexts(contexts)
        fallback_messages = self._fallback_messages(question, context_text)
        langfuse_prompt = self._fetch_langfuse_prompt()
        if langfuse_prompt is None:
            return ResolvedPrompt(messages=fallback_messages, metadata={"prompt_source": "fallback"})

        try:
            compiled = langfuse_prompt.compile(question=question, contexts=context_text or "无")
            messages = self._normalize_messages(compiled)
        except Exception as exc:  # pragma: no cover - Langfuse SDK boundary
            logger.warning("Langfuse prompt compile failed, using fallback prompt: %s", exc)
            return ResolvedPrompt(messages=fallback_messages, metadata={"prompt_source": "fallback"})

        if not messages:
            logger.warning("Langfuse prompt compiled to empty messages, using fallback prompt")
            return ResolvedPrompt(messages=fallback_messages, metadata={"prompt_source": "fallback"})

        return ResolvedPrompt(
            messages=messages,
            config=self._prompt_config(langfuse_prompt),
            langfuse_prompt=langfuse_prompt,
            metadata=self._prompt_metadata(langfuse_prompt),
        )

    def _fetch_langfuse_prompt(self) -> Optional[Any]:
        return self.observability.get_prompt(
            name=settings.rag_answer_prompt_name,
            prompt_type="chat",
            label=settings.rag_answer_prompt_label,
            cache_ttl_seconds=settings.rag_answer_prompt_cache_ttl_seconds,
            fallback=self._fallback_template(),
        )

    def _fallback_template(self) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": DEFAULT_RAG_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "问题：{{question}}\n\n知识库片段：\n{{contexts}}",
            },
        ]

    def _fallback_messages(self, question: str, context_text: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": DEFAULT_RAG_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"问题：{question}\n\n知识库片段：\n{context_text or '无'}",
            },
        ]

    def _normalize_messages(self, compiled_prompt: Any) -> List[Dict[str, str]]:
        if not isinstance(compiled_prompt, list):
            return []
        messages = []
        for message in compiled_prompt:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if not role or not content:
                continue
            messages.append({"role": role, "content": content})
        return messages

    def _format_contexts(self, contexts: Sequence[RetrievedChunk]) -> str:
        return "\n\n".join(
            f"[{index}] {chunk.filename}\n{chunk.content}"
            for index, chunk in enumerate(contexts, start=1)
        )

    def _prompt_config(self, langfuse_prompt: Any) -> Dict[str, Any]:
        config = getattr(langfuse_prompt, "config", {}) or {}
        return config if isinstance(config, dict) else {}

    def _prompt_metadata(self, langfuse_prompt: Any) -> Dict[str, Any]:
        metadata = {
            "prompt_source": "langfuse",
            "prompt_name": getattr(langfuse_prompt, "name", settings.rag_answer_prompt_name),
            "prompt_version": getattr(langfuse_prompt, "version", ""),
            "prompt_label": settings.rag_answer_prompt_label,
            "prompt_is_fallback": getattr(langfuse_prompt, "is_fallback", False),
        }
        return {key: value for key, value in metadata.items() if value not in {"", None}}
