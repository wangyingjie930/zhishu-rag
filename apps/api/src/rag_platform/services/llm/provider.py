from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Sequence

import httpx

from rag_platform.core.config import settings
from rag_platform.services.prompts import RagAnswerPromptManager, ResolvedPrompt
from rag_platform.services.retrieval.hybrid import RetrievedChunk


@dataclass(frozen=True)
class LLMGeneration:
    answer: str
    prompt: Optional[Any] = None
    prompt_metadata: Dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    async def answer(self, question: str, contexts: Sequence[RetrievedChunk]) -> str:
        ...


class MockLLMProvider:
    async def generate_answer(
        self,
        question: str,
        contexts: Sequence[RetrievedChunk],
    ) -> LLMGeneration:
        return LLMGeneration(answer=await self.answer(question, contexts))

    async def answer(self, question: str, contexts: Sequence[RetrievedChunk]) -> str:
        if not contexts:
            return "我没有在当前知识库中检索到足够相关的内容。建议补充文档或换一种问法。"
        return (
            "已完成知识库召回。当前还没有接入正式生成模型，"
            "请在下方引用来源中查看本次命中的文本片段。"
        )


class OpenAICompatibleLLMProvider:
    def __init__(
        self,
        base_url: str = settings.llm_openai_base_url,
        model: str = settings.llm_model,
        api_key: str = settings.llm_api_key or settings.openai_api_key,
        max_output_tokens: int = settings.llm_max_output_tokens,
    ) -> None:
        if not base_url:
            raise ValueError("LLM_OPENAI_BASE_URL is required")
        if not model:
            raise ValueError("LLM_MODEL is required")
        if max_output_tokens <= 0:
            raise ValueError("LLM_MAX_OUTPUT_TOKENS is required")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.prompt_manager = RagAnswerPromptManager()

    def _request_url(self) -> str:
        api_root = self.base_url if self.base_url.endswith("/v1") else f"{self.base_url}/v1"
        return f"{api_root}/chat/completions"

    def _request_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request_payload(self, prompt: ResolvedPrompt) -> dict:
        payload = {
            "model": self._model_from_config(prompt.config),
            "messages": prompt.messages,
            "temperature": self._temperature_from_config(prompt.config),
            "max_tokens": self._max_tokens_from_config(prompt.config),
        }
        for key in ("response_format", "tools", "tool_choice"):
            if key in prompt.config:
                payload[key] = prompt.config[key]
        return payload

    def _parse_answer(self, payload: dict) -> str:
        return str(payload["choices"][0]["message"]["content"]).strip()

    async def generate_answer(
        self,
        question: str,
        contexts: Sequence[RetrievedChunk],
    ) -> LLMGeneration:
        prompt = self.prompt_manager.resolve(question, contexts)
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                self._request_url(),
                headers=self._request_headers(),
                json=self._request_payload(prompt),
            )
            response.raise_for_status()
        return LLMGeneration(
            answer=self._parse_answer(response.json()),
            prompt=prompt.langfuse_prompt,
            prompt_metadata=prompt.metadata,
        )

    async def answer(self, question: str, contexts: Sequence[RetrievedChunk]) -> str:
        generation = await self.generate_answer(question, contexts)
        return generation.answer

    def _model_from_config(self, config: Dict[str, Any]) -> str:
        return str(config.get("model") or self.model)

    def _temperature_from_config(self, config: Dict[str, Any]) -> float:
        try:
            return float(config.get("temperature", 0.2))
        except (TypeError, ValueError):
            return 0.2

    def _max_tokens_from_config(self, config: Dict[str, Any]) -> int:
        try:
            value = int(config.get("max_tokens") or self.max_output_tokens)
        except (TypeError, ValueError):
            return self.max_output_tokens
        return value if value > 0 else self.max_output_tokens


def get_llm_provider() -> LLMProvider:
    provider = settings.llm_provider.strip().lower()
    if provider in {"openai", "openai-compatible"}:
        return OpenAICompatibleLLMProvider()
    return MockLLMProvider()
