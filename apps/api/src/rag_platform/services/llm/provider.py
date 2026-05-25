from typing import Protocol, Sequence

import httpx

from rag_platform.core.config import settings
from rag_platform.services.retrieval.hybrid import RetrievedChunk


class LLMProvider(Protocol):
    async def answer(self, question: str, contexts: Sequence[RetrievedChunk]) -> str:
        ...


class MockLLMProvider:
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

    def _request_url(self) -> str:
        api_root = self.base_url if self.base_url.endswith("/v1") else f"{self.base_url}/v1"
        return f"{api_root}/chat/completions"

    def _request_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request_payload(self, question: str, contexts: Sequence[RetrievedChunk]) -> dict:
        context_text = "\n\n".join(
            f"[{index}] {chunk.filename}\n{chunk.content}"
            for index, chunk in enumerate(contexts, start=1)
        )
        system_prompt = (
            "你是企业知识库问答助手。只根据给定的知识库片段回答；"
            "如果片段不足以回答，就说明没有检索到足够依据。"
        )
        user_prompt = f"问题：{question}\n\n知识库片段：\n{context_text or '无'}"
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": self.max_output_tokens,
        }

    def _parse_answer(self, payload: dict) -> str:
        return str(payload["choices"][0]["message"]["content"]).strip()

    async def answer(self, question: str, contexts: Sequence[RetrievedChunk]) -> str:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                self._request_url(),
                headers=self._request_headers(),
                json=self._request_payload(question, contexts),
            )
            response.raise_for_status()
        return self._parse_answer(response.json())


def get_llm_provider() -> LLMProvider:
    provider = settings.llm_provider.strip().lower()
    if provider in {"openai", "openai-compatible"}:
        return OpenAICompatibleLLMProvider()
    return MockLLMProvider()
