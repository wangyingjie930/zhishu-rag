import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.indices.query.query_transform.base import BaseQueryTransform
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import QueryBundle

from rag_platform.core.config import settings


@dataclass(frozen=True)
class HyDEExpansion:
    embedding_texts: List[str]
    hypothetical_document: Optional[str]


@dataclass(frozen=True)
class QueryExpansion:
    queries: List[str]
    embedding_texts: List[str]


QUERY_EXPANSION_PROMPT = PromptTemplate(
    """你是企业知识库检索查询扩展器。
请围绕用户问题生成 {num_queries} 条更适合检索的中文查询改写，覆盖同义词、术语补全和可能的文档表达。
要求：
- 不回答问题，只输出查询。
- 不引入用户问题之外的新实体。
- 输出 JSON 字符串数组，例如 ["查询一", "查询二"]。

用户问题：{query_str}
"""
)


class OpenAICompatibleQueryLLM(CustomLLM):
    base_url: str = settings.hyde_openai_base_url
    model: str = settings.hyde_llm_model
    api_key: str = settings.openai_api_key
    max_output_tokens: int = settings.hyde_max_output_tokens
    timeout_seconds: float = 30

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=32768,
            num_output=self.max_output_tokens,
            model_name=self.model,
        )

    def _request_url(self) -> str:
        if not self.base_url:
            raise ValueError("HYDE_OPENAI_BASE_URL is required")
        base_url = self.base_url.rstrip("/")
        api_root = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
        return f"{api_root}/chat/completions"

    def _request_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request_payload(self, prompt: str) -> dict:
        if not self.model:
            raise ValueError("HYDE_LLM_MODEL is required")
        if self.max_output_tokens <= 0:
            raise ValueError("HYDE_MAX_OUTPUT_TOKENS is required")
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": self.max_output_tokens,
        }

    def _parse_text(self, payload: dict) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError("HyDE OpenAI-compatible endpoint returned no choices")

        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            text = "\n".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("text")
            ).strip()
        else:
            text = str(content).strip()

        if not text:
            text = str(choices[0].get("text", "")).strip()
        if not text:
            raise ValueError("HyDE OpenAI-compatible endpoint returned empty content")
        return text

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                self._request_url(),
                headers=self._request_headers(),
                json=self._request_payload(prompt),
            )
            response.raise_for_status()

        return CompletionResponse(text=self._parse_text(response.json()))

    def stream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponseGen:
        yield self.complete(prompt, formatted=formatted, **kwargs)


class HyDEQueryExpander:
    def __init__(self, llm: Optional[CustomLLM] = None) -> None:
        self.llm = llm or OpenAICompatibleQueryLLM()

    def expand(self, query: str, include_original: bool = True) -> HyDEExpansion:
        # 这里使用 LlamaIndex 原生 HyDEQueryTransform，保留原问题可降低生成漂移风险。
        transform = HyDEQueryTransform(llm=self.llm, include_original=include_original)
        bundle = transform.run(query)
        embedding_texts = [text.strip() for text in bundle.embedding_strs if text.strip()]
        hypothetical_document = embedding_texts[0] if embedding_texts else None

        if not embedding_texts:
            embedding_texts = [query]
        return HyDEExpansion(
            embedding_texts=embedding_texts,
            hypothetical_document=hypothetical_document,
        )


class QueryExpansionTransform(BaseQueryTransform):
    def __init__(
        self,
        llm: Optional[CustomLLM] = None,
        expansion_prompt: Optional[BasePromptTemplate] = None,
        num_queries: int = 3,
        include_original: bool = True,
    ) -> None:
        super().__init__()
        self._llm = llm or OpenAICompatibleQueryLLM()
        self._expansion_prompt = expansion_prompt or QUERY_EXPANSION_PROMPT
        self._num_queries = num_queries
        self._include_original = include_original

    def _get_prompts(self) -> PromptDictType:
        return {"expansion_prompt": self._expansion_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        if "expansion_prompt" in prompts:
            self._expansion_prompt = prompts["expansion_prompt"]

    def _run(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
        query_str = query_bundle.query_str
        response = self._llm.predict(
            self._expansion_prompt,
            query_str=query_str,
            num_queries=self._num_queries,
        )
        expanded_queries = self._parse_queries(response, query_str)
        embedding_strs = expanded_queries
        if self._include_original:
            embedding_strs = [query_str, *expanded_queries]
        return QueryBundle(
            query_str=query_str,
            custom_embedding_strs=embedding_strs,
        )

    def _parse_queries(self, raw_text: str, original_query: str) -> List[str]:
        normalized = raw_text.strip()
        queries = self._parse_json_queries(normalized) or self._parse_line_queries(normalized)

        deduped: List[str] = []
        seen = {original_query.strip()}
        for query in queries:
            cleaned = query.strip().strip('"').strip("'")
            if not cleaned or cleaned in seen:
                continue
            deduped.append(cleaned)
            seen.add(cleaned)
            if len(deduped) >= self._num_queries:
                break
        return deduped

    def _parse_json_queries(self, raw_text: str) -> List[str]:
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        if isinstance(parsed, dict):
            for key in ("queries", "expanded_queries", "query_expansions"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [str(item) for item in value]
        return []

    def _parse_line_queries(self, raw_text: str) -> List[str]:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        return [re.sub(r"^[-*\d.、)\s]+", "", line).strip() for line in lines]


class QueryExpansionExpander:
    def __init__(self, llm: Optional[CustomLLM] = None, num_queries: int = 3) -> None:
        self.llm = llm or OpenAICompatibleQueryLLM()
        self.num_queries = num_queries

    def expand(self, query: str, include_original: bool = True) -> QueryExpansion:
        # 使用 LlamaIndex QueryTransform 把扩展查询写入 QueryBundle.embedding_strs。
        transform = QueryExpansionTransform(
            llm=self.llm,
            num_queries=self.num_queries,
            include_original=include_original,
        )
        bundle = transform.run(query)
        embedding_texts = [text.strip() for text in bundle.embedding_strs if text.strip()]
        queries = [text for text in embedding_texts if text != query]
        if not embedding_texts:
            embedding_texts = [query]
        return QueryExpansion(queries=queries, embedding_texts=embedding_texts)
