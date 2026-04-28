from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Protocol

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding

from rag_platform.core.config import settings

DEFAULT_EMBEDDING_MODEL_ID = "google:gemini-embedding-001"


class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> List[float]:
        ...


@dataclass(frozen=True)
class EmbeddingModelOption:
    id: str
    label: str
    provider: str
    model: str
    dimensions: int
    enabled: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


EMBEDDING_MODEL_OPTIONS = [
    EmbeddingModelOption(
        id=DEFAULT_EMBEDDING_MODEL_ID,
        label="Google Gemini Embedding 001",
        provider="google",
        model="gemini-embedding-001",
        dimensions=settings.embedding_dim,
        enabled=bool(settings.google_api_key),
        reason="" if settings.google_api_key else "未配置 GOOGLE_API_KEY",
    ),
]


def list_embedding_model_options() -> List[Dict[str, Any]]:
    return [option.to_dict() for option in EMBEDDING_MODEL_OPTIONS]


def get_embedding_model_option(option_id: Optional[str]) -> EmbeddingModelOption:
    selected = option_id or DEFAULT_EMBEDDING_MODEL_ID
    for option in EMBEDDING_MODEL_OPTIONS:
        if option.id == selected:
            return option
    raise ValueError(f"Unknown embedding model: {selected}")


class GoogleEmbeddingClient:
    """Gemini embedding HTTP client shared by ingestion and LlamaIndex semantic chunking.

    The Postgres vector column is fixed at settings.embedding_dim, so the provider requests that
    dimensionality from Gemini instead of letting the API return a model-specific default.
    """

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        dimensions: int = settings.embedding_dim,
        api_key: str = settings.google_api_key,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> None:
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for Google embeddings")
        self.model = model
        self.dimensions = dimensions
        self.api_key = api_key
        self.task_type = task_type

    def _request_url(self) -> str:
        model_name = self.model if self.model.startswith("models/") else f"models/{self.model}"
        return f"https://generativelanguage.googleapis.com/v1beta/{model_name}:embedContent"

    def _request_payload(self, text: str) -> Dict[str, Any]:
        return {
            "content": {"parts": [{"text": text}]},
            "taskType": self.task_type,
            "outputDimensionality": self.dimensions,
        }

    def _parse_embedding(self, payload: Dict[str, Any]) -> List[float]:
        values = payload["embedding"]["values"]
        if len(values) != self.dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimensions}, got {len(values)}"
            )
        return [float(value) for value in values]

    def embed(self, text: str) -> List[float]:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                self._request_url(),
                params={"key": self.api_key},
                json=self._request_payload(text),
            )
            response.raise_for_status()
        return self._parse_embedding(response.json())

    async def aembed(self, text: str) -> List[float]:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                self._request_url(),
                params={"key": self.api_key},
                json=self._request_payload(text),
            )
            response.raise_for_status()
        return self._parse_embedding(response.json())


class GoogleEmbeddingProvider:
    def __init__(
        self,
        model: str = "gemini-embedding-001",
        dimensions: int = settings.embedding_dim,
        api_key: str = settings.google_api_key,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> None:
        self.client = GoogleEmbeddingClient(
            model=model,
            dimensions=dimensions,
            api_key=api_key,
            task_type=task_type,
        )

    async def embed(self, text: str) -> List[float]:
        return await self.client.aembed(text)


class GoogleLlamaIndexEmbedding(BaseEmbedding):
    """LlamaIndex 语义分块适配器，明确只接真实 embedding 服务。"""

    google_model: str = "gemini-embedding-001"
    dimensions: int = settings.embedding_dim
    api_key: str = settings.google_api_key
    task_type: str = "RETRIEVAL_DOCUMENT"

    def _client(self) -> GoogleEmbeddingClient:
        # 语义分块是同步接口，这里复用同一套 Gemini 请求约定。
        return GoogleEmbeddingClient(
            model=self.google_model,
            dimensions=self.dimensions,
            api_key=self.api_key,
            task_type=self.task_type,
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._client().embed(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await self._client().aembed(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._client().embed(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._client().aembed(query)


class EmbeddingProviderRegistry:
    def get(self, option_id: Optional[str] = None, usage: str = "document") -> EmbeddingProvider:
        option = get_embedding_model_option(option_id)
        if not option.enabled:
            raise ValueError(option.reason or f"Embedding model is disabled: {option.id}")
        if option.provider == "google":
            task_type = "RETRIEVAL_QUERY" if usage == "query" else "RETRIEVAL_DOCUMENT"
            return GoogleEmbeddingProvider(
                model=option.model,
                dimensions=option.dimensions,
                task_type=task_type,
            )
        raise ValueError(f"Unsupported embedding provider: {option.provider}")

    def get_llama_index(self, option_id: Optional[str] = None) -> BaseEmbedding:
        option = get_embedding_model_option(option_id)
        if not option.enabled:
            raise ValueError(option.reason or f"Embedding model is disabled: {option.id}")
        if option.provider == "google":
            return GoogleLlamaIndexEmbedding(
                google_model=option.model,
                dimensions=option.dimensions,
                task_type="RETRIEVAL_DOCUMENT",
            )
        raise ValueError(f"Unsupported LlamaIndex embedding provider: {option.provider}")
