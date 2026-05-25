from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Protocol

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding

from rag_platform.core.config import settings

DEFAULT_EMBEDDING_MODEL_ID = settings.embedding_model.strip()


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


def _embedding_model_options() -> List[EmbeddingModelOption]:
    if not DEFAULT_EMBEDDING_MODEL_ID:
        return []

    provider = _provider_for_model_id(DEFAULT_EMBEDDING_MODEL_ID)
    missing_config = []
    if not settings.embedding_base_url:
        missing_config.append("EMBEDDING_BASE_URL")
    if provider == "google" and not settings.embedding_endpoint_path:
        missing_config.append("EMBEDDING_ENDPOINT_PATH")
    if settings.embedding_dim <= 0:
        missing_config.append("EMBEDDING_DIM")
    if not _embedding_api_key_for_provider(provider):
        missing_config.append("EMBEDDING_API_KEY")

    return [
        EmbeddingModelOption(
            id=DEFAULT_EMBEDDING_MODEL_ID,
            label=DEFAULT_EMBEDDING_MODEL_ID,
            provider=provider,
            model=_provider_model_name(DEFAULT_EMBEDDING_MODEL_ID),
            dimensions=settings.embedding_dim,
            enabled=not missing_config,
            reason="" if not missing_config else f"未配置 {', '.join(missing_config)}",
        ),
    ]


def list_embedding_model_options() -> List[Dict[str, Any]]:
    return [option.to_dict() for option in _embedding_model_options()]


def get_embedding_model_option(option_id: Optional[str]) -> EmbeddingModelOption:
    selected = resolve_embedding_model_id(option_id)
    if not selected:
        raise ValueError("EMBEDDING_MODEL is required")
    for option in _embedding_model_options():
        if option.id == selected:
            return option
    raise ValueError(f"Unknown embedding model: {selected}")


def resolve_embedding_model_id(option_id: Optional[str]) -> str:
    selected = (option_id or "").strip() or DEFAULT_EMBEDDING_MODEL_ID
    if not selected:
        raise ValueError("EMBEDDING_MODEL is required")

    option_ids = {option.id for option in _embedding_model_options()}
    if selected in option_ids:
        return selected
    if DEFAULT_EMBEDDING_MODEL_ID:
        return DEFAULT_EMBEDDING_MODEL_ID
    raise ValueError(f"Unknown embedding model: {selected}")


def _provider_for_model_id(model_id: str) -> str:
    return "google" if model_id.startswith("google:") else "openai-compatible"


def _provider_model_name(model_id: str) -> str:
    return model_id.split(":", 1)[1] if ":" in model_id else model_id


def _embedding_api_key_for_provider(provider: str) -> str:
    if provider == "google":
        return settings.embedding_api_key or settings.google_api_key
    return settings.embedding_api_key


class OpenAICompatibleEmbeddingClient:
    """Embedding HTTP client shared by ingestion and LlamaIndex semantic chunking."""

    def __init__(
        self,
        model: str = settings.embedding_model,
        dimensions: int = settings.embedding_dim,
        api_key: str = settings.embedding_api_key,
        base_url: str = settings.embedding_base_url,
        endpoint_path: str = settings.embedding_endpoint_path,
    ) -> None:
        if not model:
            raise ValueError("EMBEDDING_MODEL is required")
        if not base_url:
            raise ValueError("EMBEDDING_BASE_URL is required")
        if dimensions <= 0:
            raise ValueError("EMBEDDING_DIM is required")
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY is required for embeddings")
        self.model = model
        self.dimensions = dimensions
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.endpoint_path = endpoint_path.strip()

    def _request_url(self) -> str:
        if not self.endpoint_path:
            return self.base_url
        endpoint_path = self.endpoint_path.strip("/")
        return f"{self.base_url}/{endpoint_path}"

    def _request_payload(self, text: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "input": text,
            "dimensions": self.dimensions,
        }

    def _parse_embedding(self, payload: Dict[str, Any]) -> List[float]:
        values = payload["data"][0]["embedding"]
        if len(values) != self.dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimensions}, got {len(values)}"
            )
        return [float(value) for value in values]

    def embed(self, text: str) -> List[float]:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                self._request_url(),
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=self._request_payload(text),
            )
            response.raise_for_status()
        return self._parse_embedding(response.json())

    async def aembed(self, text: str) -> List[float]:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                self._request_url(),
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=self._request_payload(text),
            )
            response.raise_for_status()
        return self._parse_embedding(response.json())


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        model: str = settings.embedding_model,
        dimensions: int = settings.embedding_dim,
        api_key: str = settings.embedding_api_key,
        base_url: str = settings.embedding_base_url,
        endpoint_path: str = settings.embedding_endpoint_path,
    ) -> None:
        self.client = OpenAICompatibleEmbeddingClient(
            model=model,
            dimensions=dimensions,
            api_key=api_key,
            base_url=base_url,
            endpoint_path=endpoint_path,
        )

    async def embed(self, text: str) -> List[float]:
        return await self.client.aembed(text)


class GoogleEmbeddingClient:
    """Gemini embedding HTTP client shared by ingestion and LlamaIndex semantic chunking."""

    def __init__(
        self,
        model: str = _provider_model_name(settings.embedding_model),
        dimensions: int = settings.embedding_dim,
        api_key: str = settings.embedding_api_key or settings.google_api_key,
        base_url: str = settings.embedding_base_url,
        endpoint_path: str = settings.embedding_endpoint_path,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> None:
        if not model:
            raise ValueError("EMBEDDING_MODEL is required")
        if not base_url:
            raise ValueError("EMBEDDING_BASE_URL is required")
        if not endpoint_path:
            raise ValueError("EMBEDDING_ENDPOINT_PATH is required")
        if dimensions <= 0:
            raise ValueError("EMBEDDING_DIM is required")
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY is required for embeddings")
        self.model = model
        self.dimensions = dimensions
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.endpoint_path = endpoint_path.strip("/")
        self.task_type = task_type

    def _request_url(self) -> str:
        return f"{self.base_url}/{self.endpoint_path}"

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
                headers={"x-goog-api-key": self.api_key},
                json=self._request_payload(text),
            )
            response.raise_for_status()
        return self._parse_embedding(response.json())

    async def aembed(self, text: str) -> List[float]:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                self._request_url(),
                headers={"x-goog-api-key": self.api_key},
                json=self._request_payload(text),
            )
            response.raise_for_status()
        return self._parse_embedding(response.json())


class GoogleEmbeddingProvider:
    def __init__(
        self,
        model: str,
        dimensions: int = settings.embedding_dim,
        api_key: str = settings.embedding_api_key or settings.google_api_key,
        base_url: str = settings.embedding_base_url,
        endpoint_path: str = settings.embedding_endpoint_path,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> None:
        self.client = GoogleEmbeddingClient(
            model=model,
            dimensions=dimensions,
            api_key=api_key,
            base_url=base_url,
            endpoint_path=endpoint_path,
            task_type=task_type,
        )

    async def embed(self, text: str) -> List[float]:
        return await self.client.aembed(text)


class OpenAICompatibleLlamaIndexEmbedding(BaseEmbedding):
    """LlamaIndex 语义分块适配器，明确只接真实 embedding 服务。"""

    embedding_model: str = settings.embedding_model
    base_url: str = settings.embedding_base_url
    dimensions: int = settings.embedding_dim
    api_key: str = settings.embedding_api_key
    endpoint_path: str = settings.embedding_endpoint_path

    def _client(self) -> OpenAICompatibleEmbeddingClient:
        return OpenAICompatibleEmbeddingClient(
            model=self.embedding_model,
            dimensions=self.dimensions,
            api_key=self.api_key,
            base_url=self.base_url,
            endpoint_path=self.endpoint_path,
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._client().embed(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await self._client().aembed(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._client().embed(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._client().aembed(query)


class GoogleLlamaIndexEmbedding(BaseEmbedding):
    """LlamaIndex 语义分块适配器，明确只接真实 embedding 服务。"""

    embedding_model: str = _provider_model_name(settings.embedding_model)
    base_url: str = settings.embedding_base_url
    dimensions: int = settings.embedding_dim
    api_key: str = settings.embedding_api_key or settings.google_api_key
    endpoint_path: str = settings.embedding_endpoint_path
    task_type: str = "RETRIEVAL_DOCUMENT"

    def _client(self) -> GoogleEmbeddingClient:
        return GoogleEmbeddingClient(
            model=self.embedding_model,
            dimensions=self.dimensions,
            api_key=self.api_key,
            base_url=self.base_url,
            endpoint_path=self.endpoint_path,
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
        if option.provider == "openai-compatible":
            return OpenAICompatibleEmbeddingProvider(
                model=option.model,
                dimensions=option.dimensions,
            )
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
        if option.provider == "openai-compatible":
            return OpenAICompatibleLlamaIndexEmbedding(
                embedding_model=option.model,
                dimensions=option.dimensions,
            )
        if option.provider == "google":
            return GoogleLlamaIndexEmbedding(
                embedding_model=option.model,
                dimensions=option.dimensions,
                task_type="RETRIEVAL_DOCUMENT",
            )
        raise ValueError(f"Unsupported LlamaIndex embedding provider: {option.provider}")
