from typing import List, Optional

from ragas.embeddings.base import BaseRagasEmbedding

from rag_platform.core.config import settings
from rag_platform.services.ingestion.embeddings import (
    GoogleEmbeddingClient,
    OpenAICompatibleEmbeddingClient,
    get_embedding_model_option,
)


class RagasConfiguredEmbedding(BaseRagasEmbedding):
    """让 RAGAS 评测复用平台的 embedding 配置，避免评测和检索跑到不同向量空间。"""

    def __init__(self, model_id: Optional[str] = None) -> None:
        super().__init__()
        self.option = get_embedding_model_option(model_id or settings.embedding_model)

    def _client(self):
        if self.option.provider == "google":
            return GoogleEmbeddingClient(
                model=self.option.model,
                dimensions=self.option.dimensions,
                task_type="SEMANTIC_SIMILARITY",
            )
        if self.option.provider == "openai-compatible":
            return OpenAICompatibleEmbeddingClient(
                model=self.option.model,
                dimensions=self.option.dimensions,
            )
        raise ValueError(f"Unsupported RAGAS embedding provider: {self.option.provider}")

    def embed_text(self, text: str, **kwargs) -> List[float]:
        return self._client().embed(text)

    async def aembed_text(self, text: str, **kwargs) -> List[float]:
        return await self._client().aembed(text)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]

    async def aembed_query(self, text: str) -> List[float]:
        return await self.aembed_text(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return [await self.aembed_text(text) for text in texts]
