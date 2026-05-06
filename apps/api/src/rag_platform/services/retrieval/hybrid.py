import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import KnowledgeBase
from rag_platform.services.ingestion.embeddings import (
    DEFAULT_EMBEDDING_MODEL_ID,
    EmbeddingProvider,
    EmbeddingProviderRegistry,
)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    content: str
    score: float
    metadata: Dict[str, Any]


class HybridRetriever:
    def __init__(self, embedding_provider: EmbeddingProvider = None) -> None:
        self.embedding_provider = embedding_provider
        self.embedding_providers = EmbeddingProviderRegistry()

    async def retrieve(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
        query: str,
        top_k: int = 8,
    ) -> tuple[List[RetrievedChunk], Dict[str, Any]]:
        embedding_model = await self._load_embedding_model(session, tenant_id, kb_id)
        embedding_provider = self.embedding_provider or self.embedding_providers.get(
            embedding_model,
            usage="query",
        )
        embedding = await embedding_provider.embed(query)
        vector_rows = await self._vector_search(
            session,
            tenant_id,
            kb_id,
            embedding,
            embedding_model,
            top_k * 2,
        )
        keyword_rows = await self._keyword_search(session, tenant_id, kb_id, query, top_k * 2)
        fused = self._rrf(vector_rows, keyword_rows)[:top_k]
        trace = {
            "retriever": "hybrid_rrf",
            "vector_candidates": len(vector_rows),
            "keyword_candidates": len(keyword_rows),
            "top_k": top_k,
            "embedding_model": embedding_model,
            "reranker": "none",
        }
        return fused, trace

    async def _load_embedding_model(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
    ) -> str:
        result = await session.execute(
            select(KnowledgeBase.ingestion_policy).where(
                KnowledgeBase.tenant_id == tenant_id,
                KnowledgeBase.id == kb_id,
            )
        )
        policy = result.scalar_one_or_none() or {}
        return policy.get("embedding", {}).get("model", DEFAULT_EMBEDDING_MODEL_ID)

    async def _vector_search(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
        embedding: List[float],
        embedding_model: str,
        limit: int,
    ) -> List[RetrievedChunk]:
        vector_literal = "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"
        result = await session.execute(
            text(
                """
                SELECT
                  c.id AS chunk_id,
                  c.document_id,
                  d.filename,
                  c.content,
                  c.metadata,
                  1 - (c.embedding <=> CAST(:embedding AS vector)) AS score
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.tenant_id = :tenant_id
                  AND c.kb_id = :kb_id
                  AND c.embedding IS NOT NULL
                  AND COALESCE(
                    c.metadata->'embedding'->>'model',
                    :default_embedding_model
                  ) = :embedding_model
                ORDER BY c.embedding <=> CAST(:embedding AS vector)
                LIMIT :limit
                """
            ),
            {
                "tenant_id": tenant_id,
                "kb_id": kb_id,
                "embedding": vector_literal,
                "embedding_model": embedding_model,
                "default_embedding_model": DEFAULT_EMBEDDING_MODEL_ID,
                "limit": limit,
            },
        )
        return [self._row_to_chunk(row._mapping) for row in result]

    async def _keyword_search(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
        query: str,
        limit: int,
    ) -> List[RetrievedChunk]:
        result = await session.execute(
            text(
                """
                SELECT
                  c.id AS chunk_id,
                  c.document_id,
                  d.filename,
                  c.content,
                  c.metadata,
                  ts_rank_cd(c.search_vector, websearch_to_tsquery('simple', :query)) AS score
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.tenant_id = :tenant_id
                  AND c.kb_id = :kb_id
                  AND c.search_vector @@ websearch_to_tsquery('simple', :query)
                ORDER BY score DESC
                LIMIT :limit
                """
            ),
            {"tenant_id": tenant_id, "kb_id": kb_id, "query": query, "limit": limit},
        )
        return [self._row_to_chunk(row._mapping) for row in result]

    def _rrf(
        self,
        vector_rows: Iterable[RetrievedChunk],
        keyword_rows: Iterable[RetrievedChunk],
        k: int = 60,
    ) -> List[RetrievedChunk]:
        by_id: Dict[uuid.UUID, RetrievedChunk] = {}
        scores: Dict[uuid.UUID, float] = {}
        for rows, weight in ((vector_rows, 0.65), (keyword_rows, 0.35)):
            for rank, chunk in enumerate(rows, start=1):
                by_id[chunk.chunk_id] = chunk
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (k + rank)
        ranked = sorted(by_id.values(), key=lambda item: scores[item.chunk_id], reverse=True)
        return [
            RetrievedChunk(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                filename=item.filename,
                content=item.content,
                score=scores[item.chunk_id],
                metadata=item.metadata,
            )
            for item in ranked
        ]

    def _row_to_chunk(self, row: Any) -> RetrievedChunk:
        metadata = dict(row["metadata"] or {})
        content = row["content"]
        parent_text = metadata.get("parent_text")
        if metadata.get("chunker") == "parent_child" and isinstance(parent_text, str):
            metadata = {
                **metadata,
                "matched_child_content": content,
            }
            content = parent_text

        return RetrievedChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            filename=row["filename"],
            content=content,
            score=float(row["score"] or 0.0),
            metadata=metadata,
        )
