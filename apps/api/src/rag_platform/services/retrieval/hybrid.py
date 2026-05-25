import asyncio
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional

from sqlalchemy import bindparam, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import KnowledgeBase
from rag_platform.services.ingestion.embeddings import (
    DEFAULT_EMBEDDING_MODEL_ID,
    EmbeddingProvider,
    EmbeddingProviderRegistry,
    resolve_embedding_model_id,
)
from rag_platform.services.observability import get_langfuse_observability
from rag_platform.services.retrieval.hyde import HyDEQueryExpander, QueryExpansionExpander

KEYWORD_RETRIEVER = "pg_search_bm25"
KEYWORD_TOKENIZER = "pdb.jieba"
KEYWORD_SCORE_SOURCE = "pdb.score"
SCORE_LOG_LIMIT = 10
TRACE_CANDIDATE_LIMIT = 12

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    content: str
    score: float
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class RetrievalQueryPlan:
    keyword_query: str
    vector_embedding_texts: List[str]
    retrieval_queries: List[str]
    hyde_enabled: bool
    hyde_status: str
    query_expansion_enabled: bool = False
    query_expansion_status: str = "disabled"
    expanded_queries: List[str] = field(default_factory=list)
    query_expansion_error: Optional[str] = None
    hyde_hypothetical_document: Optional[str] = None
    hyde_error: Optional[str] = None


class HybridRetriever:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider = None,
        hyde_expander: HyDEQueryExpander = None,
        query_expansion_expander: QueryExpansionExpander = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.embedding_providers = EmbeddingProviderRegistry()
        self.hyde_expander = hyde_expander or HyDEQueryExpander()
        self.query_expansion_expander = query_expansion_expander or QueryExpansionExpander()

    async def retrieve(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
        query: str,
        top_k: int = 8,
        hyde_enabled: bool = False,
        query_expansion_enabled: bool = False,
    ) -> tuple[List[RetrievedChunk], Dict[str, Any]]:
        retrieval_run_id = uuid.uuid4().hex
        stages: List[Dict[str, Any]] = []
        started_at = time.perf_counter()
        observability = get_langfuse_observability()

        with observability.observation(
            "retrieval",
            input_data={
                "query": query,
                "tenant_id": str(tenant_id),
                "kb_id": str(kb_id),
                "top_k": top_k,
                "hyde_enabled": hyde_enabled,
                "query_expansion_enabled": query_expansion_enabled,
            },
            metadata={
                "retrieval_run_id": retrieval_run_id,
                "kb_id": str(kb_id),
                "top_k": top_k,
            },
        ) as retrieval_observation:
            with observability.observation("retrieval_policy") as policy_observation:
                with self._record_stage(stages, "retrieval_policy") as stage:
                    embedding_model = resolve_embedding_model_id(
                        await self._load_embedding_model(session, tenant_id, kb_id)
                    )
                    retrieval_policy = await self._load_retrieval_policy(
                        session,
                        tenant_id,
                        kb_id,
                    )
                    vector_weight = float(retrieval_policy.get("vector_weight", 0.65))
                    keyword_weight = float(retrieval_policy.get("keyword_weight", 0.35))
                    policy_snapshot = {
                        "top_k": top_k,
                        "vector_weight": vector_weight,
                        "keyword_weight": keyword_weight,
                        "embedding_model": embedding_model,
                        "keyword_retriever": KEYWORD_RETRIEVER,
                        "keyword_tokenizer": KEYWORD_TOKENIZER,
                        "reranker": retrieval_policy.get("reranker", "none"),
                        "score_threshold": retrieval_policy.get("score_threshold", 0),
                    }
                    stage["output_count"] = 1
                    stage["metadata"] = policy_snapshot
                observability.update_observation(
                    policy_observation,
                    output=policy_snapshot,
                    metadata={
                        "embedding_model": embedding_model,
                        "retriever": "hybrid",
                    },
                )

            with observability.observation(
                "query_transform",
                input_data={"query": query},
                metadata={
                    "hyde_enabled": hyde_enabled,
                    "query_expansion_enabled": query_expansion_enabled,
                },
            ) as query_transform_observation:
                with self._record_stage(stages, "query_transform") as stage:
                    query_plan = await self._build_query_plan(
                        query,
                        retrieval_policy,
                        hyde_enabled,
                        query_expansion_enabled,
                    )
                    query_transform_output = self._query_plan_trace(query_plan)
                    stage["output_count"] = len(query_plan.retrieval_queries)
                    stage["metadata"] = {
                        "query_transform": self._query_transform_label(query_plan),
                        "hyde_status": query_plan.hyde_status,
                        "query_expansion_status": query_plan.query_expansion_status,
                    }
                observability.update_observation(
                    query_transform_observation,
                    output=query_transform_output,
                    metadata=stage["metadata"],
                )

            embedding_provider = self.embedding_provider or self.embedding_providers.get(
                embedding_model,
                usage="query",
            )
            with observability.observation(
                "embedding_original_query",
                input_data={"texts": [query]},
                metadata={"embedding_model": embedding_model},
            ) as original_embedding_observation:
                with self._record_stage(stages, "embedding_original_query") as stage:
                    original_embedding = await embedding_provider.embed(query)
                    stage["input_count"] = 1
                    stage["output_count"] = 1
                    stage["metadata"] = {"embedding_model": embedding_model}
                observability.update_observation(
                    original_embedding_observation,
                    output={
                        "embedding_count": 1,
                        "dimensions": len(original_embedding),
                    },
                    metadata=stage["metadata"],
                )

            keyword_candidate_count = 0
            keyword_rows: List[RetrievedChunk] = []
            vector_trace: Dict[str, Any] = {
                "vector_scoring_query": "original_query",
                "vector_query_count": 1,
            }

            if query_plan.query_expansion_status == "applied":
                with observability.observation(
                    "multi_query_hybrid_recall",
                    input_data={"queries": query_plan.retrieval_queries},
                    metadata={"query_count": len(query_plan.retrieval_queries)},
                ) as multi_query_observation:
                    with self._record_stage(stages, "multi_query_hybrid_recall") as stage:
                        vector_rows, vector_trace = await self._multi_query_hybrid_candidate_search(
                            session,
                            tenant_id,
                            kb_id,
                            query,
                            query_plan.retrieval_queries,
                            original_embedding,
                            embedding_provider,
                            embedding_model,
                            top_k * 2,
                            vector_weight=vector_weight,
                            keyword_weight=keyword_weight,
                        )
                        keyword_candidate_count = int(
                            vector_trace.get("query_local_keyword_candidate_count", 0)
                        )
                        stage["input_count"] = len(query_plan.retrieval_queries)
                        stage["output_count"] = len(vector_rows)
                        stage["metadata"] = vector_trace
                    observability.update_observation(
                        multi_query_observation,
                        output={
                            "trace": vector_trace,
                            "candidates": self._candidate_snapshots(vector_rows),
                        },
                        metadata={
                            "candidate_count": len(vector_rows),
                            "keyword_candidate_count": keyword_candidate_count,
                        },
                    )

                fused = vector_rows[:top_k]
                self._log_score_rows(
                    "multi_query_q0_rerank_output",
                    fused,
                    scoring_query=query,
                    score_source="original_query_vector_similarity",
                )
            else:
                with observability.observation(
                    "embedding_retrieval_texts",
                    input_data={"texts": query_plan.vector_embedding_texts},
                    metadata={
                        "embedding_model": embedding_model,
                        "embedding_text_count": len(query_plan.vector_embedding_texts),
                    },
                ) as embedding_observation:
                    with self._record_stage(stages, "embedding_retrieval_texts") as stage:
                        embedding = await self._embed_query_texts(
                            embedding_provider,
                            query_plan.vector_embedding_texts,
                        )
                        stage["input_count"] = len(query_plan.vector_embedding_texts)
                        stage["output_count"] = 1
                        stage["metadata"] = {"embedding_model": embedding_model}
                    observability.update_observation(
                        embedding_observation,
                        output={
                            "embedding_count": 1,
                            "dimensions": len(embedding),
                        },
                        metadata=stage["metadata"],
                    )

                with observability.observation(
                    "vector_recall",
                    input_data={"query": query, "limit": top_k * 2},
                    metadata={"embedding_model": embedding_model},
                ) as vector_observation:
                    with self._record_stage(stages, "vector_recall") as stage:
                        vector_rows = await self._vector_search(
                            session,
                            tenant_id,
                            kb_id,
                            embedding,
                            embedding_model,
                            top_k * 2,
                        )
                        stage["input_count"] = 1
                        stage["output_count"] = len(vector_rows)
                        stage["metadata"] = {"limit": top_k * 2}
                    observability.update_observation(
                        vector_observation,
                        output={"candidates": self._candidate_snapshots(vector_rows)},
                        metadata={"candidate_count": len(vector_rows)},
                    )
                self._log_score_rows(
                    "vector_single_query",
                    vector_rows,
                    scoring_query=query,
                    score_source="query_vector_similarity",
                )

                with observability.observation(
                    "keyword_recall",
                    input_data={"query": query_plan.keyword_query, "limit": top_k * 2},
                    metadata={
                        "keyword_retriever": KEYWORD_RETRIEVER,
                        "keyword_tokenizer": KEYWORD_TOKENIZER,
                    },
                ) as keyword_observation:
                    with self._record_stage(stages, "keyword_recall") as stage:
                        keyword_rows = await self._keyword_search(
                            session,
                            tenant_id,
                            kb_id,
                            query_plan.keyword_query,
                            top_k * 2,
                        )
                        keyword_candidate_count = len(keyword_rows)
                        stage["input_count"] = 1
                        stage["output_count"] = keyword_candidate_count
                        stage["metadata"] = {"limit": top_k * 2}
                    observability.update_observation(
                        keyword_observation,
                        output={"candidates": self._candidate_snapshots(keyword_rows)},
                        metadata={"candidate_count": keyword_candidate_count},
                    )
                self._log_score_rows(
                    "keyword_original_query",
                    keyword_rows,
                    scoring_query=query_plan.keyword_query,
                    score_source=KEYWORD_SCORE_SOURCE,
                )

                with observability.observation(
                    "fusion_rrf",
                    input_data={
                        "vector_candidates": len(vector_rows),
                        "keyword_candidates": len(keyword_rows),
                    },
                    metadata={
                        "vector_weight": vector_weight,
                        "keyword_weight": keyword_weight,
                    },
                ) as fusion_observation:
                    with self._record_stage(stages, "fusion_rrf") as stage:
                        fused = self._rrf(
                            vector_rows,
                            keyword_rows,
                            vector_weight=vector_weight,
                            keyword_weight=keyword_weight,
                        )[:top_k]
                        stage["input_count"] = len(vector_rows) + len(keyword_rows)
                        stage["output_count"] = len(fused)
                    observability.update_observation(
                        fusion_observation,
                        output={
                            "candidates": self._candidate_snapshots(
                                fused,
                                vector_rows=vector_rows,
                                keyword_rows=keyword_rows,
                                vector_weight=vector_weight,
                                keyword_weight=keyword_weight,
                            )
                        },
                        metadata={"returned_count": len(fused)},
                    )
                self._log_score_rows(
                    "rrf_fused",
                    fused,
                    scoring_query=query,
                    score_source="weighted_rrf",
                )

            latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
            candidates = {
                "vector": self._candidate_snapshots(vector_rows),
                "keyword": self._candidate_snapshots(keyword_rows),
                "fused": self._candidate_snapshots(
                    fused,
                    vector_rows=vector_rows,
                    keyword_rows=keyword_rows,
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight,
                ),
                "returned": self._candidate_snapshots(
                    fused[:top_k],
                    vector_rows=vector_rows,
                    keyword_rows=keyword_rows,
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight,
                ),
            }
            diagnostics = self._build_diagnostics(
                vector_rows=vector_rows,
                keyword_rows=keyword_rows,
                returned_rows=fused,
                query_plan=query_plan,
                score_threshold=float(retrieval_policy.get("score_threshold", 0) or 0),
            )
            trace = {
                "trace_version": "1.0",
                "retrieval_run_id": retrieval_run_id,
                "latency_ms": latency_ms,
                "retriever": (
                    "multi_query_hybrid_q0_rerank"
                    if query_plan.query_expansion_status == "applied"
                    else "hybrid_rrf"
                ),
                "vector_candidates": len(vector_rows),
                "keyword_candidates": keyword_candidate_count,
                "top_k": top_k,
                "vector_weight": vector_weight,
                "keyword_weight": keyword_weight,
                "embedding_model": embedding_model,
                "keyword_retriever": KEYWORD_RETRIEVER,
                "keyword_tokenizer": KEYWORD_TOKENIZER,
                **vector_trace,
                "query_transform": self._query_transform_label(query_plan),
                "query_expansion_enabled": query_plan.query_expansion_enabled,
                "query_expansion_status": query_plan.query_expansion_status,
                "retrieval_queries": query_plan.retrieval_queries,
                "expanded_queries": query_plan.expanded_queries,
                "query_expansion_error": query_plan.query_expansion_error,
                "hyde_enabled": query_plan.hyde_enabled,
                "hyde_status": query_plan.hyde_status,
                "hyde_embedding_text_count": len(query_plan.vector_embedding_texts),
                "hyde_hypothetical_document": self._preview_text(
                    query_plan.hyde_hypothetical_document,
                    limit=1400,
                ),
                "hyde_hypothetical_document_preview": self._preview_text(
                    query_plan.hyde_hypothetical_document
                ),
                "hyde_error": query_plan.hyde_error,
                "reranker": retrieval_policy.get("reranker", "none"),
                "score_threshold": retrieval_policy.get("score_threshold", 0),
                "stages": stages,
                "candidates": candidates,
                "diagnostics": diagnostics,
            }
            observability.update_observation(
                retrieval_observation,
                output={
                    "trace_summary": self._trace_summary(trace),
                    "returned_candidates": candidates["returned"],
                },
                metadata={
                    "retrieval_run_id": retrieval_run_id,
                    "retriever": trace["retriever"],
                    "empty_retrieval": diagnostics["empty_retrieval"],
                    "low_confidence": diagnostics["low_confidence"],
                },
            )
            return fused, trace

    async def _build_query_plan(
        self,
        query: str,
        retrieval_policy: Dict[str, Any],
        hyde_enabled: bool = False,
        query_expansion_enabled: bool = False,
    ) -> RetrievalQueryPlan:
        keyword_query = query
        vector_embedding_texts = [query]
        retrieval_queries = [query]
        expanded_queries: List[str] = []
        query_expansion_status = "disabled"
        query_expansion_error = None

        if query_expansion_enabled:
            try:
                expansion = await asyncio.to_thread(
                    self.query_expansion_expander.expand,
                    query,
                    include_original=True,
                )
                vector_embedding_texts = self._dedupe_texts(expansion.embedding_texts)
                expanded_queries = expansion.queries
                retrieval_queries = self._dedupe_texts([query, *expanded_queries])
                query_expansion_status = "applied" if expanded_queries else "empty"
            except Exception as exc:  # pragma: no cover - defensive boundary around external LLMs
                query_expansion_status = "fallback"
                query_expansion_error = str(exc)

        if not hyde_enabled:
            return RetrievalQueryPlan(
                keyword_query=keyword_query,
                vector_embedding_texts=vector_embedding_texts,
                retrieval_queries=retrieval_queries,
                hyde_enabled=False,
                hyde_status="disabled",
                query_expansion_enabled=query_expansion_enabled,
                query_expansion_status=query_expansion_status,
                expanded_queries=expanded_queries,
                query_expansion_error=query_expansion_error,
            )

        include_original = bool(retrieval_policy.get("hyde_include_original", True))
        try:
            # LlamaIndex 的 HyDEQueryTransform 是同步接口；放到线程里避免阻塞异步请求循环。
            expansion = await asyncio.to_thread(
                self.hyde_expander.expand,
                query,
                include_original=include_original,
            )
        except Exception as exc:  # pragma: no cover - defensive boundary around external LLMs
            return RetrievalQueryPlan(
                keyword_query=keyword_query,
                vector_embedding_texts=vector_embedding_texts,
                retrieval_queries=retrieval_queries,
                hyde_enabled=True,
                hyde_status="fallback",
                query_expansion_enabled=query_expansion_enabled,
                query_expansion_status=query_expansion_status,
                expanded_queries=expanded_queries,
                query_expansion_error=query_expansion_error,
                hyde_error=str(exc),
            )

        return RetrievalQueryPlan(
            keyword_query=keyword_query,
            vector_embedding_texts=self._dedupe_texts(
                [*vector_embedding_texts, *expansion.embedding_texts]
            ),
            retrieval_queries=retrieval_queries,
            hyde_enabled=True,
            hyde_status="applied",
            query_expansion_enabled=query_expansion_enabled,
            query_expansion_status=query_expansion_status,
            expanded_queries=expanded_queries,
            query_expansion_error=query_expansion_error,
            hyde_hypothetical_document=expansion.hypothetical_document,
        )

    def _query_transform_label(self, query_plan: RetrievalQueryPlan) -> str:
        transforms = []
        if query_plan.query_expansion_enabled:
            transforms.append("query_expansion")
        if query_plan.hyde_enabled:
            transforms.append("hyde")
        return "+".join(transforms) if transforms else "none"

    def _dedupe_texts(self, texts: List[str]) -> List[str]:
        deduped = []
        seen = set()
        for text_value in texts:
            cleaned = text_value.strip()
            if not cleaned or cleaned in seen:
                continue
            deduped.append(cleaned)
            seen.add(cleaned)
        return deduped or [""]

    async def _embed_query_texts(
        self,
        embedding_provider: EmbeddingProvider,
        texts: List[str],
    ) -> List[float]:
        cleaned_texts = [text.strip() for text in texts if text.strip()]
        if not cleaned_texts:
            cleaned_texts = [""]

        embeddings = [await embedding_provider.embed(text) for text in cleaned_texts]
        if len(embeddings) == 1:
            return embeddings[0]

        dimensions = {len(embedding) for embedding in embeddings}
        if len(dimensions) != 1:
            raise ValueError("HyDE embedding dimensions are inconsistent")

        return [sum(values) / len(embeddings) for values in zip(*embeddings)]

    async def _multi_query_hybrid_candidate_search(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
        original_query: str,
        candidate_queries: List[str],
        original_embedding: List[float],
        embedding_provider: EmbeddingProvider,
        embedding_model: str,
        limit_per_query: int,
        vector_weight: float,
        keyword_weight: float,
    ) -> tuple[List[RetrievedChunk], Dict[str, Any]]:
        # Query expansion 只负责扩大候选池；最终分数统一回到原始问题 embedding。
        candidate_ids: Dict[uuid.UUID, None] = {}
        candidate_query_count = 0
        dense_candidate_count = 0
        keyword_candidate_count = 0
        query_local_rrf_candidate_count = 0
        for candidate_query in self._dedupe_texts(candidate_queries):
            candidate_embedding = (
                original_embedding
                if candidate_query == original_query
                else await embedding_provider.embed(candidate_query)
            )
            vector_rows = await self._vector_search(
                session,
                tenant_id,
                kb_id,
                candidate_embedding,
                embedding_model,
                limit_per_query,
            )
            dense_candidate_count += len(vector_rows)
            self._log_score_rows(
                "multi_query_dense_recall",
                vector_rows,
                scoring_query=candidate_query,
                score_source="candidate_query_vector_similarity",
            )
            keyword_rows = await self._keyword_search(
                session,
                tenant_id,
                kb_id,
                candidate_query,
                limit_per_query,
            )
            keyword_candidate_count += len(keyword_rows)
            self._log_score_rows(
                "multi_query_bm25_recall",
                keyword_rows,
                scoring_query=candidate_query,
                score_source=KEYWORD_SCORE_SOURCE,
            )
            query_local_rows = self._rrf(
                vector_rows,
                keyword_rows,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
            )
            query_local_rrf_candidate_count += len(query_local_rows)
            self._log_score_rows(
                "multi_query_local_rrf",
                query_local_rows,
                scoring_query=candidate_query,
                score_source="query_local_weighted_rrf",
            )
            candidate_query_count += 1
            for row in query_local_rows[:limit_per_query]:
                candidate_ids[row.chunk_id] = None

        rescored_rows = await self._score_vector_candidates(
            session,
            tenant_id,
            kb_id,
            original_embedding,
            embedding_model,
            list(candidate_ids),
        )
        self._log_score_rows(
            "multi_query_q0_rerank",
            rescored_rows,
            scoring_query=original_query,
            score_source="original_query_vector_similarity",
        )
        return rescored_rows, {
            "vector_scoring_query": "original_query",
            "vector_query_count": candidate_query_count,
            "query_local_dense_candidate_count": dense_candidate_count,
            "query_local_keyword_candidate_count": keyword_candidate_count,
            "query_local_rrf_candidate_count": query_local_rrf_candidate_count,
            "multi_query_candidate_count": len(candidate_ids),
            "rerank_method": "original_query_vector_similarity",
        }

    async def _score_vector_candidates(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
        embedding: List[float],
        embedding_model: str,
        chunk_ids: List[uuid.UUID],
    ) -> List[RetrievedChunk]:
        if not chunk_ids:
            return []

        vector_literal = "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"
        statement = text(
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
              AND c.id IN :chunk_ids
              AND c.embedding IS NOT NULL
              AND COALESCE(
                c.metadata->'embedding'->>'model',
                :default_embedding_model
              ) = :embedding_model
            ORDER BY c.embedding <=> CAST(:embedding AS vector)
            """
        ).bindparams(bindparam("chunk_ids", expanding=True))
        result = await session.execute(
            statement,
            {
                "tenant_id": tenant_id,
                "kb_id": kb_id,
                "embedding": vector_literal,
                "embedding_model": embedding_model,
                "default_embedding_model": DEFAULT_EMBEDDING_MODEL_ID,
                "chunk_ids": tuple(chunk_ids),
            },
        )
        return [self._row_to_chunk(row._mapping) for row in result]

    @contextmanager
    def _record_stage(
        self,
        stages: List[Dict[str, Any]],
        name: str,
    ) -> Iterator[Dict[str, Any]]:
        started_at = time.perf_counter()
        stage: Dict[str, Any] = {
            "name": name,
            "status": "success",
            "input_count": 0,
            "output_count": 0,
            "metadata": {},
        }
        try:
            yield stage
        except Exception as exc:
            stage["status"] = "error"
            stage["error"] = str(exc)
            raise
        finally:
            stage["duration_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
            stages.append(stage)

    def _query_plan_trace(self, query_plan: RetrievalQueryPlan) -> Dict[str, Any]:
        return {
            "keyword_query": query_plan.keyword_query,
            "retrieval_queries": query_plan.retrieval_queries,
            "vector_embedding_text_count": len(query_plan.vector_embedding_texts),
            "query_transform": self._query_transform_label(query_plan),
            "query_expansion": {
                "enabled": query_plan.query_expansion_enabled,
                "status": query_plan.query_expansion_status,
                "expanded_queries": query_plan.expanded_queries,
                "error": query_plan.query_expansion_error,
            },
            "hyde": {
                "enabled": query_plan.hyde_enabled,
                "status": query_plan.hyde_status,
                "hypothetical_document_preview": self._preview_text(
                    query_plan.hyde_hypothetical_document
                ),
                "error": query_plan.hyde_error,
            },
        }

    def _candidate_snapshots(
        self,
        rows: List[RetrievedChunk],
        limit: int = TRACE_CANDIDATE_LIMIT,
        vector_rows: Optional[List[RetrievedChunk]] = None,
        keyword_rows: Optional[List[RetrievedChunk]] = None,
        vector_weight: float = 0.65,
        keyword_weight: float = 0.35,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        vector_rank = self._rank_map(vector_rows or [])
        keyword_rank = self._rank_map(keyword_rows or [])
        vector_score = {row.chunk_id: row.score for row in vector_rows or []}
        keyword_score = {row.chunk_id: row.score for row in keyword_rows or []}

        snapshots = []
        for rank, row in enumerate(rows[:limit], start=1):
            score_components: Dict[str, Any] = {}
            if row.chunk_id in vector_rank:
                score_components["vector_rank"] = vector_rank[row.chunk_id]
                score_components["vector_score"] = round(vector_score[row.chunk_id], 6)
                score_components["vector_rrf"] = round(
                    vector_weight / (rrf_k + vector_rank[row.chunk_id]),
                    8,
                )
            if row.chunk_id in keyword_rank:
                score_components["keyword_rank"] = keyword_rank[row.chunk_id]
                score_components["keyword_score"] = round(keyword_score[row.chunk_id], 6)
                score_components["keyword_rrf"] = round(
                    keyword_weight / (rrf_k + keyword_rank[row.chunk_id]),
                    8,
                )

            snapshots.append(
                {
                    "rank": rank,
                    "chunk_id": str(row.chunk_id),
                    "document_id": str(row.document_id),
                    "filename": row.filename,
                    "score": round(row.score, 6),
                    "score_components": score_components,
                    "metadata": self._candidate_metadata(row.metadata),
                    "content_preview": self._preview_text(row.content),
                }
            )
        return snapshots

    def _candidate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        allowed_keys = (
            "chunker",
            "page",
            "page_number",
            "section",
            "heading",
        )
        return {key: str(metadata[key]) for key in allowed_keys if key in metadata}

    def _rank_map(self, rows: Iterable[RetrievedChunk]) -> Dict[uuid.UUID, int]:
        return {row.chunk_id: rank for rank, row in enumerate(rows, start=1)}

    def _build_diagnostics(
        self,
        vector_rows: List[RetrievedChunk],
        keyword_rows: List[RetrievedChunk],
        returned_rows: List[RetrievedChunk],
        query_plan: RetrievalQueryPlan,
        score_threshold: float,
    ) -> Dict[str, Any]:
        vector_ids = {row.chunk_id for row in vector_rows}
        keyword_ids = {row.chunk_id for row in keyword_rows}
        overlap_union = vector_ids | keyword_ids
        overlap_ratio = (
            round(len(vector_ids & keyword_ids) / len(overlap_union), 4)
            if overlap_union
            else 0.0
        )
        score_gap = 0.0
        if len(returned_rows) >= 2:
            score_gap = round(returned_rows[0].score - returned_rows[1].score, 6)

        warnings = []
        if not returned_rows:
            warnings.append("empty_retrieval")
        if not vector_rows:
            warnings.append("empty_vector_recall")
        if not keyword_rows and query_plan.query_expansion_status != "applied":
            warnings.append("empty_keyword_recall")
        if query_plan.hyde_status == "fallback":
            warnings.append("hyde_fallback")
        if query_plan.query_expansion_status == "fallback":
            warnings.append("query_expansion_fallback")
        if score_threshold > 0:
            warnings.append("score_threshold_configured_not_applied")

        top_score = returned_rows[0].score if returned_rows else 0.0
        low_confidence = bool(not returned_rows or (score_threshold > 0 and top_score < score_threshold))
        return {
            "empty_retrieval": not returned_rows,
            "low_confidence": low_confidence,
            "vector_keyword_overlap": overlap_ratio,
            "score_gap_top1_top2": score_gap,
            "top_score": round(top_score, 6),
            "warnings": warnings,
        }

    def _trace_summary(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "trace_version": trace["trace_version"],
            "retrieval_run_id": trace["retrieval_run_id"],
            "retriever": trace["retriever"],
            "latency_ms": trace["latency_ms"],
            "vector_candidates": trace["vector_candidates"],
            "keyword_candidates": trace["keyword_candidates"],
            "returned_candidates": len(trace["candidates"]["returned"]),
            "query_transform": trace["query_transform"],
            "diagnostics": trace["diagnostics"],
        }

    def _preview_text(self, text_value: Optional[str], limit: int = 240) -> Optional[str]:
        if not text_value:
            return None
        normalized = " ".join(text_value.split())
        return normalized[:limit]

    def _log_score_rows(
        self,
        stage: str,
        rows: List[RetrievedChunk],
        scoring_query: str,
        score_source: str,
        limit: int = SCORE_LOG_LIMIT,
    ) -> None:
        if not rows:
            logger.warning(
                "[retrieval-score] stage=%s score_source=%s scoring_query=%r rows=0",
                stage,
                score_source,
                self._preview_text(scoring_query, limit=160),
            )
            return

        for rank, row in enumerate(rows[:limit], start=1):
            logger.warning(
                (
                    "[retrieval-score] stage=%s score_source=%s rank=%s score=%.6f "
                    "chunk_id=%s document_id=%s filename=%r scoring_query=%r content=%r"
                ),
                stage,
                score_source,
                rank,
                row.score,
                row.chunk_id,
                row.document_id,
                row.filename,
                self._preview_text(scoring_query, limit=160),
                self._preview_text(row.content, limit=180),
            )

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

    async def _load_retrieval_policy(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
    ) -> Dict[str, Any]:
        result = await session.execute(
            select(KnowledgeBase.retrieval_policy).where(
                KnowledgeBase.tenant_id == tenant_id,
                KnowledgeBase.id == kb_id,
            )
        )
        return result.scalar_one_or_none() or {}

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
        keyword_query = query.strip()
        if not keyword_query:
            return []

        result = await session.execute(
            text(
                """
                SELECT
                  c.id AS chunk_id,
                  c.document_id,
                  d.filename,
                  c.content,
                  c.metadata,
                  pdb.score(c.id) AS score
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.tenant_id = :tenant_id
                  AND c.kb_id = :kb_id
                  AND c.content ||| CAST(:query AS pdb.jieba)
                ORDER BY pdb.score(c.id) DESC
                LIMIT :limit
                """
            ),
            {
                "tenant_id": tenant_id,
                "kb_id": kb_id,
                "query": keyword_query,
                "limit": limit,
            },
        )
        return [self._row_to_chunk(row._mapping) for row in result]

    def _rrf(
        self,
        vector_rows: Iterable[RetrievedChunk],
        keyword_rows: Iterable[RetrievedChunk],
        vector_weight: float = 0.65,
        keyword_weight: float = 0.35,
        k: int = 60,
    ) -> List[RetrievedChunk]:
        by_id: Dict[uuid.UUID, RetrievedChunk] = {}
        scores: Dict[uuid.UUID, float] = {}
        for rows, weight in ((vector_rows, vector_weight), (keyword_rows, keyword_weight)):
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
        window_context = metadata.get("window_context")
        if metadata.get("chunker") == "sentence_window" and isinstance(window_context, str):
            metadata = {
                **metadata,
                "matched_chunk_content": content,
            }
            content = window_context

        return RetrievedChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            filename=row["filename"],
            content=content,
            score=float(row["score"] or 0.0),
            metadata=metadata,
        )
