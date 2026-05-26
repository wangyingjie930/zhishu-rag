import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isnan
from typing import Any, Dict, Iterable, List, Optional, Protocol

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.core.config import settings
from rag_platform.db.models import (
    ChatMessage,
    ChatSession,
    EvalDataset,
    EvalRun,
    EvalRunResult,
    EvalSample,
    KnowledgeBase,
)
from rag_platform.domain.enums import MessageRole
from rag_platform.services.llm.provider import LLMGeneration, LLMProvider, get_llm_provider
from rag_platform.services.observability import get_langfuse_observability
from rag_platform.services.retrieval.hybrid import HybridRetriever, RetrievedChunk

RAGAS_METRICS = (
    "faithfulness",
    "response_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)
LLAMAINDEX_RETRIEVAL_METRICS = (
    "hit_rate",
    "mrr",
    "precision",
    "recall",
    "ap",
    "ndcg",
)
EVAL_METRICS = (*RAGAS_METRICS, *LLAMAINDEX_RETRIEVAL_METRICS)


@dataclass(frozen=True)
class ScoreBundle:
    metrics: Dict[str, float]
    reasons: Dict[str, str]


class EvaluationScorer(Protocol):
    async def score(
        self,
        user_input: str,
        response: str,
        reference: str,
        retrieved_contexts: List[str],
        retrieved_context_ids: List[str],
        expected_context_ids: List[str],
    ) -> ScoreBundle:
        ...


class RagasScorer:
    """Run the real RAGAS evaluator and normalize its result for the API contract."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        self.api_key = (
            api_key
            if api_key is not None
            else settings.ragas_openai_api_key or settings.llm_api_key or settings.openai_api_key
        )
        self.base_url = base_url if base_url is not None else settings.ragas_openai_base_url
        self.llm_model = llm_model or settings.ragas_llm_model
        self.embedding_model = embedding_model or settings.ragas_embedding_model or settings.embedding_model
        self._metrics: Optional[List[Any]] = None

    async def score(
        self,
        user_input: str,
        response: str,
        reference: str,
        retrieved_contexts: List[str],
        retrieved_context_ids: List[str],
        expected_context_ids: List[str],
    ) -> ScoreBundle:
        from ragas import EvaluationDataset, aevaluate

        dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": user_input,
                    "response": response,
                    "retrieved_contexts": retrieved_contexts,
                    "reference": reference,
                }
            ]
        )
        result = await aevaluate(
            dataset,
            metrics=self._get_metrics(),
            show_progress=False,
            raise_exceptions=True,
        )
        rows = result.to_pandas().to_dict("records")
        return self._score_bundle_from_row(rows[0] if rows else {})

    def _get_metrics(self) -> List[Any]:
        if self._metrics is not None:
            return self._metrics
        if not self.api_key:
            raise ValueError("未配置 OPENAI_API_KEY 或 RAGAS_OPENAI_API_KEY，无法运行真实 RAGAS 评测")
        if not self.llm_model:
            raise ValueError("未配置 RAGAS_LLM_MODEL，无法运行真实 RAGAS 评测")
        if not self.embedding_model:
            raise ValueError("未配置 RAGAS_EMBEDDING_MODEL，无法运行真实 RAGAS 评测")

        from openai import OpenAI
        from rag_platform.services.evaluation_embeddings import RagasConfiguredEmbedding
        from ragas.llms import llm_factory
        from ragas.metrics._answer_correctness import AnswerCorrectness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._context_precision import ContextPrecision
        from ragas.metrics._context_recall import ContextRecall
        from ragas.metrics._faithfulness import Faithfulness

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self._openai_api_base_url()
        client = OpenAI(**client_kwargs)
        llm = llm_factory(self.llm_model, client=client)
        embeddings = RagasConfiguredEmbedding(self.embedding_model)
        self._metrics = [
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings, name="response_relevancy"),
            ContextPrecision(llm=llm, name="context_precision"),
            ContextRecall(llm=llm),
            AnswerCorrectness(llm=llm, embeddings=embeddings),
        ]
        return self._metrics

    def _openai_api_base_url(self) -> str:
        base_url = self.base_url.rstrip("/")
        return base_url if base_url.endswith("/v1") else f"{base_url}/v1"

    def _score_bundle_from_row(self, row: Dict[str, Any]) -> ScoreBundle:
        metrics = {
            "faithfulness": self._metric_value(row.get("faithfulness")),
            "response_relevancy": self._metric_value(row.get("response_relevancy")),
            "context_precision": self._metric_value(row.get("context_precision")),
            "context_recall": self._metric_value(row.get("context_recall")),
            "answer_correctness": self._metric_value(row.get("answer_correctness")),
        }
        reasons = {
            "faithfulness": "由 RAGAS Faithfulness 指标评估回答是否可由检索上下文支撑。",
            "response_relevancy": "由 RAGAS AnswerRelevancy 指标评估回答与问题的相关性。",
            "context_precision": "由 RAGAS ContextPrecision 指标评估相关上下文是否排在更靠前位置。",
            "context_recall": "由 RAGAS ContextRecall 指标评估检索上下文是否覆盖标准答案信息。",
            "answer_correctness": "由 RAGAS AnswerCorrectness 指标评估回答与标准答案的一致性。",
        }
        return ScoreBundle(metrics=metrics, reasons=reasons)

    def _metric_value(self, value: Any) -> float:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if isnan(numeric_value):
            return 0.0
        return round(numeric_value, 3)


class LlamaIndexRetrievalScorer:
    """Evaluate retrieved chunk ids with LlamaIndex RetrieverEvaluator."""

    async def score(
        self,
        user_input: str,
        retrieved_contexts: List[str],
        retrieved_context_ids: List[str],
        expected_context_ids: List[str],
    ) -> ScoreBundle:
        expected_ids = [str(item) for item in expected_context_ids if str(item).strip()]
        retrieved_ids = [str(item) for item in retrieved_context_ids if str(item).strip()]
        if not expected_ids:
            return ScoreBundle(
                metrics={metric: 0.0 for metric in LLAMAINDEX_RETRIEVAL_METRICS},
                reasons={
                    metric: "未设置 expected_context_ids，无法计算确定性检索指标。"
                    for metric in LLAMAINDEX_RETRIEVAL_METRICS
                },
            )

        retriever = _StaticLlamaIndexRetriever(retrieved_ids, retrieved_contexts)

        from llama_index.core.evaluation import RetrieverEvaluator

        evaluator = RetrieverEvaluator.from_metric_names(
            list(LLAMAINDEX_RETRIEVAL_METRICS),
            retriever=retriever,
        )
        result = await evaluator.aevaluate(user_input, expected_ids)
        metrics = {
            metric: self._metric_value(result.metric_vals_dict.get(metric))
            for metric in LLAMAINDEX_RETRIEVAL_METRICS
        }
        reasons = {
            "hit_rate": "LlamaIndex RetrieverEvaluator：Top-K 命中任一期望 chunk 记为 1。",
            "mrr": "LlamaIndex RetrieverEvaluator：第一个相关 chunk 排名的倒数。",
            "precision": "LlamaIndex RetrieverEvaluator：召回结果中相关 chunk 占比。",
            "recall": "LlamaIndex RetrieverEvaluator：期望 chunk 被召回的覆盖率。",
            "ap": "LlamaIndex RetrieverEvaluator：Average Precision。",
            "ndcg": "LlamaIndex RetrieverEvaluator：排名折损后的归一化增益。",
        }
        return ScoreBundle(metrics=metrics, reasons=reasons)

    def _metric_value(self, value: Any) -> float:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if isnan(numeric_value):
            return 0.0
        return round(numeric_value, 3)


class _StaticLlamaIndexRetriever(BaseRetriever):
    def __init__(self, retrieved_ids: List[str], retrieved_contexts: List[str]) -> None:
        super().__init__()
        self.nodes = [
            NodeWithScore(
                node=TextNode(id_=chunk_id, text=_context_for_index(retrieved_contexts, index)),
                score=1.0 / (index + 1),
            )
            for index, chunk_id in enumerate(retrieved_ids)
        ]

    def _retrieve(self, query_bundle):
        return self.nodes


def _context_for_index(contexts: List[str], index: int) -> str:
    if index < len(contexts):
        return contexts[index]
    return ""


class EvaluationService:
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        llm: Optional[LLMProvider] = None,
        scorer: Optional[EvaluationScorer] = None,
        retrieval_scorer: Optional[LlamaIndexRetrievalScorer] = None,
    ) -> None:
        self.retriever = retriever or HybridRetriever()
        self.llm = llm or get_llm_provider()
        self.scorer = scorer or RagasScorer()
        self.retrieval_scorer = retrieval_scorer or LlamaIndexRetrievalScorer()
        self.observability = get_langfuse_observability()

    async def list_candidates(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
        limit: int = 80,
    ) -> List[Dict[str, Any]]:
        statement = (
            select(ChatMessage, ChatSession.id.label("chat_session_id"))
            .join(ChatSession, ChatMessage.session_id == ChatSession.id)
            .where(ChatSession.tenant_id == tenant_id, ChatSession.kb_id == kb_id)
            .order_by(ChatSession.created_at.desc(), ChatMessage.created_at.asc())
            .limit(limit * 2)
        )
        rows = (await session.execute(statement)).all()

        candidates: List[Dict[str, Any]] = []
        last_user_by_session: Dict[uuid.UUID, ChatMessage] = {}
        for message, chat_session_id in rows:
            if message.role == MessageRole.user.value:
                last_user_by_session[chat_session_id] = message
                continue
            if message.role != MessageRole.assistant.value:
                continue
            user_message = last_user_by_session.get(chat_session_id)
            if user_message is None:
                continue
            candidates.append(
                {
                    "id": message.id,
                    "session_id": chat_session_id,
                    "user_message_id": user_message.id,
                    "assistant_message_id": message.id,
                    "user_input": user_message.content,
                    "response": message.content,
                    "citations": self._normalize_citations(message.citations),
                    "retrieval_trace": message.retrieval_trace or {},
                    "created_at": message.created_at,
                }
            )
        return candidates[:limit]

    async def list_datasets(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
    ) -> List[Dict[str, Any]]:
        statement = (
            select(EvalDataset, func.count(EvalSample.id).label("sample_count"))
            .outerjoin(EvalSample, EvalSample.dataset_id == EvalDataset.id)
            .where(EvalDataset.tenant_id == tenant_id, EvalDataset.kb_id == kb_id)
            .group_by(EvalDataset.id)
            .order_by(EvalDataset.created_at.desc())
        )
        rows = (await session.execute(statement)).all()
        return [
            {
                "id": dataset.id,
                "kb_id": dataset.kb_id,
                "name": dataset.name,
                "description": dataset.description,
                "sample_count": sample_count,
                "created_at": dataset.created_at,
            }
            for dataset, sample_count in rows
        ]

    async def create_dataset(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
        name: str,
        description: str,
    ) -> EvalDataset:
        await self._require_kb(session, tenant_id, kb_id)
        dataset = EvalDataset(
            tenant_id=tenant_id,
            kb_id=kb_id,
            name=name,
            description=description,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)
        return dataset

    async def delete_dataset(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        dataset_id: uuid.UUID,
    ) -> None:
        await self._require_dataset(session, tenant_id, dataset_id)
        run_ids = select(EvalRun.id).where(
            EvalRun.tenant_id == tenant_id,
            EvalRun.dataset_id == dataset_id,
        )
        # 评测集删除会带走样本、运行和运行结果；这里显式删除，避免依赖不同数据库的级联实现差异。
        await session.execute(
            delete(EvalRunResult).where(
                EvalRunResult.tenant_id == tenant_id,
                EvalRunResult.run_id.in_(run_ids),
            )
        )
        await session.execute(
            delete(EvalRun).where(
                EvalRun.tenant_id == tenant_id,
                EvalRun.dataset_id == dataset_id,
            )
        )
        await session.execute(
            delete(EvalSample).where(
                EvalSample.tenant_id == tenant_id,
                EvalSample.dataset_id == dataset_id,
            )
        )
        await session.execute(
            delete(EvalDataset).where(
                EvalDataset.tenant_id == tenant_id,
                EvalDataset.id == dataset_id,
            )
        )
        await session.commit()

    async def list_samples(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        dataset_id: uuid.UUID,
    ) -> List[EvalSample]:
        await self._require_dataset(session, tenant_id, dataset_id)
        result = await session.execute(
            select(EvalSample)
            .where(EvalSample.tenant_id == tenant_id, EvalSample.dataset_id == dataset_id)
            .order_by(EvalSample.created_at.desc())
        )
        return list(result.scalars())

    async def add_sample(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        dataset_id: uuid.UUID,
        payload: Dict[str, Any],
    ) -> EvalSample:
        await self._require_dataset(session, tenant_id, dataset_id)
        sample = EvalSample(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            source_message_id=payload.get("source_message_id"),
            user_input=payload["user_input"],
            reference=payload.get("reference", ""),
            expected_context_ids=payload.get("expected_context_ids", []),
            tags=payload.get("tags", []),
            original_response=payload.get("original_response", ""),
            original_citations=self._normalize_citations(payload.get("original_citations", [])),
            original_retrieval_trace=payload.get("original_retrieval_trace", {}),
        )
        session.add(sample)
        await session.commit()
        await session.refresh(sample)
        return sample

    async def update_sample(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        dataset_id: uuid.UUID,
        sample_id: uuid.UUID,
        payload: Dict[str, Any],
    ) -> EvalSample:
        sample = await self._require_sample(session, tenant_id, dataset_id, sample_id)
        if "reference" in payload and payload["reference"] is not None:
            sample.reference = payload["reference"]
        if "expected_context_ids" in payload and payload["expected_context_ids"] is not None:
            sample.expected_context_ids = payload["expected_context_ids"]
        if "tags" in payload and payload["tags"] is not None:
            sample.tags = payload["tags"]
        await session.commit()
        await session.refresh(sample)
        return sample

    async def create_run(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        dataset_id: uuid.UUID,
        query_expansion_enabled: bool = False,
    ) -> EvalRun:
        dataset = await self._require_dataset(session, tenant_id, dataset_id)
        kb = await self._require_kb(session, tenant_id, dataset.kb_id)
        top_k = int((kb.retrieval_policy or {}).get("top_k", 8))
        run = EvalRun(
            tenant_id=tenant_id,
            dataset_id=dataset.id,
            kb_id=dataset.kb_id,
            status="running",
            config={
                "evaluator": "ragas",
                "metrics": list(EVAL_METRICS),
                "retrieval_evaluator": "llama_index.retriever_evaluator",
                "ragas_llm_model": settings.ragas_llm_model,
                "ragas_embedding_model": settings.ragas_embedding_model,
                "retrieval_policy": kb.retrieval_policy,
                "retrieval_options": {
                    "top_k": top_k,
                    "query_expansion_enabled": query_expansion_enabled,
                },
            },
        )
        session.add(run)
        await session.flush()

        try:
            samples = await self.list_samples(session, tenant_id, dataset.id)
            if not samples:
                raise ValueError("评测集没有样本")
            for sample in samples:
                await self._evaluate_sample(
                    session,
                    tenant_id,
                    kb,
                    run,
                    sample,
                    top_k=top_k,
                    query_expansion_enabled=query_expansion_enabled,
                )
            run.metrics = await self._aggregate_run_metrics(session, run.id)
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)
        except Exception as exc:
            run.status = "failed"
            run.error_message = str(exc)
            run.completed_at = datetime.now(timezone.utc)

        await session.commit()
        await session.refresh(run)
        return run

    async def get_run(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        run_id: uuid.UUID,
    ) -> Dict[str, Any]:
        result = await session.execute(
            select(EvalRun).where(EvalRun.tenant_id == tenant_id, EvalRun.id == run_id)
        )
        run = result.scalar_one_or_none()
        if run is None:
            raise ValueError("评测运行不存在")
        results = await self._list_run_results(session, tenant_id, run.id)
        return {"run": run, "results": results}

    async def _evaluate_sample(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        kb: KnowledgeBase,
        run: EvalRun,
        sample: EvalSample,
        top_k: int,
        query_expansion_enabled: bool,
    ) -> None:
        with self.observability.trace(
            "rag_eval_sample",
            input_data={
                "sample_id": str(sample.id),
                "user_input": sample.user_input,
                "reference": sample.reference,
                "top_k": top_k,
                "query_expansion_enabled": query_expansion_enabled,
            },
            metadata={
                "tenant_id": str(tenant_id),
                "kb_id": str(kb.id),
                "run_id": str(run.id),
                "sample_id": str(sample.id),
            },
            tags=["rag", "evaluation"],
        ) as eval_observation:
            chunks, trace = await self.retriever.retrieve(
                session,
                tenant_id,
                kb.id,
                sample.user_input,
                top_k=top_k,
                query_expansion_enabled=query_expansion_enabled,
            )
            with self.observability.observation(
                "llm_answer",
                as_type="generation",
                input_data={
                    "question": sample.user_input,
                    "contexts": self._context_previews(chunks),
                },
                metadata={
                    "context_count": len(chunks),
                    "retrieval_run_id": trace.get("retrieval_run_id"),
                },
                model=getattr(self.llm, "model", self.llm.__class__.__name__),
            ) as llm_observation:
                generation = await self._generate_answer(sample.user_input, chunks)
                response = generation.answer
                self.observability.update_observation(
                    llm_observation,
                    output={"answer": response},
                    metadata={"answer_length": len(response), **generation.prompt_metadata},
                    prompt=generation.prompt,
                )

            citations = self._chunks_to_citations(chunks)
            contexts = [citation["content"] for citation in citations]
            context_ids = [citation["chunk_id"] for citation in citations]
            score = await self.scorer.score(
                user_input=sample.user_input,
                response=response,
                reference=sample.reference,
                retrieved_contexts=contexts,
                retrieved_context_ids=context_ids,
                expected_context_ids=[str(item) for item in sample.expected_context_ids],
            )
            retrieval_score = await self.retrieval_scorer.score(
                user_input=sample.user_input,
                retrieved_contexts=contexts,
                retrieved_context_ids=context_ids,
                expected_context_ids=[str(item) for item in sample.expected_context_ids],
            )
            metrics = {**score.metrics, **retrieval_score.metrics}
            reasons = {**score.reasons, **retrieval_score.reasons}
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    self.observability.score_observation(
                        eval_observation,
                        name=metric_name,
                        value=float(metric_value),
                        comment=reasons.get(metric_name),
                    )
            self.observability.update_observation(
                eval_observation,
                output={
                    "response": response,
                    "metrics": metrics,
                    "retrieval": trace.get("diagnostics", {}),
                },
                metadata={
                    "retrieval_run_id": trace.get("retrieval_run_id"),
                    "retriever": trace.get("retriever"),
                },
            )
            session.add(
                EvalRunResult(
                    tenant_id=tenant_id,
                    run_id=run.id,
                    sample_id=sample.id,
                    user_input=sample.user_input,
                    response=response,
                    reference=sample.reference,
                    retrieved_contexts=contexts,
                    citations=citations,
                    retrieval_trace=trace,
                    metrics=metrics,
                    reasons=reasons,
                )
            )

    async def _aggregate_run_metrics(
        self,
        session: AsyncSession,
        run_id: uuid.UUID,
    ) -> Dict[str, float]:
        result = await session.execute(
            select(EvalRunResult).where(EvalRunResult.run_id == run_id)
        )
        rows = list(result.scalars())
        if not rows:
            return {metric: 0 for metric in EVAL_METRICS}
        summary = {}
        for metric in EVAL_METRICS:
            values = [float(row.metrics.get(metric, 0) or 0) for row in rows]
            summary[metric] = round(sum(values) / len(values), 3)
        return summary

    async def _list_run_results(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        run_id: uuid.UUID,
    ) -> List[EvalRunResult]:
        result = await session.execute(
            select(EvalRunResult)
            .where(EvalRunResult.tenant_id == tenant_id, EvalRunResult.run_id == run_id)
            .order_by(EvalRunResult.created_at.desc())
        )
        return list(result.scalars())

    async def _require_kb(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
    ) -> KnowledgeBase:
        result = await session.execute(
            select(KnowledgeBase).where(
                KnowledgeBase.tenant_id == tenant_id,
                KnowledgeBase.id == kb_id,
            )
        )
        kb = result.scalar_one_or_none()
        if kb is None:
            raise ValueError("知识库不存在")
        return kb

    async def _require_dataset(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        dataset_id: uuid.UUID,
    ) -> EvalDataset:
        result = await session.execute(
            select(EvalDataset).where(
                EvalDataset.tenant_id == tenant_id,
                EvalDataset.id == dataset_id,
            )
        )
        dataset = result.scalar_one_or_none()
        if dataset is None:
            raise ValueError("评测集不存在")
        return dataset

    async def _require_sample(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        dataset_id: uuid.UUID,
        sample_id: uuid.UUID,
    ) -> EvalSample:
        result = await session.execute(
            select(EvalSample).where(
                EvalSample.tenant_id == tenant_id,
                EvalSample.dataset_id == dataset_id,
                EvalSample.id == sample_id,
            )
        )
        sample = result.scalar_one_or_none()
        if sample is None:
            raise ValueError("评测样本不存在")
        return sample

    def _context_previews(self, chunks: Iterable[RetrievedChunk]) -> List[Dict[str, Any]]:
        return [
            {
                "chunk_id": str(chunk.chunk_id),
                "document_id": str(chunk.document_id),
                "filename": chunk.filename,
                "score": round(chunk.score, 6),
                "content_preview": " ".join(chunk.content.split())[:240],
            }
            for chunk in chunks
        ]

    def _chunks_to_citations(self, chunks: Iterable[RetrievedChunk]) -> List[Dict[str, Any]]:
        return [
            {
                "chunk_id": str(chunk.chunk_id),
                "document_id": str(chunk.document_id),
                "filename": chunk.filename,
                "score": chunk.score,
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]

    def _normalize_citations(self, citations: Any) -> List[Dict[str, Any]]:
        if not isinstance(citations, list):
            return []
        normalized = []
        for citation in citations:
            if hasattr(citation, "model_dump"):
                citation = citation.model_dump(mode="json")
            if not isinstance(citation, dict):
                continue
            normalized.append(
                {
                    "chunk_id": str(citation.get("chunk_id", "")),
                    "document_id": str(citation.get("document_id", "")),
                    "filename": str(citation.get("filename", "")),
                    "score": float(citation.get("score", 0) or 0),
                    "content": str(citation.get("content", "")),
                    "metadata": citation.get("metadata") if isinstance(citation.get("metadata"), dict) else {},
                }
            )
        return normalized

    async def _generate_answer(
        self,
        question: str,
        chunks: List[RetrievedChunk],
    ) -> LLMGeneration:
        generate_answer = getattr(self.llm, "generate_answer", None)
        if callable(generate_answer):
            return await generate_answer(question, chunks)
        return LLMGeneration(answer=await self.llm.answer(question, chunks))
