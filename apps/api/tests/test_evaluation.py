import uuid
from types import SimpleNamespace

import pytest

from rag_platform.schemas.evaluation import EvalRunCreate
from rag_platform.services.evaluation import (
    EvaluationService,
    LlamaIndexRetrievalScorer,
    RagasScorer,
    ScoreBundle,
)
from rag_platform.services.llm.provider import OpenAICompatibleLLMProvider
from rag_platform.services.retrieval.hybrid import RetrievedChunk


def test_ragas_scorer_normalizes_metric_names_and_values() -> None:
    scorer = RagasScorer(api_key="")

    score = scorer._score_bundle_from_row(
        {
            "faithfulness": 0.9154,
            "response_relevancy": 0.8012,
            "context_precision": 1,
            "context_recall": 0.6666,
            "answer_correctness": float("nan"),
        }
    )

    assert score.metrics == {
        "faithfulness": 0.915,
        "response_relevancy": 0.801,
        "context_precision": 1.0,
        "context_recall": 0.667,
        "answer_correctness": 0.0,
    }
    assert "RAGAS Faithfulness" in score.reasons["faithfulness"]


def test_ragas_scorer_requires_openai_key() -> None:
    scorer = RagasScorer(api_key="")

    with pytest.raises(ValueError, match="RAGAS"):
        scorer._get_metrics()


def test_ragas_scorer_prefers_ragas_api_key(monkeypatch) -> None:
    monkeypatch.setattr(
        "rag_platform.services.evaluation.settings.ragas_openai_api_key",
        "ragas-key",
    )
    monkeypatch.setattr("rag_platform.services.evaluation.settings.llm_api_key", "llm-key")
    monkeypatch.setattr("rag_platform.services.evaluation.settings.openai_api_key", "openai-key")

    scorer = RagasScorer()

    assert scorer.api_key == "ragas-key"


def test_ragas_scorer_builds_aevaluate_metric_objects() -> None:
    from ragas.metrics.base import Metric

    scorer = RagasScorer(
        api_key="test-key",
        base_url="http://localhost:8317",
        llm_model="gpt-5.5",
        embedding_model="google:gemini-embedding-001",
    )

    metrics = scorer._get_metrics()

    assert metrics
    assert all(isinstance(metric, Metric) for metric in metrics)


def test_ragas_configured_embedding_supports_legacy_metric_methods() -> None:
    from rag_platform.services.evaluation_embeddings import RagasConfiguredEmbedding

    embeddings = RagasConfiguredEmbedding("google:gemini-embedding-001")

    assert hasattr(embeddings, "embed_query")
    assert hasattr(embeddings, "embed_documents")
    assert hasattr(embeddings, "aembed_query")
    assert hasattr(embeddings, "aembed_documents")


def test_ragas_scorer_normalizes_openai_base_url() -> None:
    scorer = RagasScorer(api_key="test-key", base_url="http://localhost:8317")
    scorer_with_v1 = RagasScorer(api_key="test-key", base_url="http://localhost:8317/v1")

    assert scorer._openai_api_base_url() == "http://localhost:8317/v1"
    assert scorer_with_v1._openai_api_base_url() == "http://localhost:8317/v1"


def test_evaluation_service_uses_configured_llm_provider(monkeypatch) -> None:
    monkeypatch.setattr("rag_platform.services.llm.provider.settings.llm_provider", "openai-compatible")
    monkeypatch.setattr(
        "rag_platform.services.llm.provider.settings.llm_openai_base_url",
        "http://localhost:8317",
    )
    monkeypatch.setattr("rag_platform.services.llm.provider.settings.llm_model", "gpt-5.5")
    monkeypatch.setattr("rag_platform.services.llm.provider.settings.llm_max_output_tokens", 512)

    service = EvaluationService()

    assert isinstance(service.llm, OpenAICompatibleLLMProvider)


def test_eval_run_create_defaults_query_expansion_off() -> None:
    payload = EvalRunCreate(dataset_id=uuid.uuid4())

    assert payload.query_expansion_enabled is False


@pytest.mark.asyncio
async def test_llama_index_retrieval_scorer_scores_expected_context_ids() -> None:
    scorer = LlamaIndexRetrievalScorer()

    score = await scorer.score(
        user_input="问题",
        retrieved_contexts=["A", "B"],
        retrieved_context_ids=["chunk-a", "chunk-b"],
        expected_context_ids=["chunk-b"],
    )

    assert score.metrics == {
        "hit_rate": 1.0,
        "mrr": 0.5,
        "precision": 0.5,
        "recall": 1.0,
        "ap": 0.5,
        "ndcg": 0.387,
    }
    assert "RetrieverEvaluator" in score.reasons["ndcg"]


@pytest.mark.asyncio
async def test_llama_index_retrieval_scorer_requires_expected_context_ids() -> None:
    scorer = LlamaIndexRetrievalScorer()

    score = await scorer.score(
        user_input="问题",
        retrieved_contexts=["A"],
        retrieved_context_ids=["chunk-a"],
        expected_context_ids=[],
    )

    assert score.metrics == {
        "hit_rate": 0.0,
        "mrr": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "ap": 0.0,
        "ndcg": 0.0,
    }
    assert "expected_context_ids" in score.reasons["hit_rate"]


@pytest.mark.asyncio
async def test_evaluate_sample_passes_query_expansion_to_retriever() -> None:
    class RecordingRetriever:
        def __init__(self) -> None:
            self.query_expansion_enabled = None

        async def retrieve(
            self,
            session,
            tenant_id,
            kb_id,
            query,
            top_k=8,
            hyde_enabled=False,
            query_expansion_enabled=False,
        ):
            self.query_expansion_enabled = query_expansion_enabled
            return [
                RetrievedChunk(
                    chunk_id=uuid.uuid4(),
                    document_id=uuid.uuid4(),
                    filename="demo.md",
                    content="命中的上下文",
                    score=0.9,
                    metadata={},
                )
            ], {"query_expansion_enabled": query_expansion_enabled}

    class StaticLLM:
        async def answer(self, question, contexts):
            return "评测回答"

    class StaticScorer:
        async def score(
            self,
            user_input,
            response,
            reference,
            retrieved_contexts,
            retrieved_context_ids,
            expected_context_ids,
        ):
            return ScoreBundle(
                metrics={"faithfulness": 1.0},
                reasons={"faithfulness": "ok"},
            )

    class RecordingSession:
        def __init__(self) -> None:
            self.added = []

        def add(self, value) -> None:
            self.added.append(value)

    retriever = RecordingRetriever()
    session = RecordingSession()
    service = EvaluationService(retriever=retriever, llm=StaticLLM(), scorer=StaticScorer())

    await service._evaluate_sample(
        session=session,
        tenant_id=uuid.uuid4(),
        kb=SimpleNamespace(id=uuid.uuid4()),
        run=SimpleNamespace(id=uuid.uuid4()),
        sample=SimpleNamespace(
            id=uuid.uuid4(),
            user_input="问题",
            reference="标准答案",
            expected_context_ids=[],
        ),
        top_k=12,
        query_expansion_enabled=True,
    )

    assert retriever.query_expansion_enabled is True
    assert session.added[0].retrieval_trace["query_expansion_enabled"] is True
