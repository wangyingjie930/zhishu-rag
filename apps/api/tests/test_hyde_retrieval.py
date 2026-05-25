import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.custom import CustomLLM

from rag_platform.services.retrieval.hybrid import HybridRetriever, RetrievedChunk
from rag_platform.services.retrieval.hyde import (
    HyDEExpansion,
    HyDEQueryExpander,
    OpenAICompatibleQueryLLM,
    QueryExpansionExpander,
    QueryExpansionTransform,
)


class StaticHyDELLM(CustomLLM):
    output: str

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(num_output=128, model_name="static-hyde-test")

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=self.output)

    def stream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponseGen:
        yield self.complete(prompt, formatted=formatted, **kwargs)


class FailingHyDEExpander:
    def expand(self, query: str, include_original: bool = True) -> HyDEExpansion:
        raise RuntimeError("hyde llm unavailable")


class LengthEmbeddingProvider:
    def __init__(self) -> None:
        self.texts = []

    async def embed(self, text: str):
        self.texts.append(text)
        return [float(len(text)), float(len(text) + 2)]


class RecordingKeywordSession:
    def __init__(self) -> None:
        self.statement = ""
        self.params = {}

    async def execute(self, statement, params):
        self.statement = str(statement)
        self.params = params
        return []


def test_hyde_expander_uses_llama_index_transform() -> None:
    expander = HyDEQueryExpander(llm=StaticHyDELLM(output="这是一段用于召回的假设答案。"))

    expansion = expander.expand("如何申请系统权限？")

    assert expansion.embedding_texts == [
        "这是一段用于召回的假设答案。",
        "如何申请系统权限？",
    ]
    assert expansion.hypothetical_document == "这是一段用于召回的假设答案。"


def test_openai_compatible_hyde_llm_builds_local_chat_completion_request() -> None:
    llm = OpenAICompatibleQueryLLM(
        base_url="http://localhost:8317",
        model="gpt-5.5",
        api_key="",
    )

    assert llm._request_url() == "http://localhost:8317/v1/chat/completions"
    assert llm._request_headers() == {"Content-Type": "application/json"}
    assert llm._request_payload("生成 HyDE")["model"] == "gpt-5.5"
    assert (
        llm._parse_text({"choices": [{"message": {"content": "假设答案"}}]})
        == "假设答案"
    )


def test_query_expansion_transform_uses_llama_index_query_bundle() -> None:
    transform = QueryExpansionTransform(
        llm=StaticHyDELLM(output='["Istio 遥测数据类型", "Istio 指标 分布式追踪 访问日志"]'),
        num_queries=2,
    )

    bundle = transform.run("Istio 生成哪几类遥测数据？")

    assert bundle.embedding_strs == [
        "Istio 生成哪几类遥测数据？",
        "Istio 遥测数据类型",
        "Istio 指标 分布式追踪 访问日志",
    ]


def test_query_expansion_expander_returns_queries_and_embedding_texts() -> None:
    expander = QueryExpansionExpander(
        llm=StaticHyDELLM(output="1. Istio 遥测数据类型\n2. Istio metrics traces logs"),
        num_queries=2,
    )

    expansion = expander.expand("Istio 生成哪几类遥测数据？")

    assert expansion.queries == ["Istio 遥测数据类型", "Istio metrics traces logs"]
    assert expansion.embedding_texts[0] == "Istio 生成哪几类遥测数据？"


@pytest.mark.asyncio
async def test_hyde_query_plan_falls_back_to_original_query() -> None:
    retriever = HybridRetriever(hyde_expander=FailingHyDEExpander())

    plan = await retriever._build_query_plan("如何申请系统权限？", {}, hyde_enabled=True)

    assert plan.hyde_enabled is True
    assert plan.hyde_status == "fallback"
    assert plan.vector_embedding_texts == ["如何申请系统权限？"]
    assert plan.hyde_error == "hyde llm unavailable"


class StaticQueryExpansionExpander:
    def expand(self, query: str, include_original: bool = True):
        return SimpleNamespace(
            queries=["Istio 遥测类型", "Istio metrics tracing access logs"],
            embedding_texts=[query, "Istio 遥测类型", "Istio metrics tracing access logs"],
        )


@pytest.mark.asyncio
async def test_query_plan_includes_query_expansion_texts() -> None:
    retriever = HybridRetriever(query_expansion_expander=StaticQueryExpansionExpander())

    plan = await retriever._build_query_plan(
        "Istio 生成哪几类遥测数据？",
        {},
        query_expansion_enabled=True,
    )

    assert plan.query_expansion_status == "applied"
    assert plan.expanded_queries == ["Istio 遥测类型", "Istio metrics tracing access logs"]
    assert plan.keyword_query == "Istio 生成哪几类遥测数据？"
    assert plan.retrieval_queries == [
        "Istio 生成哪几类遥测数据？",
        "Istio 遥测类型",
        "Istio metrics tracing access logs",
    ]
    assert plan.vector_embedding_texts == [
        "Istio 生成哪几类遥测数据？",
        "Istio 遥测类型",
        "Istio metrics tracing access logs",
    ]


@pytest.mark.asyncio
async def test_keyword_search_uses_pg_search_bm25() -> None:
    session = RecordingKeywordSession()
    retriever = HybridRetriever()

    rows = await retriever._keyword_search(
        session,
        tenant_id=uuid.uuid4(),
        kb_id=uuid.uuid4(),
        query="Istio 访问日志",
        limit=5,
    )

    assert rows == []
    assert "pdb.score(c.id) AS score" in session.statement
    assert "c.content ||| CAST(:query AS pdb.jieba)" in session.statement
    assert "ORDER BY pdb.score(c.id) DESC" in session.statement
    assert "ts_rank_cd" not in session.statement
    assert "search_vector" not in session.statement
    assert session.params["query"] == "Istio 访问日志"
    assert session.params["limit"] == 5


class RecordingMultiQueryRetriever(HybridRetriever):
    def __init__(self) -> None:
        super().__init__()
        self.vector_search_embeddings = []
        self.keyword_queries = []
        self.rescore_embedding = None
        self.chunk_ids = [uuid.uuid4(), uuid.uuid4()]

    async def _vector_search(
        self,
        session,
        tenant_id,
        kb_id,
        embedding,
        embedding_model,
        limit,
    ):
        self.vector_search_embeddings.append(embedding)
        chunk_id = self.chunk_ids[min(len(self.vector_search_embeddings) - 1, 1)]
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=uuid.uuid4(),
                filename="demo.txt",
                content="demo",
                score=0.1,
                metadata={},
            )
        ]

    async def _keyword_search(
        self,
        session,
        tenant_id,
        kb_id,
        query,
        limit,
    ):
        self.keyword_queries.append(query)
        chunk_id = self.chunk_ids[min(len(self.keyword_queries) - 1, 1)]
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=uuid.uuid4(),
                filename="demo.txt",
                content="demo",
                score=0.2,
                metadata={},
            )
        ]

    async def _score_vector_candidates(
        self,
        session,
        tenant_id,
        kb_id,
        embedding,
        embedding_model,
        chunk_ids,
    ):
        self.rescore_embedding = embedding
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=uuid.uuid4(),
                filename="demo.txt",
                content="demo",
                score=0.9,
                metadata={},
            )
            for chunk_id in chunk_ids
        ]


@pytest.mark.asyncio
async def test_multi_query_hybrid_retrieval_reranks_with_original_query() -> None:
    embedding_provider = LengthEmbeddingProvider()
    retriever = RecordingMultiQueryRetriever()
    original_embedding = await embedding_provider.embed("原问题")

    rows, trace = await retriever._multi_query_hybrid_candidate_search(
        session=None,
        tenant_id=uuid.uuid4(),
        kb_id=uuid.uuid4(),
        original_query="原问题",
        candidate_queries=["原问题", "扩展查询"],
        original_embedding=original_embedding,
        embedding_provider=embedding_provider,
        embedding_model="test",
        limit_per_query=3,
        vector_weight=0.65,
        keyword_weight=0.35,
    )

    assert retriever.vector_search_embeddings == [original_embedding, [4.0, 6.0]]
    assert retriever.keyword_queries == ["原问题", "扩展查询"]
    assert retriever.rescore_embedding == original_embedding
    assert trace["multi_query_candidate_count"] == 2
    assert trace["vector_scoring_query"] == "original_query"
    assert trace["vector_query_count"] == 2
    assert trace["query_local_keyword_candidate_count"] == 2
    assert trace["rerank_method"] == "original_query_vector_similarity"
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_hyde_embedding_texts_are_averaged_for_vector_search() -> None:
    embedding_provider = LengthEmbeddingProvider()
    retriever = HybridRetriever(embedding_provider=embedding_provider)

    embedding = await retriever._embed_query_texts(
        embedding_provider,
        ["假设答案", "原问题"],
    )

    assert embedding_provider.texts == ["假设答案", "原问题"]
    assert embedding == [3.5, 5.5]
