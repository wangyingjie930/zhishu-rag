import uuid

from llama_index.core.base.embeddings.base import BaseEmbedding

from rag_platform.services.retrieval.hybrid import HybridRetriever
from rag_platform.services.ingestion.strategies import (
    ChunkerRegistry,
    ChunkingPolicy,
    DefaultPreprocessor,
    RecursiveTextChunker,
    resolve_chunking_policy,
)


class StubEmbedding(BaseEmbedding):
    dimensions: int = 16

    def _embed(self, text: str):
        values = [float((ord(char) % 17) + 1) for char in text[: self.dimensions]]
        values.extend([0.1] * (self.dimensions - len(values)))
        return values[: self.dimensions]

    def _get_text_embedding(self, text: str):
        return self._embed(text)

    def _get_query_embedding(self, query: str):
        return self._embed(query)

    async def _aget_query_embedding(self, query: str):
        return self._embed(query)


def test_preprocess_and_chunk() -> None:
    text = "标题\n\n\n第一段   内容\n\n第二段内容"
    clean = DefaultPreprocessor().clean(text)
    chunks = list(RecursiveTextChunker(chunk_size=12, overlap=3).split(clean))
    assert "\n\n\n" not in clean
    assert chunks
    assert chunks[0].token_count >= 1


def test_sentence_window_chunker_keeps_window_metadata() -> None:
    text = "第一句说明背景。第二句给出关键结论。第三句补充限制条件。"
    chunker = ChunkerRegistry().get(
        {
            "chunker": {
                "strategy": "sentence_window",
                "language": "zh",
                "chunk_size": 50,
                "window_size": 1,
            }
        }
    )

    chunks = list(chunker.split(text))

    assert chunks
    assert chunks[0].metadata["chunker"] == "sentence_window"
    assert "window_context" in chunks[0].metadata
    assert "original_text" in chunks[0].metadata


def test_sentence_window_chunk_size_changes_preview_chunks() -> None:
    text = (
        "第一句说明背景内容需要被保留。第二句给出关键结论方便观察。"
        "第三句补充限制条件用于继续撑开长度。第四句说明下一步动作。"
        "第五句用于观察窗口上下文。第六句作为最后一个句子。"
    )
    small_chunker = ChunkerRegistry().get(
        {"chunker": {"strategy": "sentence_window", "language": "zh", "chunk_size": 50}}
    )
    large_chunker = ChunkerRegistry().get(
        {"chunker": {"strategy": "sentence_window", "language": "zh", "chunk_size": 160}}
    )

    small_chunks = list(small_chunker.split(text))
    large_chunks = list(large_chunker.split(text))

    assert len(small_chunks) > len(large_chunks)
    assert small_chunks[0].metadata["chunking_policy"]["chunk_size"] == 50


def test_sentence_window_size_changes_window_context() -> None:
    text = "".join(f"第{index}句用于制造稳定窗口边界。" for index in range(1, 11))
    narrow_chunker = ChunkerRegistry().get(
        {
            "chunker": {
                "strategy": "sentence_window",
                "language": "zh",
                "chunk_size": 20,
                "overlap_ratio": 0.1,
                "window_size": 1,
            }
        }
    )
    wide_chunker = ChunkerRegistry().get(
        {
            "chunker": {
                "strategy": "sentence_window",
                "language": "zh",
                "chunk_size": 20,
                "overlap_ratio": 0.1,
                "window_size": 2,
            }
        }
    )

    narrow_chunks = list(narrow_chunker.split(text))
    wide_chunks = list(wide_chunker.split(text))

    assert len(narrow_chunks) >= 5
    assert len(narrow_chunks) == len(wide_chunks)
    assert (
        narrow_chunks[2].metadata["window_context"]
        != wide_chunks[2].metadata["window_context"]
    )
    assert wide_chunks[1].metadata["chunking_policy"]["window_size"] == 2


def test_parent_child_chunker_records_parent_metadata() -> None:
    text = (
        "第一段说明系统背景。第二句提供父块上下文。\n\n"
        "第二段包含需要召回的细节。第四句继续补充边界。第五句收束。"
    )
    chunker = ChunkerRegistry().get(
        {
            "chunker": {
                "strategy": "parent_child",
                "language": "zh",
                "chunk_size": 30,
                "overlap_ratio": 0.1,
                "max_chunk_size": 80,
            }
        }
    )

    chunks = list(chunker.split(text))

    assert chunks
    assert chunks[0].metadata["chunker"] == "parent_child"
    assert chunks[0].metadata["parent_index"] == 0
    assert chunks[0].metadata["parent_text"]
    assert chunks[0].metadata["parent_token_count"] >= chunks[0].token_count
    assert chunks[0].metadata["chunking_policy"]["strategy"] == "parent_child"


def test_parent_child_retrieval_expands_content_to_parent_text() -> None:
    parent_text = "父块包含完整上下文。子块只是命中的较短片段。"
    row = {
        "chunk_id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "filename": "demo.txt",
        "content": "子块只是命中的较短片段。",
        "score": 0.9,
        "metadata": {
            "chunker": "parent_child",
            "parent_index": 0,
            "parent_text": parent_text,
        },
    }

    chunk = HybridRetriever()._row_to_chunk(row)

    assert chunk.content == parent_text
    assert chunk.metadata["matched_child_content"] == "子块只是命中的较短片段。"


def test_semantic_hybrid_records_policy_metadata() -> None:
    text = (
        "大语言模型用于理解和生成文本。Transformer 提升了长文本建模能力。"
        "企业知识库需要稳定的检索链路。切片策略会影响召回质量。"
    )
    chunker = ChunkerRegistry().get(
        {
            "chunker": {
                "strategy": "semantic_hybrid",
                "language": "zh",
                "chunk_size": 80,
                "chunk_overlap": 10,
                "max_chunk_size": 120,
                "semantic_threshold": 95,
            }
        },
        semantic_embed_model=StubEmbedding(),
    )

    chunks = list(chunker.split(text))

    assert chunks
    assert chunks[0].metadata["chunker"] == "llama_index"
    assert chunks[0].metadata["chunking_policy"]["strategy"] == "semantic_hybrid"


def test_adaptive_policy_routes_markdown_to_structure_aware_chunker() -> None:
    policy = resolve_chunking_policy(
        {"chunker": {"strategy": "adaptive", "chunk_size": 800, "overlap_ratio": 0.2}},
        filename="handbook.md",
        mime_type="text/markdown",
        text="# 标题\n\n第一段\n\n## 小节\n\n第二段",
    )
    chunker = ChunkerRegistry().get(policy)

    chunks = list(chunker.split("# 标题\n\n第一段\n\n## 小节\n\n第二段"))

    assert chunks
    assert chunks[0].metadata["chunking_policy"]["strategy"] == "markdown_section"
    assert chunks[0].metadata["chunking_policy"]["requested_strategy"] == "adaptive"
    assert chunks[0].metadata["chunking_policy"]["chunk_overlap"] == 160


def test_overlap_ratio_is_clamped_to_recommended_range() -> None:
    policy = ChunkingPolicy(chunk_size=1000, overlap_ratio=0.5).normalized()

    assert policy.overlap_ratio == 0.2
    assert policy.chunk_overlap == 200


def test_small_preview_chunk_size_is_kept_for_manual_tuning() -> None:
    policy = ChunkingPolicy(
        strategy="sentence_window",
        chunk_size=50,
        max_chunk_size=100,
        overlap_ratio=0.15,
    ).normalized()

    assert policy.chunk_size == 50
    assert policy.chunk_overlap == 7
    assert policy.max_chunk_size == 100


def test_frontend_chunking_fields_are_normalized_into_runtime_policy() -> None:
    policy = ChunkingPolicy.from_dict(
        {
            "chunker": {
                "strategy": "semantic_hybrid",
                "chunk_size": 700,
                "overlap_ratio": 0.12,
                "max_chunk_size": 1100,
                "semantic_buffer_size": 3,
                "semantic_threshold": 88,
                "window_size": 4,
            }
        }
    ).normalized()

    assert policy.chunk_size == 700
    assert policy.overlap_ratio == 0.12
    assert policy.chunk_overlap == 84
    assert policy.max_chunk_size == 1100
    assert policy.semantic_buffer_size == 3
    assert policy.semantic_threshold == 88
    assert policy.window_size == 4
