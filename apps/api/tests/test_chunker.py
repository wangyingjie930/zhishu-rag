from rag_platform.services.ingestion.strategies import (
    ChunkerRegistry,
    DefaultPreprocessor,
    RecursiveTextChunker,
)


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
                "window_size": 1,
            }
        }
    )

    chunks = list(chunker.split(text))

    assert chunks
    assert chunks[0].metadata["chunker"] == "llama_index"
    assert "window_context" in chunks[0].metadata
    assert "original_text" in chunks[0].metadata


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
        }
    )

    chunks = list(chunker.split(text))

    assert chunks
    assert chunks[0].metadata["chunker"] == "llama_index"
    assert chunks[0].metadata["chunking_policy"]["strategy"] == "semantic_hybrid"
