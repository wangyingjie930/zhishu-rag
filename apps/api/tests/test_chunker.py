from rag_platform.services.ingestion.strategies import DefaultPreprocessor, RecursiveTextChunker


def test_preprocess_and_chunk() -> None:
    text = "标题\n\n\n第一段   内容\n\n第二段内容"
    clean = DefaultPreprocessor().clean(text)
    chunks = list(RecursiveTextChunker(chunk_size=12, overlap=3).split(clean))
    assert "\n\n\n" not in clean
    assert chunks
    assert chunks[0].token_count >= 1

