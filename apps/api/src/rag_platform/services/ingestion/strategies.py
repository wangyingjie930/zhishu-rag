import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

from llama_index.core import Document as LlamaDocument
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceSplitter,
    SentenceWindowNodeParser,
    TokenTextSplitter,
)
from llama_index.core.utils import get_tokenizer

from rag_platform.services.ingestion.embeddings import DEFAULT_EMBEDDING_MODEL_ID


class Parser(Protocol):
    def parse(self, filename: str, payload: bytes) -> str:
        ...


class TextParser:
    def parse(self, filename: str, payload: bytes) -> str:
        return payload.decode("utf-8", errors="ignore")


class AutoParser:
    def parse(self, filename: str, payload: bytes) -> str:
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix in {"txt", "md", "markdown", "csv", "json", "log"}:
            return TextParser().parse(filename, payload)
        return (
            "该文件类型已上传，但当前演示解析器只内置文本类解析。"
            "生产环境可在 ParserRegistry 中接入 PDF、Office、HTML、OCR、邮件和图片解析器。"
        )


class Preprocessor(Protocol):
    def clean(self, text: str) -> str:
        ...


class DefaultPreprocessor:
    def clean(self, text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


@dataclass(frozen=True)
class Chunk:
    index: int
    content: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class Chunker(Protocol):
    def split(self, text: str) -> Iterable[Chunk]:
        ...


@dataclass(frozen=True)
class ChunkingPolicy:
    """面向企业的文本分块配置。

    项目保持策略名称稳定，而底层的分块器库可以独立演进。
    业务用户通常应该选择一个预设策略；工程师可以对具体的数值进行调优。
    """

    strategy: str = "semantic_hybrid"
    chunk_size: int = 900
    chunk_overlap: int = 120
    window_size: int = 2
    max_chunk_size: int = 1200
    semantic_buffer_size: int = 1
    semantic_threshold: int = 95
    language: str = "zh"

    @classmethod
    def from_dict(cls, raw_policy: Optional[Dict[str, Any]]) -> "ChunkingPolicy":
        if not raw_policy:
            return cls()
        chunker_policy = raw_policy.get("chunker", raw_policy)
        allowed = set(cls.__dataclass_fields__)
        values = {key: value for key, value in chunker_policy.items() if key in allowed}
        return cls(**values)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "window_size": self.window_size,
            "max_chunk_size": self.max_chunk_size,
            "semantic_buffer_size": self.semantic_buffer_size,
            "semantic_threshold": self.semantic_threshold,
            "language": self.language,
        }


def default_ingestion_policy() -> Dict[str, Any]:
    return {
        "parser": "auto",
        "preprocessor": "default",
        "embedding": {"model": DEFAULT_EMBEDDING_MODEL_ID},
        "chunker": ChunkingPolicy().to_metadata(),
    }


def chinese_sentence_tokenizer(text: str) -> List[str]:
    """供 LlamaIndex 语义/窗口解析器使用的中文感知句子拆分器。"""

    sentences = re.findall(r"[^。！？…\n]+[。！？…\n]?", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _sentence_splitter_for(language: str):
    return chinese_sentence_tokenizer if language.lower().startswith("zh") else None


class LlamaIndexChunker:
    """LlamaIndex 节点解析器的适配器。

    LlamaIndex 负责拆分行为；此类仅将节点规范化为平台的 Chunk 约定，并记录调试、重建和检索提示所需的元数据。
    """

    def __init__(
        self,
        policy: Optional[ChunkingPolicy] = None,
        semantic_embed_model: Optional[BaseEmbedding] = None,
    ) -> None:
        self.policy = policy or ChunkingPolicy()
        self.tokenizer = get_tokenizer()
        self.semantic_embed_model = semantic_embed_model

    def split(self, text: str) -> Iterable[Chunk]:
        strategy = self.policy.strategy
        if strategy == "semantic_hybrid":
            yield from self._split_semantic_hybrid(text)
            return

        parser = self._build_parser(strategy)
        yield from self._nodes_to_chunks(parser.get_nodes_from_documents([LlamaDocument(text=text)]))

    def _build_parser(self, strategy: str):
        sentence_splitter = _sentence_splitter_for(self.policy.language)
        if strategy == "token":
            return TokenTextSplitter(
                chunk_size=self.policy.chunk_size,
                chunk_overlap=self.policy.chunk_overlap,
            )
        if strategy in {"sentence", "sentence_sliding_window"}:
            return SentenceSplitter(
                chunk_size=self.policy.chunk_size,
                chunk_overlap=self.policy.chunk_overlap,
            )
        if strategy == "sentence_window":
            return SentenceWindowNodeParser.from_defaults(
                sentence_splitter=sentence_splitter,
                window_size=self.policy.window_size,
                window_metadata_key="window_context",
                original_text_metadata_key="original_text",
            )
        if strategy == "semantic":
            if self.semantic_embed_model is None:
                raise ValueError("Semantic chunking requires a real LlamaIndex embedding model")
            return SemanticSplitterNodeParser(
                buffer_size=self.policy.semantic_buffer_size,
                breakpoint_percentile_threshold=self.policy.semantic_threshold,
                sentence_splitter=sentence_splitter,
                embed_model=self.semantic_embed_model,
            )
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    def _split_semantic_hybrid(self, text: str) -> Iterable[Chunk]:
        semantic_parser = self._build_parser("semantic")
        window_parser = SentenceSplitter(
            chunk_size=self.policy.chunk_size,
            chunk_overlap=self.policy.chunk_overlap,
        )
        primary_nodes = semantic_parser.get_nodes_from_documents([LlamaDocument(text=text)])
        final_nodes = []
        for node in primary_nodes:
            content = node.get_content()
            token_count = len(self.tokenizer(content))
            if token_count <= self.policy.max_chunk_size:
                final_nodes.append(node)
                continue

            # 非常大的语义段落对于提示词组装来说代价仍然过高，因此我们
            # 采用 LlamaIndex 考虑句子边界的滑动拆分作为第二次处理。
            final_nodes.extend(window_parser.get_nodes_from_documents([LlamaDocument(text=content)]))

        yield from self._nodes_to_chunks(final_nodes)

    def _nodes_to_chunks(self, nodes) -> Iterable[Chunk]:
        for index, node in enumerate(nodes):
            content = node.get_content().strip()
            if not content:
                continue
            metadata = dict(node.metadata or {})
            metadata.update(
                {
                    "chunker": "llama_index",
                    "chunking_policy": self.policy.to_metadata(),
                }
            )
            yield Chunk(
                index=index,
                content=content,
                token_count=max(1, len(self.tokenizer(content))),
                metadata=metadata,
            )


class RecursiveTextChunker:
    def __init__(self, chunk_size: int = 900, overlap: int = 120) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> Iterable[Chunk]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = paragraph
        if current:
            chunks.append(current)

        normalized: List[str] = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                normalized.append(chunk)
                continue
            start = 0
            while start < len(chunk):
                normalized.append(chunk[start : start + self.chunk_size])
                start += max(1, self.chunk_size - self.overlap)

        for index, content in enumerate(normalized):
            yield Chunk(
                index=index,
                content=content,
                token_count=max(1, len(content) // 4),
                metadata={
                    "chunker": "recursive_text",
                    "chunking_policy": {
                        "strategy": "recursive_text",
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.overlap,
                    },
                },
            )


class ChunkerRegistry:
    def get(
        self,
        policy: Optional[Dict[str, Any]] = None,
        semantic_embed_model: Optional[BaseEmbedding] = None,
    ) -> Chunker:
        chunking_policy = ChunkingPolicy.from_dict(policy)
        if chunking_policy.strategy == "recursive_text":
            return RecursiveTextChunker(
                chunk_size=chunking_policy.chunk_size,
                overlap=chunking_policy.chunk_overlap,
            )
        return LlamaIndexChunker(chunking_policy, semantic_embed_model=semantic_embed_model)


class ParserRegistry:
    def __init__(self) -> None:
        self._parsers = {
            "auto": AutoParser(),
            "text": TextParser(),
        }

    def get(self, name: str) -> Parser:
        return self._parsers.get(name, self._parsers["auto"])
