import re
from dataclasses import dataclass, field, replace
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Protocol, Union

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


class PdfParser:
    def parse(self, filename: str, payload: bytes) -> str:
        try:
            pages = self._extract_with_pdfium(payload)
        except Exception:
            pages = []
        if not pages:
            pages = self._extract_with_pypdf(payload)

        if not pages:
            raise ValueError(
                "未能从 PDF 中提取文本，扫描件或图片型 PDF 需要先接入 OCR。"
            )
        return "\n\n".join(
            f"PDF 第 {index} 页\n{page_text}"
            for index, page_text in enumerate(pages, start=1)
            if page_text
        )

    def _extract_with_pdfium(self, payload: bytes) -> List[str]:
        try:
            import pypdfium2 as pdfium
        except ImportError:
            return []

        pages: List[str] = []
        pdf = pdfium.PdfDocument(payload)
        try:
            for page in pdf:
                try:
                    textpage = page.get_textpage()
                    try:
                        page_text = textpage.get_text_range()
                    finally:
                        textpage.close()
                    page_text = normalize_pdf_page_text(page_text or "")
                    if page_text:
                        pages.append(page_text)
                finally:
                    page.close()
        finally:
            pdf.close()
        return pages

    def _extract_with_pypdf(self, payload: bytes) -> List[str]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "PDF 解析依赖 pypdfium2 或 pypdf，"
                "请在 API 环境安装其中一个依赖后重试。"
            ) from exc

        reader = PdfReader(BytesIO(payload), strict=False)
        pages: List[str] = []
        for page in reader.pages:
            page_text = normalize_pdf_page_text(page.extract_text() or "")
            if page_text:
                pages.append(page_text)
        return pages


class AutoParser:
    def parse(self, filename: str, payload: bytes) -> str:
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix in {"txt", "md", "markdown", "csv", "json", "log"}:
            return TextParser().parse(filename, payload)
        if suffix == "pdf":
            return PdfParser().parse(filename, payload)
        return (
            "该文件类型已上传，但当前演示解析器只内置文本类解析。"
            "生产环境可在 ParserRegistry 中继续接入 Office、HTML、OCR、邮件和图片解析器。"
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


_CJK_CHARS = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_CJK_SOFT_BREAK_LEFT = f"{_CJK_CHARS}，、；：）】》”’"
_CJK_SOFT_BREAK_RIGHT = f"{_CJK_CHARS}（【《“‘"


def normalize_pdf_page_text(text: str) -> str:
    """修复 PDF 抽取时常见的版式换行，避免把软换行误当句子边界。"""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?m)^\s*\d+\s*/\s*\d+\s*$", "", text)
    text = re.sub(r"(?m)^\s*PDF\s*第\s*\d+\s*页\s*$", "", text)
    text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)
    text = re.sub(r"(?m)^([^\n。！？.!?]{2,24})\n\1(?=[^\n])", r"\1\n", text)
    text = re.sub(
        rf"([{_CJK_SOFT_BREAK_LEFT}])\n([{_CJK_SOFT_BREAK_RIGHT}])",
        r"\1\2",
        text,
    )
    text = re.sub(
        rf"(?m)(^|\n)([{_CJK_CHARS}A-Za-z][{_CJK_CHARS}A-Za-z /-]{{1,16}})\2(?=[{_CJK_CHARS}（(])",
        r"\1\2",
        text,
    )
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

    strategy: str = "adaptive"
    chunk_size: int = 900
    chunk_overlap: int = 120
    overlap_ratio: float = 0.15
    window_size: int = 2
    max_chunk_size: int = 1200
    semantic_buffer_size: int = 1
    semantic_threshold: int = 95
    language: str = "zh"
    requested_strategy: Optional[str] = None
    routing_reason: Optional[str] = None

    @classmethod
    def from_dict(
        cls,
        raw_policy: Optional[Union[Dict[str, Any], "ChunkingPolicy"]],
    ) -> "ChunkingPolicy":
        if isinstance(raw_policy, cls):
            return raw_policy
        if not raw_policy:
            return cls()
        chunker_policy = raw_policy.get("chunker", raw_policy)
        allowed = set(cls.__dataclass_fields__)
        values = {key: value for key, value in chunker_policy.items() if key in allowed}
        return cls(**values)

    def normalized(self) -> "ChunkingPolicy":
        chunk_size = _clamp_int(self.chunk_size, 50, 1600)
        overlap_ratio = _clamp_float(self.overlap_ratio, 0.10, 0.20)
        chunk_overlap = min(int(chunk_size * overlap_ratio), chunk_size - 1)
        max_chunk_size = max(_clamp_int(self.max_chunk_size, chunk_size, 2400), chunk_size)
        return replace(
            self,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            overlap_ratio=overlap_ratio,
            max_chunk_size=max_chunk_size,
            semantic_buffer_size=_clamp_int(self.semantic_buffer_size, 1, 4),
            semantic_threshold=_clamp_int(self.semantic_threshold, 80, 98),
            window_size=_clamp_int(self.window_size, 1, 4),
        )

    def to_metadata(self) -> Dict[str, Any]:
        metadata = {
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "overlap_ratio": self.overlap_ratio,
            "window_size": self.window_size,
            "max_chunk_size": self.max_chunk_size,
            "semantic_buffer_size": self.semantic_buffer_size,
            "semantic_threshold": self.semantic_threshold,
            "language": self.language,
        }
        if self.requested_strategy:
            metadata["requested_strategy"] = self.requested_strategy
        if self.routing_reason:
            metadata["routing_reason"] = self.routing_reason
        return metadata


def default_ingestion_policy() -> Dict[str, Any]:
    return {
        "parser": "auto",
        "preprocessor": "default",
        "embedding": {"model": DEFAULT_EMBEDDING_MODEL_ID},
        "chunker": ChunkingPolicy().normalized().to_metadata(),
    }


def chinese_sentence_tokenizer(text: str) -> List[str]:
    """供 LlamaIndex 语义/窗口解析器使用的中文感知句子拆分器。"""

    sentences = re.findall(r".+?(?:[。！？…]+|(?:\n\s*){2,}|$)", text, flags=re.S)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def sentence_tokenizer_for_preview(text: str, language: str) -> List[str]:
    if language.lower().startswith("zh"):
        return chinese_sentence_tokenizer(text)
    sentences = re.findall(r".+?(?:[.!?]+|(?:\n\s*){2,}|$)", text, flags=re.S)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _sentence_splitter_for(language: str):
    return chinese_sentence_tokenizer if language.lower().startswith("zh") else None


def resolve_chunking_policy(
    raw_policy: Optional[Union[Dict[str, Any], ChunkingPolicy]],
    filename: str = "",
    mime_type: str = "",
    text: str = "",
) -> ChunkingPolicy:
    policy = ChunkingPolicy.from_dict(raw_policy).normalized()
    if policy.strategy != "adaptive":
        return policy

    profile = _inspect_document_profile(filename, mime_type, text)
    if profile["structured"]:
        return replace(
            policy,
            strategy="markdown_section",
            requested_strategy="adaptive",
            routing_reason=profile["reason"],
        )
    if profile["short"]:
        return replace(
            policy,
            strategy="sentence_window",
            requested_strategy="adaptive",
            routing_reason=profile["reason"],
        )
    return replace(
        policy,
        strategy="semantic_hybrid",
        requested_strategy="adaptive",
        routing_reason=profile["reason"],
    )


def needs_semantic_embedding(policy: ChunkingPolicy) -> bool:
    return policy.strategy in {"semantic", "semantic_hybrid"}


def _clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = minimum
    return max(minimum, min(parsed, maximum))


def _clamp_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = minimum
    return max(minimum, min(parsed, maximum))


def _inspect_document_profile(filename: str, mime_type: str, text: str) -> Dict[str, Any]:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    markdown_like = suffix in {"md", "markdown"} or bool(re.search(r"(?m)^#{1,6}\s+", text))
    table_like = "|" in text and bool(re.search(r"(?m)^\s*\|.+\|\s*$", text))
    if markdown_like or table_like:
        return {"structured": True, "short": False, "reason": "structure_aware"}
    if len(text) < 1800:
        return {"structured": False, "short": True, "reason": "short_text_window"}
    if mime_type.startswith("text/") or suffix in {"txt", "log", "json", "csv"}:
        return {"structured": False, "short": False, "reason": "semantic_text"}
    return {"structured": False, "short": False, "reason": "semantic_default"}


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
        nodes = parser.get_nodes_from_documents([LlamaDocument(text=text)])
        yield from self._nodes_to_chunks(nodes)

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
                chunking_tokenizer_fn=sentence_splitter,
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
        if strategy == "markdown_section":
            return MarkdownSectionSplitter(self.policy, self.tokenizer)
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
            fallback_nodes = window_parser.get_nodes_from_documents([LlamaDocument(text=content)])
            final_nodes.extend(fallback_nodes)

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


class MarkdownSectionSplitter:
    def __init__(self, policy: ChunkingPolicy, tokenizer) -> None:
        self.policy = policy
        self.tokenizer = tokenizer
        self.fallback = SentenceSplitter(
            chunk_size=policy.chunk_size,
            chunk_overlap=policy.chunk_overlap,
        )

    def get_nodes_from_documents(self, documents: List[LlamaDocument]):
        nodes = []
        for document in documents:
            for section in self._split_sections(document.text or ""):
                token_count = len(self.tokenizer(section))
                if token_count <= self.policy.max_chunk_size:
                    nodes.append(LlamaDocument(text=section))
                    continue
                # 结构段落过长时，继续按句子边界滑动切分。
                nodes.extend(self.fallback.get_nodes_from_documents([LlamaDocument(text=section)]))
        return nodes

    def _split_sections(self, text: str) -> Iterable[str]:
        current: List[str] = []
        for line in text.splitlines():
            is_heading = bool(re.match(r"^#{1,6}\s+", line))
            if is_heading and current:
                yield "\n".join(current).strip()
                current = []
            current.append(line)
        if current:
            yield "\n".join(current).strip()


class SentenceWindowChunker:
    """按句子边界聚合正文，并把邻近块作为窗口上下文写入 metadata。"""

    def __init__(self, policy: ChunkingPolicy) -> None:
        self.policy = policy
        self.tokenizer = get_tokenizer()
        self.sentence_parser = SentenceWindowNodeParser.from_defaults(
            sentence_splitter=lambda text: sentence_tokenizer_for_preview(
                text,
                self.policy.language,
            ),
            window_size=policy.window_size,
            window_metadata_key="sentence_window_context",
            original_text_metadata_key="sentence_original_text",
        )

    def split(self, text: str) -> Iterable[Chunk]:
        nodes = self.sentence_parser.get_nodes_from_documents([LlamaDocument(text=text)])
        sentences = [node.text.strip() for node in nodes if node.text.strip()]
        if not sentences:
            return

        raw_chunks = self._pack_sentences(sentences)
        for index, chunk_sentences in enumerate(raw_chunks):
            content = "".join(chunk_sentences).strip()
            if not content:
                continue
            window_context = self._build_window_context(raw_chunks, index)
            yield Chunk(
                index=index,
                content=content,
                token_count=max(1, len(self.tokenizer(content))),
                metadata={
                    "chunker": "sentence_window",
                    "window_context": window_context,
                    "original_text": content,
                    "sentence_count": len(chunk_sentences),
                    "chunking_policy": self.policy.to_metadata(),
                },
            )

    def _pack_sentences(self, sentences: List[str]) -> List[List[str]]:
        chunks: List[List[str]] = []
        current: List[str] = []

        for sentence in sentences:
            candidate = [*current, sentence]
            if current and self._token_count(candidate) > self.policy.chunk_size:
                chunks.append(current)
                current = self._overlap_tail(current)
            current.append(sentence)

        if current:
            chunks.append(current)
        return chunks

    def _overlap_tail(self, sentences: List[str]) -> List[str]:
        if self.policy.chunk_overlap <= 0 or len(sentences) <= 1:
            return []

        tail: List[str] = []
        for sentence in reversed(sentences):
            candidate = [sentence, *tail]
            if self._token_count(candidate) > self.policy.chunk_overlap:
                break
            tail = candidate

        # 避免重叠覆盖完整上一块，否则下一轮会产生几乎相同的块。
        return tail if len(tail) < len(sentences) else tail[1:]

    def _build_window_context(self, chunks: List[List[str]], index: int) -> str:
        start = max(0, index - self.policy.window_size)
        end = min(len(chunks), index + self.policy.window_size + 1)
        return "\n".join("".join(chunk).strip() for chunk in chunks[start:end] if chunk)

    def _token_count(self, sentences: List[str]) -> int:
        return len(self.tokenizer("".join(sentences)))


class ParentChildChunker:
    """先构造较大的父块，再把每个父块切成用于召回的子块。"""

    def __init__(self, policy: ChunkingPolicy) -> None:
        self.policy = policy
        self.tokenizer = get_tokenizer()

    def split(self, text: str) -> Iterable[Chunk]:
        parent_texts = self._split_parent_texts(text)
        child_index = 0
        for parent_index, parent_text in enumerate(parent_texts):
            parent_token_count = self._token_count(parent_text)
            child_texts = self._split_child_texts(parent_text)
            for child_offset, child_text in enumerate(child_texts):
                content = child_text.strip()
                if not content:
                    continue
                yield Chunk(
                    index=child_index,
                    content=content,
                    token_count=max(1, self._token_count(content)),
                    metadata={
                        "chunker": "parent_child",
                        "parent_index": parent_index,
                        "parent_text": parent_text,
                        "parent_token_count": parent_token_count,
                        "parent_character_count": len(parent_text),
                        "child_index": child_offset,
                        "chunking_policy": self.policy.to_metadata(),
                    },
                )
                child_index += 1

    def _split_parent_texts(self, text: str) -> List[str]:
        paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
        if not paragraphs:
            paragraphs = [text.strip()] if text.strip() else []
        return self._pack_units(paragraphs, self.policy.max_chunk_size, overlap=0)

    def _split_child_texts(self, parent_text: str) -> List[str]:
        sentences = sentence_tokenizer_for_preview(parent_text, self.policy.language)
        if not sentences:
            sentences = [parent_text]
        return self._pack_units(sentences, self.policy.chunk_size, overlap=self.policy.chunk_overlap)

    def _pack_units(self, units: List[str], target_tokens: int, overlap: int) -> List[str]:
        chunks: List[str] = []
        current: List[str] = []

        for unit in units:
            if self._token_count(unit) > target_tokens:
                if current:
                    chunks.append(self._join_units(current))
                    current = []
                chunks.extend(self._split_long_text(unit, target_tokens, overlap))
                continue

            candidate = [*current, unit]
            if current and self._token_count(self._join_units(candidate)) > target_tokens:
                chunks.append(self._join_units(current))
                current = self._overlap_tail(current, overlap)
            current.append(unit)

        if current:
            chunks.append(self._join_units(current))
        return [chunk for chunk in chunks if chunk.strip()]

    def _overlap_tail(self, units: List[str], overlap: int) -> List[str]:
        if overlap <= 0 or len(units) <= 1:
            return []

        tail: List[str] = []
        for unit in reversed(units):
            candidate = [unit, *tail]
            if self._token_count(self._join_units(candidate)) > overlap:
                break
            tail = candidate
        return tail if len(tail) < len(units) else tail[1:]

    def _split_long_text(self, text: str, target_tokens: int, overlap: int) -> List[str]:
        if not text:
            return []

        chunks: List[str] = []
        start = 0
        step_chars = max(1, target_tokens - overlap)
        while start < len(text):
            end = min(len(text), start + target_tokens)
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start += step_chars
        return [chunk for chunk in chunks if chunk]

    def _join_units(self, units: List[str]) -> str:
        if any("\n" in unit for unit in units):
            return "\n\n".join(unit.strip() for unit in units if unit.strip())
        return "".join(unit.strip() for unit in units if unit.strip())

    def _token_count(self, text: str) -> int:
        return len(self.tokenizer(text))


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
        policy: Optional[Union[Dict[str, Any], ChunkingPolicy]] = None,
        semantic_embed_model: Optional[BaseEmbedding] = None,
    ) -> Chunker:
        chunking_policy = ChunkingPolicy.from_dict(policy).normalized()
        if chunking_policy.strategy == "sentence_window":
            return SentenceWindowChunker(chunking_policy)
        if chunking_policy.strategy == "parent_child":
            return ParentChildChunker(chunking_policy)
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
            "pdf": PdfParser(),
            "text": TextParser(),
        }

    def get(self, name: str) -> Parser:
        return self._parsers.get(name, self._parsers["auto"])
