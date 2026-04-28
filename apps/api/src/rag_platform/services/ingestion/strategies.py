import re
from dataclasses import dataclass
from typing import Iterable, List, Protocol


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


class Chunker(Protocol):
    def split(self, text: str) -> Iterable[Chunk]:
        ...


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
            yield Chunk(index=index, content=content, token_count=max(1, len(content) // 4))


class ParserRegistry:
    def __init__(self) -> None:
        self._parsers = {
            "auto": AutoParser(),
            "text": TextParser(),
        }

    def get(self, name: str) -> Parser:
        return self._parsers.get(name, self._parsers["auto"])

