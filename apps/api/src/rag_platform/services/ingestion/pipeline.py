import hashlib
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.core.config import settings
from rag_platform.db.models import Document, DocumentChunk, KnowledgeBase
from rag_platform.domain.enums import DocumentStatus
from rag_platform.services.ingestion.embeddings import DeterministicEmbeddingProvider, EmbeddingProvider
from rag_platform.services.ingestion.strategies import (
    ChunkerRegistry,
    DefaultPreprocessor,
    ParserRegistry,
    default_ingestion_policy,
)


class IngestionPipeline:
    def __init__(self, embedding_provider: EmbeddingProvider = None) -> None:
        self.parsers = ParserRegistry()
        self.preprocessor = DefaultPreprocessor()
        self.chunkers = ChunkerRegistry()
        self.embedding_provider = embedding_provider or DeterministicEmbeddingProvider()

    async def ingest_upload(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
        filename: str,
        mime_type: str,
        payload: bytes,
        parser: str = "auto",
    ) -> Document:
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        checksum = hashlib.sha256(payload).hexdigest()
        object_path = Path(settings.upload_dir) / f"{checksum}-{filename}"
        object_path.write_bytes(payload)

        document = Document(
            tenant_id=tenant_id,
            kb_id=kb_id,
            filename=filename,
            mime_type=mime_type or "application/octet-stream",
            object_uri=str(object_path),
            status=DocumentStatus.processing.value,
            parser=parser,
            checksum=checksum,
            metadata_={"source": "upload", "bytes": len(payload)},
        )
        session.add(document)
        await session.flush()

        try:
            ingestion_policy = await self._load_ingestion_policy(session, tenant_id, kb_id)
            raw_text = self.parsers.get(parser).parse(filename, payload)
            clean_text = self.preprocessor.clean(raw_text)
            chunker = self.chunkers.get(ingestion_policy)
            for chunk in chunker.split(clean_text):
                embedding = await self.embedding_provider.embed(chunk.content)
                session.add(
                    DocumentChunk(
                        tenant_id=tenant_id,
                        kb_id=kb_id,
                        document_id=document.id,
                        chunk_index=chunk.index,
                        content=chunk.content,
                        token_count=chunk.token_count,
                        metadata_={
                            "parser": parser,
                            "checksum": checksum,
                            **chunk.metadata,
                        },
                        embedding=embedding,
                    )
                )
            document.status = DocumentStatus.indexed.value
            document.error_message = None
        except Exception as exc:  # pragma: no cover - defensive boundary
            document.status = DocumentStatus.failed.value
            document.error_message = str(exc)
        await session.commit()
        await session.refresh(document)
        return document

    async def _load_ingestion_policy(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
    ) -> dict:
        result = await session.execute(
            select(KnowledgeBase.ingestion_policy).where(
                KnowledgeBase.tenant_id == tenant_id,
                KnowledgeBase.id == kb_id,
            )
        )
        return result.scalar_one_or_none() or default_ingestion_policy()
