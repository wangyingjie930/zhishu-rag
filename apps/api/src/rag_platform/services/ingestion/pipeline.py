import hashlib
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.core.config import settings
from rag_platform.db.models import Document, DocumentChunk, KnowledgeBase
from rag_platform.domain.enums import DocumentStatus
from rag_platform.services.ingestion.embeddings import (
    DEFAULT_EMBEDDING_MODEL_ID,
    EmbeddingProvider,
    EmbeddingProviderRegistry,
)
from rag_platform.services.ingestion.strategies import (
    ChunkerRegistry,
    ChunkingPolicy,
    DefaultPreprocessor,
    ParserRegistry,
    default_ingestion_policy,
    needs_semantic_embedding,
    resolve_chunking_policy,
)


MAX_PREVIEW_CHUNKS = 80


class IngestionPipeline:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider = None,
        semantic_embed_model: Optional[BaseEmbedding] = None,
    ) -> None:
        self.parsers = ParserRegistry()
        self.preprocessor = DefaultPreprocessor()
        self.chunkers = ChunkerRegistry()
        self.embedding_provider = embedding_provider
        self.semantic_embed_model = semantic_embed_model
        self.embedding_providers = EmbeddingProviderRegistry()

    async def ingest_upload(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
        filename: str,
        mime_type: str,
        payload: bytes,
        parser: str = "auto",
        embedding_model: Optional[str] = None,
        chunking_policy: Optional[Dict[str, Any]] = None,
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
            ingestion_policy = await self._load_ingestion_policy(
                session,
                tenant_id,
                kb_id,
                embedding_model,
                chunking_policy=chunking_policy,
                persist_overrides=True,
            )
            embedding_model_id = ingestion_policy.get("embedding", {}).get(
                "model",
                DEFAULT_EMBEDDING_MODEL_ID,
            )
            clean_text = self._parse_and_clean(filename, payload, parser)
            chunking_policy = resolve_chunking_policy(
                ingestion_policy,
                filename=filename,
                mime_type=mime_type,
                text=clean_text,
            )
            # 保存本次入库实际生效的策略，方便后续审计与复现。
            document.metadata_ = {
                **document.metadata_,
                "ingestion_policy": {
                    "embedding": {"model": embedding_model_id},
                    "chunker": chunking_policy.to_metadata(),
                },
            }
            embedding_provider = self.embedding_provider or self.embedding_providers.get(
                embedding_model_id,
                usage="document",
            )
            chunks = await self._split_text(
                clean_text=clean_text,
                chunking_policy=chunking_policy,
                embedding_model_id=embedding_model_id,
            )
            for chunk in chunks:
                embedding = await embedding_provider.embed(chunk.content)
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
                            "embedding": {"model": embedding_model_id},
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

    async def preview_upload_chunks(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
        filename: str,
        mime_type: str,
        payload: bytes,
        parser: str = "auto",
        embedding_model: Optional[str] = None,
        chunking_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ingestion_policy = await self._load_ingestion_policy(
            session,
            tenant_id,
            kb_id,
            embedding_model,
            chunking_policy=chunking_policy,
            persist_overrides=False,
        )
        embedding_model_id = ingestion_policy.get("embedding", {}).get(
            "model",
            DEFAULT_EMBEDDING_MODEL_ID,
        )
        clean_text = self._parse_and_clean(filename, payload, parser)
        resolved_policy = resolve_chunking_policy(
            ingestion_policy,
            filename=filename,
            mime_type=mime_type,
            text=clean_text,
        )
        chunks = await self._split_text(
            clean_text=clean_text,
            chunking_policy=resolved_policy,
            embedding_model_id=embedding_model_id,
        )
        visible_chunks = chunks[:MAX_PREVIEW_CHUNKS]
        return {
            "filename": filename,
            "mime_type": mime_type,
            "parser": parser,
            "chunking_policy": resolved_policy.to_metadata(),
            "clean_text_length": len(clean_text),
            "total_chunks": len(chunks),
            "truncated": len(chunks) > len(visible_chunks),
            "chunks": [
                {
                    "index": chunk.index,
                    "content": chunk.content,
                    "token_count": chunk.token_count,
                    "character_count": len(chunk.content),
                    "metadata": chunk.metadata,
                }
                for chunk in visible_chunks
            ],
        }

    def _parse_and_clean(self, filename: str, payload: bytes, parser: str) -> str:
        raw_text = self.parsers.get(parser).parse(filename, payload)
        return self.preprocessor.clean(raw_text)

    async def _split_text(
        self,
        clean_text: str,
        chunking_policy,
        embedding_model_id: str,
    ) -> List:
        semantic_embed_model = self.semantic_embed_model
        if semantic_embed_model is None and needs_semantic_embedding(chunking_policy):
            semantic_embed_model = self.embedding_providers.get_llama_index(embedding_model_id)
        chunker = self.chunkers.get(
            chunking_policy,
            semantic_embed_model=semantic_embed_model,
        )
        return list(chunker.split(clean_text))

    async def _load_ingestion_policy(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
        embedding_model: Optional[str] = None,
        chunking_policy: Optional[Dict[str, Any]] = None,
        persist_overrides: bool = False,
    ) -> dict:
        result = await session.execute(
            select(KnowledgeBase).where(
                KnowledgeBase.tenant_id == tenant_id,
                KnowledgeBase.id == kb_id,
            )
        )
        kb = result.scalar_one_or_none()
        policy = deepcopy(
            kb.ingestion_policy if kb and kb.ingestion_policy else default_ingestion_policy()
        )
        policy.setdefault("embedding", {})
        policy["embedding"].setdefault("model", DEFAULT_EMBEDDING_MODEL_ID)

        if embedding_model:
            # 上传、语义分块和检索必须保持在同一个真实 embedding 空间里。
            policy["embedding"]["model"] = embedding_model
        if chunking_policy:
            policy.setdefault("chunker", {})
            policy["chunker"].update(chunking_policy.get("chunker", chunking_policy))
        policy["chunker"] = ChunkingPolicy.from_dict(policy).normalized().to_metadata()

        if persist_overrides and kb is not None and (embedding_model or chunking_policy):
            # 用户在导入向导提交的参数会固化到知识库，后续上传默认沿用同一套策略。
            kb.ingestion_policy = policy
            await session.flush()

        return policy
