import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import Document, DocumentChunk
from rag_platform.db.session import get_session
from rag_platform.domain.enums import DocumentStatus
from rag_platform.schemas.rag import DocumentChunkPreviewOut, DocumentOut, DocumentReindexRequest
from rag_platform.services.ingestion.pipeline import IngestionPipeline
from rag_platform.services.security.context import RequestContext, get_request_context

router = APIRouter()


@router.get("", response_model=List[DocumentOut])
async def list_documents(
    kb_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> List[Document]:
    result = await session.execute(
        select(Document)
        .where(Document.tenant_id == context.tenant_id, Document.kb_id == kb_id)
        .order_by(Document.created_at.desc())
    )
    return list(result.scalars())


@router.post("/preview", response_model=DocumentChunkPreviewOut)
async def preview_document_chunks(
    kb_id: uuid.UUID = Form(...),
    parser: str = Form("auto"),
    embedding_model: str = Form(""),
    chunking_policy: str = Form("{}"),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> Dict[str, Any]:
    payload = await file.read()
    pipeline = IngestionPipeline()
    return await pipeline.preview_upload_chunks(
        session=session,
        tenant_id=context.tenant_id,
        kb_id=kb_id,
        filename=file.filename or "uploaded-document",
        mime_type=file.content_type or "application/octet-stream",
        payload=payload,
        parser=parser,
        embedding_model=embedding_model or None,
        chunking_policy=_parse_chunking_policy(chunking_policy),
    )


@router.post("/upload", response_model=DocumentOut)
async def upload_document(
    kb_id: uuid.UUID = Form(...),
    parser: str = Form("auto"),
    embedding_model: str = Form(""),
    chunking_policy: str = Form("{}"),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> Document:
    payload = await file.read()
    pipeline = IngestionPipeline()
    return await pipeline.ingest_upload(
        session=session,
        tenant_id=context.tenant_id,
        kb_id=kb_id,
        filename=file.filename or "uploaded-document",
        mime_type=file.content_type or "application/octet-stream",
        payload=payload,
        parser=parser,
        embedding_model=embedding_model or None,
        chunking_policy=_parse_chunking_policy(chunking_policy),
    )


@router.post("/{document_id}/preview-reindex", response_model=DocumentChunkPreviewOut)
async def preview_document_reindex(
    document_id: uuid.UUID,
    payload: DocumentReindexRequest,
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> Dict[str, Any]:
    document = await _require_document(session, context.tenant_id, document_id)
    file_payload = _read_document_payload(document)
    pipeline = IngestionPipeline()
    return await pipeline.preview_upload_chunks(
        session=session,
        tenant_id=context.tenant_id,
        kb_id=document.kb_id,
        filename=document.filename,
        mime_type=document.mime_type,
        payload=file_payload,
        parser=payload.parser or document.parser or "auto",
        embedding_model=payload.embedding_model or None,
        chunking_policy=payload.chunking_policy,
    )


@router.post("/{document_id}/reindex", response_model=DocumentOut)
async def reindex_document(
    document_id: uuid.UUID,
    payload: DocumentReindexRequest,
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> Document:
    document = await _require_document(session, context.tenant_id, document_id)
    pipeline = IngestionPipeline()

    try:
        document.status = DocumentStatus.processing.value
        document.parser = payload.parser or document.parser or "auto"
        await session.flush()

        await _delete_document_chunks(
            session,
            context.tenant_id,
            document.id,
        )
        ingestion_policy = await pipeline._load_ingestion_policy(
            session,
            context.tenant_id,
            document.kb_id,
            embedding_model=payload.embedding_model or None,
            chunking_policy=payload.chunking_policy,
        )
        await pipeline.index_existing_document(
            session=session,
            tenant_id=context.tenant_id,
            document=document,
            ingestion_policy=ingestion_policy,
        )
        document.status = DocumentStatus.indexed.value
        document.error_message = None
        await session.commit()
    except Exception as exc:  # pragma: no cover - defensive boundary around external parsers/models
        await session.rollback()
        document = await _require_document(session, context.tenant_id, document_id)
        document.status = DocumentStatus.failed.value
        document.error_message = str(exc)
        await session.commit()

    await session.refresh(document)
    return document


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> None:
    result = await session.execute(
        select(Document).where(
            Document.tenant_id == context.tenant_id,
            Document.id == document_id,
        )
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # 数据库级彻底删除：先删向量分块，再删文档记录，避免留下不可见的召回数据。
    await session.execute(
        delete(DocumentChunk).where(
            DocumentChunk.tenant_id == context.tenant_id,
            DocumentChunk.document_id == document.id,
        )
    )
    await session.delete(document)
    await session.commit()


async def _require_document(
    session: AsyncSession,
    tenant_id: uuid.UUID,
    document_id: uuid.UUID,
) -> Document:
    result = await session.execute(
        select(Document).where(
            Document.tenant_id == tenant_id,
            Document.id == document_id,
        )
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


async def _delete_document_chunks(
    session: AsyncSession,
    tenant_id: uuid.UUID,
    document_id: uuid.UUID,
) -> None:
    await session.execute(
        delete(DocumentChunk).where(
            DocumentChunk.tenant_id == tenant_id,
            DocumentChunk.document_id == document_id,
        )
    )


def _read_document_payload(document: Document) -> bytes:
    path = Path(document.object_uri)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document source file not found")
    return path.read_bytes()


def _parse_chunking_policy(raw_policy: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw_policy or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="chunking_policy 必须是合法 JSON") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="chunking_policy 必须是 JSON 对象")
    return parsed
