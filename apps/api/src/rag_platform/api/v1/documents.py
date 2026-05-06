import json
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import Document
from rag_platform.db.session import get_session
from rag_platform.schemas.rag import DocumentChunkPreviewOut, DocumentOut
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


def _parse_chunking_policy(raw_policy: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw_policy or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="chunking_policy 必须是合法 JSON") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="chunking_policy 必须是 JSON 对象")
    return parsed
