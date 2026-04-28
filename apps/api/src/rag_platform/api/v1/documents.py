import uuid
from typing import List

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import Document
from rag_platform.db.session import get_session
from rag_platform.schemas.rag import DocumentOut
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


@router.post("/upload", response_model=DocumentOut)
async def upload_document(
    kb_id: uuid.UUID = Form(...),
    parser: str = Form("auto"),
    embedding_model: str = Form(""),
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
    )
