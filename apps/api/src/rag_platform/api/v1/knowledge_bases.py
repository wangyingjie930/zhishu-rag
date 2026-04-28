import uuid
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import KnowledgeBase
from rag_platform.db.session import get_session
from rag_platform.schemas.rag import KnowledgeBaseCreate, KnowledgeBaseOut
from rag_platform.services.security.context import RequestContext, get_request_context

router = APIRouter()


@router.get("", response_model=List[KnowledgeBaseOut])
async def list_knowledge_bases(
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> List[KnowledgeBase]:
    result = await session.execute(
        select(KnowledgeBase).where(KnowledgeBase.tenant_id == context.tenant_id).order_by(KnowledgeBase.created_at)
    )
    return list(result.scalars())


@router.post("", response_model=KnowledgeBaseOut)
async def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> KnowledgeBase:
    kb = KnowledgeBase(
        id=uuid.uuid4(),
        tenant_id=context.tenant_id,
        name=payload.name,
        description=payload.description,
        visibility=payload.visibility,
        retrieval_policy={
            "top_k": 8,
            "vector_weight": 0.65,
            "keyword_weight": 0.35,
            "reranker": "none",
        },
    )
    session.add(kb)
    await session.commit()
    await session.refresh(kb)
    return kb

