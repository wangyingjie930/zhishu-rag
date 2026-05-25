import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import KnowledgeBase
from rag_platform.db.session import get_session
from rag_platform.schemas.rag import KnowledgeBaseCreate, KnowledgeBaseOut, RetrievalPolicyCreate
from rag_platform.services.ingestion.strategies import default_ingestion_policy
from rag_platform.services.security.context import RequestContext, get_request_context

router = APIRouter()


@router.get("", response_model=List[KnowledgeBaseOut])
async def list_knowledge_bases(
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> List[KnowledgeBase]:
    result = await session.execute(
        select(KnowledgeBase)
        .where(KnowledgeBase.tenant_id == context.tenant_id)
        .order_by(KnowledgeBase.created_at)
    )
    return list(result.scalars())


@router.post("", response_model=KnowledgeBaseOut)
async def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> KnowledgeBase:
    policy = default_ingestion_policy()
    retrieval_policy = (
        payload.retrieval_policy.model_dump()
        if payload.retrieval_policy
        else {
            "top_k": 8,
            "vector_weight": 0.65,
            "keyword_weight": 0.35,
            "reranker": "none",
            "score_threshold": 0,
        }
    )

    kb = KnowledgeBase(
        id=uuid.uuid4(),
        tenant_id=context.tenant_id,
        name=payload.name,
        description=payload.description,
        visibility=payload.visibility,
        retrieval_policy=retrieval_policy,
        ingestion_policy=policy,
    )
    session.add(kb)
    await session.commit()
    await session.refresh(kb)
    return kb


@router.put("/{kb_id}/retrieval-policy", response_model=KnowledgeBaseOut)
async def update_retrieval_policy(
    kb_id: uuid.UUID,
    payload: RetrievalPolicyCreate,
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> KnowledgeBase:
    result = await session.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.tenant_id == context.tenant_id,
            KnowledgeBase.id == kb_id,
        )
    )
    kb = result.scalar_one_or_none()
    if kb is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    kb.retrieval_policy = payload.model_dump()
    await session.commit()
    await session.refresh(kb)
    return kb


@router.delete("/{kb_id}", status_code=204)
async def delete_knowledge_base(
    kb_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> None:
    result = await session.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.tenant_id == context.tenant_id,
            KnowledgeBase.id == kb_id,
        )
    )
    kb = result.scalar_one_or_none()
    if kb is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    await session.delete(kb)
    await session.commit()
