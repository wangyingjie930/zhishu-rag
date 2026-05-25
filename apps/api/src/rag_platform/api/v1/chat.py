from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.session import get_session
from rag_platform.schemas.rag import ChatRequest, ChatResponse
from rag_platform.services.chat import ChatService
from rag_platform.services.security.context import RequestContext, get_request_context

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> ChatResponse:
    service = ChatService()
    return await service.chat(
        session=session,
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        kb_id=payload.kb_id,
        session_id=payload.session_id,
        message=payload.message,
        top_k=payload.top_k,
        hyde_enabled=payload.hyde_enabled,
        query_expansion_enabled=payload.query_expansion_enabled,
    )
