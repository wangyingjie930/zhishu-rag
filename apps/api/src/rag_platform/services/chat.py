import uuid
from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import ChatMessage, ChatSession
from rag_platform.domain.enums import MessageRole
from rag_platform.schemas.rag import ChatResponse, Citation
from rag_platform.services.llm.provider import MockLLMProvider
from rag_platform.services.retrieval.hybrid import HybridRetriever


class ChatService:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.llm = MockLLMProvider()

    async def chat(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        kb_id: uuid.UUID,
        message: str,
        session_id: uuid.UUID = None,
        top_k: int = 8,
    ) -> ChatResponse:
        chat_session = ChatSession(
            id=session_id or uuid.uuid4(),
            tenant_id=tenant_id,
            kb_id=kb_id,
            user_id=user_id,
            title=message[:80],
        )
        if session_id is None:
            session.add(chat_session)

        session.add(ChatMessage(session_id=chat_session.id, role=MessageRole.user.value, content=message))
        chunks, trace = await self.retriever.retrieve(session, tenant_id, kb_id, message, top_k=top_k)
        answer = await self.llm.answer(message, chunks)
        citations: List[Citation] = [
            Citation(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                score=chunk.score,
                content=chunk.content,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]
        assistant_message = ChatMessage(
            session_id=chat_session.id,
            role=MessageRole.assistant.value,
            content=answer,
            citations=[citation.model_dump(mode="json") for citation in citations],
            retrieval_trace=trace,
        )
        session.add(assistant_message)
        await session.commit()
        return ChatResponse(
            session_id=chat_session.id,
            answer=answer,
            citations=citations,
            retrieval_trace=trace,
        )

