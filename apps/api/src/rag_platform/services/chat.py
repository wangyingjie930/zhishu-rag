import uuid
from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import ChatMessage, ChatSession
from rag_platform.domain.enums import MessageRole
from rag_platform.schemas.rag import ChatResponse, Citation
from rag_platform.services.llm.provider import get_llm_provider
from rag_platform.services.observability import get_langfuse_observability
from rag_platform.services.retrieval.hybrid import HybridRetriever


class ChatService:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.llm = get_llm_provider()
        self.observability = get_langfuse_observability()

    async def chat(
        self,
        session: AsyncSession,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        kb_id: uuid.UUID,
        message: str,
        session_id: uuid.UUID = None,
        top_k: int = 8,
        hyde_enabled: bool = False,
        query_expansion_enabled: bool = False,
    ) -> ChatResponse:
        if session_id is None:
            chat_session = ChatSession(
                id=uuid.uuid4(),
                tenant_id=tenant_id,
                kb_id=kb_id,
                user_id=user_id,
                title=message[:80],
            )
            session.add(chat_session)
            await session.flush()
        else:
            result = await session.execute(
                select(ChatSession).where(
                    ChatSession.id == session_id,
                    ChatSession.tenant_id == tenant_id,
                    ChatSession.kb_id == kb_id,
                )
            )
            chat_session = result.scalar_one_or_none()
            if chat_session is None:
                chat_session = ChatSession(
                    id=session_id,
                    tenant_id=tenant_id,
                    kb_id=kb_id,
                    user_id=user_id,
                    title=message[:80],
                )
                session.add(chat_session)
                await session.flush()

        with self.observability.trace(
            "rag_chat",
            input_data={
                "message": message,
                "kb_id": str(kb_id),
                "top_k": top_k,
                "hyde_enabled": hyde_enabled,
                "query_expansion_enabled": query_expansion_enabled,
            },
            metadata={
                "tenant_id": str(tenant_id),
                "kb_id": str(kb_id),
                "top_k": top_k,
            },
            user_id=str(user_id),
            session_id=str(chat_session.id),
            tags=["rag", "chat"],
        ) as root_observation:
            session.add(
                ChatMessage(
                    session_id=chat_session.id,
                    role=MessageRole.user.value,
                    content=message,
                )
            )
            chunks, trace = await self.retriever.retrieve(
                session,
                tenant_id,
                kb_id,
                message,
                top_k=top_k,
                hyde_enabled=hyde_enabled,
                query_expansion_enabled=query_expansion_enabled,
            )
            with self.observability.observation(
                "llm_answer",
                as_type="generation",
                input_data={
                    "question": message,
                    "contexts": self._context_previews(chunks),
                },
                metadata={
                    "context_count": len(chunks),
                    "retrieval_run_id": trace.get("retrieval_run_id"),
                },
                model=getattr(self.llm, "model", self.llm.__class__.__name__),
            ) as llm_observation:
                answer = await self.llm.answer(message, chunks)
                self.observability.update_observation(
                    llm_observation,
                    output={"answer": answer},
                    metadata={"answer_length": len(answer)},
                )

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
            self.observability.update_observation(
                root_observation,
                output={
                    "answer": answer,
                    "citation_count": len(citations),
                    "retrieval": trace.get("diagnostics", {}),
                },
                metadata={
                    "retrieval_run_id": trace.get("retrieval_run_id"),
                    "retriever": trace.get("retriever"),
                },
            )
            return ChatResponse(
                session_id=chat_session.id,
                answer=answer,
                citations=citations,
                retrieval_trace=trace,
            )

    def _context_previews(self, chunks) -> List[dict]:
        return [
            {
                "chunk_id": str(chunk.chunk_id),
                "document_id": str(chunk.document_id),
                "filename": chunk.filename,
                "score": round(chunk.score, 6),
                "content_preview": " ".join(chunk.content.split())[:240],
            }
            for chunk in chunks
        ]
