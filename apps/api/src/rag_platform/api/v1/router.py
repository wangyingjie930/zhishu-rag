from fastapi import APIRouter

from rag_platform.api.v1 import (
    chat,
    documents,
    embedding_models,
    evaluation,
    health,
    knowledge_bases,
)

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(knowledge_bases.router, prefix="/knowledge-bases", tags=["knowledge-bases"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(embedding_models.router, prefix="/embedding-models", tags=["embedding-models"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(evaluation.router, prefix="/eval", tags=["eval"])
