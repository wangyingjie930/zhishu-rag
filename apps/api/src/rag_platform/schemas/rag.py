import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class KnowledgeBaseOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    visibility: str
    retrieval_policy: Dict[str, Any]
    ingestion_policy: Dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}


class RetrievalPolicyCreate(BaseModel):
    top_k: int = Field(default=8, ge=1, le=30)
    vector_weight: float = Field(default=0.65, ge=0, le=1)
    keyword_weight: float = Field(default=0.35, ge=0, le=1)
    reranker: str = "none"
    score_threshold: float = Field(default=0, ge=0, le=1)


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    visibility: str = "private"
    retrieval_policy: Optional[RetrievalPolicyCreate] = None


class DocumentOut(BaseModel):
    id: uuid.UUID
    kb_id: uuid.UUID
    filename: str
    mime_type: str
    status: str
    parser: str
    metadata_: Dict[str, Any] = Field(default_factory=dict, serialization_alias="metadata")
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChunkPreviewItem(BaseModel):
    index: int
    content: str
    token_count: int
    character_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunkPreviewOut(BaseModel):
    filename: str
    mime_type: str
    parser: str
    chunking_policy: Dict[str, Any]
    clean_text_length: int
    total_chunks: int
    chunks: List[ChunkPreviewItem]
    truncated: bool = False


class DocumentReindexRequest(BaseModel):
    parser: str = "auto"
    embedding_model: str = ""
    chunking_policy: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    score: float
    content: str
    metadata: Dict[str, Any] = {}


class ChatRequest(BaseModel):
    kb_id: uuid.UUID
    message: str = Field(min_length=1)
    session_id: Optional[uuid.UUID] = None
    top_k: int = Field(default=8, ge=1, le=30)
    hyde_enabled: bool = False
    query_expansion_enabled: bool = False


class ChatResponse(BaseModel):
    session_id: uuid.UUID
    answer: str
    citations: List[Citation]
    retrieval_trace: Dict[str, Any]
