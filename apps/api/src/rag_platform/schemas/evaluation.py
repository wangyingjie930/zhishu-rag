import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


MetricMap = Dict[str, Optional[float]]


class EvalCitation(BaseModel):
    chunk_id: str
    document_id: Optional[str] = None
    filename: str = ""
    score: float = 0
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvalCandidateOut(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    user_message_id: uuid.UUID
    assistant_message_id: uuid.UUID
    user_input: str
    response: str
    citations: List[EvalCitation] = Field(default_factory=list)
    retrieval_trace: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class EvalDatasetCreate(BaseModel):
    kb_id: uuid.UUID
    name: str = Field(min_length=1, max_length=120)
    description: str = ""


class EvalDatasetOut(BaseModel):
    id: uuid.UUID
    kb_id: uuid.UUID
    name: str
    description: str
    sample_count: int = 0
    created_at: datetime


class EvalSampleCreate(BaseModel):
    source_message_id: Optional[uuid.UUID] = None
    user_input: str = Field(min_length=1)
    reference: str = ""
    expected_context_ids: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    original_response: str = ""
    original_citations: List[EvalCitation] = Field(default_factory=list)
    original_retrieval_trace: Dict[str, Any] = Field(default_factory=dict)


class EvalSampleUpdate(BaseModel):
    reference: Optional[str] = None
    expected_context_ids: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class EvalSampleOut(BaseModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    source_message_id: Optional[uuid.UUID] = None
    user_input: str
    reference: str
    expected_context_ids: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    original_response: str
    original_citations: List[EvalCitation] = Field(default_factory=list)
    original_retrieval_trace: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class EvalRunCreate(BaseModel):
    dataset_id: uuid.UUID
    query_expansion_enabled: bool = False


class EvalRunResultOut(BaseModel):
    id: uuid.UUID
    run_id: uuid.UUID
    sample_id: uuid.UUID
    user_input: str
    response: str
    reference: str
    retrieved_contexts: List[str] = Field(default_factory=list)
    citations: List[EvalCitation] = Field(default_factory=list)
    retrieval_trace: Dict[str, Any] = Field(default_factory=dict)
    metrics: MetricMap = Field(default_factory=dict)
    reasons: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime


class EvalRunOut(BaseModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    kb_id: uuid.UUID
    status: str
    metrics: MetricMap = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    error_message: str = ""
    created_at: datetime
    completed_at: Optional[datetime] = None
    results: List[EvalRunResultOut] = Field(default_factory=list)
