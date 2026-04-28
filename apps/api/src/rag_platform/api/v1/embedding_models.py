from typing import Any, Dict, List

from fastapi import APIRouter

from rag_platform.services.ingestion.embeddings import list_embedding_model_options

router = APIRouter()


@router.get("", response_model=List[Dict[str, Any]])
async def list_embedding_models() -> List[Dict[str, Any]]:
    return list_embedding_model_options()
