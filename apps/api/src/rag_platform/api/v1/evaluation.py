import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import EvalDataset, EvalRun, EvalSample
from rag_platform.db.session import get_session
from rag_platform.schemas.evaluation import (
    EvalCandidateOut,
    EvalDatasetCreate,
    EvalDatasetOut,
    EvalRunCreate,
    EvalRunOut,
    EvalRunResultOut,
    EvalSampleCreate,
    EvalSampleOut,
    EvalSampleUpdate,
)
from rag_platform.services.evaluation import EvaluationService
from rag_platform.services.security.context import RequestContext, get_request_context

router = APIRouter()


def _service() -> EvaluationService:
    return EvaluationService()


def _http_error(exc: ValueError) -> HTTPException:
    message = str(exc)
    status_code = 404 if "不存在" in message else 400
    return HTTPException(status_code=status_code, detail=message)


def _dataset_out(dataset: EvalDataset, sample_count: int = 0) -> EvalDatasetOut:
    return EvalDatasetOut(
        id=dataset.id,
        kb_id=dataset.kb_id,
        name=dataset.name,
        description=dataset.description,
        sample_count=sample_count,
        created_at=dataset.created_at,
    )


def _sample_out(sample: EvalSample) -> EvalSampleOut:
    return EvalSampleOut(
        id=sample.id,
        dataset_id=sample.dataset_id,
        source_message_id=sample.source_message_id,
        user_input=sample.user_input,
        reference=sample.reference,
        expected_context_ids=[str(item) for item in sample.expected_context_ids],
        tags=[str(item) for item in sample.tags],
        original_response=sample.original_response,
        original_citations=sample.original_citations,
        original_retrieval_trace=sample.original_retrieval_trace,
        created_at=sample.created_at,
    )


def _run_out(run: EvalRun, results: List[EvalRunResultOut] = None) -> EvalRunOut:
    return EvalRunOut(
        id=run.id,
        dataset_id=run.dataset_id,
        kb_id=run.kb_id,
        status=run.status,
        metrics=run.metrics,
        config=run.config,
        error_message=run.error_message,
        created_at=run.created_at,
        completed_at=run.completed_at,
        results=results or [],
    )


@router.get("/candidates", response_model=List[EvalCandidateOut])
async def list_eval_candidates(
    kb_id: uuid.UUID,
    limit: int = Query(default=80, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> List[EvalCandidateOut]:
    try:
        return await _service().list_candidates(session, context.tenant_id, kb_id, limit=limit)
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.get("/datasets", response_model=List[EvalDatasetOut])
async def list_eval_datasets(
    kb_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> List[EvalDatasetOut]:
    rows = await _service().list_datasets(session, context.tenant_id, kb_id)
    return [EvalDatasetOut(**row) for row in rows]


@router.post("/datasets", response_model=EvalDatasetOut)
async def create_eval_dataset(
    payload: EvalDatasetCreate,
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> EvalDatasetOut:
    try:
        dataset = await _service().create_dataset(
            session,
            context.tenant_id,
            payload.kb_id,
            payload.name,
            payload.description,
        )
        return _dataset_out(dataset)
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.delete("/datasets/{dataset_id}", status_code=204)
async def delete_eval_dataset(
    dataset_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> None:
    try:
        await _service().delete_dataset(session, context.tenant_id, dataset_id)
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.get("/datasets/{dataset_id}/samples", response_model=List[EvalSampleOut])
async def list_eval_samples(
    dataset_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> List[EvalSampleOut]:
    try:
        samples = await _service().list_samples(session, context.tenant_id, dataset_id)
        return [_sample_out(sample) for sample in samples]
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.post("/datasets/{dataset_id}/samples", response_model=EvalSampleOut)
async def add_eval_sample(
    dataset_id: uuid.UUID,
    payload: EvalSampleCreate,
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> EvalSampleOut:
    try:
        sample = await _service().add_sample(
            session,
            context.tenant_id,
            dataset_id,
            payload.model_dump(mode="json"),
        )
        return _sample_out(sample)
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.put("/datasets/{dataset_id}/samples/{sample_id}", response_model=EvalSampleOut)
async def update_eval_sample(
    dataset_id: uuid.UUID,
    sample_id: uuid.UUID,
    payload: EvalSampleUpdate,
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> EvalSampleOut:
    try:
        sample = await _service().update_sample(
            session,
            context.tenant_id,
            dataset_id,
            sample_id,
            payload.model_dump(exclude_unset=True),
        )
        return _sample_out(sample)
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.post("/runs", response_model=EvalRunOut)
async def create_eval_run(
    payload: EvalRunCreate,
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> EvalRunOut:
    try:
        run = await _service().create_run(
            session,
            context.tenant_id,
            payload.dataset_id,
            query_expansion_enabled=payload.query_expansion_enabled,
        )
        return _run_out(run)
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.get("/runs/{run_id}", response_model=EvalRunOut)
async def get_eval_run(
    run_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    context: RequestContext = Depends(get_request_context),
) -> EvalRunOut:
    try:
        payload = await _service().get_run(session, context.tenant_id, run_id)
        results = [
            EvalRunResultOut(
                id=result.id,
                run_id=result.run_id,
                sample_id=result.sample_id,
                user_input=result.user_input,
                response=result.response,
                reference=result.reference,
                retrieved_contexts=result.retrieved_contexts,
                citations=result.citations,
                retrieval_trace=result.retrieval_trace,
                metrics=result.metrics,
                reasons=result.reasons,
                created_at=result.created_at,
            )
            for result in payload["results"]
        ]
        return _run_out(payload["run"], results)
    except ValueError as exc:
        raise _http_error(exc) from exc
