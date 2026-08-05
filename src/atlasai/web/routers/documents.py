import asyncio
import json
import tempfile
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from atlasai.application.documents import DocumentService
from atlasai.application.quotas import QuotaService
from atlasai.application.usage import UsageService
from atlasai.infrastructure.telemetry import usage_record_from_callback
from atlasai.web.dependencies import (
    RequestContext,
    get_demo_web_settings,
    get_document_service,
    get_quota_service,
    get_request_context,
    get_usage_service,
    get_vector_service,
)
from atlasai.store.hybrid_store import PostgresVectorService
from atlasai.web.schemas.document import DocumentAcceptedResponse, DocumentResponse

router = APIRouter(prefix="/api/v1", tags=["documents"])

MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024
PDF_CONTENT_TYPES = {"application/pdf"}


def get_stream_ingest_pdf():
    from atlasai.rag.rag_ingestion import stream_ingest_pdf

    return stream_ingest_pdf


@router.post("/documents", response_model=DocumentAcceptedResponse, status_code=202)
async def upload_document(
    request_context: Annotated[RequestContext, Depends(get_request_context)],
    document_service: Annotated[DocumentService, Depends(get_document_service)],
    quota_service: Annotated[QuotaService, Depends(get_quota_service)],
    usage_service: Annotated[UsageService, Depends(get_usage_service)],
    vector_service: Annotated[PostgresVectorService, Depends(get_vector_service)],
    settings=Depends(get_demo_web_settings),
    file: UploadFile = File(...),
):
    filename = file.filename or "upload.pdf"
    if file.content_type not in PDF_CONTENT_TYPES and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="PDF exceeds the 25 MB upload limit.")

    admission = quota_service.claim_upload(user_id=request_context.session.session.user_id)
    if not admission.allowed:
        raise HTTPException(status_code=429, detail=admission.reason)

    document, job = document_service.create_document_job(
        user_id=request_context.session.session.user_id,
        filename=filename,
        size_bytes=len(file_bytes),
        ttl_seconds=settings.session_max_age_seconds,
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
        prefix="atlasai-document-",
    ) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    async def run_job() -> None:
        try:
            await document_service.run_job(
                job_id=job.job_id,
                file_path=temp_path,
                file_name=filename,
                user_id=request_context.session.session.user_id,
                stream_ingest_pdf=get_stream_ingest_pdf(),
                storage=vector_service.get_store("raggidy_docs"),
            )
        finally:
            usage_service.record_usage(
                user_id=request_context.session.session.user_id,
                record=usage_record_from_callback(
                    operation="ingestion",
                    payload=None,
                    trace_id=request_context.trace_id,
                    run_id=job.job_id,
                ),
            )
            quota_service.release_ingestion(
                user_id=request_context.session.session.user_id
            )

    asyncio.create_task(run_job())

    return DocumentAcceptedResponse(
        document_id=document.document_id,
        job_id=job.job_id,
        status=job.state,
    )


@router.get("/documents/active", response_model=DocumentResponse | None)
def get_active_document(
    request_context: Annotated[RequestContext, Depends(get_request_context)],
    document_service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentResponse | None:
    document = document_service.get_active_document(
        user_id=request_context.session.session.user_id
    )
    if document is None:
        return None

    return DocumentResponse(
        document_id=document.document_id,
        filename=document.filename,
        size_bytes=document.size_bytes,
        status=document.status,
        created_at=document.created_at,
        expires_at=document.expires_at,
    )


@router.get("/ingestions/{job_id}/events")
async def stream_ingestion_events(
    job_id: str,
    request_context: Annotated[RequestContext, Depends(get_request_context)],
    document_service: Annotated[DocumentService, Depends(get_document_service)],
):
    job = document_service.get_owned_job(
        user_id=request_context.session.session.user_id,
        job_id=job_id,
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")

    async def response_stream():
        seen_event_id = 0
        while True:
            current_job = document_service.get_owned_job(
                user_id=request_context.session.session.user_id,
                job_id=job_id,
            )
            if current_job is None:
                break

            pending_events = [
                event for event in current_job.events if event.event_id > seen_event_id
            ]
            for event in pending_events:
                seen_event_id = event.event_id
                yield f"id: {event.event_id}\n"
                yield f"data: {json.dumps(event.payload)}\n\n"

            if current_job.state in {"ready", "failed"}:
                break

            await asyncio.sleep(0.05)

    return StreamingResponse(
        response_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
