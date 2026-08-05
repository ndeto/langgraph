import asyncio
import json
import re
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse

from atlasai.application.documents import DocumentService
from atlasai.application.quotas import QuotaService
from atlasai.web.dependencies import (
    RequestContext,
    get_demo_web_settings,
    get_document_service,
    get_quota_service,
    get_request_context,
)
from atlasai.web.schemas.document import DocumentAcceptedResponse, DocumentResponse

router = APIRouter(prefix="/api/v1", tags=["documents"])

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
PDF_CONTENT_TYPES = {"application/pdf"}


def _sanitize_staged_filename(filename: str) -> str:
    base_name = Path(filename).name or "upload.pdf"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name).strip("._")
    if not sanitized:
        return "upload.pdf"
    if not sanitized.lower().endswith(".pdf"):
        sanitized = f"{sanitized}.pdf"
    return sanitized


@router.post("/documents", response_model=DocumentAcceptedResponse, status_code=202)
async def upload_document(
    request_context: Annotated[RequestContext, Depends(get_request_context)],
    document_service: Annotated[DocumentService, Depends(get_document_service)],
    quota_service: Annotated[QuotaService, Depends(get_quota_service)],
    settings=Depends(get_demo_web_settings),
    file: UploadFile = File(...),
):
    filename = file.filename or "upload.pdf"
    if file.content_type not in PDF_CONTENT_TYPES and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    admission = quota_service.claim_upload(user_id=request_context.session.session.user_id)
    if not admission.allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": admission.message, "code": admission.code},
        )

    staging_dir = Path(settings.staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    safe_filename = _sanitize_staged_filename(filename)
    temp_path = staging_dir / (
        f"{request_context.session.session.user_id}-{uuid4()}-{safe_filename}"
    )

    try:
        size_bytes = 0
        header = b""
        with temp_path.open("wb") as staged_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                if len(header) < 5:
                    header += chunk
                size_bytes += len(chunk)
                if size_bytes > MAX_UPLOAD_SIZE_BYTES:
                    temp_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=400,
                        detail="PDF exceeds the 10 MB upload limit.",
                    )
                staged_file.write(chunk)

        await file.close()

        if size_bytes == 0:
            temp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
        if not header.startswith(b"%PDF"):
            temp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

        document, job = document_service.create_document_job(
            user_id=request_context.session.session.user_id,
            filename=filename,
            size_bytes=size_bytes,
            ttl_seconds=settings.session_max_age_seconds,
            source_path=str(temp_path),
        )
    except Exception:
        temp_path.unlink(missing_ok=True)
        quota_service.release_ingestion(
            user_id=request_context.session.session.user_id
        )
        raise

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
