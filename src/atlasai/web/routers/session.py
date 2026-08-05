from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from atlasai.application.documents import DocumentService
from atlasai.application.quotas import QuotaService
from atlasai.application.sessions import SessionResolution, SessionService
from atlasai.application.threads import ThreadService
from atlasai.application.usage import UsageService
from atlasai.infrastructure.postgres_repositories import (
    InMemoryRepositoryBundle,
    PostgresRepositoryBundle,
)
from atlasai.web.cookies import set_demo_session_cookie
from atlasai.web.dependencies import (
    get_client_key,
    get_document_service,
    get_request_key,
    get_quota_service,
    get_repositories,
    get_session_service,
    get_thread_service,
    get_usage_service,
    resolve_demo_session,
)
from atlasai.web.schemas.session import (
    QuotaBucketResponse,
    SessionDocumentResponse,
    SessionQuotaResponse,
    SessionResponse,
    TokenUsageResponse,
)

router = APIRouter(prefix="/api/v1", tags=["session"])


def _build_session_response(
    *,
    session_resolution: SessionResolution,
    document_service: DocumentService,
    quota_service: QuotaService,
    usage_service: UsageService,
) -> SessionResponse:
    token_usage = usage_service.get_token_summary(
        user_id=session_resolution.session.user_id
    )
    quota = quota_service.get_session_quota_summary(
        user_id=session_resolution.session.user_id,
        token_usage=token_usage,
    )
    active_document = document_service.get_active_document(
        user_id=session_resolution.session.user_id
    )
    uploaded_documents = document_service.list_documents(
        user_id=session_resolution.session.user_id
    )
    return SessionResponse(
        user_id=session_resolution.session.user_id,
        expires_at=session_resolution.session.expires_at,
        active_document=(
            active_document.document_id
            if active_document is not None
            else session_resolution.session.active_document
        ),
        active_thread=session_resolution.session.active_thread,
        uploaded_documents=[
            SessionDocumentResponse(
                id=document.document_id,
                name=document.filename,
                status=document.status,
            )
            for document in uploaded_documents
        ],
        quota=SessionQuotaResponse(
            questions=QuotaBucketResponse(
                limit=quota.questions.limit,
                used=quota.questions.used,
                remaining=quota.questions.remaining,
            ),
            uploads=QuotaBucketResponse(
                limit=quota.uploads.limit,
                used=quota.uploads.used,
                remaining=quota.uploads.remaining,
            ),
            tokens=TokenUsageResponse(
                input=quota.tokens.input,
                output=quota.tokens.output,
                total=quota.tokens.total,
            ),
        ),
    )


@router.get("/session", response_model=SessionResponse)
def get_session(
    response: Response,
    session_resolution: Annotated[SessionResolution, Depends(resolve_demo_session)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    document_service: Annotated[DocumentService, Depends(get_document_service)],
    quota_service: Annotated[QuotaService, Depends(get_quota_service)],
    thread_service: Annotated[ThreadService, Depends(get_thread_service)],
    usage_service: Annotated[UsageService, Depends(get_usage_service)],
) -> SessionResponse:
    active_thread_id = session_resolution.session.active_thread
    if active_thread_id:
        thread = thread_service.get_owned_thread(
            user_id=session_resolution.session.user_id,
            thread_id=active_thread_id,
        )
        if thread is None:
            updated_session = session_service.with_active_thread(
                session_resolution.session,
                None,
            )
            session_resolution = SessionResolution(
                session=updated_session,
                set_cookie=True,
                cookie_value=session_service.serialize_session(updated_session),
            )

    if session_resolution.set_cookie:
        set_demo_session_cookie(
            response,
            session_service,
            cookie_value=session_resolution.cookie_value,
        )

    return _build_session_response(
        session_resolution=session_resolution,
        document_service=document_service,
        quota_service=quota_service,
        usage_service=usage_service,
    )


@router.post("/session/rotate", response_model=SessionResponse)
def rotate_session(
    request: Request,
    response: Response,
    session_resolution: Annotated[SessionResolution, Depends(resolve_demo_session)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    repositories: Annotated[
        PostgresRepositoryBundle | InMemoryRepositoryBundle, Depends(get_repositories)
    ],
    document_service: Annotated[DocumentService, Depends(get_document_service)],
    quota_service: Annotated[QuotaService, Depends(get_quota_service)],
    usage_service: Annotated[UsageService, Depends(get_usage_service)],
    client_key: Annotated[str, Depends(get_client_key)],
    request_key: Annotated[str, Depends(get_request_key)],
) -> SessionResponse:
    cookie_value = request.cookies.get(session_service.settings.name)
    if cookie_value is None:
        raise HTTPException(status_code=400, detail="Session not found.")
    if session_service.resolve_existing_session(cookie_value) is None:
        raise HTTPException(status_code=400, detail="Session not found.")

    repositories.enqueue_cleanup_job(user_id=session_resolution.session.user_id)
    repositories.expire_user_data(user_id=session_resolution.session.user_id)
    quota_service.repository.reset_client_bucket(client_key=client_key)
    quota_service.repository.reset_ip_bucket(ip_hash=request_key)
    next_session = session_service.rotate_session(session_resolution.session)
    set_demo_session_cookie(
        response,
        session_service,
        cookie_value=next_session.cookie_value,
    )
    return _build_session_response(
        session_resolution=next_session,
        document_service=document_service,
        quota_service=quota_service,
        usage_service=usage_service,
    )
