from typing import Annotated

from fastapi import APIRouter, Depends, Response

from atlasai.application.documents import DocumentService
from atlasai.application.quotas import QuotaService
from atlasai.application.sessions import SessionResolution, SessionService
from atlasai.application.usage import UsageService
from atlasai.web.cookies import set_demo_session_cookie
from atlasai.web.dependencies import (
    get_document_service,
    get_quota_service,
    get_session_service,
    get_usage_service,
    resolve_demo_session,
)
from atlasai.web.schemas.session import (
    QuotaBucketResponse,
    SessionQuotaResponse,
    SessionResponse,
    TokenUsageResponse,
)

router = APIRouter(prefix="/api/v1", tags=["session"])


@router.get("/session", response_model=SessionResponse)
def get_session(
    response: Response,
    session_resolution: Annotated[SessionResolution, Depends(resolve_demo_session)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    document_service: Annotated[DocumentService, Depends(get_document_service)],
    quota_service: Annotated[QuotaService, Depends(get_quota_service)],
    usage_service: Annotated[UsageService, Depends(get_usage_service)],
) -> SessionResponse:
    if session_resolution.set_cookie:
        set_demo_session_cookie(
            response,
            session_service,
            cookie_value=session_resolution.cookie_value,
        )

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
    return SessionResponse(
        user_id=session_resolution.session.user_id,
        expires_at=session_resolution.session.expires_at,
        active_document=(
            active_document.document_id
            if active_document is not None
            else session_resolution.session.active_document
        ),
        active_thread=session_resolution.session.active_thread,
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
