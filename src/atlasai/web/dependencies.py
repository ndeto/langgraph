import os
import tempfile
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from atlasai.application.quotas import QuotaPolicy, QuotaService
from atlasai.application.documents import DocumentService
from atlasai.application.sessions import (
    SessionCookieSettings,
    SessionResolution,
    SessionService,
)
from atlasai.application.threads import ThreadService
from atlasai.application.usage import UsageService
from atlasai.infrastructure.postgres_repositories import (
    InMemoryRepositoryBundle,
    PostgresRepositoryBundle,
)
from atlasai.infrastructure.telemetry import TelemetryContext, ensure_trace_id
from atlasai.service.contracts import GraphRunner
from atlasai.store.hybrid_store import PostgresVectorService


@dataclass(frozen=True)
class DemoWebSettings:
    """Web demo settings."""

    session_cookie_name: str
    session_secret: str
    session_max_age_seconds: int
    secure_cookies: bool
    question_limit: int
    upload_limit: int
    ip_question_limit: int
    ip_upload_limit: int
    concurrent_ingestions_per_user: int
    concurrent_agent_runs_per_user: int
    staging_dir: str
    ip_hash_secret: str
    trusted_proxies: tuple[str, ...]


@dataclass(frozen=True)
class RequestContext:
    """Trusted request context."""

    session: SessionResolution
    request_key: str
    trace_id: str


def load_demo_web_settings() -> DemoWebSettings:
    secure_value = os.getenv("ATLAS_DEMO_SECURE_COOKIES", "").strip().lower()
    secure_cookies = secure_value in {"1", "true", "yes", "on"}

    return DemoWebSettings(
        session_cookie_name=os.getenv(
            "ATLAS_DEMO_SESSION_COOKIE", "atlas_demo_session"
        ),
        session_secret=os.getenv("ATLAS_DEMO_SESSION_SECRET", "atlas-demo-dev-secret"),
        session_max_age_seconds=int(
            os.getenv("ATLAS_DEMO_SESSION_TTL_SECONDS", str(24 * 60 * 60))
        ),
        secure_cookies=secure_cookies,
        question_limit=int(os.getenv("ATLAS_DEMO_QUESTION_LIMIT", "10")),
        upload_limit=int(os.getenv("ATLAS_DEMO_UPLOAD_LIMIT", "2")),
        ip_question_limit=int(os.getenv("ATLAS_DEMO_IP_QUESTION_LIMIT", "30")),
        ip_upload_limit=int(os.getenv("ATLAS_DEMO_IP_UPLOAD_LIMIT", "4")),
        concurrent_ingestions_per_user=int(
            os.getenv("ATLAS_DEMO_CONCURRENT_INGESTIONS", "1")
        ),
        concurrent_agent_runs_per_user=int(
            os.getenv("ATLAS_DEMO_CONCURRENT_AGENT_RUNS", "1")
        ),
        staging_dir=os.getenv(
            "ATLASAI_STAGING_DIR",
            str(
                os.path.join(
                    tempfile.gettempdir(),
                    "atlasai",
                    "staged_uploads",
                )
            ),
        ),
        ip_hash_secret=os.getenv("IP_HASH_SECRET", "atlas-demo-ip-dev-secret"),
        trusted_proxies=tuple(
            proxy.strip()
            for proxy in os.getenv("ATLAS_DEMO_TRUSTED_PROXIES", "").split(",")
            if proxy.strip()
        ),
    )


def get_graph_service(request: Request) -> GraphRunner:
    return request.app.state.graph_service


def get_demo_web_settings(request: Request) -> DemoWebSettings:
    return request.app.state.demo_web_settings


def get_repositories(
    request: Request,
) -> PostgresRepositoryBundle | InMemoryRepositoryBundle:
    return request.app.state.repositories


def get_vector_service(request: Request) -> PostgresVectorService:
    return request.app.state.vector_service


def get_session_service(
    settings: Annotated[DemoWebSettings, Depends(get_demo_web_settings)],
    repositories: Annotated[
        PostgresRepositoryBundle | InMemoryRepositoryBundle, Depends(get_repositories)
    ],
) -> SessionService:
    cookie_settings = SessionCookieSettings(
        name=settings.session_cookie_name,
        secret=settings.session_secret,
        max_age_seconds=settings.session_max_age_seconds,
        secure=settings.secure_cookies,
    )
    return SessionService(cookie_settings, repository=repositories.sessions)


def get_quota_policy(
    settings: Annotated[DemoWebSettings, Depends(get_demo_web_settings)],
) -> QuotaPolicy:
    return QuotaPolicy(
        user_question_limit=settings.question_limit,
        user_upload_limit=settings.upload_limit,
        ip_question_limit=settings.ip_question_limit,
        ip_upload_limit=settings.ip_upload_limit,
        concurrent_ingestions_per_user=settings.concurrent_ingestions_per_user,
        concurrent_agent_runs_per_user=settings.concurrent_agent_runs_per_user,
    )


def get_request_key(request: Request) -> str:
    request_key = getattr(request.state, "request_key_hash", None)
    if isinstance(request_key, str) and request_key:
        return request_key
    return "unknown-network"


def get_client_key(request: Request) -> str:
    header_value = request.headers.get("x-atlas-client-key")
    if header_value:
        return header_value
    return "anonymous-browser"


def get_quota_service(
    policy: Annotated[QuotaPolicy, Depends(get_quota_policy)],
    repositories: Annotated[
        PostgresRepositoryBundle | InMemoryRepositoryBundle, Depends(get_repositories)
    ],
    client_key: Annotated[str, Depends(get_client_key)],
    request_key: Annotated[str, Depends(get_request_key)],
) -> QuotaService:
    return QuotaService(
        policy=policy,
        repository=repositories.quotas,
        client_key=client_key,
        ip_hash=request_key,
    )


def get_usage_service(
    repositories: Annotated[
        PostgresRepositoryBundle | InMemoryRepositoryBundle, Depends(get_repositories)
    ],
) -> UsageService:
    return UsageService(repository=repositories.usage)


def get_thread_service(
    repositories: Annotated[
        PostgresRepositoryBundle | InMemoryRepositoryBundle, Depends(get_repositories)
    ],
) -> ThreadService:
    return ThreadService(repository=repositories.threads)


def get_document_service(
    repositories: Annotated[
        PostgresRepositoryBundle | InMemoryRepositoryBundle, Depends(get_repositories)
    ],
) -> DocumentService:
    return DocumentService(repository=repositories.documents)


def resolve_demo_session(
    request: Request,
    session_service: Annotated[SessionService, Depends(get_session_service)],
) -> SessionResolution:
    return session_service.resolve_session(
        request.cookies.get(session_service.settings.name)
    )


def get_trace_id(request: Request) -> str:
    return ensure_trace_id(request.headers.get("x-trace-id"))


def get_request_context(
    session: Annotated[SessionResolution, Depends(resolve_demo_session)],
    request_key: Annotated[str, Depends(get_request_key)],
    trace_id: Annotated[str, Depends(get_trace_id)],
) -> RequestContext:
    return RequestContext(
        session=session,
        request_key=request_key,
        trace_id=trace_id,
    )


def build_request_telemetry(
    *,
    operation: str,
    context: RequestContext,
    thread_id: str | None = None,
    document_id: str | None = None,
    result: str | None = None,
) -> TelemetryContext:
    return TelemetryContext(
        operation=operation,
        user_id=context.session.session.user_id,
        thread_id=thread_id,
        document_id=document_id,
        trace_id=context.trace_id,
        result=result,
    )
