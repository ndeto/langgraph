import inspect
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, cast

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from atlasai.application.quotas import QuotaService
from atlasai.application.usage import UsageService
from atlasai.config.sys_config import SysConfig, bootstrap_config
from atlasai.infrastructure.postgres_repositories import (
    InMemoryRepositoryBundle,
    PostgresRepositoryBundle,
)
from atlasai.infrastructure.telemetry import (
    log_usage_resolution,
    usage_record_from_callback,
)
from atlasai.lib.logging import configure_logging
from atlasai.service.contracts import GraphRunner, InvokePayload
from atlasai.store.hybrid_store import PostgresVectorService
from atlasai.web.dependencies import (
    RequestContext,
    get_graph_service,
    get_quota_service,
    get_request_context,
    get_usage_service,
    load_demo_web_settings,
)
from atlasai.web.operating_hours import (
    is_new_work_route,
    is_within_operating_hours,
    operating_hours_enforced,
)
from atlasai.web.request_identity import build_request_key_hash
from atlasai.web.routers import (
    assets_router,
    documents_router,
    session_router,
    threads_router,
)
from atlasai.web.streaming import (
    extract_usage_payload,
    format_graph_chunk,
    stream_with_keepalive,
)
from atlasai.web.streaming import build_usage_event, encode_ndjson_event

LifecycleHook = Callable[[], Awaitable[None] | None]

configure_logging()

sys_config: SysConfig = bootstrap_config()
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"
StreamFormat = Literal["text", "ndjson"]


async def _run_lifecycle_hook(hook: object) -> None:
    if not callable(hook):
        return
    result = cast(LifecycleHook, hook)()
    if inspect.isawaitable(result):
        await result


def create_app(
    graph_service: GraphRunner | None = None,
    vector_service: PostgresVectorService | None = None,
    repositories: PostgresRepositoryBundle | InMemoryRepositoryBundle | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        warmed_vector_service = vector_service or PostgresVectorService()
        warmed_repositories = repositories or PostgresRepositoryBundle()
        vector_startup = getattr(warmed_vector_service, "startup", None)
        vector_shutdown = getattr(warmed_vector_service, "shutdown", None)
        repositories_startup = getattr(warmed_repositories, "startup", None)
        repositories_shutdown = getattr(warmed_repositories, "shutdown", None)

        await _run_lifecycle_hook(vector_startup)
        await _run_lifecycle_hook(repositories_startup)

        if graph_service is not None:
            warmed_graph_service = graph_service
        else:
            from atlasai.service.graph_service import GraphService

            warmed_graph_service = GraphService(
                config=sys_config,
                vector_service=warmed_vector_service,
            )

        startup_graph = getattr(warmed_graph_service, "startup", None)
        shutdown_graph = getattr(warmed_graph_service, "shutdown", None)

        await _run_lifecycle_hook(startup_graph)

        app.state.graph_service = warmed_graph_service
        app.state.vector_service = warmed_vector_service
        app.state.repositories = warmed_repositories
        app.state.demo_web_settings = load_demo_web_settings()
        app.state.lifecycle_ready = True
        try:
            yield
        finally:
            app.state.lifecycle_ready = False

        await _run_lifecycle_hook(shutdown_graph)
        await _run_lifecycle_hook(vector_shutdown)
        await _run_lifecycle_hook(repositories_shutdown)

    app = FastAPI(title="Atlas Agent Web API", lifespan=lifespan)
    app.state.lifecycle_ready = False
    app.state.demo_web_settings = load_demo_web_settings()
    if graph_service is not None:
        app.state.graph_service = graph_service
    app.state.repositories = repositories
    app.state.vector_service = vector_service
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(assets_router)
    app.include_router(documents_router)
    app.include_router(session_router)
    app.include_router(threads_router)

    @app.middleware("http")
    async def attach_request_identity(request, call_next):
        request.state.request_key_hash = build_request_key_hash(
            request,
            app.state.demo_web_settings,
        )
        if (
            operating_hours_enforced()
            and is_new_work_route(request.method, request.url.path)
            and not is_within_operating_hours()
        ):
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "The demo accepts new work daily from 08:00 to 22:00 EAT.",
                    "code": "demo_closed",
                },
            )
        return await call_next(request)

    @app.get("/")
    def _():
        return FileResponse(INDEX_FILE)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        repositories_ready = _repositories_are_ready(
            getattr(app.state, "repositories", None)
        )
        vector_ready = await _vector_service_is_ready(
            getattr(app.state, "vector_service", None)
        )
        graph_ready = _graph_service_is_warmed(
            getattr(app.state, "graph_service", None)
        )
        checks = {
            "lifecycle": bool(getattr(app.state, "lifecycle_ready", False)),
            "repositories": repositories_ready,
            "vector_store": vector_ready,
            "graph": graph_ready,
        }
        if not all(checks.values()):
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "checks": checks},
            )
        return {"status": "ready", "checks": checks}

    @app.post("/invoke")
    async def _(
        input: InvokePayload,
        graph_service: Annotated[GraphRunner, Depends(get_graph_service)],
        request_context: Annotated[RequestContext, Depends(get_request_context)],
        quota_service: Annotated[QuotaService, Depends(get_quota_service)],
        usage_service: Annotated[UsageService, Depends(get_usage_service)],
        stream_format: StreamFormat = "text",
    ):
        admission = quota_service.claim_question(
            user_id=request_context.session.session.user_id
        )
        if not admission.allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": admission.message, "code": admission.code},
                headers={"X-Trace-Id": request_context.trace_id},
            )

        stream_headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Trace-Id": request_context.trace_id,
        }
        payload: InvokePayload = {
            **input,
            "user_id": request_context.session.session.user_id,
            "trace_id": request_context.trace_id,
        }

        async def response_stream():
            usage_recorded = False
            try:
                async for chunk in graph_service.stream(payload):
                    if isinstance(chunk, dict) and chunk.get("type") == "final":
                        usage_payload = extract_usage_payload(chunk.get("data"))
                        record = usage_record_from_callback(
                            operation="agent",
                            payload=usage_payload,
                            provider=sys_config["model"]["provider"],
                            model=sys_config["model"]["model"],
                            trace_id=request_context.trace_id,
                        )
                        log_usage_resolution(
                            source="invoke.final",
                            payload=usage_payload,
                            record=record,
                        )
                        usage_service.record_usage(
                            user_id=request_context.session.session.user_id,
                            record=record,
                        )
                        if stream_format == "ndjson":
                            yield encode_ndjson_event(build_usage_event(record))
                        usage_recorded = True

                    formatted_chunk = format_graph_chunk(chunk, stream_format)
                    if formatted_chunk is not None:
                        yield formatted_chunk
            finally:
                if not usage_recorded:
                    record = usage_record_from_callback(
                        operation="agent",
                        payload=None,
                        provider=sys_config["model"]["provider"],
                        model=sys_config["model"]["model"],
                        trace_id=request_context.trace_id,
                        run_id=request_context.trace_id,
                    )
                    log_usage_resolution(
                        source="invoke.missing-final",
                        payload=None,
                        record=record,
                    )
                    usage_service.record_usage(
                        user_id=request_context.session.session.user_id,
                        record=record,
                    )
                quota_service.release_agent_run(
                    user_id=request_context.session.session.user_id
                )

        stream = response_stream()
        response_body = (
            stream_with_keepalive(stream) if stream_format == "ndjson" else stream
        )
        return StreamingResponse(
            response_body,
            media_type=(
                "application/x-ndjson" if stream_format == "ndjson" else "text/plain"
            ),
            headers=stream_headers,
        )

    return app


def _repositories_are_ready(repositories: object) -> bool:
    if isinstance(repositories, PostgresRepositoryBundle):
        if repositories.engine is None or not all(
            repository is not None
            for repository in (
                repositories.sessions,
                repositories.documents,
                repositories.quotas,
                repositories.threads,
                repositories.usage,
                repositories.assets,
            )
        ):
            return False
        try:
            with repositories.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception:
            return False
        return True
    if isinstance(repositories, InMemoryRepositoryBundle):
        return True
    return False


async def _vector_service_is_ready(vector_service: object) -> bool:
    if isinstance(vector_service, PostgresVectorService):
        if vector_service.engine is None or vector_service.pg_engine is None:
            return False
        try:
            async with vector_service.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception:
            return False
    if vector_service is None:
        return False
    try:
        getattr(vector_service, "get_store")("raggidy_docs")
    except (AttributeError, RuntimeError, KeyError):
        return False
    return True


def _graph_service_is_warmed(graph_service: object) -> bool:
    if graph_service is None:
        return False
    graph = getattr(graph_service, "graph", graph_service)
    return graph is not None


app = create_app()
