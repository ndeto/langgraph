import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, cast

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from atlasai.application.quotas import QuotaService
from atlasai.application.usage import UsageService
from atlasai.config.sys_config import SysConfig, bootstrap_config
from atlasai.infrastructure.postgres_repositories import (
    InMemoryRepositoryBundle,
    PostgresRepositoryBundle,
)
from atlasai.infrastructure.telemetry import usage_record_from_callback
from atlasai.service.contracts import GraphRunner, InvokePayload
from atlasai.store.hybrid_store import PostgresVectorService
from atlasai.web.dependencies import (
    RequestContext,
    get_graph_service,
    get_quota_service,
    get_request_context,
    get_usage_service,
    get_vector_service,
    load_demo_web_settings,
)
from atlasai.web.routers import documents_router, session_router, threads_router
from atlasai.web.streaming import extract_usage_payload, format_graph_chunk

LifecycleHook = Callable[[], Awaitable[None]]

sys_config: SysConfig = bootstrap_config()
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024
PDF_CONTENT_TYPES = {"application/pdf"}
StreamFormat = Literal["text", "ndjson"]


def get_stream_ingest_pdf():
    from atlasai.rag.rag_ingestion import stream_ingest_pdf

    return stream_ingest_pdf


def create_app(
    graph_service: GraphRunner | None = None,
    vector_service: PostgresVectorService | None = None,
    repositories: PostgresRepositoryBundle | InMemoryRepositoryBundle | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from atlasai.service.graph_service import GraphService

        warmed_vector_service = vector_service or PostgresVectorService()
        warmed_repositories = repositories or PostgresRepositoryBundle()
        vector_startup = getattr(warmed_vector_service, "startup", None)
        vector_shutdown = getattr(warmed_vector_service, "shutdown", None)
        repositories_startup = getattr(warmed_repositories, "startup", None)
        repositories_shutdown = getattr(warmed_repositories, "shutdown", None)

        if callable(vector_startup):
            await cast(LifecycleHook, vector_startup)()
        if callable(repositories_startup):
            await cast(LifecycleHook, repositories_startup)()

        warmed_graph_service = (
            graph_service
            if graph_service is not None
            else GraphService(
                config=sys_config,
                vector_service=warmed_vector_service,
            )
        )

        startup_graph = getattr(warmed_graph_service, "startup", None)
        shutdown_graph = getattr(warmed_graph_service, "shutdown", None)

        if callable(startup_graph):
            await cast(LifecycleHook, startup_graph)()

        app.state.graph_service = warmed_graph_service
        app.state.vector_service = warmed_vector_service
        app.state.repositories = warmed_repositories
        app.state.demo_web_settings = load_demo_web_settings()
        yield

        if callable(shutdown_graph):
            await cast(LifecycleHook, shutdown_graph)()
        if callable(vector_shutdown):
            await cast(LifecycleHook, vector_shutdown)()
        if callable(repositories_shutdown):
            await cast(LifecycleHook, repositories_shutdown)()

    app = FastAPI(title="Atlas Agent Web API", lifespan=lifespan)
    app.state.demo_web_settings = load_demo_web_settings()
    if graph_service is not None:
        app.state.graph_service = graph_service
    app.state.repositories = repositories
    app.state.vector_service = vector_service
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(documents_router)
    app.include_router(session_router)
    app.include_router(threads_router)

    @app.get("/")
    def _():
        return FileResponse(INDEX_FILE)

    @app.get("/health")
    def health():
        return {"status": "ok"}

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
                content={"detail": admission.reason},
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
        }

        async def response_stream():
            usage_recorded = False
            try:
                async for chunk in graph_service.stream(payload):
                    if isinstance(chunk, dict) and chunk.get("type") == "final":
                        usage_service.record_usage(
                            user_id=request_context.session.session.user_id,
                            record=usage_record_from_callback(
                                operation="agent",
                                payload=extract_usage_payload(chunk.get("data")),
                                provider=sys_config["model"]["provider"],
                                model=sys_config["model"]["model"],
                                trace_id=request_context.trace_id,
                                run_id=request_context.trace_id,
                            ),
                        )
                        usage_recorded = True

                    formatted_chunk = format_graph_chunk(chunk, stream_format)
                    if formatted_chunk is not None:
                        yield formatted_chunk
            finally:
                if not usage_recorded:
                    usage_service.record_usage(
                        user_id=request_context.session.session.user_id,
                        record=usage_record_from_callback(
                            operation="agent",
                            payload=None,
                            provider=sys_config["model"]["provider"],
                            model=sys_config["model"]["model"],
                            trace_id=request_context.trace_id,
                            run_id=request_context.trace_id,
                        ),
                    )
                quota_service.release_agent_run(
                    user_id=request_context.session.session.user_id
                )

        return StreamingResponse(
            response_stream(),
            media_type=(
                "application/x-ndjson" if stream_format == "ndjson" else "text/plain"
            ),
            headers=stream_headers,
        )

    @app.post("/ingest/pdf")
    async def ingest_pdf(
        request_context: Annotated[RequestContext, Depends(get_request_context)],
        quota_service: Annotated[QuotaService, Depends(get_quota_service)],
        usage_service: Annotated[UsageService, Depends(get_usage_service)],
        vector_service: Annotated[PostgresVectorService, Depends(get_vector_service)],
        file: UploadFile = File(...),
        stream_format: Literal["ndjson"] = "ndjson",
    ):
        del stream_format

        filename = file.filename or "upload.pdf"
        if file.content_type not in PDF_CONTENT_TYPES and not filename.lower().endswith(
            ".pdf"
        ):
            return JSONResponse(
                status_code=400,
                content={"detail": "Only PDF uploads are supported."},
            )

        file_bytes = await file.read()
        if not file_bytes:
            return JSONResponse(
                status_code=400,
                content={"detail": "The uploaded PDF is empty."},
            )

        if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": "PDF exceeds the 25 MB upload limit.",
                    "max_size_bytes": MAX_UPLOAD_SIZE_BYTES,
                },
            )

        admission = quota_service.claim_upload(
            user_id=request_context.session.session.user_id
        )
        if not admission.allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": admission.reason},
                headers={"X-Trace-Id": request_context.trace_id},
            )

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
                prefix="atlasai-upload-",
            ) as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name

            async def response_stream():
                try:
                    stream_ingest_pdf = get_stream_ingest_pdf()
                    async for event in stream_ingest_pdf(
                        temp_path,
                        file_name=filename,
                        user_id=request_context.session.session.user_id,
                        storage=vector_service.get_store("raggidy_docs"),
                        store_name="raggidy_docs",
                    ):
                        yield json.dumps(event) + "\n"
                except Exception as exc:
                    yield json.dumps({"type": "error", "text": str(exc)}) + "\n"
                finally:
                    usage_service.record_usage(
                        user_id=request_context.session.session.user_id,
                        record=usage_record_from_callback(
                            operation="ingestion",
                            payload=None,
                            provider=sys_config["model"]["provider"],
                            model=sys_config["model"]["model"],
                            trace_id=request_context.trace_id,
                            run_id=request_context.trace_id,
                        ),
                    )
                    quota_service.release_ingestion(
                        user_id=request_context.session.session.user_id
                    )
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)

            return StreamingResponse(
                response_stream(),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Trace-Id": request_context.trace_id,
                },
            )
        except HTTPException:
            quota_service.release_ingestion(
                user_id=request_context.session.session.user_id
            )
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    return app


app = create_app()
