import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, cast

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from atlasai.config.sys_config import SysConfig, bootstrap_config
from atlasai.rag.rag_ingestion import stream_ingest_pdf
from atlasai.service.graph_service import GraphRunner, GraphService, InvokePayload

StreamFormat = Literal["text", "ndjson"]
LifecycleHook = Callable[[], Awaitable[None]]

sys_config: SysConfig = bootstrap_config()
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024
PDF_CONTENT_TYPES = {"application/pdf"}


def format_graph_chunk(chunk: object, stream_format: StreamFormat) -> str | None:
    if not isinstance(chunk, dict):
        return None

    chunk_type = chunk.get("type")
    data = chunk.get("data")

    match stream_format:
        case "text":
            return format_text_chunk(chunk_type, data)
        case "ndjson":
            return format_ndjson_chunk(chunk_type, data)


def format_text_chunk(chunk_type: object, data: object) -> str | None:
    match chunk_type:
        case "status" if isinstance(data, str):
            return f"{data}\n"
        case "token" if isinstance(data, str):
            return data
        case _:
            return None


def format_ndjson_chunk(chunk_type: object, data: object) -> str | None:
    match chunk_type:
        case "status" | "token" if isinstance(data, str):
            return json.dumps({"type": chunk_type, "text": data}) + "\n"
        case "rag_images" if isinstance(data, str):
            return json.dumps({"type": "rag_images", "markdown": data}) + "\n"
        case "final":
            return json.dumps({"type": "done"}) + "\n"
        case _:
            return None


def create_app(graph_service: GraphRunner | None = None) -> FastAPI:
    service = (
        graph_service if graph_service is not None else GraphService(config=sys_config)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        startup = getattr(service, "startup", None)
        shutdown = getattr(service, "shutdown", None)

        if callable(startup):
            await cast(LifecycleHook, startup)()

        app.state.graph_service = service
        yield

        if callable(shutdown):
            await cast(LifecycleHook, shutdown)()

    def get_graph_service(request: Request) -> GraphRunner:
        return request.app.state.graph_service

    app = FastAPI(title="Atlas Agent Web API", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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
        stream_format: StreamFormat = "text",
    ):
        async def response_stream():
            async for chunk in graph_service.stream(input):
                formatted_chunk = format_graph_chunk(chunk, stream_format)
                if formatted_chunk is not None:
                    yield formatted_chunk

        return StreamingResponse(
            response_stream(),
            media_type=(
                "application/x-ndjson" if stream_format == "ndjson" else "text/plain"
            ),
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/ingest/pdf")
    async def ingest_pdf(
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
                    async for event in stream_ingest_pdf(
                        temp_path,
                        file_name=filename,
                        store_name="raggidy_docs",
                    ):
                        yield json.dumps(event) + "\n"
                except Exception as exc:
                    yield json.dumps({"type": "error", "text": str(exc)}) + "\n"
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)

            return StreamingResponse(
                response_stream(),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        except HTTPException:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    return app


app = create_app()
