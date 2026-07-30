import json
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from atlasai.config.sys_config import SysConfig, bootstrap_config
from atlasai.service.graph_service import GraphRunner, GraphService, InvokePayload

sys_config: SysConfig = bootstrap_config()
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"


def create_app(graph_service: GraphRunner) -> FastAPI:
    def get_graph_service() -> GraphRunner:
        return graph_service

    app = FastAPI(title="Atlas Agent Web API", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def _():
        return FileResponse(INDEX_FILE)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/invoke")
    def _(
        input: InvokePayload,
        graph_service: Annotated[GraphRunner, Depends(get_graph_service)],
        stream_format: Literal["text", "ndjson"] = "text",
    ):
        async def response_stream():
            async for chunk in graph_service.stream(input):
                if chunk.get("type") == "status":
                    status = chunk.get("data")
                    if isinstance(status, str):
                        if stream_format == "ndjson":
                            yield (
                                json.dumps({"type": "status", "text": status}) + "\n"
                            )
                        else:
                            yield f"{status}\n"
                elif chunk.get("type") == "token":
                    token = chunk.get("data")
                    if isinstance(token, str):
                        if stream_format == "ndjson":
                            yield (json.dumps({"type": "token", "text": token}) + "\n")
                        else:
                            yield token
                elif chunk.get("type") == "final" and stream_format == "ndjson":
                    yield json.dumps({"type": "done"}) + "\n"

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

    return app

graph_service = GraphService(config=sys_config)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await graph_service.startup()
    app.state.graph_service = graph_service
    yield
    await graph_service.shutdown()

app = create_app(graph_service)
