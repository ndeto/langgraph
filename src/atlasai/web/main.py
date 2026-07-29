from logging import Logger
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from atlasai.config.sys_config import SysConfig, bootstrap_config
from atlasai.service.graph_service import GraphRunner, GraphService, InvokePayload

sys_config: SysConfig = bootstrap_config()
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"


def get_logger_service() -> Logger:
    return Logger(name="atlas-logger")


def create_app(graph_service: GraphRunner) -> FastAPI:
    def get_graph_service() -> GraphRunner:
        return graph_service

    app = FastAPI(title="Atlas Agent Web API")
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
    ):
        async def response_stream():
            async for chunk in graph_service.stream(input):
                if chunk.get("type") == "status":
                    status = chunk.get("data")
                    if isinstance(status, str):
                        yield f"{status}\n"
                elif chunk.get("type") == "token":
                    token = chunk.get("data")
                    if isinstance(token, str):
                        yield token

        return StreamingResponse(
            response_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


graph_service = GraphService(config=sys_config)
app = create_app(graph_service)
