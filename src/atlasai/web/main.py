from logging import Logger
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.responses import PlainTextResponse

from atlasai.config.sys_config import SysConfig, bootstrap_config
from atlasai.service.graph import GraphRunner, GraphService, InvokePayload

sys_config: SysConfig = bootstrap_config()


def get_logger_service() -> Logger:
    return Logger(name="atlas-logger")


def create_app(graph_service: GraphRunner) -> FastAPI:
    def get_graph_service() -> GraphRunner:
        return graph_service

    app = FastAPI(title="Atlas Agent Web API")

    @app.get("/")
    def _():
        return {"Hello": "World"}

    @app.post("/invoke", response_class=PlainTextResponse)
    def _(
        input: InvokePayload,
        graph_service: Annotated[GraphRunner, Depends(get_graph_service)],
    ):
        res = graph_service.run(input)
        return res

    return app


graph_service = GraphService(config=sys_config)
app = create_app(graph_service)
