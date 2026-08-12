import asyncio
import logging
import tracemalloc
from collections.abc import AsyncIterator
from typing import cast

import requests
from langchain.messages import HumanMessage, RemoveMessage, SystemMessage, trim_messages
from langchain.tools import BaseTool, tool
from langchain_core.callbacks.usage import UsageMetadataCallbackHandler
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime
from langgraph.store.postgres.aio import AsyncPostgresStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from pydantic import BaseModel, Field, PrivateAttr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from typing_extensions import TypedDict

from atlasai.config.models import search_model, system_model
from atlasai.config.sys_config import SysConfig
from atlasai.internal.graphs.template import print_graph
from atlasai.lib.session import Context, StateContext
from atlasai.rag.utils import (
    build_document_context_prompt,
    build_retrieved_image_assets,
)
from atlasai.service.contracts import GraphRunner, InvokePayload
from atlasai.store import PostgresVectorService, tool_store
from atlasai.tools import math_tools

if not tracemalloc.is_tracing():
    tracemalloc.start(25)

logger = logging.getLogger(__name__)

SUMMARY_TRIGGER_MESSAGE_COUNT = 12
SUMMARY_KEEP_LAST_MESSAGES = 6


class PriceInput(BaseModel):
    ticker: str = Field(description="The ticker for the price we are fetching")


class CryptoPriceToolInputSchema(TypedDict):
    ticker: str


class CryptoPriceTool(BaseTool):
    name: str = "get_crypto_prices"
    description: str = "Coin gecko API used to retrieve cryptocurrency prices"

    _config: SysConfig = PrivateAttr()

    def __init__(self, config: SysConfig, **kwargs):
        super().__init__(**kwargs)
        self._config = config

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True,
    )
    def _run(self, ticker: str):
        if not self.validate_schema(ticker):
            return ValueError("Schema not Valid")

        # Request the current USD price from CoinGecko.
        coin_gecko_url = f"https://api.coingecko.com/api/v3/simple/price?vs_currencies=usd&ids={ticker}&x_cg_demo_api_key={self._config['cg_api_key']}"
        res = requests.get(coin_gecko_url)
        res.raise_for_status()

        return res.json()

    def validate_schema(self, ticker: str) -> bool:
        return ticker is not None


class GraphService(GraphRunner):
    """Orchestrates graph execution, tool access, and persistence lifecycle."""

    sys_config: SysConfig
    tools_list: list[BaseTool] = []
    get_crypto_prices: BaseTool

    def __init__(
        self,
        config: SysConfig,
        *,
        vector_service: PostgresVectorService,
    ) -> None:
        self.sys_config = config
        self.vector_service = vector_service
        self.get_crypto_prices = CryptoPriceTool(config)
        self.tool_retriever = tool_store(self.tools_list).as_retriever(
            search_kwargs={"k": 5}
        )
        self.checkpointer = None
        self.store = None
        self.graph = None

        @tool
        async def search_user_info(query: str):
            """Search indexed websites for a named person's biography or professional details. Use for questions like 'Who is X?' or 'What does X do?'"""
            return await self.vector_service.asimilarity_search(
                table_name="websites",
                query=query,
            )

        self.tools_list = [
            search_user_info,
            self.get_crypto_prices,
            *math_tools,
            create_manage_memory_tool(namespace=("memories")),
            create_search_memory_tool(namespace=("memories")),
        ]
        self._tools_by_name = {tool.name: tool for tool in self.tools_list}

    async def startup(self):
        """Open Postgres resources, initialize storage, and compile graphs."""

        self.pg_store_ctx = AsyncPostgresStore.from_conn_string(
            self.sys_config["pg_store"]
        )
        self.checkpointer_ctx = AsyncPostgresSaver.from_conn_string(
            self.sys_config["pg_store"]
        )

        self.checkpointer = await self.checkpointer_ctx.__aenter__()
        self.pg_store = await self.pg_store_ctx.__aenter__()

        await self.checkpointer.setup()
        await self.pg_store.setup()
        self.store = self.pg_store

        self.graph = self.build_main_graph().compile(
            checkpointer=self.checkpointer, store=self.store
        )
        print_graph(self.graph, "graph.png")

    async def shutdown(self):
        """Close Postgres resource context managers."""

        if self.pg_store_ctx is not None:
            await self.pg_store_ctx.__aexit__(None, None, None)
        if self.checkpointer_ctx is not None:
            await self.checkpointer_ctx.__aexit__(None, None, None)

    def retrieve_tools(self, tool_names: list[str]):
        return [
            self._tools_by_name[tool_name]
            for tool_name in dict.fromkeys(tool_names)
            if tool_name in self._tools_by_name
        ]

    async def tool_resolver_node(self, state: StateContext, runtime: Runtime[Context]):
        query = state.get("user_input")
        if not query:
            return {"resolved_tools": []}

        status = "[Atlas AI] Running..."
        print(status, flush=True)
        runtime.stream_writer({"type": "status", "data": status})

        matched_tools = self.tool_retriever.invoke(query)
        tool_names = list(
            dict.fromkeys(
                tool.metadata["name"]
                for tool in matched_tools
                if tool.metadata.get("name") in self._tools_by_name
            )
        )

        return {"resolved_tools": tool_names}

    async def agent_llm_node(self, state: StateContext, runtime: Runtime[Context]):
        context = runtime.context or {
            "user_instruction_prompt": self.sys_config["user_instruction_prompt"],
            "soul": self.sys_config["soul"],
        }

        instruction_prompt = self.sys_config.get("user_instruction_prompt")
        system_instruction = f"Available Tools: {state.get('resolved_tools')}"
        user_input = state.get("user_input") or ""
        user_id = state.get("user_id")
        trace_id = state.get("trace_id")
        summary = state.get("summary")

        filter_payload: dict[str, str] | None = None
        if user_id:
            filter_payload = {"user_id": user_id}
        rag_res = await self.vector_service.asimilarity_search(
            table_name="raggidy_docs",
            query=user_input,
            filter=filter_payload,
        )

        rag_context = build_document_context_prompt(
            user_input, rag_res, "knowledge-base"
        )
        related_image_assets = build_retrieved_image_assets(rag_res)
        logger.info(
            "rag_image_assets trace_id=%s user_id=%s retrieved_chunks=%s assets=%s",
            trace_id,
            user_id,
            len(rag_res),
            related_image_assets,
        )

        history = state.get("messages") or []
        system_context = "\n\n".join(
            part
            for part in [
                system_instruction,
                f"RAG DOCUMENT CONTEXT:\n{rag_context}",
                f"CONVERSATION SUMMARY:\n{summary}" if summary else None,
                f"INSTRUCTIONS PROMPT: {instruction_prompt}",
                f"SOUL FILE: {context['soul']}",
            ]
            if part
        )

        messages = [
            SystemMessage(content=system_context),
            *history,
        ]

        trimmed_messages = trim_messages(
            messages,
            max_tokens=1000000,
            strategy="last",
            token_counter="approximate",
            # Start history with a human message, optionally preceded by a system message.
            start_on="human",
            # Preserve system instructions while trimming older conversation turns.
            include_system=True,
            allow_partial=False,
        )

        tools_list: list[str] = state.get("resolved_tools") or []
        tools: list[BaseTool] = self.retrieve_tools(tools_list)

        llm = system_model(self.sys_config)

        llm_status = "[Atlas AI] LLM is working..."
        print(llm_status, flush=True)
        runtime.stream_writer({"type": "status", "data": llm_status})
        usage_callback = UsageMetadataCallbackHandler()
        callbacks = [usage_callback]
        response = await llm.bind_tools(tools).ainvoke(
            trimmed_messages,
            config=cast(
                RunnableConfig,
                {
                    "callbacks": callbacks,
                    "run_name": "atlas-main-agent",
                },
            ),
        )

        for tool_call in response.tool_calls:
            tool_status = f"[Atlas AI] Calling tool: {tool_call['name']}"
            print(tool_status, flush=True)
            runtime.stream_writer({"type": "status", "data": tool_status})

        usage_payload = None
        response_usage_metadata = getattr(response, "usage_metadata", None)
        if response_usage_metadata:
            usage_payload = {
                "model_name": getattr(response, "response_metadata", {}).get(
                    "model_name",
                    getattr(response, "model_name", None),
                ),
                "usage_metadata": response_usage_metadata,
            }
        elif usage_callback.usage_metadata:
            model_name, usage_metadata = next(
                iter(usage_callback.usage_metadata.items())
            )
            usage_payload = {
                "model_name": model_name,
                "usage_metadata": usage_metadata,
            }

        logger.info(
            "agent_usage_payload trace_id=%s has_payload=%s response_usage_keys=%s "
            "callback_models=%s",
            trace_id,
            usage_payload is not None,
            sorted(response_usage_metadata.keys())
            if isinstance(response_usage_metadata, dict)
            else [],
            sorted(usage_callback.usage_metadata.keys()),
        )
        if usage_payload is not None:
            runtime.stream_writer({"type": "usage_payload", "data": usage_payload})

        return {
            "messages": [response],
            "selected_image_assets": related_image_assets or None,
            "usage_payload": usage_payload,
        }

    @staticmethod
    def _messages_to_summarize(messages: list[object]) -> list[object]:
        """Return the older thread messages that should be summarized."""

        if len(messages) <= SUMMARY_TRIGGER_MESSAGE_COUNT:
            return []

        kept_messages = messages[-SUMMARY_KEEP_LAST_MESSAGES:]
        if not kept_messages:
            return []

        keep_start_index = len(messages) - len(kept_messages)
        for index, message in enumerate(kept_messages, start=keep_start_index):
            if getattr(message, "type", None) in {"human", "system"}:
                return messages[:index]

        return []

    async def summarize_conversation_node(self, state: StateContext):
        messages = state.get("messages") or []
        summarize_messages = self._messages_to_summarize(messages)
        if not summarize_messages:
            return {}

        existing_summary = (state.get("summary") or "").strip()
        llm = search_model(self.sys_config)
        summary_prefix = (
            f"Current summary:\n{existing_summary}\n\n" if existing_summary else ""
        )
        summary_instruction = f"""
        {summary_prefix},
        Summarize the older conversation messages above.
        Keep durable user facts, active work, decisions, preferences, unresolved questions, and important document context.
        Return only the updated running summary."""

        response = await llm.ainvoke(
            [
                *summarize_messages,
                HumanMessage(content=summary_instruction),
            ],
            config={"run_name": "atlas-thread-summarizer"},
        )

        updated_summary = self._message_chunk_text(response).strip()
        if not updated_summary:
            return {}

        return {
            "summary": updated_summary,
        }

    def build_main_graph(self):
        tool_node = ToolNode(self.tools_list)
        builder = StateGraph(state_schema=StateContext, context_schema=Context)
        builder.add_node("tool_resolver", self.tool_resolver_node)
        builder.add_node("agent", self.agent_llm_node)
        builder.add_node("tool", tool_node)
        builder.add_edge(START, "tool_resolver")
        builder.add_edge("tool_resolver", "agent")
        builder.add_conditional_edges(
            "agent", tools_condition, {"tools": "tool", END: END}
        )
        builder.add_edge("tool", "agent")

        return builder

    def build_graph_config(self, thread_id: str, run_name: str):
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "run_name": run_name,
        }

        return config

    async def run_summarizer(self, thread_id: str) -> None:
        if self.graph is None:
            raise RuntimeError("Graph service startup must run before summarizer use.")

        config = self.build_graph_config(thread_id, "atlasai-thread-summary")
        state_snapshot = await self.graph.aget_state(config)
        state = cast(StateContext, state_snapshot.values)
        summarize_messages = self._messages_to_summarize(state.get("messages") or [])
        if not summarize_messages:
            return

        summary_update = await self.summarize_conversation_node(state)
        updated_summary = summary_update.get("summary")
        if not isinstance(updated_summary, str) or not updated_summary.strip():
            return

        delete_messages = [
            RemoveMessage(id=message.id)
            for message in summarize_messages
            if getattr(message, "id", None)
        ]
        await self.graph.aupdate_state(
            config,
            {
                "summary": updated_summary.strip(),
                "messages": delete_messages,
            },
            as_node="agent",
        )

    def _log_background_task(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception as exc:
            print(f"Background summarizer failed: {exc}")

    @staticmethod
    def _message_chunk_text(message: object) -> str:
        """Extract plain text from a streamed LangChain message chunk."""

        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content

        if not isinstance(content, list):
            return ""

        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])

        return "".join(text_parts)

    async def stream(
        self,
        payload: InvokePayload,
    ) -> AsyncIterator[object]:
        if self.graph is None:
            raise RuntimeError("Graph service startup must run before streaming.")

        query = payload["user_input"]
        thread_id = payload["thread_id"]
        trace_id = payload.get("trace_id") or thread_id
        try:
            current_state: StateContext = {
                "user_input": query,
                "thread_id": thread_id,
                "trace_id": trace_id,
                "messages": [HumanMessage(content=query)],
            }
            user_id = payload.get("user_id")
            if user_id is not None:
                current_state["user_id"] = user_id
            if "document_id" in payload:
                current_state["document_id"] = payload["document_id"]

            last_state: StateContext | None = None
            latest_usage_payload: object | None = None

            async for chunk in self.graph.astream(
                current_state,
                config=self.build_graph_config(thread_id, "atlas-main"),
                version="v2",
                stream_mode=["custom", "messages", "values"],
            ):
                # Full graph state after a step
                if chunk["type"] == "values":
                    last_state = chunk["data"]
                elif chunk["type"] == "custom":
                    status = chunk["data"]
                    if isinstance(status, dict) and status.get("type") == "status":
                        yield status
                    elif (
                        isinstance(status, dict)
                        and status.get("type") == "usage_payload"
                    ):
                        latest_usage_payload = status.get("data")
                elif chunk["type"] == "messages":
                    # Stream Node Status
                    message, metadata = chunk["data"]
                    if metadata.get("langgraph_node") != "agent":
                        continue

                    token = self._message_chunk_text(message)
                    if token:
                        yield {"type": "token", "data": token}

            if last_state is not None:
                selected_image_assets = last_state.get("selected_image_assets")
                if isinstance(selected_image_assets, list) and selected_image_assets:
                    yield {"type": "sources", "data": selected_image_assets}

                final_state: StateContext = last_state
                if latest_usage_payload is not None and not last_state.get(
                    "usage_payload"
                ):
                    final_state = {**last_state, "usage_payload": latest_usage_payload}

                yield {"type": "final", "data": final_state}

                summary_task = asyncio.create_task(self.run_summarizer(thread_id))
                summary_task.add_done_callback(self._log_background_task)

        except Exception as e:
            print(e)
            raise
