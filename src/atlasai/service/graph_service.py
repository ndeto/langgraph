import asyncio
import json
import os
import tracemalloc
from collections.abc import AsyncIterator
from typing import Protocol

import requests
from langchain.messages import HumanMessage, RemoveMessage, SystemMessage, trim_messages
from langchain.tools import BaseTool, tool
from langchain_classic.prompts import PromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_postgres import PGVectorStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
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
from atlasai.lib.session import Context, MemoryUpdateResult, StateContext
from atlasai.rag.utils import build_document_context_prompt
from atlasai.store import get_or_create_store, tool_store
from atlasai.tools import math_tools
from atlasai.util.utils import load_file

if not tracemalloc.is_tracing():
    tracemalloc.start(25)


class InvokePayload(TypedDict):
    user_input: str
    thread_id: str


class GraphRunner(Protocol):
    def stream(self, payload: InvokePayload) -> AsyncIterator[object]: ...


class GraphService(GraphRunner):
    sys_config: SysConfig
    tools_list: list[BaseTool] = []
    get_crypto_prices: BaseTool

    def __init__(self, config: SysConfig) -> None:
        self.sys_config = config
        self.get_crypto_prices = self.CryptoPriceTool(config)

        @tool
        async def search_user_info(query: str):
            """This searches available website sources for user information like personal information and biographies for personalities"""
            store: PGVectorStore = await get_or_create_store("websites")
            res = await store.asimilarity_search(query)

            return res

        self.tools_list = [
            search_user_info,
            self.get_crypto_prices,
            *math_tools,
            create_manage_memory_tool(namespace=("memories")),
            create_search_memory_tool(namespace=("memories")),
        ]

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

    def retrieve_tools(self, tool_names: list[str]):
        tools = []
        for t in self.tools_list:
            for tool_name in tool_names:
                if t.name == tool_name:
                    tools.append(t)

        return tools

    def tool_resolver_node(self, state: StateContext):
        # Select the tools most relevant to the current request.
        tools = []
        if "user_input" in state:
            query = state["user_input"]
            tool_retriever = tool_store(self.tools_list).as_retriever(
                search_kwargs={"k": 5}
            )

            matched_tools = tool_retriever.invoke(query)

            for tool in matched_tools:
                for t in self.tools_list:
                    if tool.metadata["name"] == t.name:
                        tools.append(t.name)

        return {"resolved_tools": tools}

    async def agent_llm_node(self, state: StateContext, runtime: Runtime[Context]):
        context = runtime.context or {
            "user_instruction_prompt": self.sys_config["user_instruction_prompt"],
            "soul": self.sys_config["soul"],
        }

        fact_context = []
        if "recent_facts" in state:
            for f in state["recent_facts"]:
                fact_context.append(json.dumps(f))

        instruction_prompt = self.sys_config.get("user_instruction_prompt")
        system_instruction = f"Available Tools: {state.get('resolved_tools')}"
        user_input = state.get("user_input") or ""

        rag_store = await get_or_create_store("raggidy_docs")
        rag_res = await rag_store.asimilarity_search(user_input)

        rag_context = build_document_context_prompt(
            user_input, rag_res, "knowledge-base"
        )

        history = state.get("messages") or []
        system_context = "\n\n".join(
            [
                system_instruction,
                f"RAG DOCUMENT CONTEXT:\n{rag_context}",
                f"INSTRUCTIONS PROMPT: {instruction_prompt}",
                f"SOUL FILE: {context['soul']}",
                f"RECENT FACTS:\n{json.dumps(fact_context)}",
            ]
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
        response = await llm.bind_tools(tools).ainvoke(trimmed_messages)

        for tool_call in response.tool_calls:
            tool_status = f"[Atlas AI] Calling tool: {tool_call['name']}"
            print(tool_status, flush=True)
            runtime.stream_writer({"type": "status", "data": tool_status})

        return {"messages": [response]}

    def prune_messages_node(self, state: StateContext):
        messages = state.get("messages") or []

        if len(messages) <= 10:
            return {}

        kept_messages = messages[-5:]

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *kept_messages,
            ]
        }

    def memory_extractor_node(self, state: StateContext):
        prompt_temp = PromptTemplate.from_template("""
                                                    You are maintaining the user's conversational memory.

                                                    User input:
                                                    {user_input}

                                                    Recent messages:
                                                    {messages}

                                                    Existing long-term memory:
                                                    {existing_memory}

                                                    Your tasks:
                                                    1. Extract short-term facts that are useful for the next immediate reply.
                                                    2. Infer the user's mood if it is helpful and reasonably clear from the conversation.
                                                    3. Update the existing long-term memory by merging in only durable facts from the recent interaction.

                                                    Durable long-term memory includes:
                                                    - who the user is
                                                    - their interests
                                                    - recurring goals and projects
                                                    - stable preferences
                                                    - behavioral patterns in how they like to work, communicate, or receive answers
                                                    - repeated constraints, habits, or priorities

                                                    Rules for long-term memory:
                                                    - Do not store one-off requests or temporary turn-specific details.
                                                    - Deduplicate overlapping facts.
                                                    - Merge facts that describe the same thing.
                                                    - Update stale or conflicting memory with the newer information.
                                                    - Preserve a clean category and sub_category structure.
                                                    - Avoid repeated keys, repeated entries, and near-duplicate categories.
                                                    - If there is no meaningful durable update, return the existing long-term memory unchanged.

                                                    Return:
                                                    - short_term_facts
                                                    - updated_long_term_memory
                                                    - user_mood
                                                """)

        llm = search_model(self.sys_config)

        extractor = prompt_temp | llm.with_structured_output(MemoryUpdateResult)

        messages = state.get("messages") or []
        memory_path = "structured_memory.json"

        if not os.path.exists(memory_path):
            with open(memory_path, "w", encoding="utf-8") as file:
                json.dump([], file)

        existing_memory = load_file(memory_path)

        res = extractor.invoke(
            {
                "messages": messages[-5:],
                "user_input": state.get("user_input"),
                "existing_memory": existing_memory,
            }
        )

        updated_long_term_memory = (
            res.get("updated_long_term_memory") or existing_memory
        )

        with open(memory_path, "w", encoding="utf-8") as file:
            json.dump(updated_long_term_memory, file, indent=2)

        return {
            "recent_facts": res.get("short_term_facts"),
            "user_mood": res.get("user_mood"),
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

    def build_memory_graph(self):
        mem_graph = StateGraph(state_schema=StateContext)

        mem_graph.add_node("memory", self.memory_extractor_node)
        mem_graph.add_node("prune_messages", self.prune_messages_node)

        mem_graph.add_edge(START, "prune_messages")
        mem_graph.add_edge("prune_messages", "memory")
        mem_graph.add_edge("memory", END)

        return mem_graph

    def build_graph_config(self, thread_id: str, run_name: str):
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "run_name": run_name,
        }

        return config

    async def run_memory_graph(self, thread_id, state):
        memory_graph = self.build_memory_graph()

        async with (
            AsyncPostgresSaver.from_conn_string(
                self.sys_config["pg_store"]
            ) as checkpointer,
            AsyncPostgresStore.from_conn_string(
                self.sys_config["pg_store"]
            ) as postgres_store,
        ):
            await checkpointer.setup()
            await postgres_store.setup()
            graph = memory_graph.compile(
                checkpointer=checkpointer, store=postgres_store
            )
            return await graph.ainvoke(
                state,
                config=self.build_graph_config(thread_id, "atlasai-memory-graph"),
            )

    def _log_background_task(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception as exc:
            print(f"Background memory graph failed: {exc}")

    @staticmethod
    def _message_chunk_text(message: object) -> str:
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

    async def stream(self, payload: InvokePayload) -> AsyncIterator[object]:
        query = payload.get("user_input")

        main_graph = self.build_main_graph()

        async with (
            AsyncPostgresSaver.from_conn_string(
                self.sys_config["pg_store"]
            ) as checkpointer,
            AsyncPostgresStore.from_conn_string(
                self.sys_config["pg_store"]
            ) as postgres_store,
        ):
            await checkpointer.setup()
            await postgres_store.setup()
            try:
                current_state: StateContext = {
                    "user_input": query,
                    "messages": [HumanMessage(content=query)],
                }
                thread_id = payload.get("thread_id")
                graph = main_graph.compile(
                    checkpointer=checkpointer, store=postgres_store
                )
                last_state: StateContext | None = None

                async for chunk in graph.astream(
                    current_state,
                    config=self.build_graph_config(thread_id, "atlas-main"),
                    version="v2",
                    stream_mode=["custom", "messages", "values"],
                ):
                    if chunk["type"] == "values":
                        last_state = chunk["data"]
                    elif chunk["type"] == "custom":
                        status = chunk["data"]
                        if isinstance(status, dict) and status.get("type") == "status":
                            yield status
                    elif chunk["type"] == "messages":
                        message, metadata = chunk["data"]
                        if metadata.get("langgraph_node") != "agent":
                            continue

                        token = self._message_chunk_text(message)
                        if token:
                            yield {"type": "token", "data": token}

                if last_state is not None:
                    yield {"type": "final", "data": last_state}

                    memory_task = asyncio.create_task(
                        self.run_memory_graph(thread_id, last_state)
                    )
                    memory_task.add_done_callback(self._log_background_task)

            except Exception as e:
                print(e)
                raise
