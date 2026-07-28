import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import requests
from langchain.messages import HumanMessage, RemoveMessage, SystemMessage, trim_messages
from langchain.tools import BaseTool, tool
from langchain_classic.prompts import PromptTemplate
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime
from langgraph.store.postgres import PostgresStore
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
from atlasai.store import doc_store, tool_store
from atlasai.store.rag_retriever import get_store
from atlasai.tools import math_tools
from atlasai.util.utils import load_file


class InvokePayload(TypedDict):
    user_input: str
    thread_id: str


class GraphRunner(Protocol):
    def run(self, payload: InvokePayload) -> str: ...


class GraphService(GraphRunner):
    sys_config: SysConfig
    tools_list: list[BaseTool] = []
    get_crypto_prices: BaseTool

    def __init__(self, config: SysConfig) -> None:
        self.sys_config = config
        self.get_crypto_prices = self.CryptoPriceTool(config)
        self.tools_list = [
            self.search_user_info,
            self.get_crypto_prices,
            *math_tools,
            create_manage_memory_tool(namespace=("memories")),
            create_search_memory_tool(namespace=("memories")),
        ]

    @tool
    def search_user_info(self, query: str):
        """This searches available website sources for user information like personal information and biographies for personalities"""
        retriever = get_store("websites").as_retriever()
        res = retriever.invoke(query)

        json_results = []
        for r in res:
            json_results.append(json.dumps(r))

        return json_results

    @tool
    def offloaded_context_memory_search(self, queries: list[str]):
        """Retrieve stored long-term memory and past conversation context about the user or current thread. Use this when the answer depends on prior discussions, remembered preferences, personal background, ongoing work, or previously established context."""

        offloaded_context = load_file("structured_memory.json")

        matches = []

        for item in offloaded_context:
            text = json.dumps(item)
            if any(query.lower() in text for query in queries):
                matches.append(item)

        return matches

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

            # Call
            coin_gecko_url = f"https://api.coingecko.com/api/v3/simple/price?vs_currencies=usd&ids={ticker}&x_cg_demo_api_key={self._config['cg_api_key']}"
            res = requests.get(coin_gecko_url)
            res.raise_for_status()

            return res.json()

        def validate_schema(self, ticker: str) -> bool:
            if ticker is None:
                return False

            return True

    def retrieve_tools(self, tool_names: list[str]):
        tools = []
        for t in self.tools_list:
            for tool_name in tool_names:
                if t.name == tool_name:
                    tools.append(t)

        return tools

    def tool_resolver_node(self, state: StateContext):
        # Tool Loadout
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

    def agent_llm_node(self, state: StateContext, runtime: Runtime[Context]):
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

        rag_store = doc_store().as_retriever()
        rag_res = rag_store.invoke(user_input)

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
            # Most chat models expect that chat history starts with either:
            # (1) a HumanMessage or
            # (2) a SystemMessage followed by a HumanMessage
            start_on="human",
            # Usually, we want to keep the SystemMessage
            # if it's present in the original history.
            # The SystemMessage has special instructions for the model.
            include_system=True,
            allow_partial=False,
        )

        tools_list: list[str] = state.get("resolved_tools") or []

        tools: list[BaseTool] = self.retrieve_tools(tools_list)

        llm = system_model(self.sys_config)

        response = llm.bind_tools(tools).invoke(trimmed_messages)

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

    def build_graph_config(self, thread_id: str, run_name: str) -> dict[str, Any]:
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "run_name": run_name,
        }

        return config

    def run_main_graph(self, graph, thread_id, state: StateContext):
        return graph.invoke(
            state,
            config=self.build_graph_config(thread_id, "atlasai-main-graph"),
        )

    def run_memory_graph(self, graph, thread_id, state):
        return graph.invoke(
            state,
            config=self.build_graph_config(thread_id, "atlasai-memory-graph"),
        )

    def run(self, payload: InvokePayload):
        query = payload.get("user_input")

        main_graph = self.build_main_graph()
        memory_graph = self.build_memory_graph()
        memory_executor = ThreadPoolExecutor(max_workers=1)

        with PostgresSaver.from_conn_string(self.sys_config["db_conn"]) as checkpointer:
            with PostgresStore.from_conn_string(
                self.sys_config["pg_store"]
            ) as postgres_store:
                checkpointer.setup()
                postgres_store.setup()
                try:
                    current_state: StateContext = {
                        "user_input": query,
                        "messages": [HumanMessage(content=query)],
                    }
                    thread_id = payload.get("thread_id")
                    graph = main_graph.compile(
                        checkpointer=checkpointer, store=postgres_store
                    )
                    final_state = self.run_main_graph(graph, thread_id, current_state)

                    mem_graph = memory_graph.compile(
                        checkpointer=checkpointer, store=postgres_store
                    )

                    memory_executor.submit(
                        self.run_memory_graph, mem_graph, thread_id, final_state
                    )

                    return final_state["messages"][-1].content

                except Exception as e:
                    print(e)
                    raise
                finally:
                    memory_executor.shutdown(wait=True, cancel_futures=True)
