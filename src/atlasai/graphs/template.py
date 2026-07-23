import json
import logging
import os
from pprint import pprint
from typing import Any, TypedDict
import warnings
from concurrent.futures import ThreadPoolExecutor

from langchain_chroma import Chroma
from langchain_unstructured.document_loaders import Element
from pydantic import BaseModel, Field
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from atlasai.config.models import system_model, search_model
from atlasai.lib.session import Context, MemoryUpdateResult, StateContext
from atlasai.config.sys_config import SysConfig, bootstrap_config
from atlasai.rag.rag_ingestion import summarize_chunks
from atlasai.tools import math_tools
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    trim_messages,
    RemoveMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_unstructured import UnstructuredLoader
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
import Stemmer
from langchain_core.prompts import PromptTemplate
from langfuse import propagate_attributes
from langfuse.langchain import CallbackHandler
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title

from atlasai.util.utils import load_file


load_dotenv()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
warnings.filterwarnings(
    "ignore",
    message="The v3 streaming protocol on Pregel is experimental.",
)
config: SysConfig = bootstrap_config()
model = system_model(config)
search_llm = search_model(config)

LANGFUSE_TAGS = ["atlasai", "template"]
LANGFUSE_ENV_VARS = (
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_BASE_URL",
)
langfuse_handler = (
    CallbackHandler()
    if all(os.getenv(env_var) for env_var in LANGFUSE_ENV_VARS)
    else None
)


def build_graph_config(thread_id: str, run_name: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "run_name": run_name,
    }

    if langfuse_handler is not None:
        config["callbacks"] = [langfuse_handler]

    return config


def run_memory_graph(memory_graph, latest_state: StateContext, thread_id: str):
    if langfuse_handler is None:
        return memory_graph.invoke(
            latest_state,
            config=build_graph_config(thread_id, "atlasai-memory-graph"),
        )

    with propagate_attributes(
        trace_name="atlasai-memory-update",
        session_id=thread_id,
        tags=LANGFUSE_TAGS,
        metadata={"graph": "memory"},
    ):
        return memory_graph.invoke(
            latest_state,
            config=build_graph_config(thread_id, "atlasai-memory-graph"),
        )


def run_main_graph(graph, current_state: StateContext, thread_id: str):
    config = build_graph_config(thread_id, "atlasai-main-graph")

    if langfuse_handler is None:
        return graph.stream_events(current_state, config=config, version="v3")

    with propagate_attributes(
        trace_name="atlasai-chat-turn",
        session_id=thread_id,
        tags=LANGFUSE_TAGS,
        metadata={"graph": "main"},
    ):
        stream = graph.stream_events(current_state, config=config, version="v3")

        for message in stream.messages:
            if message.node != "agent":
                continue

            for token in message.text:
                print(token, end="", flush=True)

        return stream


def load_user_info():
    urls = [
        "https://ndeto.eth.limo",
    ]
    chunks: list[Element] = []

    for url in urls:
        data = UnstructuredLoader(web_url=url).load()
        elements = []
        for page in data:
            elements = partition_html(text=page.page_content)

            chunk = chunk_by_title(
                elements=elements,
                max_characters=3000,
                new_after_n_chars=2400,
                combine_text_under_n_chars=500,
            )

            chunks.extend(chunk)

    doc_chunks = summarize_chunks(chunks)

    return doc_chunks


# doc_splits = load_user_info()


# def _retrieve_user_info():
#     vector_store = InMemoryVectorStore.from_documents(
#         documents=doc_splits, embedding=OpenAIEmbeddings()
#     )

#     return vector_store.as_retriever()


@tool
def search_user_info(query: str):
    """This searches available sources for user information"""
    # retriever = _retrieve_user_info()
    # res = retriever.invoke(query)

    # json_results = []
    # for r in res:
    #     json_results.append(json.dumps(r))

    # return json_results
    return "Ndeto"


def doc_retriever(persisted_dir="db/chroma.db"):
    store = Chroma(
        persist_directory=persisted_dir, embedding_function=OpenAIEmbeddings(), collection_metadata={"hnsw:space": "cosine"}
    )

    return store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.3 
        }
    )

rag_retriever = doc_retriever()

def build_document_context_prompt(
    query: str, chunks: list[Any], source_label: str
) -> str:
    def read_json_field(value: Any) -> Any:
        if not isinstance(value, str):
            return value

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def extract_chunk_sections(chunk: Any) -> tuple[str, list[Any], list[Any]]:
        metadata = getattr(chunk, "metadata", {}) or {}
        original_content = metadata.get("original_content")

        if original_content:
            original_data = read_json_field(original_content)
            if not isinstance(original_data, dict):
                original_data = {}

            raw_text = read_json_field(original_data.get("raw_text", ""))
            if not isinstance(raw_text, str):
                raw_text = str(raw_text)

            tables_html = read_json_field(original_data.get("tables_html", []))
            if isinstance(tables_html, str):
                tables_html = [tables_html] if tables_html.strip() else []

            images = read_json_field(original_data.get("images", []))
            if isinstance(images, str):
                images = [images] if images.strip() else []

            return raw_text.strip(), tables_html, images

        page_content = getattr(chunk, "page_content", "")
        if not isinstance(page_content, str):
            page_content = str(page_content)

        return page_content.strip(), [], []

    prompt_text = f"""Based on the following {source_label} documents, answer this question: {query}

CONTENT TO ANALYZE:
"""
    document_count = 0

    for chunk in chunks:
        raw_text, tables_html, images = extract_chunk_sections(chunk)
        if not raw_text and not tables_html and not images:
            continue

        document_count += 1
        prompt_text += f"--- Document {document_count} ---\n"

        if raw_text:
            prompt_text += f"TEXT:\n{raw_text}\n\n"

        if tables_html:
            prompt_text += "TABLES:\n"
            for j, table in enumerate(tables_html):
                prompt_text += f"Table {j + 1}:\n{table}\n\n"

        if images:
            prompt_text += "IMAGES:\n"
            for j, _ in enumerate(images):
                prompt_text += f"Image {j + 1}: image data exists in this chunk.\n"
            prompt_text += "\n"

    if document_count == 0:
        prompt_text += "No non-empty documents were retrieved for this source.\n\n"

    prompt_text += """Use the document content above when it is relevant.
If the documents do not contain enough information, say so instead of guessing."""

    return prompt_text


@tool
def offloaded_context_memory_search(queries: list[str]):
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
        coin_gecko_url = f"https://api.coingecko.com/api/v3/simple/price?vs_currencies=usd&ids={ticker}&x_cg_demo_api_key={config['cg_api_key']}"
        res = requests.get(coin_gecko_url)
        res.raise_for_status()

        return res.json()

    def validate_schema(self, ticker: str) -> bool:
        if ticker is None:
            return False

        return True


get_crypto_prices = CryptoPriceTool()

tools_list = [
    search_user_info,
    get_crypto_prices,
    offloaded_context_memory_search,
    *math_tools,
]
tool_node = ToolNode(tools_list)

stemmer = Stemmer.Stemmer("english")


def initialize_and_index_tools():
    dynamic_tool_registry = []
    descriptions = []
    metadata = []

    for t in tools_list:
        descriptions.append(f"Tool Name: {t.name}. Description: {t.description}")
        metadata.append(
            {
                "name": t.name,
                "parameters": t.args,  # The exact Pydantic/JSON schema definition
            }
        )

        dynamic_tool_registry.append(
            {"name": t.name, "description": t.description, "parameters": t.args}
        )

    tool_store = InMemoryVectorStore.from_texts(
        texts=descriptions, metadatas=metadata, embedding=OpenAIEmbeddings()
    )

    return tool_store.as_retriever(search_kwargs={"k": 5})


tool_retriever = initialize_and_index_tools()


def agent_llm_node(state: StateContext, runtime: Runtime[Context]):
    context = runtime.context or {
        "user_instruction_prompt": config["user_instruction_prompt"],
        "soul": config["soul"],
    }

    fact_context = []
    if "recent_facts" in state:
        for f in state["recent_facts"]:
            fact_context.append(json.dumps(f))

    instruction_prompt = config.get("user_instruction_prompt")
    system_instruction = f"Available Tools: {state.get('resolved_tools')}"
    user_input = state.get("user_input") or ""
    rag_res = rag_retriever.invoke(user_input)

    rag_context = build_document_context_prompt(user_input, rag_res, "knowledge-base")

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

    tools: list[BaseTool] = retrieve_tools(tools_list)

    response = model.bind_tools(tools).invoke(trimmed_messages)

    return {"messages": [response]}


def prune_messages_node(state: StateContext):
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


def retrieve_tools(tool_names: list[str]):
    tools = []
    for t in tools_list:
        for tool_name in tool_names:
            if t.name == tool_name:
                tools.append(t)

    return tools


def context_resolver_node(state: StateContext):
    # Tool Loadout
    tools = []
    if "user_input" in state:
        query = state["user_input"]
        matched_tools = tool_retriever.invoke(query)

        for tool in matched_tools:
            for t in tools_list:
                if tool.metadata["name"] == t.name:
                    tools.append(t.name)

    return {"resolved_tools": tools}


def print_graph(graph, name):
    try:
        graph.get_graph().draw_png(output_file_path=name)
    except ImportError:
        print(
            "You likely need to install dependencies for pygraphviz, see more here https://github.com/pygraphviz/pygraphviz/blob/main/INSTALL.txt"
        )


def memory_extractor_node(state: StateContext):
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

    extractor = prompt_temp | search_llm.with_structured_output(MemoryUpdateResult)

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

    updated_long_term_memory = res.get("updated_long_term_memory") or existing_memory

    with open(memory_path, "w", encoding="utf-8") as file:
        json.dump(updated_long_term_memory, file, indent=2)

    return {
        "recent_facts": res.get("short_term_facts"),
        "user_mood": res.get("user_mood"),
    }


def main():
    builder = StateGraph(state_schema=StateContext, context_schema=Context)
    builder.add_node("context_resolver", context_resolver_node)
    builder.add_node("agent", agent_llm_node)
    builder.add_node("tool", tool_node)
    builder.add_edge(START, "context_resolver")
    builder.add_edge("context_resolver", "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tool", END: END})
    builder.add_edge("tool", "agent")

    mem_graph = StateGraph(state_schema=StateContext)
    mem_graph.add_node("memory", memory_extractor_node)
    mem_graph.add_node("prune_messages", prune_messages_node)

    mem_graph.add_edge(START, "prune_messages")
    mem_graph.add_edge("prune_messages", "memory")
    mem_graph.add_edge("memory", END)

    latest_state = None

    with SqliteSaver.from_conn_string("memory.sqlite") as memory:
        graph = builder.compile(checkpointer=memory)
        memory_graph = mem_graph.compile(checkpointer=memory)
        memory_executor = ThreadPoolExecutor(max_workers=1)
        print_graph(graph, "graph.png")
        print_graph(memory_graph, "memory.png")

        try:
            while True:
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit"]:
                    print("Good bye!")
                    break

                if not user_input.strip():
                    continue

                current_state: StateContext = {
                    "user_input": user_input,
                    "messages": [HumanMessage(content=user_input)],
                }
                thread_id = "test1"

                print("Agent: ", end="", flush=True)

                stream = run_main_graph(graph, current_state, thread_id)

                if langfuse_handler is None:
                    for message in stream.messages:
                        if message.node != "agent":
                            continue

                        for token in message.text:
                            print(token, end="", flush=True)

                latest_state = stream.output
                pprint("\n")

                memory_executor.submit(
                    run_memory_graph,
                    memory_graph,
                    latest_state,
                    thread_id,
                )
        finally:
            memory_executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
