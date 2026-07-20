from atlasai.config.models import system_model
from atlasai.lib.session import Context, StateContext
from atlasai.config.sys_config import SysConfig, bootstrap_config
from atlasai.tools import math_tools
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import display
from langgraph.checkpoint.sqlite import SqliteSaver
import Stemmer

load_dotenv()
config: SysConfig = bootstrap_config()
model = system_model(config)


def load_user_info():
    urls = [
        "https://ndeto.eth.limo",
        "https://en.wikipedia.org/wiki/Albert_Einstein",
    ]
    chunks = []
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    for url in urls:
        data = UnstructuredLoader(web_url=url).load()
        doc_chunks = splitter.split_documents(data)
        for chunk in doc_chunks:
            chunks.append(chunk)

    return chunks


doc_splits = load_user_info()


def _retrieve_user_info():
    vector_store = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=OpenAIEmbeddings()
    )

    return vector_store.as_retriever()


@tool
def search_user_info(query: str):
    """This searches available sources for user information"""
    retriever = _retrieve_user_info()
    res = retriever.invoke(query)

    return res


tools_list = [search_user_info, *math_tools]
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

    user_input = state.get("user_input") or ""

    user_message = HumanMessage(user_input)
    user_instruction_prompt = SystemMessage(
        content=config.get("user_instruction_prompt")
    )
    soul = SystemMessage(context["soul"])
    system_instruction = (
        f"Available Tools: {state.get('resolved_tools')}"
    )

    history = state.get("messages") or []
    messages = [
        user_message,
        soul,
        user_instruction_prompt,
        SystemMessage(content=system_instruction),
        *history,
    ]

    tools: list[BaseTool] = state.get("resolved_tools") or []

    response = model.bind_tools(tools).invoke(messages)

    return {"messages": [response]}


def retrieve_tools(tool_names: list[str]):
    tools = []
    for t in tools_list:
        for tool_name in tool_names:
            if t.name == tool_name:
                tools.append(t)

    return tools


def context_resolver_node(state: StateContext):
    if "user_input" in state:
        query = state["user_input"]
        matched_tools = tool_retriever.invoke(query)

        tools = []
        for tool in matched_tools:
            for t in tools_list:
                if tool.metadata["name"] == t.name:
                    tools.append(t)

        return {"resolved_tools": tools}


def print_graph(graph):
    try:
        display(graph.get_graph().draw_png(output_file_path="graph.png"))
    except ImportError:
        print(
            "You likely need to install dependencies for pygraphviz, see more here https://github.com/pygraphviz/pygraphviz/blob/main/INSTALL.txt"
        )


def main():
    builder = StateGraph(state_schema=StateContext, context_schema=Context)
    builder.add_node("context_resolver", context_resolver_node)
    builder.add_node("agent", agent_llm_node)
    builder.add_node("tool", tool_node)
    builder.add_edge(START, "context_resolver")
    builder.add_edge("context_resolver", "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tool", END: END})
    builder.add_edge("tool", "agent")

    with SqliteSaver.from_conn_string("memory.sqlite") as memory:
        graph = builder.compile(checkpointer=memory)
        graph = builder.compile()
        print_graph(graph)

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Good bye!")
                break

            if not user_input.strip():
                continue

            current_state: StateContext = {
                "user_input": user_input,
            }

            final_state = graph.invoke(
                current_state,
                config={"configurable": {"thread_id": "tool_loadout"}},
            )

            assistant_response = final_state["messages"][-1]
            print(f"Agent: {assistant_response.content}\n")

if __name__ == "__main__":
    raise SystemExit(main())
