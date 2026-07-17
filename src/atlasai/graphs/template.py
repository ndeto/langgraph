from typing import Literal, TypedDict
from atlasai.config.models import system_model
from atlasai.lib.session import Context, GraderResult, StateContext
from atlasai.config.sys_config import SysConfig, bootstrap_config
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import ToolNode

load_dotenv()
config: SysConfig = bootstrap_config()
model = system_model(config)


def load_user_info():
    url = "https://ndeto.eth.limo"

    data = UnstructuredLoader(web_url=url).load()

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )

    chunks = splitter.split_documents(data)

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


search_user_info_tool = search_user_info


def search_user_info_node(state: StateContext):
    if "tool_calls" in state and len(state["tool_calls"]) > 0:
        query = state["tool_calls"][0]["args"]
        res = search_user_info_tool.invoke(query)
        if res is not None:
            return {"tool_results": res}


def research_user_node(state: StateContext, runtime: Runtime[Context]):
    refine_prompt = ""

    if  state.get("grader_result") is not None:
        refine_prompt = SystemMessage(
            content=f"""Results already exist but they are insufficiaent, expand the search query to produce relevant results
                                                    The existing tool search results are: {state.get("tool_results")}
                                                    Grader Recommendation and Result: {state.get("grader_result")}
                                                    """
        )

    context = runtime.context or {
        "system_prompt": config["system_prompt"],
        "soul": config["soul"],
    }

    if "user_input" not in state:
        raise ValueError("No user input")

    user_input = state.get("user_input")
    user_message = HumanMessage(user_input)
    system_prompt = SystemMessage(context["system_prompt"])

    soul = SystemMessage(context["soul"])
    messages = [user_message, system_prompt, soul, refine_prompt]
    response = model.bind_tools([search_user_info_tool]).invoke(messages)
    tool_calls = []

    for t in response.tool_calls:
        if "name" in t:
            tool_calls.append({"name": t["name"], "args": t["args"]})

    return {"tool_calls": tool_calls}


def grader_node(state: StateContext):
    prompt_template = PromptTemplate.from_template(
        """
        SYSTEM: Your work is to evaluate the relevance of the data against the systesm prompt: {system_prompt}
        This is the users original input: {user_input}
        The tool results are: {tool_results}
        
        Your work is to determine whether it meets criteria and return the following single strings for field res:
        "generate_node": If the info and context is sufficient
        "research_user_node": If you deem it does not meet the criteria

        For field: "sentiment" : Add suggestions on more info needed to make this query valid or more relevant

        class GraderResult(TypedDict):
            res: Literal["generate_node", "research_user_node"]
            sentiment: str
        """
    )

    prompt_summary = prompt_template | model.with_structured_output(GraderResult)
    user_input = ""
    tool_results = ""
    
    if "user_input" in state:
        user_input = state["user_input"] or None
        
    if "tool_results" in state:
        tool_results = state["tool_results"] or None
    

    res = prompt_summary.invoke(
        {
            "system_prompt": config["system_prompt"],
            "user_input": user_input,
            "tool_results": tool_results,
        }
    )

    refine_iterations = state.get("refine_iterations", 0)
    if res["res"] == "research_user_node":
        refine_iterations += 1

    return {"grader_result": res, "refine_iterations": refine_iterations}


def route_after_grader(
    state: StateContext,
) -> Literal["generate_node", "research_user_node"]:
    if state.get("refine_iterations", 0) >= 3:
        return "generate_node"

    grader_result = state.get("grader_result")
    if grader_result is None:
        return "generate_node"

    return grader_result["res"]


def generate_node(state: StateContext):
    """Generate Final Answer"""
    prompt = PromptTemplate.from_template("""
    SYSTEM: Your work is to generate the final human friendly answer to the users questions: 
        The system prompt was: {system_prompt}
        This is the users original input: {user_input}
        The tool results are: {tool_results}
        The Grading logic answer is {grader_result}
        SOUL: {soul}
                                          
        Use the Soul instructions to learn the personality to respond in:
    """)

    prompt_sum = prompt | model

    res = prompt_sum.invoke(
        {
            "system_prompt": config["system_prompt"],
            "soul": config["soul"],
            "user_input": state.get("user_input"),
            "tool_results": state.get("tool_results"),
            "grader_result": state.get("grader_result"),
        }
    )

    return {"final_response": res.content}
    
def main():
    builder = StateGraph(state_schema=StateContext, context_schema=Context)
    builder.add_node(research_user_node)
    builder.add_node(search_user_info_node)
    builder.add_node(grader_node)
    builder.add_node(generate_node)

    builder.add_edge(START, "research_user_node")
    builder.add_edge("research_user_node", "search_user_info_node")
    builder.add_edge("search_user_info_node", "grader_node")
    builder.add_conditional_edges("grader_node", route_after_grader)
    builder.add_edge("generate_node", END)

    graph = builder.compile()

    input: StateContext = {"user_input": "Who is Ndeto"}

    res = graph.invoke(input)

    print(res.get("final_response"))


if __name__ == "__main__":
    raise SystemExit(main())
