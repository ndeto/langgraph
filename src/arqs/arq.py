import os
from pathlib import Path
import sys
from typing import Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, SecretStr

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser

SRC_ROOT = Path(__file__).resolve().parents[2]
RAG_LESSON_ROOT = SRC_ROOT / "lessons" / "03_rag"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(RAG_LESSON_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_LESSON_ROOT))

from src.rag.retrieval import _get_retriever
from langchain_core.tools import tool

load_dotenv()

class State(TypedDict, total=False):
    user_input: str | None
    arq: dict | None
    tool_result: str | None
    final_answer: str
    query: str

class ToolPlan(BaseModel):
    needs_tool: bool = Field(description="Whether a tool is required.")
    current_context: str = Field(description="Short summary of user's situation.")
    active_constraints: list[str] = Field(description="Constraints that matter right now")
    tool_name: str | None = Field(default=None, description="Tool to call if any")
    tool_input: str | None = Field(default=None, description="Args for the tool")
    answer_strategy: str = Field(description="How should the assistant respond next")

api_key: str | None = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("No Key")

retriever = _get_retriever()

@tool
def retrieve_blog_posts(query: str) -> str:
    """Search and return info"""
    retriever = _get_retriever()
    retrieved_docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in retrieved_docs])

retrieve_blog_post_tool = retrieve_blog_posts

llm = ChatOpenAI(
    api_key=SecretStr(api_key),
    model="gpt-5.4-mini",
)

reasoner = llm.bind_tools(tools=[retrieve_blog_post_tool], response_format=ToolPlan)

def reason_node(state: State) -> dict:
    reason_template = ChatPromptTemplate.from_template(
        """
        User Input: {user_input}
        You are filling an Attentive Reasoning Query form.
        Before answering:
        - restate the current context
        - identify active constraints
        - decide if a tool is needed
        - if needed, name the tool and tool input
        - describe the answer strategy                     
        """)
    
    user_input = state.get("user_input") or "Yoh"

    reasoner_summary = reason_template | reasoner

    plan = reasoner_summary.invoke({"user_input": user_input})

    return {"arq": plan}

def route_after_reason(state: State) -> Literal["tool_node", "final_node"]:
    if state.get("arq", "needs_tool"):
        return "tool_node"
    return "final_node"

@tool
def tool_node(state: State) -> dict:
    tool_name = state.get("arq", "tool_name")
    tool_input = state.get("arg", "tool_input")

    if tool_name == "search_docs":
        result = f"Mock doc results for: {tool_input}"
    else:
        result = "Unknown tool"
    
    return {"tool_result": result}

def final_node(state: State) -> dict:
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are answering the user.

        User Input: {user_input}

        Current Context: {current_context}

        Constraints: {active_constraints}

        Tool Result: {tool_result}

        Answer Strategy: {answer_strategy}
        """
    )

    prompt_summarizer = prompt_template | llm

    res = prompt_summarizer.invoke({
        "user_input": state.get("user_input"),
        "current_context": state.get("arq", "current_context"),
        "active_constraints": state.get("arq", "active_contraints"),
        "tool_result": state.get("tool_result"),
        "answer_strategy": state.get("arq", "current_context"),
    })

    return {"final_answer": res.content}

graph_builder = StateGraph(state_schema=State)
graph_builder.add_node(reason_node)
graph_builder.add_node(tool_node)
graph_builder.add_node(final_node)

graph_builder.add_edge(START, "reason_node")
graph_builder.add_conditional_edges("reason_node", route_after_reason, ["tool_node", "final_node"])
graph_builder.add_edge("tool_node", "final_node")
graph_builder.add_edge("final_node", END)

graph = graph_builder.compile()

def main():
    res = graph.invoke({
            "user_input": "Who is Martin",
            "arq": {
                "needs_tool": False,
                "current_context": "",
                "active_constraints": [],
                "tool_name": [retrieve_blog_posts],
                "tool_input": None,
                "answer_strategy": "",
            }
            })

    answer = res.get("final_answer")
    print(answer)
    
if __name__ == "__main__":
    main()
        
