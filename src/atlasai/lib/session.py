from typing import Any, Literal, TypedDict
from langchain_core.documents import Document

class GraderResult(TypedDict):
    res: Literal["generate_node", "research_user_node"]
    sentiment: str

class Context(TypedDict):
    system_prompt: str | None
    soul: str | None

class StateContext(TypedDict, total=False):
    user_input: str
    recent_messages: dict[str, str] | None
    llm_response: str | list[str | dict[Any, Any]]
    tool_calls: list[dict]
    final_response: str | None
    tool_results: list[Document] | None
    grader_result: GraderResult | None
    refine_iterations: int



