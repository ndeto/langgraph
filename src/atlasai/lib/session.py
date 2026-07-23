from typing import Annotated,TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class Facts(TypedDict):
    category: str
    sub_category: str | None
    facts: dict
    user_mood: str


class FactResults(TypedDict):
    short_term_facts: list[Facts]
    long_term_facts: list[Facts]


class MemoryUpdateResult(TypedDict):
    short_term_facts: list[Facts]
    updated_long_term_memory: list[Facts]
    user_mood: str | None


class Context(TypedDict):
    system_prompt: str | None
    soul: str | None

class StateContext(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str
    available_tools_metadata: list | None
    resolved_tools: list[str]
    recent_facts: list[Facts]
    user_mood: str | None
