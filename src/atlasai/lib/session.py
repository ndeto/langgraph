from typing import Annotated,TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import BaseTool
class Context(TypedDict):
    system_prompt: str | None
    soul: str | None

class StateContext(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str
    available_tools_metadata: list | None
    resolved_tools: list[BaseTool]