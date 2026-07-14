from typing import TypedDict

class RuntimeContext(TypedDict):
    system_prompt: str | None
    soul: str | None

class StateContext(TypedDict):
    user_input: str
    recent_messages: dict[str, str]
    llm_response: str | None
    tool_name: str | None
    tool_input: str | None
    final_response: str | None



