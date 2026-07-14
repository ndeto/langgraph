from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict


# State is the data that flows through the graph and can be updated by nodes.
class State(TypedDict):
    input_string: str
    response: str


# Context is per-invocation configuration that nodes can read at runtime.
class Context(TypedDict):
    system_prompt: str
    soul: str


# A node reads the current state/context and returns a partial state update.
def node(state: State, runtime: Runtime[Context]) -> dict[str, str]:
    context: Context = runtime.context or {
        "system_prompt": "You are helpful.",
        "soul": "",
    }
    system_prompt = context["system_prompt"]
    user_input = state["input_string"]
    return {
        "response": f"{system_prompt} | User said {user_input}"
    }


# The builder wires nodes together before the graph is compiled.
graph_builder = StateGraph(
    state_schema=State,
    context_schema=Context,
)
graph_builder.add_node("respond", node)
graph_builder.set_entry_point("respond")
graph_builder.set_finish_point("respond")
graph = graph_builder.compile()


if __name__ == "__main__":
    # Invoke the graph with concrete state and context values for one run.
    result = graph.invoke(
        {
            "input_string": "hello",
            "response": "",
        },
        context={
            "system_prompt": "You are helpful.",
            "soul": "",
        },
    )
    print(result)
