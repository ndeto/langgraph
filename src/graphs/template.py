from pathlib import Path
import sys

from langgraph.runtime import Runtime
from langgraph.graph import StateGraph
from dotenv import load_dotenv

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.config.models import system_model
from src.lib.session import RuntimeContext, StateContext
from config.config import SysConfig, bootstrap_config
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
config: SysConfig = bootstrap_config()

def llm_call_node(state: StateContext, runtime: Runtime[RuntimeContext]):
    context = runtime.context or {
        "system_prompt": config["system_prompt"],
        "soul": config["soul"]
    }

    user_input = state["user_input"]
    model = system_model(config)
    user_message = HumanMessage(user_input)
    system_prompt = SystemMessage(context["system_prompt"])
    soul = SystemMessage(context["soul"])
    messages = [user_message, system_prompt, soul]
    response = model.invoke(messages)
    return {"llm_response": response.content}

def post_llm_node(state: StateContext, runtime: Runtime[RuntimeContext]):
    return ""


# def tool_call_node(state: StateContext, runtime:)

def main():
    """
    Learning objective:
    Understand agentic engineering as layers around a model call.

    1. Prompt engineering:
       Shape the message for one call.

    2. Context engineering:
       Choose what the model sees for that call, including system prompt,
       soul, recent messages, tools, and capabilities.

    3. Harness engineering:
       Build the code around the model so one pass is useful in a real system.
       This includes model calls, tool routing, validation, and state updates.

    4. Loop engineering:
       Decide whether the system should run again, call a tool, verify a result,
       or stop with a final answer.

    Core lessons:
    - Do not confuse the model with the system.
    - A good prompt alone is not enough.
    - A large context window is not a strategy.
    - Reliability comes from orchestration: tools, retries, verifiers,
      sub-agents, and stop conditions.
    - The model is the commodity; the loop around it is the engineering.

    Practical questions for this template:
    - What is the input for one turn?
    - What context should enter the LLM call?
    - What tools and capabilities are available?
    - How do we verify success?
    - What tells the loop to stop?
    """
    graph_builder = StateGraph(state_schema=StateContext, context_schema=RuntimeContext)
    graph_builder.add_node("respond", llm_call_node)
    graph_builder.set_entry_point("respond")
    graph_builder.set_finish_point("respond")
    graph = graph_builder.compile()

    result = graph.invoke(
        {
            "user_input": "can you help with google connection?",
            "recent_messages": {},
            "llm_response": None,
            "tool_name": None,
            "tool_input": None,
            "final_response": None,
        }
    )
    print(result)
if __name__ == "__main__":
    raise SystemExit(main())
