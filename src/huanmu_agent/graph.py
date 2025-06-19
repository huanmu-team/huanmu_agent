"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import asyncio
from datetime import UTC, datetime
from typing import Dict, List, Literal, cast, Any
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from huanmu_agent.configuration import Configuration
from huanmu_agent.state import InputState, State, SalesAgentStateOutput
from huanmu_agent.tools import TOOLS
from huanmu_agent.utils.langchain_utils import load_chat_model


# Define the function that calls the model


async def call_model(state: State) -> Dict[str, Any]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_context()

    # Initialize the model with tool binding in a background thread to avoid blocking.
    model = load_chat_model(configuration.model, configuration.temperature).bind_tools(TOOLS)

    def _prepare_system_message():
        # Use Beijing time (UTC+8) instead of UTC and append the current time in Chinese.
        beijing_now = datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat()
        # First, substitute any {system_time} placeholder in the prompt if present.
        system_prompt_text = configuration.system_prompt.format(system_time=beijing_now)
        # Then, explicitly append the current time in Chinese for clarity.
        return f"目前时间：{beijing_now}\n{system_prompt_text}"

    system_message = await asyncio.to_thread(_prepare_system_message)

    # Get the model's response
    response = cast(
        AIMessage,
        await asyncio.to_thread(
            model.invoke,
            [{"role": "system", "content": system_message}, *state.messages],
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        content = "Sorry, I could not find an answer to your question in the specified number of steps."
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content=content,
                )
            ],
            "last_message": content,
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response], "last_message": response.content}


# Define a new graph

builder = StateGraph(State, input=InputState, config_schema=Configuration, output=SalesAgentStateOutput)

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
graph = builder.compile(name="HuanMu Agent")
