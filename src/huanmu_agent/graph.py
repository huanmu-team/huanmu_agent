"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import asyncio
from datetime import UTC, datetime
from typing import Dict, List, Literal, cast, Any
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from huanmu_agent.configuration import Configuration
from huanmu_agent.prompts import TRIAGE_SYSTEM_PROMPT
from huanmu_agent.state import InputState, State, SalesAgentStateOutput, HumanControlState
from huanmu_agent.tools import TOOLS, request_human_assistance
from huanmu_agent.utils.langchain_utils import load_chat_model

# Define the new routing node
async def route_to_human_or_ai(state: State) -> Dict[str, Any]:
    """
    Routes the request to a human or the AI.
    This is the new entrypoint to the graph.
    """
    # ---- 类型校正：如果 human_control 被 REST API 更新成普通 dict，则自动转回数据类 ----
    if isinstance(state.human_control, dict):
        try:
            state.human_control = HumanControlState(**state.human_control)  # type: ignore[arg-type]
        except TypeError:
            # 字段不完整时也容忍；只取可能存在的键
            state.human_control = HumanControlState(
                is_human_active=state.human_control.get("is_human_active", False),
                human_operator_id=state.human_control.get("human_operator_id"),
                transfer_reason=state.human_control.get("transfer_reason"),
                transfer_time=state.human_control.get("transfer_time"),
            )

    # Check for a resume command first
    last_message = state.messages[-1]
    if isinstance(last_message, HumanMessage) and last_message.content == "起床了小七":
        return {
            "human_control": HumanControlState(is_human_active=False),
            "messages": [
                AIMessage(
                    content="好\n(●`ε´●)",
                    name="resume_ai_confirmation",
                )
            ],
            "last_message": "好\n(●`ε´●)",
        }

    # 检查是否已经处于人工接管状态
    if state.human_control.is_human_active:
        # 在人工接管状态下，静默忽略新的用户消息
        # 不添加任何内容到消息历史，直接结束流程
        if isinstance(last_message, HumanMessage):
            # 返回空状态，让系统直接结束，不处理用户消息
            return {"last_message": ""}
        
        # 如果不是用户消息（比如状态更新），则允许通过
        return {"last_message": ""}
        
    configuration = Configuration.from_context()
    triage_model = load_chat_model(configuration.model, 0.0).bind_tools(
        [request_human_assistance]
    )
    
    response = cast(
        AIMessage,
        await asyncio.to_thread(
            triage_model.invoke,
            [{"role": "system", "content": TRIAGE_SYSTEM_PROMPT}, state.messages[-1]],
        ),
    )
    
    # If the router decides to call a tool, it means we need human assistance.
    # We return the response to be added to the state. The graph will then
    # route to the human assistance tool node.
    if response.tool_calls:
        return {"messages": [response], "last_message": ""}
    # Otherwise, we don't add anything to the state and proceed to the main model.
    # The return value is empty, so state remains unchanged.
    return {"last_message": ""}


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
    # 类型校正，防止 human_control 是 dict
    if isinstance(state.human_control, dict):
        try:
            state.human_control = HumanControlState(**state.human_control)  # type: ignore[arg-type]
        except TypeError:
            state.human_control = HumanControlState(
                is_human_active=state.human_control.get("is_human_active", False),
                human_operator_id=state.human_control.get("human_operator_id"),
                transfer_reason=state.human_control.get("transfer_reason"),
                transfer_time=state.human_control.get("transfer_time"),
            )

    # 如果处于人工接管状态，直接返回人工回复，不调用AI模型
    if state.human_control.is_human_active:
        last_message = state.messages[-1]
        
        # 如果最后一条消息是人工发送的，直接返回
        if isinstance(last_message, HumanMessage):
            return {
                "messages": [AIMessage(content="人工客服正在处理您的问题...")],
                "last_message": "人工客服正在处理您的问题...",
            }
        else:
            # 如果不是人工消息，返回现有的最后消息
            return {
                "messages": [last_message],
                "last_message": last_message.content if isinstance(last_message.content, str) else "",
            }
        
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

    cleaned_messages = state.messages
    # Get the model's response using cleaned messages
    response = cast(
        AIMessage,
        await asyncio.to_thread(
            model.invoke,
            [{"role": "system", "content": system_message}, *cleaned_messages],
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

    # ========== 正则表达式规范输出：去除Markdown格式 ==========
    # 对主回复做去Markdown格式处理，保证输出为纯文本
    last_message_content = response.content if isinstance(response.content, str) else str(response.content) if response.content else ""
    last_message_content = remove_markdown(last_message_content)
    # 对AI回复内容本身也做去Markdown处理，防止后续流程中再次出现格式符号
    response.content = remove_markdown(response.content) if isinstance(response.content, str) else response.content
    # ========== 正则表达式规范输出结束 ==========

    return {"messages": [response], "last_message": last_message_content}


# Define a new graph
builder = StateGraph(State, input=InputState, config_schema=Configuration, output=SalesAgentStateOutput)

# Define the nodes
builder.add_node("route_to_human_or_ai", route_to_human_or_ai)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Add a dedicated node for the human assistance tool
async def enter_human_takeover(state: State) -> dict:
    """
    This node is responsible for entering the human takeover state.
    It does two things:
    1. It creates a ToolMessage for each tool call in the last AIMessage.
    2. It sets the `is_human_active` flag to True.
    """
    tool_messages = []
    last_message = state.messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            # Here we can call the actual tool, but for now we'll just mock the response
            # with the query that was passed to the tool.
            tool_messages.append(
                ToolMessage(
                    tool_call_id=tool_call["id"],
                    content=f"Human assistance requested with query: {tool_call['args']['query']}",
                    name=tool_call["name"],
                )
            )

    return {
        "messages": tool_messages,
        "human_control": HumanControlState(
            is_human_active=True
        ),
        "last_message": "",
    }

builder.add_node("human_assistance", enter_human_takeover)

# Set the entrypoint to the new router node
builder.add_edge(START, "route_to_human_or_ai")

def should_route_to_human(state: State) -> Literal["human_assistance", "call_model", "__end__"]:
    """Determines whether the next step is human assistance or the main AI model."""
    # ---- 类型校正 ----
    if isinstance(state.human_control, dict):
        try:
            state.human_control = HumanControlState(**state.human_control)  # type: ignore[arg-type]
        except TypeError:
            state.human_control = HumanControlState(
                is_human_active=state.human_control.get("is_human_active", False),
                human_operator_id=state.human_control.get("human_operator_id"),
                transfer_reason=state.human_control.get("transfer_reason"),
                transfer_time=state.human_control.get("transfer_time"),
            )
    
    # 如果已经处于人工接管状态，直接结束，不进行任何处理
    if state.human_control.is_human_active:
        last_message = state.messages[-1]
        # 只有唤醒命令才允许继续处理
        if isinstance(last_message, HumanMessage) and last_message.content == "起床了小七":
            return "call_model"  # 唤醒命令会在route_to_human_or_ai中处理
        return "__end__"
        
    # Check the last message in the state
    last_message = state.messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # If the last message has tool calls, it's from our router.
        # The tool should be request_human_assistance.
        return "human_assistance"
    # Otherwise, proceed to the main AI model
    return "call_model"

# Add a conditional edge from the router
builder.add_conditional_edges(
    "route_to_human_or_ai",
    should_route_to_human,
    {
        "human_assistance": "human_assistance", 
        "call_model": "call_model",
        "__end__": "__end__"
    },
)

def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    # ---- 类型校正 ----
    if isinstance(state.human_control, dict):
        try:
            state.human_control = HumanControlState(**state.human_control)  # type: ignore[arg-type]
        except TypeError:
            state.human_control = HumanControlState(
                is_human_active=state.human_control.get("is_human_active", False),
                human_operator_id=state.human_control.get("human_operator_id"),
                transfer_reason=state.human_control.get("transfer_reason"),
                transfer_time=state.human_control.get("transfer_time"),
            )
    # 如果处于人工接管状态，直接结束
    if state.human_control.is_human_active:
        return "__end__"
        
    last_message = state.messages[-1]
    
    # End the conversation if AI confirms resumption
    if isinstance(last_message, AIMessage) and last_message.name == "resume_ai_confirmation":
        return "__end__"
    
    # 如果最后一条消息是工具响应，且是人工处理完成的响应，直接结束
    if (isinstance(last_message, ToolMessage) and 
        last_message.name == "request_human_assistance" and 
        "人工处理已完成" in last_message.content):
        return "__end__"
    
    if not isinstance(last_message, AIMessage):
        return "__end__"
    
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
        
    # Otherwise we execute the requested actions
    return "tools"

# The main conversation loop
builder.add_conditional_edges(
    "call_model",
    route_model_output,
)
builder.add_edge("tools", "call_model")

# After human assistance, we loop back to the main model
builder.add_edge("human_assistance", "__end__")

# Compile the builder into an executable graph
graph = builder.compile(name="HuanMu Agent")

# =========================
# 正则表达式规范输出区域
# 该函数用于去除模型输出中的 Markdown 格式，保证最终输出为纯文本，便于展示和后续处理
# =========================
def remove_markdown(text: str) -> str:
    import re
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # 去除加粗（**文本**）
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # 去除斜体（*文本*）
    text = re.sub(r'# ', '', text)                  # 去除标题（# 标题）
    text = re.sub(r'- ', '', text)                  # 去除无序列表（- 列表项）
    text = re.sub(r'\d+\. ', lambda m: m.group(0).replace('*', ''), text)  # 去除有序列表前的*
    return text