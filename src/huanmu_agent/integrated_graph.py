"""
整合图 - 将MAS复杂对话逻辑与人工接管功能结合
既保留原有的人工接管机制，又提供MAS的智能对话能力
"""

import asyncio
from datetime import UTC, datetime
from typing import Dict, List, Literal, cast, Any
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from .configuration import Configuration
from .prompts import TRIAGE_SYSTEM_PROMPT
from .state import InputState, State, SalesAgentStateOutput, HumanControlState
from .tools import TOOLS, request_human_assistance
from .utils.langchain_utils import load_chat_model

# 导入MAS图的节点
from .mas_graph import (
    initialize_state_node,
    analyze_sentiment_node, 
    design_node,
    generate_and_evaluate_node,
    self_verification_node
)

# 人工接管相关节点（从原图复制）
async def route_to_human_or_ai(state: State) -> Dict[str, Any]:
    """
    路由节点：决定是人工处理还是AI处理
    """
    # 类型校正
    if isinstance(state.human_control, dict):
        try:
            state.human_control = HumanControlState(**state.human_control)
        except TypeError:
            state.human_control = HumanControlState(
                is_human_active=state.human_control.get("is_human_active", False),
                human_operator_id=state.human_control.get("human_operator_id"),
                transfer_reason=state.human_control.get("transfer_reason"),
                transfer_time=state.human_control.get("transfer_time"),
            )

    # 检查唤醒命令
    last_message = state.messages[-1] if state.messages else None
    if isinstance(last_message, HumanMessage) and last_message.content == "起床了小七":
        return {
            "human_control": HumanControlState(is_human_active=False),
            "messages": [AIMessage(content="好\n(●`ε´●)", name="resume_ai_confirmation")],
            "last_message": "好\n(●`ε´●)",
        }

    # 检查是否已经处于人工接管状态
    if state.human_control.is_human_active:
        if isinstance(last_message, HumanMessage):
            return {"last_message": ""}
        return {"last_message": ""}
        
    configuration = Configuration.from_context()
    triage_model = load_chat_model(configuration.model, 0.0).bind_tools([request_human_assistance])
    
    response = cast(
        AIMessage,
        await asyncio.to_thread(
            triage_model.invoke,
            [{"role": "system", "content": TRIAGE_SYSTEM_PROMPT}, last_message],
        ),
    )
    
    if response.tool_calls:
        return {"messages": [response], "last_message": ""}
    return {"last_message": ""}

async def enter_human_takeover(state: State) -> dict:
    """进入人工接管状态"""
    tool_messages = []
    last_message = state.messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_messages.append(
                ToolMessage(
                    tool_call_id=tool_call["id"],
                    content=f"Human assistance requested with query: {tool_call['args']['query']}",
                    name=tool_call["name"],
                )
            )

    return {
        "messages": tool_messages,
        "human_control": HumanControlState(is_human_active=True),
        "last_message": "",
    }

def should_route_to_human(state: State) -> Literal["human_assistance", "mas_pipeline", "__end__"]:
    """决定下一步的路由"""
    # 类型校正
    if isinstance(state.human_control, dict):
        try:
            state.human_control = HumanControlState(**state.human_control)
        except TypeError:
            state.human_control = HumanControlState(
                is_human_active=state.human_control.get("is_human_active", False),
                human_operator_id=state.human_control.get("human_operator_id"),
                transfer_reason=state.human_control.get("transfer_reason"),
                transfer_time=state.human_control.get("transfer_time"),
            )
    
    # 如果已经处于人工接管状态，直接结束
    if state.human_control.is_human_active:
        last_message = state.messages[-1] if state.messages else None
        if isinstance(last_message, HumanMessage) and last_message.content == "起床了小七":
            return "mas_pipeline"  # 唤醒命令会在route_to_human_or_ai中处理
        return "__end__"
        
    # 检查最后一条消息
    last_message = state.messages[-1] if state.messages else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "human_assistance"
    
    # 否则进入MAS流水线
    return "mas_pipeline"

# MAS流水线节点 - 将MAS的5个节点串联起来
def mas_pipeline_node(state: State) -> Dict[str, Any]:
    """
    MAS流水线节点 - 执行完整的MAS对话逻辑
    """
    # 1. 初始化状态
    state_updates = initialize_state_node(state)
    for key, value in state_updates.items():
        if hasattr(state, key):
            setattr(state, key, value)
    
    # 2. 情感分析
    sentiment_updates = analyze_sentiment_node(state)
    for key, value in sentiment_updates.items():
        if hasattr(state, key):
            setattr(state, key, value)
    
    # 3. 策略设计
    design_updates = design_node(state)
    for key, value in design_updates.items():
        if hasattr(state, key):
            setattr(state, key, value)
    
    # 4. 生成和评估
    eval_updates = generate_and_evaluate_node(state)
    for key, value in eval_updates.items():
        if hasattr(state, key):
            setattr(state, key, value)
    
    # 5. 自我验证
    final_updates = self_verification_node(state)
    
    return final_updates

def remove_markdown(text: str) -> str:
    """去除Markdown格式"""
    import re
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # 去除加粗
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # 去除斜体
    text = re.sub(r'# ', '', text)                  # 去除标题
    text = re.sub(r'^\s*[-*•] ', '', text, flags=re.MULTILINE)  # 去除无序列表
    text = re.sub(r'\d+\.\s*', '', text)            # 去除有序列表
    return text

def build_integrated_graph() -> StateGraph:
    """构建整合图"""
    builder = StateGraph(State, input=InputState, config_schema=Configuration, output=SalesAgentStateOutput)
    
    # 添加节点
    builder.add_node("route_to_human_or_ai", route_to_human_or_ai)
    builder.add_node("human_assistance", enter_human_takeover)
    builder.add_node("mas_pipeline", mas_pipeline_node)
    builder.add_node("tools", ToolNode(TOOLS))
    
    # 设置入口点
    builder.add_edge(START, "route_to_human_or_ai")
    
    # 添加条件边
    builder.add_conditional_edges(
        "route_to_human_or_ai",
        should_route_to_human,
        {
            "human_assistance": "human_assistance", 
            "mas_pipeline": "mas_pipeline",
            "__end__": "__end__"
        },
    )
    
    # MAS流水线直接结束
    builder.add_edge("mas_pipeline", "__end__")
    
    # 人工接管后结束
    builder.add_edge("human_assistance", "__end__")
    
    return builder

# 编译图
integrated_graph = build_integrated_graph().compile(name="Integrated MAS Agent") 