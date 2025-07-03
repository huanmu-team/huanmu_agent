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

# 原有的条件边已被下面的增强版替代
# builder.add_conditional_edges(...)

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

# 直接使用原有的图构建方式，避免复杂的导入
# 我们将MAS逻辑直接整合到这个文件中

# 导入MAS相关模块
try:
    from huanmu_agent.blocks import create_block
    from huanmu_agent.blocks.state_evaluator import evaluate_state
    from huanmu_agent.blocks.intent_analyzer import analyze_customer_intent, update_appointment_info
    from huanmu_agent.mas_prompts import load_prompt
    from huanmu_agent.state import EmotionalState, DebugInfo
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import json
    
    HAS_MAS_SUPPORT = True
except ImportError as e:
    print(f"Warning: MAS modules not available: {e}")
    HAS_MAS_SUPPORT = False

# === 原始MAS流水线的5个独立节点 ===

def initialize_mas_state_node(state: State) -> Dict[str, Any]:
    """MAS初始化状态节点"""
    if not HAS_MAS_SUPPORT:
        return {}
    
    # 增加对话轮次
    new_turn_count = getattr(state, 'turn_count', 0) + 1
    
    # 初始化情感状态
    current_emotional_state = getattr(state, 'emotional_state', None)
    if not current_emotional_state:
        from huanmu_agent.state import EmotionalState
        current_emotional_state = EmotionalState()
    
    return {
        "turn_count": new_turn_count,
        "emotional_state": current_emotional_state,
        "internal_monologue": [],
        "current_stage": getattr(state, 'current_stage', 'initial_contact'),
        "customer_intent_level": getattr(state, 'customer_intent_level', 'low'),
    }

def analyze_mas_sentiment_node(state: State) -> Dict[str, Any]:
    """MAS情感分析节点"""
    if not HAS_MAS_SUPPORT:
        return {}
    
    internal_monologue = getattr(state, 'internal_monologue', [])
    current_emotional_state = getattr(state, 'emotional_state', None)
    
    # 动态设定温度
    comfort = current_emotional_state.comfort_level if current_emotional_state else 0.0
    familiarity = current_emotional_state.familiarity_level if current_emotional_state else 0.0
    agent_temperature = 0.6
    if comfort > 0.6 and familiarity > 0.5:
        agent_temperature = 0.7
    elif comfort < 0.3:
        agent_temperature = 0.5
    
    internal_monologue.append(f"温度设定：根据当前情感 (舒适度:{comfort:.2f}, 熟悉度:{familiarity:.2f})，设定温度为 {agent_temperature}。")
    
    return {
        "agent_temperature": agent_temperature,
        "internal_monologue": internal_monologue,
    }

def design_mas_strategy_node(state: State) -> Dict[str, Any]:
    """MAS策略设计节点 - 核心决策逻辑"""
    if not HAS_MAS_SUPPORT:
        return {}
    
    internal_monologue = getattr(state, 'internal_monologue', [])
    
    # 1. 状态评估
    current_emotional_state = getattr(state, 'emotional_state', None)
    customer_intent = "low"
    
    try:
        evaluation_result = evaluate_state(state)
        if evaluation_result:
            new_emotional_state = evaluation_result.get("emotional_state")
            if new_emotional_state:
                current_emotional_state = new_emotional_state
            customer_intent = evaluation_result.get("customer_intent_level", "low")
    except Exception as e:
        print(f"状态评估失败: {e}")
    
    internal_monologue.append(f"情感评估完成: {current_emotional_state.model_dump_json() if current_emotional_state else 'None'}")
    internal_monologue.append(f"客户意向评估: {customer_intent}")
    
    # 2. 意图分析
    customer_intent_obj = None
    try:
        intent_result = analyze_customer_intent(state)
        customer_intent_obj = intent_result.get("customer_intent")
        if customer_intent_obj:
            internal_monologue.append(f"行为意图识别: {customer_intent_obj.intent_type} (置信度: {customer_intent_obj.confidence:.2f})")
    except Exception as e:
        print(f"意图分析失败: {e}")
    
    # 3. 阶段推进逻辑
    current_stage = getattr(state, 'current_stage', 'initial_contact')
    turn_count = getattr(state, 'turn_count', 0)
    
    if current_emotional_state:
        trust_level = current_emotional_state.trust_level
        comfort_level = current_emotional_state.comfort_level
        familiarity_level = current_emotional_state.familiarity_level
    else:
        trust_level = comfort_level = familiarity_level = 0.0
    
    new_stage = current_stage
    if current_stage == "initial_contact":
        if turn_count >= 1 and comfort_level > 0.2:
            new_stage = "ice_breaking"
    elif current_stage == "ice_breaking":
        if familiarity_level > 0.3:
            new_stage = "subtle_expertise"
    elif current_stage == "subtle_expertise":
        if trust_level > 0.4:
            new_stage = "pain_point_mining"
    elif current_stage == "pain_point_mining":
        if trust_level > 0.6 and customer_intent in ["medium", "high"]:
            new_stage = "solution_visualization"
    elif current_stage == "solution_visualization":
        if trust_level > 0.7 and customer_intent == "high":
            new_stage = "natural_invitation"
    
    if new_stage != current_stage:
        internal_monologue.append(f"自然流程推进: '{current_stage}' → '{new_stage}' (信任{trust_level:.2f}/舒适{comfort_level:.2f}/熟悉{familiarity_level:.2f})")
    
    # 4. 策略决策
    candidate_actions = []
    
    # 根据意图调整候选动作
    if customer_intent_obj and customer_intent_obj.intent_type == "info_seeking":
        candidate_actions = ["value_display", "needs_analysis"]
        internal_monologue.append("客户寻求信息，优先提供项目介绍")
    elif customer_intent_obj and customer_intent_obj.intent_type == "general_chat":
        candidate_actions = ["rapport_building"]
        internal_monologue.append("一般聊天，建立关系")
    else:
        # 基于阶段的默认策略
        if new_stage == "initial_contact":
            candidate_actions = ["greeting"]
        elif new_stage == "ice_breaking":
            candidate_actions = ["rapport_building"]
        elif new_stage == "subtle_expertise":
            candidate_actions = ["value_display"]
        elif new_stage == "pain_point_mining":
            candidate_actions = ["needs_analysis", "pain_point_test"]
        elif new_stage == "solution_visualization":
            candidate_actions = ["value_pitch", "value_display"]
        elif new_stage == "natural_invitation":
            candidate_actions = ["active_close"]
    
    # 扩展搜索空间
    if len(candidate_actions) == 1:
        primary_action = candidate_actions[0]
        internal_monologue.append(f"扩展搜索空间: {primary_action} → {candidate_actions}")
    
    # 确保搜索空间合理
    candidate_actions = list(set(candidate_actions))[:3]
    if not candidate_actions:
        candidate_actions = ["rapport_building"]
    
    decision_context = f"阶段:{new_stage}, 情感:{customer_intent}, 信任:{trust_level:.2f}"
    if customer_intent_obj:
        decision_context += f", 意图:{customer_intent_obj.intent_type}"
    internal_monologue.append(f"策略决策 ({decision_context}) -> 候选动作: {candidate_actions}")
    
    return {
        "emotional_state": current_emotional_state,
        "current_stage": new_stage,
        "customer_intent_level": customer_intent,
        "candidate_actions": candidate_actions,
        "internal_monologue": internal_monologue,
    }

def generate_mas_responses_node(state: State) -> Dict[str, Any]:
    """MAS并行生成回复节点"""
    if not HAS_MAS_SUPPORT:
        return {}
    
    internal_monologue = getattr(state, 'internal_monologue', [])
    candidate_actions = getattr(state, 'candidate_actions', [])
    agent_temperature = getattr(state, 'agent_temperature', 0.6)
    
    evaluated_responses = []
    
    def generate_single_response(action_name):
        try:
            configuration = Configuration.from_context()
            node_model = getattr(state, 'node_model', configuration.model)
            
            # 创建LangChain模型作为sampler
            model_sampler = load_chat_model(node_model, agent_temperature)
            
            block = create_block(action_name, model_sampler, node_model)
            if not block:
                return None
            
            response = block.forward(state.messages, agent_temperature)
            if not response:
                return None
            
            # 增强评估逻辑
            score = 0.5
            reasoning = "基础评估"
            
            # 阶段适配度评分
            current_stage = getattr(state, 'current_stage', 'initial_contact')
            if current_stage == "ice_breaking" and action_name == "rapport_building":
                score += 0.15
                reasoning = "阶段匹配度高"
            elif current_stage == "initial_contact" and action_name == "greeting":
                score += 0.2
                reasoning = "初次接触策略"
            
            # 内容质量评估
            if "服务" in response and action_name == "value_display":
                score += 0.1
                reasoning += "，提供服务信息"
            
            return {
                "action": action_name,
                "response": response,
                "score": min(1.0, score),
                "reasoning": reasoning
            }
        except Exception as e:
            print(f"生成回复失败 [{action_name}]: {e}")
            return None
    
    # 并行生成
    from concurrent.futures import ThreadPoolExecutor, as_completed
    max_workers = min(3, len(candidate_actions))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_action = {executor.submit(generate_single_response, action): action for action in candidate_actions}
        
        for future in as_completed(future_to_action):
            try:
                result = future.result()
                if result:
                    evaluated_responses.append(result)
                    internal_monologue.append(f"  - [{result['action']}] 生成回复: '{result['response'][:30]}...' -> 评估得分: {result['score']:.2f} (原因: {result['reasoning']})")
            except Exception as e:
                action = future_to_action[future]
                internal_monologue.append(f"  - [{action}] 生成失败: {e}")
    
    if not evaluated_responses:
        internal_monologue.append("所有候选回复生成失败，使用兜底回复")
    
    return {
        "evaluated_responses": evaluated_responses,
        "internal_monologue": internal_monologue,
    }

async def verify_mas_response_node(state: State) -> Dict[str, Any]:
    """MAS自我验证节点"""
    if not HAS_MAS_SUPPORT:
        return await call_model(state)
    
    internal_monologue = getattr(state, 'internal_monologue', [])
    evaluated_responses = getattr(state, 'evaluated_responses', [])
    
    if not evaluated_responses:
        final_response = "您好！有什么可以帮您的吗？"
        internal_monologue.append("使用兜底回复")
    elif len(evaluated_responses) == 1:
        final_response = evaluated_responses[0]['response']
        internal_monologue.append(f"自我验证：只有1个高质量选项，直接选择 '{evaluated_responses[0]['action']}'。")
    else:
        # 选择最佳回复
        best_response = max(evaluated_responses, key=lambda x: x['score'])
        final_response = best_response['response']
        internal_monologue.append(f"自我验证：从 {len(evaluated_responses)} 个选项中选择得分最高的回复 (模块: {best_response['action']}, 得分: {best_response['score']:.2f})。")
    
    internal_monologue.append("将AI的最新回复返回，由LangGraph自动更新历史。")
    
    # 生成调试信息
    debug_info = None
    if getattr(state, 'verbose', False):
        debug_info = DebugInfo(
            current_stage=getattr(state, 'current_stage', 'initial_contact'),
            emotional_state=getattr(state, 'emotional_state').model_dump() if getattr(state, 'emotional_state') else None,
            internal_monologue=internal_monologue,
        )
    
    return {
        "last_message": final_response,
        "messages": [AIMessage(content=final_response)],
        "debug_info": debug_info,
        "internal_monologue": internal_monologue,
    }

# 修改路由逻辑以支持MAS
def should_route_to_human_enhanced(state: State) -> Literal["human_assistance", "initialize_mas_state", "call_model", "__end__"]:
    """增强的路由函数，支持MAS流水线"""
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
            return "initialize_mas_state" if HAS_MAS_SUPPORT else "call_model"
        return "__end__"
        
    # 检查最后一条消息
    last_message = state.messages[-1] if state.messages else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "human_assistance"
    
    # 根据是否支持MAS决定路由
    return "initialize_mas_state" if HAS_MAS_SUPPORT else "call_model"

# 添加MAS节点到构建器
if HAS_MAS_SUPPORT:
    builder.add_node("initialize_mas_state", initialize_mas_state_node)
    builder.add_node("analyze_mas_sentiment", analyze_mas_sentiment_node)
    builder.add_node("design_mas_strategy", design_mas_strategy_node)
    builder.add_node("generate_mas_responses", generate_mas_responses_node)
    builder.add_node("verify_mas_response", verify_mas_response_node)
    
    # 连接MAS流水线的5个节点
    builder.add_edge("initialize_mas_state", "analyze_mas_sentiment")
    builder.add_edge("analyze_mas_sentiment", "design_mas_strategy")
    builder.add_edge("design_mas_strategy", "generate_mas_responses")
    builder.add_edge("generate_mas_responses", "verify_mas_response")
    builder.add_edge("verify_mas_response", "__end__")

# 更新条件边
builder.add_conditional_edges(
    "route_to_human_or_ai",
    should_route_to_human_enhanced,
    {
        "human_assistance": "human_assistance", 
        "initialize_mas_state": "initialize_mas_state" if HAS_MAS_SUPPORT else "call_model",
        "call_model": "call_model",
        "__end__": "__end__"
    },
)

# Compile the builder into an executable graph
graph = builder.compile(name="HuanMu Agent")

# =========================
# 正则表达式规范输出区域
# 该函数用于去除模型输出中的 Markdown 格式，保证最终输出为纯文本，便于展示和后续处理
# =========================
def remove_markdown(text: str) -> str:
    import re
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # 去除加粗
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # 去除斜体
    text = re.sub(r'# ', '', text)                  # 去除标题
    text = re.sub(r'^\s*[-*•] ', '', text, flags=re.MULTILINE)  # 去除无序列表
    text = re.sub(r'\d+\.\s*', '', text)            # 去除有序列表
    return text