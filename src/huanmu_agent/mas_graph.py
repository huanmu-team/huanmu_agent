"""
MAS Agent Graph - 整合mas_graph.py的复杂对话逻辑到当前框架
保留人工接管功能，使用当前框架的模型调用方式
"""
import os
import json
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from .state import State, EmotionalState, DebugInfo, SalesAgentStateOutput
from .utils.langchain_utils import load_chat_model
from .mas_prompts import load_prompt

# 从blocks导入必要模块
from .blocks import create_block
from .blocks.state_evaluator import evaluate_state
from .blocks.intent_analyzer import analyze_customer_intent, update_appointment_info

from langgraph.graph import StateGraph, END, START

def initialize_state_node(state: State) -> Dict[str, Any]:
    """
    初始化状态节点 - 为图的执行提供所有必要的默认值
    """
    # 处理user_input，转换为HumanMessage（兼容性处理）
    new_messages_to_add = []
    if hasattr(state, 'user_input') and state.user_input:
        new_messages_to_add.append(HumanMessage(content=state.user_input))
    
    # 增加对话轮次
    new_turn_count = getattr(state, 'turn_count', 0) + 1
    
    # 重置运行时状态
    updated_state = {
        "turn_count": new_turn_count,
        "internal_monologue": [],
        "candidate_actions": [],
        "evaluated_responses": [],
        "final_response": "",
    }
    
    # 清除user_input（如果存在）
    if hasattr(state, 'user_input'):
        updated_state["user_input"] = None
    
    # 如果有新消息要添加，更新messages
    if new_messages_to_add:
        updated_state["messages"] = new_messages_to_add
    
    return updated_state

def analyze_sentiment_node(state: State) -> Dict[str, Any]:
    """
    情感分析节点 - 根据情感状态动态调整温度
    """
    internal_monologue = state.internal_monologue or []
    emotional_state = state.emotional_state
    verbose = state.verbose
    
    if not emotional_state:
        return {"agent_temperature": 0.6}
    
    # 基于舒适度和熟悉度设定温度
    comfort = emotional_state.comfort_level
    familiarity = emotional_state.familiarity_level
    
    agent_temperature = 0.6  # 默认值
    if comfort > 0.6 and familiarity > 0.5:
        agent_temperature = 0.6  # 更富创造性
    elif comfort < 0.3:
        agent_temperature = 0.6  # 更保守
    
    new_monologue = internal_monologue + [
        f"温度设定：根据当前情感 (舒适度:{comfort:.2f}, 熟悉度:{familiarity:.2f})，设定温度为 {agent_temperature}。"
    ]
    
    if verbose:
        print(f"[DEBUG] 情感分析节点: 温度设定为 {agent_temperature}")
    
    return {
        "agent_temperature": agent_temperature,
        "internal_monologue": new_monologue,
    }

def design_node(state: State) -> Dict[str, Any]:
    """
    智能决策节点 - 基于情感状态和客户意图选择对话策略
    """
    internal_monologue = state.internal_monologue or []
    verbose = state.verbose
    
    # 1. 状态评估
    evaluation_result = evaluate_state(state)
    current_emotional_state = evaluation_result.get("emotional_state", state.emotional_state)
    customer_intent = evaluation_result.get("customer_intent_level", state.customer_intent_level)
    
    # 2. 客户意图分析
    intent_result = analyze_customer_intent(state)
    current_customer_intent = intent_result.get("customer_intent")
    
    # 3. 预约信息更新
    appointment_updates = {}
    if current_customer_intent:
        appointment_updates = update_appointment_info(state, current_customer_intent)
    
    current_appointment_info = state.appointment_info
    if appointment_updates.get("appointment_info"):
        current_appointment_info = appointment_updates["appointment_info"]
    
    internal_monologue.append(f"情感评估完成: {current_emotional_state.model_dump_json()}")
    internal_monologue.append(f"客户意向评估: {customer_intent}")
    if current_customer_intent:
        internal_monologue.append(f"行为意图识别: {current_customer_intent.intent_type} (置信度: {current_customer_intent.confidence:.2f})")
    
    # 4. 对话阶段推进逻辑
    current_stage = state.current_stage
    trust_level = current_emotional_state.trust_level
    comfort_level = current_emotional_state.comfort_level
    familiarity_level = current_emotional_state.familiarity_level
    turn_count = state.turn_count
    
    new_stage = current_stage
    
    # 阶段推进逻辑
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
    
    # 自然回退机制
    if comfort_level < 0.3 and current_stage not in ["initial_contact", "ice_breaking"]:
        new_stage = "ice_breaking"
        internal_monologue.append(f"检测到舒适度过低 ({comfort_level:.2f})，自然回退到轻松破冰")
    
    if new_stage != current_stage:
        internal_monologue.append(f"自然流程推进: '{current_stage}' → '{new_stage}'")
    
    # 5. 候选动作决策
    candidate_actions = []
    
    # 基于当前阶段决定候选动作
    if new_stage == "initial_contact":
        candidate_actions = ["greeting"]
    elif new_stage == "ice_breaking":
        candidate_actions = ["rapport_building"]
    elif new_stage == "subtle_expertise":
        candidate_actions = ["value_display"]
        if familiarity_level > 0.4:
            candidate_actions.append("needs_analysis")
    elif new_stage == "pain_point_mining":
        candidate_actions = ["needs_analysis", "pain_point_test"]
        if trust_level > 0.6:
            candidate_actions.append("value_display")
    elif new_stage == "solution_visualization":
        candidate_actions = ["value_pitch", "value_display"]
        if customer_intent == "high":
            candidate_actions.append("active_close")
    elif new_stage == "natural_invitation":
        candidate_actions = ["active_close"]
        if customer_intent != "high":
            candidate_actions.append("value_pitch")
    
    # 确保至少有一个候选动作
    if not candidate_actions:
        candidate_actions = ["rapport_building"]
    
    internal_monologue.append(f"策略决策 -> 候选动作: {candidate_actions}")
    
    if verbose:
        print(f"[DEBUG] 策略设计节点: 客户意向={customer_intent}, 信任度={trust_level:.2f}")
    
    result = {
        "emotional_state": current_emotional_state,
        "current_stage": new_stage,
        "customer_intent_level": customer_intent,
        "candidate_actions": candidate_actions,
        "internal_monologue": internal_monologue,
    }
    
    if current_customer_intent:
        result["customer_intent"] = current_customer_intent
    if current_appointment_info:
        result["appointment_info"] = current_appointment_info
    
    return result

def _format_messages(messages: List[Any]) -> str:
    """将 LangChain BaseMessage 对象的列表格式化为单个字符串。"""
    if not messages:
        return "（无历史记录）"
    
    formatted_string = ""
    for message in messages:
        role = "客户" if message.type == "human" else "林医生"
        formatted_string += f"{role}: {message.content}\n"
    return formatted_string.strip()

def _generate_and_evaluate_action(action: str, state: State) -> tuple:
    """为单个动作生成并评估回复"""
    messages = state.messages
    node_model = state.node_model
    agent_temperature = state.agent_temperature
    feedback_model = state.feedback_model
    
    try:
        # 加载模型
        model = load_chat_model(node_model, agent_temperature)
        
        # 创建对话模块
        block = create_block(action, lambda msgs, temp, **kwargs: (model.invoke(msgs).content, 0), node_model)
        if not block:
            return None, f"模块 '{action}' 创建失败，已跳过。"
        
        # 生成回复
        response = block.forward(list(messages), agent_temperature)
        if response is None:
            return None, f"模块 '{action}' 执行失败，已跳过。"
        
        # 简化评估：使用基础规则评估
        score = 0.7  # 默认分数
        reasoning = "生成成功"
        
        evaluated_response = {
            "action": action,
            "response": response,
            "score": score,
            "reasoning": reasoning
        }
        
        monologue_entry = f"  - [{action}] 生成回复: '{response[:30]}...' -> 评估得分: {score}"
        return evaluated_response, monologue_entry
        
    except Exception as e:
        return None, f"  - [{action}] 处理时出现异常: {e}"

def generate_and_evaluate_node(state: State) -> Dict[str, Any]:
    """并行生成和评估候选回复"""
    internal_monologue = state.internal_monologue or []
    candidate_actions = state.candidate_actions or []
    
    evaluated_responses = []
    new_monologue = list(internal_monologue)
    
    # 限制并发数量
    max_concurrent_requests = min(3, len(candidate_actions) or 1)
    
    with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
        future_to_action = {
            executor.submit(_generate_and_evaluate_action, action, state): action 
            for action in candidate_actions
        }
        
        for future in as_completed(future_to_action):
            try:
                evaluated_response, monologue_entry = future.result()
                if evaluated_response:
                    evaluated_responses.append(evaluated_response)
                if monologue_entry:
                    new_monologue.append(monologue_entry)
            except Exception as e:
                action = future_to_action[future]
                new_monologue.append(f"  - [{action}] 在并行执行中捕获到致命错误: {e}")
    
    if state.verbose:
        print(f"[DEBUG] 生成评估节点: 评估了 {len(evaluated_responses)} 个候选回复")
    
    if not evaluated_responses:
        new_monologue.append("所有模块都执行失败了，使用紧急兜底回复")
        final_response = "抱歉，我现在有点问题，能稍后再联系我吗？"
        return {
            "final_response": final_response,
            "last_message": final_response,
            "internal_monologue": new_monologue
        }
    
    return {
        "evaluated_responses": evaluated_responses,
        "internal_monologue": new_monologue,
    }

def self_verification_node(state: State) -> Dict[str, Any]:
    """从评估过的候选项中选择最佳响应"""
    evaluated_responses = state.evaluated_responses or []
    internal_monologue = state.internal_monologue or []
    
    # 选择得分最高的回复
    if not evaluated_responses:
        fallback_response = "嗯嗯，好的"
        new_monologue = internal_monologue + ["自我验证失败：没有可供选择的候选回复，使用紧急回复"]
        return {
            "final_response": fallback_response,
            "last_message": fallback_response,
            "messages": [AIMessage(content=fallback_response)],
            "internal_monologue": new_monologue
        }
    
    if len(evaluated_responses) == 1:
        final_response = evaluated_responses[0]['response']
        new_monologue = internal_monologue + [f"自我验证：只有1个选项，直接选择 '{evaluated_responses[0]['action']}'。"]
    else:
        best_response = sorted(evaluated_responses, key=lambda x: x['score'], reverse=True)[0]
        final_response = best_response['response']
        new_monologue = internal_monologue + [
            f"自我验证：从 {len(evaluated_responses)} 个选项中选择得分最高的回复 (模块: {best_response['action']}, 得分: {best_response['score']:.2f})。"
        ]
    
    new_monologue.append("将AI的最新回复返回，由LangGraph自动更新历史。")
    
    if state.verbose:
        print(f"[DEBUG] 最终回复: {final_response}")
    
    # 生成调试信息
    debug_info = None
    if state.verbose:
        debug_info = DebugInfo(
            current_stage=state.current_stage,
            emotional_state=state.emotional_state.model_dump() if state.emotional_state else None,
            internal_monologue=new_monologue,
        )
    
    return {
        "final_response": final_response,
        "last_message": final_response,
        "messages": [AIMessage(content=final_response)],
        "internal_monologue": new_monologue,
        "debug_info": debug_info,
    }

def build_mas_graph() -> StateGraph:
    """构建MAS代理图"""
    workflow = StateGraph(State, output=SalesAgentStateOutput)
    
    # 添加节点
    workflow.add_node("initialize_state", initialize_state_node)
    workflow.add_node("analyze_sentiment", analyze_sentiment_node)
    workflow.add_node("meta_design", design_node)
    workflow.add_node("generate_and_evaluate", generate_and_evaluate_node)
    workflow.add_node("self_verify", self_verification_node)
    
    # 设置入口点
    workflow.set_entry_point("initialize_state")
    
    # 添加边
    workflow.add_edge("initialize_state", "analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "meta_design")
    workflow.add_edge("meta_design", "generate_and_evaluate")
    workflow.add_edge("generate_and_evaluate", "self_verify")
    workflow.add_edge("self_verify", END)
    
    return workflow

# 编译图
mas_app = build_mas_graph().compile() 