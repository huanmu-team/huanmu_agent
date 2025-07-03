import sys
import os
import json
from typing import List, Dict, Any, TypedDict, NotRequired
from operator import itemgetter
from concurrent.futures import ThreadPoolExecutor, as_completed

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver  # 在LangGraph Cloud中不需要，会自动提供
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

# from sampler.chat_completion_sampler import ChatCompletionSampler
# from sampler.o_chat_completion_sampler import OChatCompletionSampler
# from sampler.together_completion_sampler import ChatCompletionSampler as ToChatCompletionSampler
# from sampler.vllm_completion_sampler import ChatCompletionSampler as VllmChatCompletionSampler
from sampler.factory import SamplerFactory # 导入新的采样器工厂
from blocks import create_block
from utils import extract_xml
from common import AgentState, EmotionalState, MasAgentOutput

# 模型名称现在将从图状态(GraphState)中读取，该状态在初始化时从环境变量加载
# node_model = "gpt-4o_chatgpt"
# feedback_model = "gpt-4o_chatgpt"

# 使用新的 AgentState 作为图的状态
GraphState = AgentState

def initialize_state_node(state: GraphState) -> Dict[str, Any]:
    """
    为状态图的首次运行或每次迭代提供所有字段的默认值。
    确保所有需要的键都存在，即使它们是空的。
    """
    if state is None:
        state = {}

    # 创建一个全新的状态副本，以确保数据在图的执行过程中持久存在
    persisted_state = state.copy()

    # --- 核心改动：处理 user_input ---
    # 将单次的 user_input 转换为 HumanMessage，准备让 add_messages 追加
    new_messages_to_add = []
    if user_input := persisted_state.pop("user_input", None):
        new_messages_to_add.append(HumanMessage(content=user_input))

    # 归一化已有的消息格式（在从检查点加载时可能需要）
    raw_messages = persisted_state.get("messages", [])
    normalized_messages = []
    for msg in raw_messages:
        if isinstance(msg, BaseMessage):
            normalized_messages.append(msg)
        elif isinstance(msg, dict):
            # 处理通过API等方式传入的、被序列化为字典的消息
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            if role in ["human", "user"]:
                normalized_messages.append(HumanMessage(content=content))
            elif role in ["ai", "assistant"]:
                normalized_messages.append(AIMessage(content=content))
    
    # 将归一化后的历史消息放回持久化状态，并追加新的消息
    persisted_state["messages"] = normalized_messages + new_messages_to_add

    # 使用 .get() 安全地访问并为我们模型的核心字段提供默认值
    initialized_state = {
        **persisted_state, # 将所有已有字段带入新状态
        # messages 字段已经在 persisted_state 中更新，无需再次覆盖
        "current_stage": persisted_state.get("current_stage", "initial_contact"),
        "emotional_state": persisted_state.get("emotional_state", EmotionalState()),
        "user_profile": persisted_state.get("user_profile", {}),
        "turn_count": persisted_state.get("turn_count", 0),
        
        # 为从旧GraphState合并过来的字段提供默认值
        "internal_monologue": [], # 总是重置内心独白
        "candidate_actions": [],  # 总是重置候选行动
        "evaluated_responses": [],# 总是重置评估过的响应
        "final_response": "",     # 总是重置最终响应
        "last_message": "",       # 重置API输出消息
        "agent_temperature": persisted_state.get("agent_temperature", 0.5),
        "node_model": persisted_state.get("node_model", os.environ.get("NODE_MODEL", "o3")),
        "feedback_model": persisted_state.get("feedback_model", os.environ.get("FEEDBACK_MODEL", "o3")),
        "verbose": persisted_state.get("verbose", False), 
        # 默认关闭调试模式,需要进行回复策略的分析时，只需要在messages字段后添加"verbose": True
        
        # 新增：行为意图和预约管理字段
        "customer_intent": persisted_state.get("customer_intent"),
        "appointment_info": persisted_state.get("appointment_info"),
    }

    # 增加对话轮次
    initialized_state["turn_count"] += 1

    # 清理运行时状态（这一步现在通过上面的重置来完成，注释掉以避免冗余）
    # initialized_state["internal_monologue"] = []
    # initialized_state["candidate_actions"] = []
    # initialized_state["evaluated_responses"] = []
    # initialized_state["final_response"] = ""


    return initialized_state

def _format_messages(messages: List[Any]) -> str:
    """将 LangChain BaseMessage 对象的列表格式化为单个字符串。"""
    if not messages:
        return "（无历史记录）"
    
    formatted_string = ""
    for message in messages:
        # message.type 在 BaseMessage 对象中是 'human', 'ai', 'system' 等
        role = "客户" if message.type == "human" else "林医生"
        formatted_string += f"{role}: {message.content}\n"
    return formatted_string.strip()

def analyze_sentiment_node(state: GraphState) -> Dict[str, Any]:
    """
    根据当前的情感状态，动态设置助手的温度（创造性）。
    前期其实可以不用这个模块7/3
    """
    internal_monologue = state.get("internal_monologue", [])
    emotional_state = state.get("emotional_state") # 我们从这里获取情感
    verbose = state.get("verbose", False)

    if not emotional_state:
        # 如果没有情感状态，使用默认温度
        return {"agent_temperature": 0.6}

    # 基于舒适度和熟悉度来设定温度
    # 如果用户感到舒适和高兴，我们说话的方式就会更活泼，更像朋友。
    #但其实这部分，应该在模型敲定后再做评估，因为每个模型的风格并不同。
    comfort = emotional_state.comfort_level
    familiarity = emotional_state.familiarity_level
    
    agent_temperature = 0.6 # 默认值，qwen使用低温，避免过度活跃
    if comfort > 0.6 and familiarity > 0.5:
        agent_temperature = 0.6 # 更富创造性、更像朋友
    elif comfort < 0.3:
        agent_temperature = 0.6 # 更保守、更谨慎

    new_monologue = internal_monologue + [f"温度设定：根据当前情感 (舒适度:{comfort:.2f}, 熟悉度:{familiarity:.2f})，设定温度为 {agent_temperature}。"]
    
    # 只在verbose模式下输出调试信息
    if verbose:
        print(f"[DEBUG] 情感分析节点: 温度设定为 {agent_temperature}")
    
    return {
        "agent_temperature": agent_temperature,
        "internal_monologue": new_monologue,
    }

def _design_node(state: GraphState) -> Dict[str, Any]:
    """
    智能决策节点，重新设计为服务导向而非销售导向。
    允许"正确的缺点"，表现得更加自然和人性化。
    """
    internal_monologue = state.get("internal_monologue", [])
    verbose = state.get("verbose", False)
    
    # 1. 调用状态评估器，获取最新的情感状态
    from blocks.state_evaluator import evaluate_state
    from blocks.intent_analyzer import analyze_customer_intent, update_appointment_info
    
    # 直接调用状态评估
    evaluation_result = evaluate_state(state)
    
    # 更新状态。如果评估失败，则使用旧状态
    current_emotional_state = evaluation_result.get("emotional_state", state["emotional_state"])
    customer_intent = evaluation_result.get("customer_intent_level", state.get("customer_intent_level", "low"))
    
    # 2. 新增：分析客户行为意图
    intent_result = analyze_customer_intent(state)
    current_customer_intent = intent_result.get("customer_intent")
    
    # 3. 新增：更新预约信息
    appointment_updates = {}
    if current_customer_intent:
        appointment_updates = update_appointment_info(state, current_customer_intent)
    
    # 合并预约信息更新
    current_appointment_info = state.get("appointment_info")
    if appointment_updates.get("appointment_info"):
        current_appointment_info = appointment_updates["appointment_info"]
    
    internal_monologue.append(f"情感评估完成: {current_emotional_state.model_dump_json()}")
    internal_monologue.append(f"客户意向评估: {customer_intent}")
    if current_customer_intent:
        internal_monologue.append(f"行为意图识别: {current_customer_intent.intent_type} (置信度: {current_customer_intent.confidence:.2f})")
        if current_customer_intent.extracted_info:
            internal_monologue.append(f"提取信息: {current_customer_intent.extracted_info}")
    if current_appointment_info:
        internal_monologue.append(f"预约状态: {current_appointment_info.appointment_status}, 时间: {current_appointment_info.preferred_time or '未定'}")
    
    if verbose:
        print(f"[DEBUG] 策略设计节点: 客户意向={customer_intent}, 信任度={current_emotional_state.trust_level:.2f}")
    
    # 2. 改进对话阶段 - 更自然的推进逻辑
    current_stage = state["current_stage"]
    trust_level = current_emotional_state.trust_level
    comfort_level = current_emotional_state.comfort_level
    familiarity_level = current_emotional_state.familiarity_level
    turn_count = state.get("turn_count", 0)
    
    new_stage = current_stage # 默认保持当前阶段
    
    # 改进的阶段推进逻辑（保持原有阶段名称）
    if current_stage == "initial_contact":
        # 阶段1：初次接触 - 自然问候，建立基础连接
        if turn_count >= 1 and comfort_level > 0.2:
            new_stage = "ice_breaking"
    elif current_stage == "ice_breaking":
        # 阶段2：轻松破冰 - 建立真实连接，允许"缺陷"
        if familiarity_level > 0.3:
            new_stage = "subtle_expertise"
    elif current_stage == "subtle_expertise":
        # 阶段3：展示专业 - 客观展示，非夸大宣传
        if trust_level > 0.4:
            new_stage = "pain_point_mining"
    elif current_stage == "pain_point_mining":
        # 阶段4：了解需求 - 真诚询问，非推销式
        if trust_level > 0.6 and customer_intent in ["medium", "high"]:
            new_stage = "solution_visualization"
    elif current_stage == "solution_visualization":
        # 阶段5：解决方案 - 协助决策，非强制推销
        if trust_level > 0.7 and customer_intent == "high":
            new_stage = "natural_invitation"
        
    # 自然回退机制：如果客户不舒服，回到更轻松的阶段
    if comfort_level < 0.3 and current_stage not in ["initial_contact", "ice_breaking"]:
        new_stage = "ice_breaking"  # 自然回退到轻松破冰
        internal_monologue.append(f"检测到舒适度过低 ({comfort_level:.2f})，自然回退到轻松破冰")
    elif trust_level < 0.2 and current_stage not in ["initial_contact"]:
        new_stage = "initial_contact"  # 重新开始
        internal_monologue.append(f"检测到信任度过低 ({trust_level:.2f})，重新开始对话")
        
    if new_stage != current_stage:
        internal_monologue.append(f"自然流程推进: '{current_stage}' → '{new_stage}' (信任{trust_level:.2f}/舒适{comfort_level:.2f}/熟悉{familiarity_level:.2f})")
    
    # 3. 改进动作决策 - 基于现有模块，让行为更自然
    candidate_actions = []
    
    # 优先级1：处理明确的预约意图
    if current_customer_intent and current_customer_intent.intent_type in ["appointment_request", "time_confirmation", "ready_to_book"]:
        if current_customer_intent.confidence > 0.8:
            # 高置信度：使用自然邀约
            candidate_actions = ["active_close", "value_display"]
            internal_monologue.append(f"检测到明确预约需求，进行自然邀约")
        else:
            # 低置信度：先了解需求
            candidate_actions = ["needs_analysis", "value_display"]
            internal_monologue.append(f"预约意图不明确，先了解具体需求")
    
    # 优先级2：处理信息咨询
    elif current_customer_intent and current_customer_intent.intent_type == "info_seeking":
        # 明确需求时优先提供信息，而不是挖掘需求
        candidate_actions = ["value_display"]
        # 只有在提供基本信息后，才考虑了解细节
        if familiarity_level > 0.4:  # 已经有一定基础时才询问细节
            candidate_actions.append("needs_analysis")
        internal_monologue.append(f"客户寻求信息，优先提供项目介绍")
    
    # 优先级3：处理价格询问（真实回应而非销售话术）
    elif current_customer_intent and current_customer_intent.intent_type == "price_inquiry":
        candidate_actions = ["value_display", "value_pitch"]
        if trust_level > 0.5:
            candidate_actions.append("active_close")  # 高信任时可以推进
        internal_monologue.append(f"价格咨询，提供真实信息")
    
    # 优先级4：处理顾虑（理解而非反驳）
    elif current_customer_intent and current_customer_intent.intent_type == "concern_raised":
        candidate_actions = ["stress_response", "rapport_building"]
        if comfort_level < 0.4:
            candidate_actions.append("rapport_building")  # 舒适度低时重建关系
        internal_monologue.append(f"客户有顾虑，给予理解和缓解")
    
    # 优先级5：基于阶段的自然对话流程
    else:
        # 根据当前阶段决定自然回应策略
        if new_stage == "initial_contact":
            candidate_actions = ["greeting"]
        elif new_stage == "ice_breaking":
            candidate_actions = ["rapport_building"]
            # 偶尔允许"缺陷"：简短回复
            if turn_count % 4 == 0:  # 偶尔表现得不那么完美
                candidate_actions = ["rapport_building"]  # 保持简洁
        elif new_stage == "subtle_expertise":
            candidate_actions = ["value_display"]
            if familiarity_level > 0.4:
                candidate_actions.append("needs_analysis")
        elif new_stage == "pain_point_mining":
            # 根据客户需求明确程度调整策略
            if current_customer_intent and current_customer_intent.intent_type == "info_seeking":
                # 如果客户已经表达明确需求，直接提供信息
                candidate_actions = ["value_display", "needs_analysis"]
            else:
                # 否则才进行需求挖掘
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
    
    # 智能策略调整：让行为更自然和贴近真人
    
    # 1. 根据情感状态调整策略
    if trust_level < 0.3:
        # 信任度低时，优先建立关系
        candidate_actions = ["rapport_building"]
        internal_monologue.append(f"信任度过低 ({trust_level:.2f})，优先建立关系")
    elif comfort_level < 0.2 and new_stage in ["solution_visualization", "natural_invitation"]:
        # 舒适度低时，回退到缓解压力
        candidate_actions.insert(0, "stress_response")
        internal_monologue.append(f"舒适度过低 ({comfort_level:.2f})，优先缓解压力")
    
    # 2. 意向等级特殊处理
    if customer_intent == "fake_high" and "reverse_probe" not in candidate_actions:
        candidate_actions.append("reverse_probe")  # 识别虚假高意向
        internal_monologue.append(f"检测到虚假高意向，添加反向试探")
    elif customer_intent == "low" and new_stage in ["solution_visualization", "natural_invitation"]:
        # 低意向客户不应该进入高压销售阶段
        candidate_actions = ["rapport_building", "needs_analysis"]
        internal_monologue.append(f"低意向客户，回退到基础交流")
    
    # 3. 自然搜索空间管理
    search_space_size = len(candidate_actions)
    
    if search_space_size == 1:
        # 适当扩展，保持灵活性
        primary_action = candidate_actions[0]
        
        if primary_action == "active_close":
            if comfort_level < 0.6:
                candidate_actions.append("stress_response")
            if trust_level > 0.7:
                candidate_actions.append("value_display")
        elif primary_action in ["value_display", "value_pitch"]:
            candidate_actions.append("needs_analysis")
            if trust_level > 0.6:
                candidate_actions.append("active_close")
        elif primary_action == "stress_response":
            candidate_actions.append("rapport_building")
        
        internal_monologue.append(f"扩展搜索空间: {primary_action} → {candidate_actions}")
    
    elif search_space_size > 3:
        # 保持合理范围
        candidate_actions = candidate_actions[:3]
        internal_monologue.append(f"限制搜索空间为3个选项")
    
    # 确保至少有基础回应能力
    if not candidate_actions:
        candidate_actions = ["rapport_building"]
        internal_monologue.append(f"兜底策略：使用基础关系建立")
    
    final_search_space = len(candidate_actions)
    decision_context = f"阶段:{new_stage}, 情感:{customer_intent}, 信任:{trust_level:.2f}"
    if current_customer_intent:
        decision_context += f", 意图:{current_customer_intent.intent_type}"
    internal_monologue.append(f"策略决策 ({decision_context}) -> 候选动作: {candidate_actions}")

    # 构建返回状态
    result = {
        "emotional_state": current_emotional_state,
        "current_stage": new_stage,
        "customer_intent_level": customer_intent,
        "candidate_actions": list(set(candidate_actions)),
        "internal_monologue": internal_monologue,
    }
    
    # 添加新的状态字段
    if current_customer_intent:
        result["customer_intent"] = current_customer_intent
    if current_appointment_info:
        result["appointment_info"] = current_appointment_info

    return result

def _fallback_evaluation(action: str, response: str, current_stage: str, emotional_state, customer_intent_level: str) -> float:
    """
    基于规则的兜底评估机制，当评估模型失败时使用
    """
    score = 0.5  # 默认中等分数
    
    # 检查回复是否过短或过长
    if len(response.strip()) < 3:
        return 0.2  # 过短回复
    if len(response) > 500:
        return 0.4  # 过长回复
    
    # 根据阶段调整基础分数
    stage_scores = {
        "initial_contact": {"greeting": 0.8, "rapport_building": 0.7},
        "ice_breaking": {"rapport_building": 0.8, "needs_analysis": 0.6},
        "subtle_expertise": {"value_display": 0.8, "needs_analysis": 0.7},
        "pain_point_mining": {"needs_analysis": 0.8, "pain_point_test": 0.7},
        "solution_visualization": {"value_pitch": 0.8, "value_display": 0.7},
        "natural_invitation": {"active_close": 0.8, "value_pitch": 0.6}
    }
    
    if current_stage in stage_scores and action in stage_scores[current_stage]:
        score = stage_scores[current_stage][action]
    
    # 根据情感状态调整
    trust_level = emotional_state.trust_level if emotional_state else 0.5
    comfort_level = emotional_state.comfort_level if emotional_state else 0.5
    
    # 信任度低时，优先关系建立
    if trust_level < 0.3:
        if action in ["rapport_building", "greeting"]:
            score += 0.1
        elif action in ["active_close", "value_pitch"]:
            score -= 0.2
    
    # 舒适度低时，避免压力过大的动作
    if comfort_level < 0.3:
        if action in ["stress_response", "rapport_building"]:
            score += 0.1
        elif action in ["active_close"]:
            score -= 0.1
    
    # 根据客户意向调整
    if customer_intent_level == "high" and action == "active_close":
        score += 0.1
    elif customer_intent_level == "low" and action == "active_close":
        score -= 0.2
    
    # 🎯 新增：明确需求时优先提供信息
    # 简单的关键词检测来判断回复类型
    need_keywords = ["我想", "想了解", "想做", "需要", "咨询", "价格", "多少钱"]
    
    # 检查回复是否直接提供了信息而不是继续提问
    if action == "value_display":
        if any(keyword in response for keyword in ["项目", "方法", "价格", "效果", "可以"]):
            score += 0.15  # 奖励提供信息的回复
    elif action == "needs_analysis":
        if any(keyword in response for keyword in ["什么", "怎么", "哪种", "为什么"]):
            score -= 0.1  # 降低继续提问的回复分数
    
    return max(0.1, min(1.0, score))  # 确保在合理范围内

def _generate_and_evaluate_action(
    action: str, 
    state: GraphState, # 传递整个状态以获取更丰富的上下文
) -> tuple:
    """
    为单个动作生成并评估回复。这是一个辅助函数，用于并行执行。
    """
    # 从 state 中解构所需变量
    messages = state["messages"]
    node_model = state["node_model"]
    agent_temperature = state["agent_temperature"]
    feedback_model = state["feedback_model"]
    current_stage = state["current_stage"]
    emotional_state = state["emotional_state"]
    customer_intent_level = state.get("customer_intent_level", "low")

    try:
        # 使用工厂动态获取采样器
        node_sampler, _ = SamplerFactory.get_sampler_and_cost(node_model)
        
        # block 的创建逻辑现在只需要采样器实例
        block = create_block(action, node_sampler, node_model)
        if not block:
            return None, f"模块 '{action}' 创建失败，已跳过。"

        # 所有模块都使用统一的forward接口
        response = block.forward(messages, agent_temperature)
            
        if response is None:
            return None, f"模块 '{action}' 执行失败，已跳过。"

        if "[INFO_SUFFICIENT]" in response:
            score, reasoning = 0.0, "内部指令，不应直接输出"
        else:
            # 🔧 优化评估逻辑：更宽松的评估标准 + 更强的兜底机制
            try:
                # 动态获取评估模型的采样器
                feedback_sampler, _ = SamplerFactory.get_sampler_and_cost(feedback_model)
                
                # 🎯 改进的评估prompt：重点关注需求满足
                
                # 检查最后一条用户消息
                last_user_message = ""
                for msg in reversed(messages):
                    if hasattr(msg, 'content') and not msg.content.startswith("小文"):
                        last_user_message = msg.content
                        break
                
                feedback_prompt = f"""
你是对话质量评估专家。评估这个回复是否合适。

**关键原则：当客户表达明确需求时，优先满足需求而不是继续挖掘**

**用户最后说：** "{last_user_message}"

**候选回复 (策略: {action}):**
"{response}"

**评估要点：**
1. 如果用户说"我想美白"、"想了解XX"等明确需求，优先给分高的回复应该是：
   - 直接提供相关信息/项目介绍 (高分)
   - 而不是继续问"您想改善什么问题" (低分)

2. 如果用户已经选择了具体项目，优先给分高的回复应该是：
   - 进入预约流程/提供案例 (高分)
   - 而不是继续了解需求 (低分)

**评分标准:**
- 0.8-1.0: 回复直接满足了用户需求
- 0.6-0.7: 回复基本合适，略有偏离但可接受
- 0.4-0.5: 回复一般，没有很好满足需求
- 0.2-0.3: 回复偏离了用户意图
- 0.0-0.1: 回复完全不合适

JSON格式: {{"score": 数值, "reasoning": "简短理由"}}
"""
                raw_feedback, _ = feedback_sampler(
                    [{"role": "user", "content": feedback_prompt}],
                    temperature=0.1,  # 降低温度提高稳定性
                    response_format='json_object'
                )
                
                # 增强JSON解析逻辑
                if raw_feedback is None or raw_feedback.strip() == "":
                    raise ValueError("评估模型返回空内容")
                
                # 尝试直接解析
                try:
                    feedback_data = json.loads(raw_feedback)
                except json.JSONDecodeError:
                    # 使用正则提取JSON
                    import re
                    match = re.search(r'\{.*?\}', raw_feedback, re.DOTALL)
                    if match:
                        feedback_data = json.loads(match.group(0))
                    else:
                        raise ValueError("无法找到有效JSON")

                score = float(feedback_data.get("score", 0.5))  # 默认给中等分
                reasoning = feedback_data.get("reasoning", "评估成功")
                
                # 确保分数在合理范围内
                score = max(0.0, min(1.0, score))

            except Exception as eval_error:
                # 🛡️ 强化兜底策略：基于规则的快速评估
                score = _fallback_evaluation(action, response, current_stage, emotional_state, customer_intent_level)
                reasoning = f"评估模型失败，使用规则评估: {eval_error}"

        evaluated_response = {
            "action": action,
            "response": response,
            "score": score,
            "reasoning": reasoning
        }
        monologue_entry = f"  - [{action}] 生成回复: '{response[:30]}...' -> 评估得分: {score} (原因: {reasoning})"
        return evaluated_response, monologue_entry

    except Exception as e:
        return None, f"  - [{action}] 处理时出现异常: {e}"


def generate_and_evaluate_node(state: GraphState) -> Dict[str, Any]:
    """
    并行地为每个候选动作生成回复并获取反馈。
    """
    internal_monologue = state.get("internal_monologue", [])
    candidate_actions = state.get("candidate_actions", [])
    
    evaluated_responses = []
    new_monologue = list(internal_monologue)

    # 限制并发数量，避免API服务器过载和CancelledError
    max_concurrent_requests = min(3, len(candidate_actions) or 1)
    with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
        future_to_action = {
            executor.submit(
                _generate_and_evaluate_action, 
                action, 
                state, # 传递整个状态以提供更丰富的评估上下文
            ): action for action in candidate_actions
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

    verbose = state.get("verbose", False)
    if verbose:
        print(f"[DEBUG] 生成评估节点: 评估了 {len(evaluated_responses)} 个候选回复")

    if not evaluated_responses:
        new_monologue.append("所有模块都执行失败了，使用紧急兜底回复")
        # 🛡️ 多级兜底机制
        try:
            # 尝试人工转接
            node_sampler, _ = SamplerFactory.get_sampler_and_cost(state["node_model"])
            block = create_block("human_handoff", node_sampler, state["node_model"])
            final_response = block.forward(state.get("messages", []), state.get("agent_temperature", 0.5))
        except Exception as e:
            new_monologue.append(f"人工转接模块也失败了: {e}")
            # 最终兜底：固定回复
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

def self_verification_node(state: GraphState) -> Dict[str, Any]:
    """
    从评估过的候选项中选择最佳响应。
    """
    evaluated_responses = state.get("evaluated_responses", [])
    messages = state.get("messages", [])
    internal_monologue = state.get("internal_monologue", [])
    
    # 不再需要从这里获取采样器，因为评分已在 generate_and_evaluate_node 完成
    # sampler = ...

    # 🔧 优化选择逻辑：降低质量门槛，确保总是有回复
    
    # 先尝试0.3以上的回复
    high_quality_responses = [r for r in evaluated_responses if r.get('score', 0.0) > 0.3]
    
    # 如果没有0.3以上的，尝试0.2以上的
    if not high_quality_responses:
        high_quality_responses = [r for r in evaluated_responses if r.get('score', 0.0) > 0.2]
    
    # 如果还是没有，选择所有回复中得分最高的
    if not high_quality_responses and evaluated_responses:
        high_quality_responses = sorted(evaluated_responses, key=lambda x: x.get('score', 0.0), reverse=True)
    
    # 极端情况：没有任何回复
    if not high_quality_responses:
        new_monologue = internal_monologue + ["自我验证失败：没有可供选择的候选回复，使用紧急回复"]
        fallback_response = "嗯嗯，好的"  # 简单自然的兜底回复
        return {
            "final_response": fallback_response,
            "last_message": fallback_response,
            "messages": [AIMessage(content=fallback_response)],
            "internal_monologue": new_monologue
        }

    if len(high_quality_responses) == 1:
        final_response = high_quality_responses[0]['response']
        new_monologue = internal_monologue + [f"自我验证：只有1个高质量选项，直接选择 '{high_quality_responses[0]['action']}'。"]
    else:
        # 直接按得分排序选择最高分的回复
        best_response = sorted(high_quality_responses, key=lambda x: x['score'], reverse=True)[0]
        final_response = best_response['response']
        new_monologue = internal_monologue + [
            f"自我验证：从 {len(high_quality_responses)} 个选项中选择得分最高的回复 (模块: {best_response['action']}, 得分: {best_response['score']:.2f})。"
        ]

    # --- 关键改动：将AI的最终回复更新回消息历史中 ---
    # With `add_messages`, we just need to return the new message(s) in a list.
    # LangGraph will handle appending it to the state.
    new_monologue.append("将AI的最新回复返回，由LangGraph自动更新历史。")

    verbose = state.get("verbose", False)
    if verbose:
        print(f"[DEBUG] 最终回复: {final_response}")

    # --- 新增：如果 verbose 模式开启，则准备调试信息 ---
    debug_info = None
    if state.get("verbose", False):
        from common import DebugInfo
        debug_info = DebugInfo(
            current_stage=state.get("current_stage"),
            emotional_state=state.get("emotional_state").model_dump() if state.get("emotional_state") else None,
            internal_monologue=new_monologue,
        )

    return {
        "final_response": final_response, # 保留此字段用于本地测试的即时打印
        "last_message": final_response,   # 新增：用于API输出
        "messages": [AIMessage(content=final_response)], # 只返回新增的消息
        "internal_monologue": new_monologue,
        "debug_info": debug_info, # 将调试信息添加到返回字典中
    }



def build_graph():
    """构建并编译 LangGraph 状态图。"""
    # 使用MasAgentOutput控制API响应格式，仿照huanmu_agent-test的设计
    workflow = StateGraph(GraphState, output=MasAgentOutput)

    workflow.add_node("initialize_state", initialize_state_node)
    workflow.add_node("analyze_sentiment", analyze_sentiment_node)
    workflow.add_node("meta_design", _design_node)
    workflow.add_node("generate_and_evaluate", generate_and_evaluate_node)
    workflow.add_node("self_verify", self_verification_node)

    workflow.set_entry_point("initialize_state")

    workflow.add_edge("initialize_state", "analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "meta_design")
    workflow.add_edge("meta_design", "generate_and_evaluate")
    workflow.add_edge("generate_and_evaluate", "self_verify")
    workflow.add_edge("self_verify", END)
    
    return workflow

app = build_graph().compile()

if __name__ == "__main__":
    print("重构后的 MAS-Zero (LangGraph 版本)")
    
    # For local testing, compile with a checkpointer
    try:
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()
        local_app = build_graph().compile(checkpointer=memory)
    except ImportError:
        # In cloud deployment, checkpointer is provided automatically
        local_app = build_graph().compile()

    config = {"configurable": {"thread_id": "user-123-cli-session"}}
    
    print("\n您好，我是林医生。请问您主要想了解哪些口腔方面的问题呢？")

    while True:
        user_input = input("你: ")
        if user_input.lower() in ["退出", "exit"]:
            print("再见！")
            break
        
        # 使用新的 user_input 字段作为输入
        inputs = {"user_input": user_input}
        
        # stream() 方法可以让我们实时看到每一步的状态更新
        # 在这里我们不关心 stream 的中间产物，只关心最终状态
        for _ in local_app.stream(inputs, config=config):
            pass


        # 从 checkpointer 获取最终的、最完整的状态
        snapshot = local_app.get_state(config)
        final_state_values = snapshot.values
        
        response = final_state_values.get("final_response", "抱歉，我好像走神了，能再说一遍吗？")
        print(f"林医生: {response}")
        
        print("\n--- 内部状态 (仅供调试) ---")
        # 直接访问状态中的值
        current_stage = final_state_values.get('current_stage')
        emotional_state = final_state_values.get('emotional_state')
        internal_monologue = final_state_values.get('internal_monologue', [])

        print(f"当前阶段: {current_stage}")
        if emotional_state:
            print(f"情感状态: {emotional_state}")
        else:
            print("情感状态: (未评估)")
            
        print("内心独白:")
        if isinstance(internal_monologue, list):
            for line in internal_monologue:
                print(f"- {line}")
        print("----------------\n")