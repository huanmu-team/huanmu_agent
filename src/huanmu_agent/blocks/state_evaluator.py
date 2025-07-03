from typing import Dict, List, Any
from ..state import State as AgentState, EmotionalState
from ..mas_prompts import load_prompt
from ..utils.langchain_utils import load_chat_model
import json
import re

def evaluate_state(state: AgentState) -> Dict[str, Any]:
    """
    评估当前对话状态，包括情感和客户意图。
    使用当前框架的模型调用方式。
    """
    # 从状态中获取评估所需的模型名称
    eval_model_name = getattr(state, "feedback_model", "openai/gpt-4o-mini-2024-07-18") # 使用 feedback_model 进行评估

    # 格式化对话历史
    messages = getattr(state, "messages", [])
    history = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in messages]
    )

    # 加载 Prompt
    try:
        prompt_template = load_prompt("state_evaluator")
    except ValueError as e:
        print(f"错误: 无法加载状态评估prompt: {e}")
        return {} # 评估失败

    prompt = prompt_template.format(
        message_history=history,
        current_stage=getattr(state, "current_stage", "initial_contact"),
        user_profile=getattr(state, "user_profile", {})
    )
    
    # 使用当前框架的模型调用方式
    try:
        model = load_chat_model(eval_model_name, 0.0)
        # 绑定JSON输出格式
        model_with_format = model.bind(response_format={"type": "json_object"})
        
        # 调用模型
        response = model_with_format.invoke([{"role": "user", "content": prompt}])
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # 处理API返回None的情况
        if response_text is None:
            print("状态评估API返回None，使用默认值")
            llm_output = {}
        else:
            # 增强JSON解析的鲁棒性：先用正则提取出JSON部分
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                llm_output = json.loads(json_str)
            else:
                # 如果找不到JSON，打印原始返回并抛出错误
                print(f"状态评估原始返回 (未找到JSON): {response_text}")
                raise ValueError("响应中未找到有效的JSON对象")
            
    except Exception as e:
        # 如果解析失败，打印原始返回以供调试
        if 'response_text' in locals() and response_text is not None:
            print(f"状态评估原始返回 (解析失败): {response_text}")
        print(f"错误: 状态评估的LLM调用失败或返回了无效JSON。 {e}")
        llm_output = {} # 出错时返回空字典

    # 从LLM的输出中提取并构建状态
    emotional_state_data = llm_output.get("emotional_state", {})
    
    return {
        "emotional_state": EmotionalState(**emotional_state_data),
        "customer_intent_level": llm_output.get("customer_intent_level", "low")
    } 