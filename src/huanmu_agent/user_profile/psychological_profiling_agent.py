"""心理侧写智能体模块
基于心理学理论对用户进行深度心理分析，支撑高情商对话
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
import asyncio

# 导入项目根目录的常量
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from constant import GOOGLE_GEMINI_PRO_MODEL

PSYCHOLOGICAL_PROFILING_SYSTEM_PROMPT = """
你是一位专业的心理学专家和心理分析师，擅长通过对话分析用户的深层心理特征。
请基于心理学理论和实践经验，对用户进行全面的心理侧写分析。

心理侧写必须严格按以下JSON格式返回：
{
  "cognitive_style": "认知风格分析",
  "emotional_patterns": "情感模式分析", 
  "behavioral_tendencies": "行为倾向分析",
  "defense_mechanisms": "心理防御机制",
  "communication_preferences": "沟通偏好",
  "attachment_style": "依恋风格",
  "stress_response": "压力应对模式",
  "motivation_drivers": "内在动机驱动",
  "self_concept": "自我概念",
  "interpersonal_patterns": "人际关系模式",
  "emotional_intelligence": "情商特征",
  "psychological_needs": "核心心理需求",
  "potential_vulnerabilities": "潜在心理脆弱性",
  "growth_opportunities": "心理成长机会",
  "recommended_approach": "建议的沟通和互动方式"
}

分析要求：
1. 基于心理学理论（如大五人格、依恋理论、认知行为理论等）
2. 深度挖掘隐藏的心理动机和情感需求
3. 识别防御机制和应对策略
4. 分析沟通风格和情感表达模式
5. 提供具体可操作的互动建议
6. 如果对话内容不足，相应字段返回"信息不足，需要更多观察"
7. 必须返回纯JSON格式，不包含任何额外文本或注释
"""

class PsychologicalProfilingStructure(BaseModel):
    """心理侧写数据结构"""
    cognitive_style: str = Field(description="认知风格分析", default="")
    emotional_patterns: str = Field(description="情感模式分析", default="")
    behavioral_tendencies: str = Field(description="行为倾向分析", default="")
    defense_mechanisms: str = Field(description="心理防御机制", default="")
    communication_preferences: str = Field(description="沟通偏好", default="")
    attachment_style: str = Field(description="依恋风格", default="")
    stress_response: str = Field(description="压力应对模式", default="")
    motivation_drivers: str = Field(description="内在动机驱动", default="")
    self_concept: str = Field(description="自我概念", default="")
    interpersonal_patterns: str = Field(description="人际关系模式", default="")
    emotional_intelligence: str = Field(description="情商特征", default="")
    psychological_needs: str = Field(description="核心心理需求", default="")
    potential_vulnerabilities: str = Field(description="潜在心理脆弱性", default="")
    growth_opportunities: str = Field(description="心理成长机会", default="")
    recommended_approach: str = Field(description="建议的沟通和互动方式", default="")

class PsychologicalProfilingAgentState(AgentState):
    error_message: Optional[str] = None
    structured_response: Optional[PsychologicalProfilingStructure] = None

class PsychologicalProfilingConfigSchema(TypedDict):
    pass

class PsychologicalProfilingStateInput(TypedDict):
    messages: List[BaseMessage]

class PsychologicalProfilingResponseFormat(BaseModel):
    psychological_profile: PsychologicalProfilingStructure = Field(description="心理侧写分析结果")
    error_message: Optional[str] = Field(description="错误信息", default=None)

class PsychologicalProfilingStateOutput(TypedDict):
    structured_response: PsychologicalProfilingStructure

# 初始化模型
chat_model = init_chat_model(
    model=GOOGLE_GEMINI_PRO_MODEL,
    model_provider="google_vertexai",
    temperature=0.3,  # 较低温度确保专业性和一致性
)

def build_psychological_profiling_prompt(state: AgentState, config: RunnableConfig) -> List[BaseMessage]:
    """构建心理侧写提示词"""
    messages = [{
        "role": "system", 
        "content": PSYCHOLOGICAL_PROFILING_SYSTEM_PROMPT
    }]

    # 添加历史对话作为分析材料
    if isinstance(state, dict):
        messages.extend(state.get("messages", []))

    # 添加心理侧写分析指令
    messages.append(HumanMessage(content="请根据以上对话历史进行心理侧写分析"))

    return messages

# 创建心理侧写agent
psychological_profiling_agent = create_react_agent(
    model=chat_model,
    tools=[],
    name="psychological_profiling_agent",
    state_schema=PsychologicalProfilingAgentState,
    config_schema=PsychologicalProfilingConfigSchema,
    response_format=PsychologicalProfilingResponseFormat,
    prompt=build_psychological_profiling_prompt,
)

async def psychological_profiling_agent_node(state: PsychologicalProfilingAgentState, config: RunnableConfig):
    """心理侧写分析节点"""
    current_conversation_messages = state.get("messages", [])

    # 如果没有对话消息，返回默认结构
    if not current_conversation_messages:
        return {
            "structured_response": PsychologicalProfilingStructure(),
            "error_message": None,
            "messages": [],
        }
    
    try:
        # 异步调用心理侧写agent
        agent_response = await asyncio.to_thread(
            psychological_profiling_agent.invoke,
            {"messages": current_conversation_messages},
            config,
        )
        
        return {
            "structured_response": agent_response.get("structured_response"),
            "error_message": None,
            "messages": current_conversation_messages,
        }
    
    except Exception as e:
        print(f"心理侧写分析过程中发生错误: {e}")
        return {"error_message": str(e)}

# 构建心理侧写工作流
psychological_profiling_graph = (
    StateGraph(
        PsychologicalProfilingAgentState, 
        input=PsychologicalProfilingStateInput, 
        config_schema=PsychologicalProfilingConfigSchema, 
        output=PsychologicalProfilingStateOutput
    )
    .add_node("psychological_profiling_generator", psychological_profiling_agent_node)
    .add_edge(START, "psychological_profiling_generator")
    .compile()
)

# 辅助函数：格式化心理侧写结果
def format_psychological_profile(profile: PsychologicalProfilingStructure) -> str:
    """格式化心理侧写结果为可读文本"""
    if not profile:
        return "暂无心理侧写数据"
    
    sections = [
        ("认知风格", profile.cognitive_style),
        ("情感模式", profile.emotional_patterns),
        ("行为倾向", profile.behavioral_tendencies),
        ("心理防御机制", profile.defense_mechanisms),
        ("沟通偏好", profile.communication_preferences),
        ("依恋风格", profile.attachment_style),
        ("压力应对", profile.stress_response),
        ("内在动机", profile.motivation_drivers),
        ("自我概念", profile.self_concept),
        ("人际模式", profile.interpersonal_patterns),
        ("情商特征", profile.emotional_intelligence),
        ("心理需求", profile.psychological_needs),
        ("潜在脆弱性", profile.potential_vulnerabilities),
        ("成长机会", profile.growth_opportunities),
        ("建议互动方式", profile.recommended_approach),
    ]
    
    formatted_text = "## 心理侧写分析报告\n\n"
    for title, content in sections:
        if content and content.strip():
            formatted_text += f"**{title}**: {content}\n\n"
    
    return formatted_text

# 辅助函数：获取关键心理特征摘要  
def get_key_psychological_insights(profile: PsychologicalProfilingStructure) -> dict:
    """提取关键心理洞察，用于快速参考"""
    if not profile:
        return {}
    
    insights = {}
    
    # 提取核心特征
    if profile.communication_preferences and profile.communication_preferences.strip():
        insights["沟通建议"] = profile.communication_preferences
    
    if profile.emotional_patterns and profile.emotional_patterns.strip():
        insights["情感特征"] = profile.emotional_patterns
    
    if profile.psychological_needs and profile.psychological_needs.strip():
        insights["核心需求"] = profile.psychological_needs
    
    if profile.recommended_approach and profile.recommended_approach.strip():
        insights["互动方式"] = profile.recommended_approach
    
    return insights 