"""用户画像生成模块"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
import asyncio
from pathlib import Path

from constant import GOOGLE_GEMINI_FLASH_MODEL, GOOGLE_GEMINI_PRO_MODEL

# 加载profile_variables.py
def load_profile_variables() -> Dict[str, Any]:
    """安全加载profile_variables字典"""
    profile_vars_path = Path(__file__).parent / "profile_variables.py"
    with open(profile_vars_path, 'r', encoding='utf-8') as f:
        code = compile(f.read(), profile_vars_path, 'exec')
        exec_globals = {}
        exec(code, exec_globals)
        return exec_globals['profile_variables']

# 初始化profile_variables
profile_variables = load_profile_variables()

PROFILE_SYSTEM_PROMPT = """
你是一个专业的医美/美容行业用户画像标签生成助手，必须严格遵循以下规则：

# 强制要求
分析用户聊天记录，生成用户画像标签，用户画像标签必须符合以下数据结构规范，并且只输出已有的标签

标签内容
{
    "social_profile": {
        "occupation": %(occupation_list)s,
        "age": %(age_list)s, 
        "region": %(region_list)s,
        "lifestyle": %(lifestyle_list)s,
        "family_status": %(family_status_list)s,
        "emotion": %(emotion_list)s
    },
    "personality_traits": {
        "character": %(character_list)s,
        "values": %(values_list)s,
        "aesthetic_style": %(aesthetic_style_list)s
    },
    "consumption_profile": {
        "ability": %(ability_list)s,
        "willingness": %(willingness_list)s,
        "preferences": %(preferences_list)s
    },
    "product_intent": {
        "current_use": %(current_use_list)s,
        "potential_needs": %(potential_needs_list)s,
        "decision_factors": %(decision_factors_list)s,
        "purchase_intent_score": %(purchase_intent_score_list)s
    },
    "customer_lifecycle": {
        "stage": %(stage_list)s,
        "value": %(value_list)s,
        "retention_strategy": %(retention_strategy_list)s
    }
}
你只能生成json格式的标签，不要输出其他内容
""" % {
    "occupation_list": str(profile_variables["social_profile"]["occupation"]),
    "age_list": str(profile_variables["social_profile"]["age"]),
    "region_list": str(profile_variables["social_profile"]["region"]),
    "lifestyle_list": str(profile_variables["social_profile"]["lifestyle"]),
    "family_status_list": str(profile_variables["social_profile"]["family_status"]),
    "emotion_list": str(profile_variables["social_profile"]["emotion"]),
    "character_list": str(profile_variables["personality_traits"]["character"]),
    "values_list": str(profile_variables["personality_traits"]["values"]),
    "aesthetic_style_list": str(profile_variables["personality_traits"]["aesthetic_style"]),
    "ability_list": str(profile_variables["consumption_profile"]["ability"]),
    "willingness_list": str(profile_variables["consumption_profile"]["willingness"]),
    "preferences_list": str(profile_variables["consumption_profile"]["preferences"]),
    "current_use_list": str(profile_variables["product_intent"]["current_use"]),
    "potential_needs_list": str(profile_variables["product_intent"]["potential_needs"]),
    "decision_factors_list": str(profile_variables["product_intent"]["decision_factors"]),
    "purchase_intent_score_list": str(profile_variables["product_intent"]["purchase_intent_score"]),
    "stage_list": str(profile_variables["customer_lifecycle"]["stage"]),
    "value_list": str(profile_variables["customer_lifecycle"]["value"]),
    "retention_strategy_list": str(profile_variables["customer_lifecycle"]["retention_strategy"])
}

class socialProfilestructure(BaseModel):
    occupation: Optional[str] = Field(default=None)
    age: Optional[str] = Field(default=None) 
    region: Optional[str] = Field(default=None)
    lifestyle: Optional[str] = Field(default=None)
    family_status: Optional[str] = Field(default=None)
    emotion: Optional[str] = Field(default=None)

class personality_traits_structure(BaseModel):
    character: Optional[str] = Field(default=None)
    values: Optional[str] = Field(default=None)
    aesthetic_style: Optional[str] = Field(default=None)

class consumption_profile_structure(BaseModel):
    ability: Optional[str] = Field(default=None)
    willingness: Optional[str] = Field(default=None) 
    preferences: Optional[str] = Field(default=None)

class product_intent_structure(BaseModel):
    current_use: Optional[str] = Field(default=None)
    potential_needs: Optional[str] = Field(default=None)
    decision_factors: Optional[str] = Field(default=None)
    purchase_intent_score: Optional[str] = Field(default=None)

class customer_lifecycle_structure(BaseModel):
    stage: Optional[str] = Field(default=None)
    value: Optional[str] = Field(default=None)
    retention_strategy: Optional[str] = Field(default=None)
    

class UserProfileStructure(BaseModel):
    """用户画像数据结构"""
    occupation: Optional[str] = Field(default=None)
    age: Optional[str] = Field(default=None) 
    region: Optional[str] = Field(default=None)
    lifestyle: Optional[str] = Field(default=None)
    family_status: Optional[str] = Field(default=None)
    emotion: Optional[str] = Field(default=None)
    
    character: Optional[str] = Field(default=None)
    values: Optional[str] = Field(default=None)
    aesthetic_style: Optional[str] = Field(default=None)
    
    ability: Optional[str] = Field(default=None)
    willingness: Optional[str] = Field(default=None) 
    preferences: Optional[str] = Field(default=None)
    
    current_use: Optional[str] = Field(default=None)
    potential_needs: Optional[str] = Field(default=None)
    decision_factors: Optional[str] = Field(default=None)
    purchase_intent_score: Optional[str] = Field(default=None)
    
    stage: Optional[str] = Field(default=None)
    value: Optional[str] = Field(default=None)
    retention_strategy: Optional[str] = Field(default=None)


class ProfileLabelAgentState(AgentState):
    error_message: Optional[str]
    structured_response: Optional[UserProfileStructure]

class ProfileLabelAgentConfigSchema(TypedDict):
    pass

class ProfileLabelAgentStateInput(TypedDict):
    messages: List[BaseMessage]

class ProfileLabelAgentResponseFormat(BaseModel):
    user_profile_label: UserProfileStructure = Field(description="用户画像标签，可以为空")
    error_message: Optional[str] = Field(description="错误信息", default=None)

class ProfileLabelAgentStateOutput(TypedDict):
    structured_response: ProfileLabelAgentResponseFormat

# 初始化模型
chat_model = init_chat_model(
    model=GOOGLE_GEMINI_PRO_MODEL,
    model_provider="google_vertexai",
    temperature=0.7,  # Balanced creativity
)

def build_profile_prompt(state: AgentState, config: RunnableConfig) -> List[BaseMessage]:
    # 直接使用硬编码的提示词
    formatted_prompt = PROFILE_SYSTEM_PROMPT
    messages = [{
        "role": "system", 
        "content": formatted_prompt
    }]
    messages.extend(state.get("messages", []))
    return messages

# 创建agent
profile_agent = create_react_agent(
    model=chat_model,
    tools=[],
    name="profile_agent",
    state_schema=ProfileLabelAgentState,
    config_schema=ProfileLabelAgentConfigSchema,
    response_format=ProfileLabelAgentResponseFormat,
    prompt=build_profile_prompt,
)

async def profile_agent_node(state: ProfileLabelAgentState, config: RunnableConfig):
    """调用用户画像生成agent"""
    current_conversation_messages = state.get("messages", [])
    
    if not current_conversation_messages:
        system_msg = [{"role": "system", "content": PROFILE_SYSTEM_PROMPT}]
        user_msg = [
            {
                "role": "user",
                "content": f"请帮我生成用户画像标签。",
            }
        ]
        current_conversation_messages = system_msg + user_msg
    
    try:
        agent_response = await asyncio.to_thread(
            profile_agent.invoke,
            {"messages": current_conversation_messages},
            config,
        )
        
        # 直接返回UserProfileStructure实例
        return {
            "structured_response": agent_response.get("structured_response"),
            "messages": current_conversation_messages,
            "error_message": None,
        }

    except Exception as e:
        print(f"Error during profile agent invocation: {e}")
        return {
            "error_message": str(e),
        }

# 构建工作流
profile_label_graph = (
    StateGraph(ProfileLabelAgentState, input=ProfileLabelAgentStateInput, config_schema=ProfileLabelAgentConfigSchema, output=ProfileLabelAgentStateOutput)
    .add_node("profile_generator", profile_agent_node)
    .add_edge(START, "profile_generator")
    .compile()
)