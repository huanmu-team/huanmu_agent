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

from constant import GOOGLE_GEMINI_FLASH_MODEL

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
你是一个专业的医美/美容行业用户画像生成助手，必须严格遵循以下规则：

# 强制要求
1. 输出必须完全符合指定的JSON结构，任何偏差都将导致系统错误
2. 每个标签必须从以下预定义类别中选择，禁止自行发明新标签
3. 必须执行同义词归一化，确保相同概念使用统一表述
4. 对于不确定的信息必须使用["未知"]，禁止猜测或编造
5. 必须从用户输入中推断尽可能多的信息，填充到各个字段中，确保信息全面且准确
6. 在持续对话中，必须严格保留之前提取的所有信息，除非新输入信息与之前信息明确冲突，否则不得更改已有信息

# 数据结构规范
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

# 执行流程
1. 严格分析对话内容，提取有效信息
2. 将提取的信息映射到预定义标签
3. 执行同义词归一化处理
4. 根据用户输入的语气、用词和上下文推断额外信息（如情感、消费意愿等），确保推断合理且准确
5. 在持续对话中，基于之前的画像信息进行更新，严格保留已有信息，除非新信息与旧信息明确冲突，否则仅补充新信息
6. 验证JSON结构完整性
7. 确保所有字段都有值（或["未知"]）

# 归一化规则示例
- occupation: "医生"、"医师"、"大夫" → "医生"
- age: "20多岁"、"20岁左右" → "18-25岁"
- lifestyle: "经常健身"、"爱运动" → "健身习惯"
- emotion: "满意"、"很高兴" → "积极"; "担心"、"忧虑" → "消极"
- character: "果断"、"坚决" → "果断型"
- current_use: "热玛吉"、"热玛吉疗程" → "热玛吉"
- potential_needs: "想瘦脸"、"希望面部紧致" → "瘦脸需求"
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


class SocialProfileStructure(BaseModel):
    occupation: str = Field(description="职业")
    age: str = Field(description="年龄")
    region: str = Field(description="地区")
    lifestyle: str = Field(description="生活方式")
    family_status: str = Field(description="家庭状况")
    emotion: str = Field(description="情感状态")

class UserProfileStructure(BaseModel):
    """用户画像数据结构"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    social_profile: SocialProfileStructure = Field(default_factory=dict, description="社会属性")
    # personality_traits: dict = Field(default_factory=dict, description="性格特质")
    # consumption_profile: dict = Field(default_factory=dict, description="消费偏好")
    # product_intent: dict = Field(default_factory=dict, description="产品意向")
    # customer_lifecycle: dict = Field(default_factory=dict, description="客户生命周期")

class ProfileAgentState(AgentState):
    error_message: Optional[str]
    structured_response: Optional[UserProfileStructure]

class ProfileConfigSchema(TypedDict):
    pass

class ProfileAgentStateInput(TypedDict):
    messages: List[BaseMessage]

class FinalProfileResponseFormat(BaseModel):
    """The final output response format of the Profile generator agent."""

    structured_response: UserProfileStructure = Field(description="用户画像标签")
    error_message: Optional[str] = Field(description="错误信息", default=None)

# 初始化模型
chat_model = init_chat_model(
    model=GOOGLE_GEMINI_FLASH_MODEL,
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

    # 添加历史对话作为上下文
    if isinstance(state, dict):
        messages.extend(state.get("messages", []))

    # 添加生成画像的指令
    messages.append(HumanMessage(content="请严格按照指定的 JSON 结构输出用户画像，分析以上对话内容，结合其中的用户表达和行为线索，尽量完善五大维度的用户画像内容。若某些信息无法判断，请填写\"未知\"，但不要随意编造与上下文无关的内容。如果这是持续对话，请基于之前的画像信息进行更新和补充，严格保留之前提取的所有信息，除非新信息与旧信息明确冲突，否则仅补充新信息。特别注意年龄等关键信息的准确性，确保与用户提供的信息一致。禁止输出除 JSON 结构外的任何其他文本或问候语。"))

    return messages

# 创建agent
profile_agent = create_react_agent(
    model=chat_model,
    tools=[],
    name="profile_agent",
    state_schema=ProfileAgentState,
    config_schema=ProfileConfigSchema,
    response_format=FinalProfileResponseFormat,
    prompt=build_profile_prompt,
)

async def profile_agent_node(state: ProfileAgentState, config: RunnableConfig):
    """调用用户画像生成agent"""
    current_conversation_messages = state.get("messages", [])
    
    try:
        agent_response = await asyncio.to_thread(
            profile_agent.invoke,
            {"messages": current_conversation_messages},
            config,
        )
        
        # 直接使用agent返回的结构化数据
        return {
            "structured_response": agent_response.get("structured_response"),
            "error_message": None,
            "messages": current_conversation_messages,
        }
    
    except Exception as e:
        print(f"Error during profile agent invocation: {e}")
        return {
            "structured_response": UserProfileStructure(),
            "error_message": str(e)
        }

# 构建工作流
profile_label_graph = (
    StateGraph(ProfileAgentState, input=ProfileAgentStateInput, config_schema=ProfileConfigSchema)
    .add_node("profile_generator", profile_agent_node)
    .add_edge(START, "profile_generator")
    .compile()
)