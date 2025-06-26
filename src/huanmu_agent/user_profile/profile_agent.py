"""用户画像生成模块"""
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

from constant import GOOGLE_GEMINI_FLASH_MODEL

PROFILE_SYSTEM_PROMPT = """
你是一个专业的用户画像生成助手。根据用户提供的基本信息、行为数据和偏好，生成详细的用户画像。

用户画像必须严格按以下JSON格式返回：
{
  "demographic": "人口统计特征(年龄、性别、职业等)",
  "behavioral": "行为特征(消费习惯、使用频率等)", 
  "psychological": "心理特征(价值观、兴趣爱好等)",
  "pain_points": "需求痛点"
}

要求：
1. 如果聊天记录为空，则对应字段返回空字符串
2. 必须返回纯JSON格式,不要包含任何额外文本或注释
"""

class UserProfileStructure(BaseModel):
    """用户画像数据结构"""
    demographic: str = Field(description="人口统计特征，如果聊天记录为空 可以为空", default="")
    behavioral: str = Field(description="行为特征，可以为空", default="") 
    psychological: str = Field(description="心理特征，可以为空", default="")
    pain_points: str = Field(description="需求痛点，可以为空", default="")

class ProfileAgentState(AgentState):
    error_message: Optional[str] = None
    structured_response: Optional[UserProfileStructure] = None

class ProfileConfigSchema(TypedDict):
    pass

class ProfileAgentStateInput(TypedDict):
    messages: List[BaseMessage]

class ProfileAgentResponseFormat(BaseModel):
    user_profile: UserProfileStructure = Field(description="用户画像，可以为空")
    error_message: Optional[str] = None

class ProfileAgentStateOutput(TypedDict):
    structured_response: UserProfileStructure


# 初始化模型
chat_model = init_chat_model(
    model=GOOGLE_GEMINI_FLASH_MODEL,
    model_provider="google_vertexai",
    temperature=0.7,  # Balanced creativity
)

def build_profile_prompt(state: AgentState, config: RunnableConfig) -> List[BaseMessage]:
    messages = [{
        "role": "system", 
        "content": PROFILE_SYSTEM_PROMPT
    }]

    # 添加历史对话作为上下文
    if isinstance(state, dict):
        messages.extend(state.get("messages", []))

    # 添加生成画像的指令
    messages.append(HumanMessage(content="请根据以上对话历史生成用户画像"))

    return messages

# 创建agent
profile_agent = create_react_agent(
    model=chat_model,
    tools=[],
    name="profile_agent",
    state_schema=ProfileAgentState,
    config_schema=ProfileConfigSchema,
    response_format=ProfileAgentResponseFormat,
    prompt=build_profile_prompt,
)

async def profile_agent_node(state: ProfileAgentState, config: RunnableConfig):
    """调用用户画像生成agent"""
    current_conversation_messages = state.get("messages", [])

    if not current_conversation_messages:
        return {
            "structured_response": UserProfileStructure(),
            "error_message": None,
            "messages": [],
        }
    
    try:
        agent_response = await asyncio.to_thread(
            profile_agent.invoke,
            {"messages": current_conversation_messages},
            config,
        )
        
        return {
            "structured_response": agent_response.get("structured_response"),
            "error_message": None,
            "messages": current_conversation_messages,
        }
    
    except Exception as e:
        print(f"Error during profile agent invocation: {e}")
        return {"error_message": str(e)}

# 构建工作流
profile_graph = (
    StateGraph(ProfileAgentState, input=ProfileAgentStateInput, config_schema=ProfileConfigSchema, output=ProfileAgentStateOutput)
    .add_node("profile_generator", profile_agent_node)
    .add_edge(START, "profile_generator")
    .compile()
)