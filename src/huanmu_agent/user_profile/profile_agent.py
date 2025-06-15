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
1. 每个字段必须提供具体内容, 不能为空
2. 内容要详细具体，包含示例说明
3. 必须返回纯JSON格式,不要包含任何额外文本或注释
"""

class UserProfileStructure(BaseModel):
    """用户画像数据结构"""
    demographic: str = Field(description="人口统计特征")
    behavioral: str = Field(description="行为特征") 
    psychological: str = Field(description="心理特征")
    pain_points: str = Field(description="需求痛点")

class ProfileAgentState(AgentState):
    error_message: Optional[str]
    structured_response: Optional[UserProfileStructure]

class ProfileConfigSchema(TypedDict):
    pass

class ProfileAgentStateInput(TypedDict):
    messages: List[BaseMessage]

# 初始化模型
profile_model = init_chat_model(
    model="gpt-3.5-turbo",
    model_provider="openai",
    temperature=0.5
)

def build_profile_prompt(state: AgentState, config: RunnableConfig) -> List[BaseMessage]:
    messages = [{
        "role": "system", 
        "content": PROFILE_SYSTEM_PROMPT
    }]

    # 添加历史对话作为上下文
    if isinstance(state, dict):
        messages.extend(state.get("messages", []))
    else:
        messages.extend(state.messages)

    # 添加生成画像的指令
    messages.append(HumanMessage(content="请根据以上对话历史生成用户画像"))

    return messages

# 创建agent
profile_agent = create_react_agent(
    model=profile_model,
    tools=[],
    name="profile_agent",
    state_schema=ProfileAgentState,
    config_schema=ProfileConfigSchema,
    response_format=UserProfileStructure,
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
        
        return {
            "structured_response": agent_response.get("structured_response"),
            "error_message": None,
        }
    
    except Exception as e:
        print(f"Error during profile agent invocation: {e}")
        return {"error_message": str(e)}

# 构建工作流
profile_graph = (
    StateGraph(ProfileAgentState, input=ProfileAgentStateInput, config_schema=ProfileConfigSchema)
    .add_node("profile_generator", profile_agent_node)
    .add_edge(START, "profile_generator")
    .compile()
)
