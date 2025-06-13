from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain.chat_models import init_chat_model
from typing import List, Optional
from langchain_core.runnables import RunnableConfig
import asyncio
from constant import GOOGLE_GEMINI_FLASH_MODEL

MOMENT_SYSTEM_PROMPT = """
你是一个微信朋友圈文案生成助手。
根据以下输入参数，内容主题, 行业, 语言, 每个内容的字数限制, 生成数量, 来生成朋友圈文案。

你需要从以下三个模板中选择一个最合适的来生成文案:

模板1：生活分享类
公式：[心情/槽点] + [讲个小故事] + [求互动/求推荐]
示例 (晒咖啡)：续命水来了！😴 一上午被三个会轰炸，感觉灵魂已掏空。这杯拿铁是最后的倔强。大家今天都还好吗？

模板2：工作/成长类
公式：[抛出痛点/金句] + [你的感悟/解决方案] + [引发共鸣]
示例 (晒加班)：所谓"稳定"，不是呆在原地，而是在任何变化中都有破局的能力。又是一个奋斗到深夜的晚上，敬给所有在路上奔跑的我们。#晚安，打工人#

模板3：知识/好物分享类
公式：[吸睛标题] + [核心亮点1, 2, 3] + [号召行动/在哪买]
示例 (推荐一本书)：这本书，治好了我的精神内耗！年度必读Top3！
作者关于"课题分离"的观点，简直醍醐灌顶。
学会了如何拒绝别人，太爽了！
如果你也常感到焦虑，一定要读读看！

若数量大于1，则生成多个内容。
你的输出应该是一个 list[WeChatMomentStructure] 对象。
WeChatMomentStructure包含以下字段:
- improved_moment: str (朋友圈文案内容)

请根据 row_moments 的具体内容来决定使用哪个模板，并生成优化后的朋友圈文案。
"""

class WeChatMomentStructure(BaseModel):
    """Structure for a single WeChat Moments post."""
    improved_moment: str = Field(description="优化后的朋友圈文案内容， 不要包含标签")

class FinalWeChatMomentResponseFormat(BaseModel):
    """The final output response format of the WeChat Moments generator agent."""
    moments: List[WeChatMomentStructure] = Field(description="朋友圈文案列表")
    error_message: Optional[str] = Field(description="错误信息", default=None)

# --- Agent State ---

class WeChatAgentState(AgentState):
    # Input parameters
    row_moment: str
    moment_number: int
    # Output
    error_message: Optional[str]
    structured_response: Optional[FinalWeChatMomentResponseFormat]

class WeChatMomentConfigSchema(TypedDict):
    system_prompt: str = MOMENT_SYSTEM_PROMPT
    topic: str

class WeChatAgentStateInput(TypedDict):
    row_moment: str
    moment_number: int

# --- System Prompt ---

# Using the model ID found in constant.py to avoid import issues.

chat_model = init_chat_model(
    model=GOOGLE_GEMINI_FLASH_MODEL,
    model_provider="google_vertexai",
    temperature=0.7 # A bit more creative for social media
)

def prompt(
    state: AgentState,
    config: RunnableConfig,
) -> list[AnyMessage]:
    topic = config["configurable"].get("topic","")
    system_prompt = config["configurable"].get("system_prompt", MOMENT_SYSTEM_PROMPT)
    if not topic == "":
        system_msg = f"{system_prompt} User's topic is {topic}"
    else:
        system_msg = system_prompt
    return [{"role": "system", "content": system_msg}] + state["messages"]

wechat_generator_agent = create_react_agent(
    model=chat_model,
    tools=[],  # No external tools needed for this agent
    name="wechat_moment_agent",
    state_schema=WeChatAgentState,
    config_schema=WeChatMomentConfigSchema,
    response_format=FinalWeChatMomentResponseFormat,
    prompt=prompt
)

async def wechat_agent_node(state: WeChatAgentState, config: RunnableConfig):
    """
    Node that invokes the WeChat Moments content generator agent asynchronously.
    """    
    row_moment = state.get("row_moment", "N/A")
    moment_number = state.get("moment_number", 1)
    topic = config["configurable"].get("topic", "N/A")
    system_prompt = config["configurable"].get("system_prompt", MOMENT_SYSTEM_PROMPT)
    print(f"-------------------------------topic-----------------------------------------{topic}")
    print(f"-------------------------------system_prompt-----------------------------------------{system_prompt}")
    current_conversation_messages = state.get("messages", [])
    print(f"current_conversation_messages: {current_conversation_messages}")
    # If this is the first turn, we need to create the initial message.
    if not current_conversation_messages:
        system_msg = [{"role": "system", "content": system_prompt}]
        user_msg = [{"role": "user", "content": f"请帮我生成朋友圈文案。\n\n微信朋友圈原始内容：{row_moment}\n\n用户主题：{topic}\n\n生成数量：{moment_number}"}]
        print(f"system_msg: {system_msg}")
        print(f"user_msg: {user_msg}")
        current_conversation_messages = system_msg + user_msg
    
    try:
        print("---WECHAT AGENT EXECUTING---")
        print(f"Invoking WeChat agent with topic: {row_moment}")
        
        # run the blocking agent in a thread
        agent_response = await asyncio.to_thread(
            wechat_generator_agent.invoke,   # sync version
            {"messages": current_conversation_messages},
            config
        )
        
        return {
            "structured_response": agent_response.get("structured_response"),
            "error_message": None,
            "messages": agent_response.get("messages", [])
        }
        
    except Exception as e:
        print(f"Error during WeChat agent invocation: {e}")
        error_message = f"Error generating WeChat Moments content: {e}"
        return {"error_message": error_message}

# --- Graph Definition ---

wechat_moment_graph = (
    StateGraph(WeChatAgentState, input=WeChatAgentStateInput, config_schema=WeChatMomentConfigSchema)
    .add_node("wechat_generator", wechat_agent_node)
    .add_edge(START, "wechat_generator")
    .compile()
)
