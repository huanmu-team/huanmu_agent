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
from constant import GOOGLE_GEMINI_FLASH_MODEL, GOOGLE_GEMINI_PRO_MODEL

# --- System Prompt ---

REPLY_SYSTEM_PROMPT = """
你是一个聊天回复建议助手。 你的任务是为销售人员和客服人员提供示例回复，帮助他们更好地与客户沟通。
根据用户的输入内容、对话上下文、语言风格，提供指定条示例回复。

输出应该是一个 list[ChatReplyStructure] 对象。
ChatReplyStructure 包含以下字段:
- suggestion: str  (单条聊天回复建议，避免包含标签或多余标点)

请确保回复自然、符合中文表达习惯，并避免与输入重复。
"""

# --- Response Schemas ---

class ChatReplyStructure(BaseModel):
    """Structure for a single chat reply suggestion."""

    suggestion: str = Field(description="聊天回复建议内容")


class FinalChatReplyResponseFormat(BaseModel):
    """The final output response format of the Chat Reply generator agent."""

    suggestions: List[ChatReplyStructure] = Field(description="聊天回复建议列表")
    error_message: Optional[str] = Field(description="错误信息", default=None)


# --- Agent State ---

class ChatReplyAgentState(AgentState):
    # Outputs
    error_message: Optional[str]
    structured_response: Optional[FinalChatReplyResponseFormat]

class ChatReplyConfigSchema(TypedDict):
    number: int
    
class ChatReplyAgentStateInput(TypedDict):
    messages: List[BaseMessage]

class ChatReplyAgentOutput(TypedDict):
    structured_response: Optional[FinalChatReplyResponseFormat]

# Initialize chat model (reuse constant to avoid import issues)
chat_model = init_chat_model(
    model=GOOGLE_GEMINI_PRO_MODEL,
    model_provider="google_vertexai",
    temperature=0.7,  # Balanced creativity
)
# Prompt builder

def prompt(state: AgentState, config: RunnableConfig) -> List[AnyMessage]:
    reply_number = config["configurable"].get("number", 3)
    system_msg_content = f"{REPLY_SYSTEM_PROMPT} 回复数量: {reply_number}"
    return [{"role": "system", "content": system_msg_content}] + state["messages"] + [{"role": "user", "content": "根据对话记录，请帮我生成销售或客服人员的聊天回复。"}]


# Create agent
chat_reply_generator_agent = create_react_agent(
    model=chat_model,
    tools=[],  # No external tools needed for this agent
    name="chat_reply_generator_agent",
    state_schema=ChatReplyAgentState,
    config_schema=ChatReplyConfigSchema,
    response_format=FinalChatReplyResponseFormat,
    prompt=prompt,
)
# --- Agent Node ---

async def chat_reply_agent_node(state: ChatReplyAgentState, config: RunnableConfig):
    """Node that invokes the Chat Reply generator agent asynchronously."""

    current_conversation_messages = state.get("messages", [])
    print(f"current_conversation_messages: {current_conversation_messages}")
    reply_number = config["configurable"].get("number", 3)
    user_msg = [
            {
                "role": "user",
                "content": f"请帮我生成聊天回复。\n\n 回复数量：{reply_number}",
            }
        ]
    # If this is the first turn, construct initial messages
    if not current_conversation_messages:
        system_msg = [{"role": "system", "content": REPLY_SYSTEM_PROMPT}]
        
        new_current_conversation_messages = system_msg + user_msg
    else:
        new_current_conversation_messages = current_conversation_messages+user_msg
    try:

        # Run blocking agent in a thread
        agent_response = await asyncio.to_thread(
            chat_reply_generator_agent.invoke,
            {"messages": new_current_conversation_messages},
            config,
        )
        
        print(f"agent_response: {agent_response}")

        return {
            "structured_response": agent_response.get("structured_response"),
            "error_message": None,
            "messages": current_conversation_messages,
        }

    except Exception as e:
        print(f"Error during Chat Reply agent invocation: {e}")
        error_message = f"Error generating Chat Reply suggestions: {e}"
        return {"error_message": error_message}


# --- Graph Definition ---

sales_chat_suggestion_graph = (
    StateGraph(ChatReplyAgentState, input=ChatReplyAgentStateInput, config_schema=ChatReplyConfigSchema, output=ChatReplyAgentOutput)
    .add_node("chat_reply_generator", chat_reply_agent_node)
    .add_edge(START, "chat_reply_generator")
    .compile()
)
