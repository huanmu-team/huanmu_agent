"""朋友圈评论 Agent - 处理文本和图片，生成评论."""
from langchain_core.messages import AnyMessage, BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import asyncio
from huanmu_agent.Post_Comments.url_to_text import process_images_to_descriptions
from constant import GOOGLE_GEMINI_FLASH_MODEL

# 数据模型定义
class CommentResponse(BaseModel):
    """朋友圈评论响应."""
    comment: Optional[str] = Field(description="朋友圈评论")
    error_message: Optional[str] = Field(default=None, description="出错时的错误信息")

class CharacterprofileConfigSchema(TypedDict):
    agent_name: str
    agent_gender: str
    agent_personality: str


class CommentReplyAgentStateInput(TypedDict):
    """评论 Agent 输入状态."""
    context: str
    urls: Optional[List[str]]  # 支持多个图片URL

class CommentAgentState(AgentState):
    """评论 Agent 状态."""
    context: Optional[str]
    urls: Optional[List[str]]  # 支持多个图片URL
    structured_response: Optional[CommentResponse]
    error_message: Optional[str]

class CommentAnalysisAgentStateOutput(TypedDict):
    """评论分析 Agent 输出状态."""
    structured_response: Optional[str]

# 初始化模型 - 使用OpenAI避免Google Cloud配置问题
llm = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai",
    temperature=0.7,
)
# llm = init_chat_model(
#     model="gemini-2.5-flash",
#     model_provider="google_vertexai",
#     temperature=0.7,
# )
# 如果要使用Google Vertex AI，需要先配置认证：
# llm = init_chat_model(
#     model=GOOGLE_GEMINI_FLASH_MODEL,
#     model_provider="google_vertexai",
#     temperature=0.7,
# )

# 评论生成提示词
def prompt_comment_generation(state: AgentState, config: RunnableConfig) -> List[AnyMessage]:
    """评论生成提示词."""
    # 从state中获取处理后的消息内容
    messages_content = ""
    if state.get("messages"):
        latest_message = state["messages"][-1]
        messages_content = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
    
    system_msg = f"""
你是一个叫{config["configurable"].get("agent_name", "小七")}的{config["configurable"].get("agent_gender", "女")}性，性格{config["configurable"].get("agent_personalityget", "热情")}。

【核心目标】
你的评论目的是给朋友留下好印象，展现你是一个{config["configurable"].get("agent_personalityget", "热情")}的人。通过合适的评论来维护和增进人际关系。

【评论策略】
1. 优先原则：能评论就评论，给人温暖正面的感受
2. 内容判断：先识别朋友圈的情绪/意图和内容类型，选择最合适的回应方式
3. 特殊情况：对于不适合评论的内容，选择沉默（返回None）

【什么时候评论】
- 日常生活分享：积极互动，表达关心
- 开心喜悦：真诚祝福，分享快乐
- 悲伤困难：给予支持，传递温暖
- 成就展示：给予认可，表达赞美
- 求助征询：提供建议，展现关心

【什么时候保持沉默（返回None）】
- 涉政、涉宗教、涉色情、涉暴力、涉歧视、敏感话题：避免争议
- 负面情绪爆发：避免火上浇油
- 纯商业广告链接：避免显得过于商业化
- 内容不清晰或无法理解：避免误解
- 内容太庄重或者太严肃：保持沉默

【评论风格要求】
- 字数：3-25字，简洁而有温度
- 语调：符合你的人设{config["configurable"].get("agent_personalityget", "热情")}
- 互动性：体现关心，鼓励进一步交流
- 避免：敷衍客套、过度表情符号、具体邀约安排
- 减少使用标点符号装饰（如"~~""##"等）

【不同情境的评论示例】
• 美食分享："看起来好香啊" "这家店在哪里"
• 风景照片："好美的地方" "心情都变好了"
• 工作成就："太棒了" "为你开心"
• 生活日常："哈哈同感" "生活真美好"
• 困难求助："抱抱，会好起来的" "需要帮忙随时找我"

直接输出你的评论内容，不要包含任何解释或额外说明。

朋友圈内容：{messages_content}
"""
    return [{"role": "system", "content": system_msg}] + state["messages"]

# 创建评论生成 Agent
comment_generation_agent = create_react_agent(
    model=llm,
    tools=[],
    name="comment_generation_agent", 
    state_schema=CommentAgentState,
    config_schema=CharacterprofileConfigSchema,
    response_format=CommentResponse,
    prompt=prompt_comment_generation,
)



# 处理图片和文本的节点函数
async def process_content_node(state: CommentAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """处理朋友圈内容，包括文字和图片，转换为合适的文本格式传给大模型."""
    try:
        context = state.get("context", "")
        urls = state.get("urls", [])
        
        print(f"[DEBUG] 开始处理内容，context: {context}")
        print(f"[DEBUG] URLs: {urls}")
        
        if not context and not urls:
            print("[DEBUG] 没有输入内容，返回错误")
            return {
                "structured_response": "",
                "error_message": "没有输入任何内容",
                "messages": []
            }
        
        # 从文本内容开始
        if not context:
            enhanced_content = ""
        else:
            enhanced_content = context

        print(f"[DEBUG] 初始化enhanced_content: {enhanced_content}")
        
        # 处理图片URL（如果提供了urls参数）
        urls_to_process = []
        if urls:
            urls_to_process.extend(urls)
            
        if urls_to_process:
            print(f"[DEBUG] 开始处理{len(urls_to_process)}个图片URL...")
            image_descriptions = await process_images_to_descriptions(urls_to_process,llm)
            
            if image_descriptions:
                enhanced_content += f"\n\n图片描述：{' '.join(image_descriptions)}"
                print(f"[DEBUG] 添加图片描述后的enhanced_content: {enhanced_content}")
        
        # 创建消息用于评论生成
        enhanced_messages = [HumanMessage(content=enhanced_content)]
        enhanced_state = {**state, "messages": enhanced_messages}
        print(f"[DEBUG] 准备调用评论生成agent，enhanced_content: {enhanced_content}")
        
        # 调用评论生成agent
        agent_response = await asyncio.to_thread(
            comment_generation_agent.invoke,
            enhanced_state,
            config
        )
        print(f"[DEBUG] Agent响应: {agent_response}")
        
        # 提取评论内容
        if agent_response.get("messages"):
            last_msg = agent_response["messages"][-1]
            final_comment = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        else:
            final_comment = "评论生成失败"
        
        print(f"[DEBUG] 最终评论: {final_comment}")
        
        return {
            "structured_response": final_comment,
            "error_message": None,
            "messages": []
        }
        
    except Exception as e:
        return {
            "structured_response": f"处理失败: {str(e)}",
            "error_message": str(e),
            "messages": []
        }

# 创建朋友圈评论 Graph
comment_analysis_graph = (
    StateGraph(
        CommentAgentState,
        input=CommentReplyAgentStateInput,
        config_schema=CharacterprofileConfigSchema,
        output=CommentAnalysisAgentStateOutput
    )
    .add_node("process_content", process_content_node)
    .add_edge(START, "process_content")
    .add_edge("process_content", END)
    .compile()
)


