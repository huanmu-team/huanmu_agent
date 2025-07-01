"""朋友圈评论 Agent - 处理文本和图片，生成评论."""
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import asyncio
from huanmu_agent.Post_Comments.url_to_text import process_images_to_descriptions
from constant import GOOGLE_GEMINI_FLASH_MODEL
class CharacterprofileConfigSchema(TypedDict):
    """人物参数"""
    agent_name: str
    agent_gender: str
    agent_personality: str

class CommentInput(TypedDict):
    """评论输入."""
    context: str
    urls: Optional[List[str]]  # 支持多个图片URL

class CommentState(TypedDict):
    """评论状态."""
    context: Optional[str]
    urls: Optional[List[str]]  # 支持多个图片URL
    enhanced_content: Optional[str]
    structured_response: Optional[str]
    error_message: Optional[str]

class CommentOutput(BaseModel):
    """评论输出."""
    structured_response: Optional[str]
    error_message: Optional[str] = Field(default=None, description="出错时的错误信息")

# 初始化模型
llm = init_chat_model(
    model=GOOGLE_GEMINI_FLASH_MODEL,
    model_provider="google_vertexai",
    temperature=0.7,
)

async def process_content_node(state: CommentState) -> Dict[str, Any]:
    """处理朋友圈内容，包括文字和图片，转换为合适的文本格式."""
    try:
        context = state.get("context", "")
        urls = state.get("urls") or []  # 确保urls不是None
        
        print(f"[DEBUG] 开始处理内容，context: {context}")
        print(f"[DEBUG] URLs: {urls}")
        
        if not context and not urls:
            print("[DEBUG] 没有输入内容，返回错误")
            return {
                "enhanced_content": "",
                "error_message": "没有输入任何内容"
            }
        
        # 从文本内容开始
        enhanced_content = context if context else ""
        print(f"[DEBUG] 初始化enhanced_content: {enhanced_content}")
        
        # 处理图片URL（如果提供了urls参数）
        if urls and isinstance(urls, list):
            print(f"[DEBUG] 开始处理{len(urls)}个图片URL...")
            image_descriptions = await process_images_to_descriptions(urls, llm)

            if image_descriptions:
                enhanced_content += f"\n\n{' '.join(image_descriptions)}"
                print(f"[DEBUG] 添加图片描述后的enhanced_content: {enhanced_content}")
        
        return {
            "enhanced_content": enhanced_content,
            "error_message": None
        }
        
    except Exception as e:
        return {
            "enhanced_content": "",
            "error_message": str(e)
        }

async def generate_comment_node(state: CommentState, config: RunnableConfig) -> Dict[str, Any]:
    """生成朋友圈评论."""
    try:
        enhanced_content = state.get("enhanced_content", "")
        
        if not enhanced_content:
            return {
                "structured_response": "",
                "error_message": "没有处理的内容可用于生成评论"
            }
        
        # 安全获取配置参数
        configurable = config.get("configurable", {}) if config else {}
        agent_name = configurable.get("agent_name", "小七")
        agent_gender = configurable.get("agent_gender", "女")
        agent_personality = configurable.get("agent_personality", "热情")
        
        system_msg = f"""
你是一个叫{agent_name}的{agent_gender}性，性格{agent_personality}。

【核心目标】
你的评论目的是给朋友留下好印象，展现你是一个{agent_personality}的人。通过合适的评论来维护和增进人际关系。

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
- 语调：符合你的人设{agent_personality}
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
========================================================
朋友圈内容：
{enhanced_content}
========================================================
"""
        
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content="请根据以上朋友圈内容生成评论")
        ]
        
        print(f"[DEBUG] 准备调用LLM生成评论")
        
        # 直接调用LLM
        response = await asyncio.to_thread(llm.invoke, messages)
        final_comment = response.content if hasattr(response, 'content') else str(response)
        
        print(f"[DEBUG] 生成的评论: {final_comment}")

        return {
            "structured_response": final_comment,
            "error_message": None
        }
        
    except Exception as e:
        return {
            "structured_response": "",
            "error_message": str(e)
        }

# 创建朋友圈评论 Workflow
comment_analysis_graph = (
    StateGraph(
        CommentState,
        input=CommentInput,
        config_schema=CharacterprofileConfigSchema,
        output=CommentOutput
    )
    .add_node("process_content", process_content_node)
    .add_node("generate_comment", generate_comment_node)
    .add_edge(START, "process_content")
    .add_edge("process_content", "generate_comment")
    .add_edge("generate_comment", END)
    .compile()
)