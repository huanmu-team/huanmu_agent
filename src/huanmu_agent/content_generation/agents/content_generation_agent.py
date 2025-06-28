import json
from typing import List, Dict, Any, TypedDict
import re
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from huanmu_agent.configuration import Configuration
import requests
from constant import GOOGLE_GEMINI_FLASH_MODEL
from pydantic import BaseModel, Field

class ContentGenerationState(TypedDict):
    messages: List[BaseMessage]
    product_info: Dict[str, Any]
    xhs_insights: Dict[str, Any]

class FinalContent(BaseModel):
    """最终的营销内容对象。"""
    marketing_copy: str = Field(description="生成的营销文案")
    poster_prompt: str = Field(description="生成的海报提示词")

CONTENT_GENERATION_PROMPT = """你是一个专业的营销内容创作专家。你的任务是：
1. 从历史消息中提取产品信息和小红书洞察。
2. 调用`generate_marketing_copy`工具生成营销文案。
3. 使用上一步生成的文案，调用`generate_poster_prompt`工具创建海报提示词。
4. 最后，将营销文案和海报提示词传递给`create_final_output`工具来生成最终的JSON对象。
5. 将`create_final_output`工具返回的JSON对象作为你的最终答案。"""

# --- LLM and Tools Definition ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
config = Configuration.from_context()

@tool
def generate_marketing_copy(product_info: Dict[str, Any], xhs_insights: Dict[str, Any]) -> str:
    """根据产品信息和小红书洞察生成营销文案。"""
    system_prompt = "你是一位顶级的社交媒体营销文案专家。请根据以下信息，创作一篇引人入胜、符合小红书风格的营销文案。"
    user_prompt = f"产品信息: {json.dumps(product_info, ensure_ascii=False)}\n\n小红书洞察: {json.dumps(xhs_insights, ensure_ascii=False)}"
    
    copy_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
    response = copy_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return response.content

@tool
def generate_poster_prompt(marketing_copy: str) -> str:
    """根据营销文案生成一个详细的、用于AI绘画的海报提示词。"""
    system_prompt = """你是一名富有创意的艺术总监。请根据给定的营销文案，生成一个用于AI文生图模型的详细海报提示词。
这个提示词必须包含以下几个方面：
1.  **画面风格**: 整体的视觉风格，例如：赛博朋克、水彩、极简主义、超现实等。
2.  **构图与主体**: 描述画面的主要元素（如产品、人物、场景）以及它们的位置和关系。
3.  **色彩与光影**: 定义主色调、辅助色以及光线的氛围（如柔和、霓虹、丁达尔效应）。
4.  **文案与排版**: 从营销文案中提炼出主标题、副标题和核心卖点，并规划它们在海报上的大致位置（如：顶部居中、左下角）和推荐的字体风格（如：优雅的衬线字体、现代的无衬线字体）。
5.  **氛围与细节**: 描述海报希望传达的整体感觉，并可以添加一些能增强感染力的细节元素。"""
    user_prompt = f"营销文案:\n{marketing_copy}"
    
    prompt_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    response = prompt_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return response.content

@tool
def create_final_output(marketing_copy: str, poster_prompt: str) -> Dict[str, str]:
    """将营销文案和海报提示词合并成一个最终的JSON对象。"""
    return FinalContent(marketing_copy=marketing_copy, poster_prompt=poster_prompt).dict()

# This tool is removed from the agent's direct access to simplify the flow.
# The orchestrator can decide to call it separately if needed.
def generate_marketing_poster(prompt: str) -> str:
    """(This tool is not used by the agent directly)"""
    # ... implementation remains the same
    pass

# --- Agent and Graph Definition ---
tools = [generate_marketing_copy, generate_poster_prompt, create_final_output]

content_generation_agent = create_react_agent(llm, tools)

def agent_node(state: ContentGenerationState):
    """
    包装节点，用于在调用代理前注入系统提示。
    """
    # The orchestrator will prepare a message with all necessary info.
    messages = [SystemMessage(content=CONTENT_GENERATION_PROMPT)] + state["messages"]
    
    result = content_generation_agent.invoke({"messages": messages})
    
    return {"messages": result["messages"]}

workflow = StateGraph(ContentGenerationState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.set_finish_point("agent")
content_generation_graph = workflow.compile() 