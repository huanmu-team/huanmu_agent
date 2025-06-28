import json
from typing import List, Dict, Any, TypedDict
import re
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from constant import GOOGLE_GEMINI_FLASH_MODEL
from pydantic import BaseModel, Field

class XiaohongshuResearchState(TypedDict):
    """小红书研究代理状态"""
    messages: List[BaseMessage]

XIAOHONGSHU_RESEARCH_PROMPT = """你是一个专业的小红书内容研究专家。你的任务是：
1. 从用户最新的消息中识别出需要研究的产品信息。
2. 生成一个合适的搜索查询。
3. 调用`search_xiaohongshu`工具在小红书上进行搜索。
4. 将搜索结果传递给`analyze_xhs_content`工具进行分析。
5. 将最终的JSON分析报告作为你的最终答案。"""

class ContentTrends(BaseModel):
    popular_topics: List[str] = Field(description="热门话题")
    title_patterns: List[str] = Field(description="标题特点")
    content_structure: List[str] = Field(description="内容结构特点")

class UserInsights(BaseModel):
    pain_points: List[str] = Field(description="用户痛点")
    preferences: List[str] = Field(description="用户偏好")
    engagement_triggers: List[str] = Field(description="互动触发点")

class VisualInsights(BaseModel):
    image_styles: List[str] = Field(description="图片风格")
    color_schemes: List[str] = Field(description="配色方案")

class XhsAnalysisReport(BaseModel):
    """小红书内容分析报告的结构化输出。"""
    content_trends: ContentTrends
    user_insights: UserInsights
    visual_insights: VisualInsights

# --- LLM and Tools Definition ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

@tool
def search_xiaohongshu(query: str) -> List[Dict[str, Any]]:
    """根据给定的查询词，在小红书（通过Tavily）上搜索相关内容。"""
    try:
        tavily = TavilySearchResults(max_results=5)
        results = tavily.invoke({"query": f"小红书 {query}"})
        return [{"content": r["content"], "url": r["url"]} for r in results]
    except Exception as e:
        return [{"error": f"小红书搜索失败: {e}"}]

@tool
def analyze_xhs_content(search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析小红书搜索结果，提取洞察并以JSON格式返回。"""
    system_prompt = "分析以下小红书内容，并根据提供的JSON schema格式化你的输出。"
    analyzer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2).with_structured_output(XhsAnalysisReport)
    
    content_text = "\n\n".join([f"URL: {r.get('url', 'N/A')}\n内容: {r.get('content', '')}" for r in search_results])
    user_prompt = f"请分析以下小红书内容：\n\n{content_text}"
    
    response = analyzer_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    return response.dict()

# --- Agent and Graph Definition ---
tools = [search_xiaohongshu, analyze_xhs_content]

xiaohongshu_agent = create_react_agent(llm, tools)

def agent_node(state: XiaohongshuResearchState):
    """
    包装节点，用于在调用代理前注入系统提示。
    """
    # This agent needs the product info, which is passed in the messages.
    # The orchestrator will prepare a message like:
    # "产品信息如下，请进行小红书研究:\n{...json...}"
    # So, we just need to prepend the static system prompt.
    messages = [SystemMessage(content=XIAOHONGSHU_RESEARCH_PROMPT)] + state["messages"]
    
    result = xiaohongshu_agent.invoke({"messages": messages})
    
    return {"messages": result["messages"]}

workflow = StateGraph(XiaohongshuResearchState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.set_finish_point("agent")
xiaohongshu_research_graph = workflow.compile() 