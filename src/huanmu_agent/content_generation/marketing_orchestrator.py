from __future__ import annotations
import json
from typing import List, Dict, Any, TypedDict
import re
from http import HTTPStatus
from dashscope import ImageSynthesis

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from huanmu_agent.configuration import Configuration

from huanmu_agent.content_generation.agents.product_research_agent import (
    product_research_graph,
    ProductResearchState,
)
from huanmu_agent.content_generation.agents.xiaohongshu_research_agent import (
    xiaohongshu_research_graph,
    XiaohongshuResearchState,
)
from huanmu_agent.content_generation.agents.content_generation_agent import (
    content_generation_graph,
    ContentGenerationState,
)

# --- Poster Generation Logic ---
def generate_marketing_poster(prompt: str) -> str:
    """
    根据提示词，使用DashScope SDK生成营销海报。
    """
    config = Configuration.from_context()
    try:
        response = ImageSynthesis.call(
            model=config.dashscope_model,
            prompt=prompt,
            api_key=config.dashscope_api_key,
            n=1,
            size='720*1280'
        )
        if response.status_code == HTTPStatus.OK:
            if response.output and response.output.results:
                return response.output.results[0].url
            else:
                return f"海报生成任务成功，但输出中没有结果。响应: {response}"
        else:
            return f"海报生成API调用失败: Code: {response.code}, Message: {response.message}"
    except Exception as e:
        return f"调用DashScope SDK时出现异常: {e}"

# --- Agent State ---
class OrchestratorState(TypedDict):
    """
    编排器的状态，负责在不同子代理之间传递信息。
    """
    messages: List[BaseMessage]  # 初始用户输入和最终结果
    product_info: Dict[str, Any]  # 产品研究结果
    xhs_insights: Dict[str, Any]  # 小红书研究结果
    generated_content: Dict[str, Any]  # 最终生成的内容
    final_poster_url: str  # New field for the final output
    next_step: str # 下一步要执行的节点

# --- Helper function to extract JSON from the last AI message ---
def get_final_json_output(messages: List[BaseMessage]) -> Dict[str, Any]:
    """从最后一条AI或工具消息中提取JSON内容。"""
    last_message = messages[-1]
    content = ""
    if isinstance(last_message, AIMessage):
        content = last_message.content
    elif isinstance(last_message, ToolMessage):
        content = last_message.content

    # The agent might return a JSON blob wrapped in ```json ... ```
    match = re.search(r"```json\n(.*)\n```", content, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = content

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"error": "Final output was not valid JSON", "raw_content": json_str}

# --- Graph Nodes ---

def run_product_research(state: OrchestratorState) -> Dict[str, Any]:
    """运行产品研究子图"""
    print("--- 1. 开始产品研究 ---")
    initial_product_state = ProductResearchState(messages=state["messages"])
    result = product_research_graph.invoke(initial_product_state)
    
    product_info = get_final_json_output(result["messages"])
    
    if "error" in product_info:
        raise ValueError(f"产品研究失败: {product_info.get('raw_response', product_info['error'])}")
        
    print("--- 产品研究完成 ---")
    return {"product_info": product_info}

def run_xiaohongshu_research(state: OrchestratorState) -> Dict[str, Any]:
    """运行小红书研究子图"""
    print("--- 2. 开始小红书研究 ---")
    product_info_message = HumanMessage(content=f"产品信息如下，请进行小红书研究:\n{json.dumps(state['product_info'], ensure_ascii=False)}")
    initial_xhs_state = XiaohongshuResearchState(messages=[product_info_message])
    result = xiaohongshu_research_graph.invoke(initial_xhs_state)
    
    xhs_insights = get_final_json_output(result["messages"])

    if "error" in xhs_insights:
        raise ValueError(f"小红书研究失败: {xhs_insights.get('raw_response', xhs_insights['error'])}")
        
    print("--- 小红书研究完成 ---")
    return {"xhs_insights": xhs_insights}

def run_content_generation(state: OrchestratorState) -> Dict[str, Any]:
    """运行内容生成子图"""
    print("--- 3. 开始内容生成 ---")
    content_generation_prompt = (
        f"请根据以下信息生成营销内容:\n\n"
        f"--- 产品信息 ---\n{json.dumps(state['product_info'], ensure_ascii=False)}\n\n"
        f"--- 小红书洞察 ---\n{json.dumps(state['xhs_insights'], ensure_ascii=False)}"
    )
    content_gen_message = HumanMessage(content=content_generation_prompt)
    initial_content_state = ContentGenerationState(messages=[content_gen_message])
    result = content_generation_graph.invoke(initial_content_state)
    
    generated_content = get_final_json_output(result["messages"])
    
    if "error" in generated_content:
        raise ValueError(f"内容生成失败: {generated_content.get('raw_response', generated_content['error'])}")
        
    print("--- 内容生成完成 ---")
    return {"generated_content": generated_content}

def generate_final_poster(state: OrchestratorState) -> Dict[str, str]:
    """最后一步：使用生成的海报提示词来创建最终的海报。"""
    print("--- 4. 开始生成最终海报 ---")
    poster_prompt = state.get("generated_content", {}).get("poster_prompt")
    
    if not poster_prompt:
        print("警告: 未找到海报提示词，跳过海报生成。")
        return {"final_poster_url": "N/A - No prompt provided"}

    poster_url = generate_marketing_poster(poster_prompt)
    print(f"--- 海报生成完成, URL: {poster_url} ---")
    
    return {"final_poster_url": poster_url}

# --- Conditional Edges ---

def decide_next_step(state: OrchestratorState) -> str:
    """根据当前状态决定下一个节点"""
    if not state.get("product_info"):
        return "product_research"
    if not state.get("xhs_insights"):
        return "xiaohongshu_research"
    if not state.get("generated_content"):
        return "content_generation"
    return "end"

# --- Build the graph ---
workflow = StateGraph(OrchestratorState)

workflow.add_node("product_research", run_product_research)
workflow.add_node("xiaohongshu_research", run_xiaohongshu_research)
workflow.add_node("content_generation", run_content_generation)
workflow.add_node("generate_poster", generate_final_poster)

workflow.set_entry_point("product_research")

workflow.add_edge("product_research", "xiaohongshu_research")
workflow.add_edge("xiaohongshu_research", "content_generation")
workflow.add_edge("content_generation", "generate_poster")
workflow.add_edge("generate_poster", END)

orchestrator_graph = workflow.compile() 