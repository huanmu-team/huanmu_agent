"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast
import datetime
import zoneinfo
from langchain_core.tools import tool
from langgraph.types import interrupt
from langchain_core.messages import HumanMessage, SystemMessage

from huanmu_agent.configuration import Configuration


@tool
async def request_human_assistance(query: str) -> str:
    """When you need help from a human to answer a question, use this tool.

    For example:
    - The user is asking for personal opinions or feelings.
    - The user's question is very ambiguous and you need clarification.
    - The user is expressing strong emotions (e.g., anger, sadness) and may need empathy.
    - The question is beyond your capabilities (e.g., requires real-world actions).
    """
    # 获取当前北京时间
    beijing_tz = zoneinfo.ZoneInfo("Asia/Shanghai")
    current_time = datetime.datetime.now(datetime.timezone.utc).astimezone(beijing_tz)
    time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # 从query中提取转人工原因
    reason = "rescue"  # 默认为救火型
    if "推进" in query or "犹豫" in query:
        reason = "progress"
    elif "成交" in query or "价格" in query:
        reason = "deal"
    
    # 构造人工接管的状态更新
    human_control_update = {
        "is_human_active": True,
        "human_operator_id": "pending",  # 待分配客服ID
        "transfer_reason": reason,
        "transfer_time": time_str
    }
    
    # 调用interrupt并传递状态更新
    interrupt({"human_control": human_control_update})
    
    return f"对话已暂停，等待人工客服介入。转接时间：{time_str}"


@tool
def get_current_time(target_timezone: str = "Asia/Shanghai") -> str:
    """
    Get the current time in a specified IANA timezone.
    This tool ensures time accuracy by getting the server's UTC time and then converting it to your specified target timezone, especially handling daylight saving time.

    Args:
        target_timezone: The name of the target timezone, following IANA timezone database format (e.g., 'Asia/Singapore', 'America/New_York', 'Europe/London').

    Returns:
        Returns the current time string in the target timezone formatted as 'YYYY-MM-DD HH:MM:SS',
        or an error message if the timezone name is invalid.
    """
    try:
        # 使用UTC时间作为可靠基准
        utc_now = datetime.datetime.now(datetime.timezone.utc)

        # 转换为目标时区
        target_tz = zoneinfo.ZoneInfo(target_timezone)
        target_time = utc_now.astimezone(target_tz)

        return target_time.strftime('%Y-%m-%d %H:%M:%S')
    except zoneinfo.ZoneInfoNotFoundError:
        return f"无效的时区: '{target_timezone}'. 请使用有效的 IANA 时区名称 (例如, 'Asia/Singapore')."
    except Exception as e:
        return f"出现错误: {e}"


@tool
async def resume_ai_control(reason: str = "人工处理完成") -> str:
    """恢复AI控制，结束人工接管状态
    
    Args:
        reason: 恢复AI控制的原因
    """
    # 获取当前北京时间
    beijing_tz = zoneinfo.ZoneInfo("Asia/Shanghai")
    current_time = datetime.datetime.now(datetime.timezone.utc).astimezone(beijing_tz)
    time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # 构造恢复AI控制的状态更新
    resume_control_update = {
        "is_human_active": False,
        "human_operator_id": None,
        "transfer_reason": None,
        "transfer_time": None
    }
    
    # 使用interrupt来更新状态（这次是恢复，不是中断）
    interrupt({"human_control": resume_control_update})
    
    # 返回一个工具响应消息，表示人工处理已完成
    return f"人工处理已完成，恢复AI控制。恢复时间：{time_str}，原因：{reason}"


# async def search(query: str) -> Optional[dict[str, Any]]:
#     """Search for general web results.

#     This function performs a search using the Tavily search engine, which is designed
#     to provide comprehensive, accurate, and trusted results. It's particularly useful
#     for answering questions about current events.
#     """
#     configuration = Configuration.from_context()
#     wrapped = TavilySearch(max_results=configuration.max_search_results)
#     return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


TOOLS: List[Callable[..., Any]] = [request_human_assistance, get_current_time]
