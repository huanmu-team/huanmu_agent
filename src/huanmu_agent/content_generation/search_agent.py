import datetime
import json
import requests
from typing import List, Optional

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# --- 1. 结构化输出模型 (Pydantic Models for Structured Output) ---
class FestivalInfo(BaseModel):
    """定义单个节日的信息结构。"""
    date: str = Field(description="节日日期，格式为 YYYY-MM-DD")
    name: str = Field(description="节日、纪念日或营销节点的名称")
    description: Optional[str] = Field(description="关于该节日的简短描述", default="")
    is_holiday: bool = Field(description="当天是否为官方节假日或周末")

class HotTopicInfo(BaseModel):
    """定义单个热点话题的信息结构。"""
    title: str = Field(description="热点标题")
    summary: str = Field(description="热点摘要")

class SearchAgentResponse(BaseModel):
    """定义最终的、完整的、结构化的搜索结果。"""
    festivals: List[FestivalInfo] = Field(description="未来7天内节日和周末列表")
    hot_topics: List[HotTopicInfo] = Field(description="近期美妆医美领域的热点话题列表")


# --- 2. Agent 状态定义 (Agent State Definition) ---
class SearchAgentState(TypedDict):
    messages: List[BaseMessage]


# --- 3. 工具定义 (Tool Definitions) ---
@tool
def get_holidays_from_api(days: int = 7) -> str:
    """
    通过调用一个专门的API，获取未来指定天数内的所有中国节假日和周末。
    返回一个包含节日信息的JSON字符串。
    """
    today = datetime.date.today()
    holidays = []
    
    # 循环获取未来`days`天的日期信息
    for i in range(days):
        current_date = today + datetime.timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        api_url = f"https://publicapi.xiaoai.me/holiday/day?date={date_str}"
        
        try:
            response = requests.get(api_url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # API返回的数据结构是 {"code": 0, "msg": "ok", "data": [{...}]}
            if data.get("code") == 0 and data.get("data"):
                day_info = data["data"][0]
                # daytype: 1-节假日, 2-双休日
                if day_info.get("daytype") in [1, 2]:
                    holidays.append({
                        "date": day_info["date"],
                        "name": day_info["holiday"] or day_info["week_desc_cn"],
                        "is_holiday": True
                    })
        except requests.exceptions.RequestException as e:
            print(f"调用节假日API失败 (日期: {date_str}): {e}")
            # 即使单日失败也继续
            continue
            
    return json.dumps(holidays, ensure_ascii=False)

@tool
def search_marketing_and_trendy_festivals() -> str:
    """
    通过网页搜索，查找未来一周内中国的营销日历、网络热点节日和纪念日。
    这个工具是对官方节假日API的补充。
    """
    try:
        today = datetime.datetime.now()
        end_date = today + datetime.timedelta(days=7)
        date_range_str = f"{today.strftime('%Y年%m月%d日')}到{end_date.strftime('%Y年%m月%d日')}"
        query = f"中国在 {date_range_str} 期间，除了法定节假日外，还有哪些网络节日、纪念日、或者营销活动节点？"
        tavily = TavilySearchResults(max_results=3)
        results = tavily.invoke({"query": query, "include_answer": True, "country": "cn"})
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"搜索营销节日失败: {e}"

@tool
def get_recent_hot_topics_search_results() -> str:
    """
    搜索近期在美容、美妆、医美领域的营销话题、社交媒体趋势和热点事件。
    返回原始的、未经处理的搜索结果JSON字符串。
    """
    try:
        current_date = datetime.datetime.now()
        year_month = current_date.strftime("%Y年%m月")
        query = f"{year_month} 美妆医美行业热门营销话题、社交媒体热门事件、美容护肤最新趋势"
        tavily = TavilySearchResults(max_results=3)
        results = tavily.invoke({"query": query, "include_answer": True, "country": "cn"})
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"获取热点话题失败: {e}"

@tool
def structure_search_results(
    official_holidays_json: str, 
    marketing_festivals_json: str, 
    hot_topics_json: str
) -> SearchAgentResponse:
    """
    接收来自API的官方节假日、来自搜索的营销节日、以及来自搜索的热点话题，
    然后将它们清洗、合并、去重，并最终以一个结构化的JSON对象返回。
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2).with_structured_output(SearchAgentResponse)
    
    system_prompt = """你是一个数据整合专家。你的任务是处理三个JSON字符串数据源：
1.  `official_holidays_json`: 来自权威API的官方节假日和周末列表。
2.  `marketing_festivals_json`: 来自网页搜索的网络节日和营销节点。
3.  `hot_topics_json`: 来自网页搜索的近期热点话题。

请执行以下操作：
1.  合并`official_holidays_json`和`marketing_festivals_json`中的信息，提取出所有未来7天内的节日/纪念日，并为它们生成`date`, `name`, `description`, `is_holiday`字段。注意去重，如果同一天有多个事件，可以合并描述。
2.  从`hot_topics_json`中，提取3-5个最重要、最相关的近期美妆医美领域营销热点。
3.  将所有信息整合成一个`SearchAgentResponse` JSON对象返回。"""

    user_prompt = f"""这是原始数据:
--- 官方节假日API数据 ---
{official_holidays_json}

--- 营销节日搜索结果 ---
{marketing_festivals_json}

--- 热点话题搜索结果 ---
{hot_topics_json}

请根据你的任务进行处理和结构化。"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    
    return response


# --- 4. Agent 和图的定义 (Agent and Graph Definition) ---
SEARCH_SYSTEM_PROMPT = """你是一个智能搜索助理。你的任务是为用户全面地搜集未来的节日和近期的热点话题。
请按以下步骤操作：
1.  首先，并行调用 `get_holidays_from_api`, `search_marketing_and_trendy_festivals`, 和 `get_recent_hot_topics_search_results` 这三个工具，以全面地获取官方假日、营销节点和行业热点。
2.  然后，将这三个工具返回的所有结果，作为参数传递给 `structure_search_results` 工具进行最终的整合和分析。
3.  `structure_search_results` 工具会生成最终的、干净的JSON结果。请直接将这个JSON结果作为你的最终答案输出。"""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
tools = [
    get_holidays_from_api,
    search_marketing_and_trendy_festivals,
    get_recent_hot_topics_search_results,
    structure_search_results,
]

# 1. 创建一个不含提示的纯净代理
search_agent = create_react_agent(llm, tools)

# 2. 创建一个包装节点
def agent_node(state: SearchAgentState) -> dict:
    """
    一个包装节点，用于在调用代理前注入系统提示，并处理空输入。
    """
    if "messages" not in state or not state["messages"]:
        state["messages"] = [HumanMessage(content="请帮我搜索未来一周的节日和近期的热点话题。")]

    messages_with_prompt = [SystemMessage(content=SEARCH_SYSTEM_PROMPT)] + state["messages"]
    
    result = search_agent.invoke({"messages": messages_with_prompt})
    
    return result

# 3. 在图中注册包装节点
search_agent_graph = StateGraph(SearchAgentState)
search_agent_graph.add_node("agent", agent_node)
search_agent_graph.set_entry_point("agent")
search_agent_graph.set_finish_point("agent")
search_agent_graph = search_agent_graph.compile()


# --- 5. 示例调用 (for direct testing) ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    def get_final_response_from_state(result: dict) -> Optional[dict]:
        """一个辅助函数，用于从最终状态的消息中提取结构化输出。"""
        last_message_content = result["messages"][-1].content
        try:
            return json.loads(last_message_content)
        except (json.JSONDecodeError, TypeError):
            print("警告: 代理的最终响应不是一个有效的JSON字符串。")
            print("原始最终响应:", last_message_content)
            return None

    try:
        user_input = "请告诉我未来一周的节日和近期热点"
        initial_state = {"messages": [HumanMessage(content=user_input)]}
        
        print(f"--- 正在为输入运行搜索代理: '{user_input}' ---")
        result_state = search_agent_graph.invoke(initial_state)
        
        print("\n--- 代理执行完毕 ---")
        final_response = get_final_response_from_state(result_state)
        
        if final_response and not final_response.get("error"):
            print("\n✅ 搜索成功！")
            print("结构化输出:")
            print(json.dumps(final_response, indent=2, ensure_ascii=False))
        else:
            print("\n❌ 搜索失败。")
            
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
