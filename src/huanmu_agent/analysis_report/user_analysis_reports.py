from langchain_core.messages import AnyMessage, BaseMessage, AIMessage,HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain.chat_models import init_chat_model
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import asyncio

from constant import GOOGLE_GEMINI_FLASH_MODEL
class UserChunkSummaryResponse(BaseModel):
    user_chunk_summary: Optional[str] = Field(description="用户情况总结")
    error_message: Optional[str] = Field(default=None, description="出错时的错误信息")

class AiDialogStyleResponse(BaseModel):
    ai_dialog_style: Optional[str] = Field(description="AI对话风格描述")
    error_message: Optional[str] = Field(default=None, description="出错时的错误信息")

class RecommendationResponse(BaseModel):
    recommendation: Optional[str] = Field(description="针对用户的服务建议或推荐")
    error_message: Optional[str] = Field(default=None, description="出错时的错误信息")

class UserSummaryAgentState(AgentState):
    structured_response: Optional[UserChunkSummaryResponse]
    error_message: Optional[str]

class AiStyleAgentState(AgentState):
    structured_response: Optional[AiDialogStyleResponse]
    error_message: Optional[str]

class RecommendationAgentState(AgentState):
    structured_response: Optional[RecommendationResponse]
    error_message: Optional[str]

class UserAnalysisAgentState(AgentState):
    structured_response: Optional[str]
    error_message: Optional[str]

class ChatReplyAgentStateInput(TypedDict):
    messages: List[BaseMessage]


# llm = init_chat_model(model="gpt-4o", temperature=0.7, model_provider="openai")
llm = init_chat_model(
    model=GOOGLE_GEMINI_FLASH_MODEL,
    model_provider="google_vertexai",
    temperature=0.7,  # Balanced creativity
)
def prompt_recommendation(state: AgentState) -> List[AnyMessage]:
    system_msg = f"""
你是一名资深医美顾问，以下是医美AI客服与用户的完整多轮对话内容：{state["messages"]}

请你基于该对话内容，撰写一份专业、结构化的医美客户风险预警与个性化服务建议。该报告将用于门店接待前会议或CRM系统录入，需内容详实、分类清晰、具备执行参考价值，覆盖以下六大核心模块：
📌 时间线分析要求：(如果有对话时间或者用户提到的时间）
- 请结合对话时间，评估信息的“新鲜度”，对用户最近表达的信息给予更高权重；
- 若客户长时间未回复、表达犹豫、信息中断，请特别标注“潜在流失预警”；
- 若对时间点表达不明，请使用括号标注“需确认”。

📌 输出格式要求：
- 报告需按照以上六个部分分章节呈现；
- 各部分建议应具体、具执行性，避免泛泛而谈；
- 不得遗漏模块；如信息不足，可注明“暂无明确信息”或“暂无法判断”。
- 本次输出有且只需要"医美客户风险预警与个性化服务建议"部分，请勿将其他内容放进来，且不要放重复内容。

请用正式、专业、中文报告风格撰写。
---------------------------------------------------------------------------------------------------------
一、医美客户风险预警与个性化服务建议
【1、客户阶段判断】（实在无法判断写暂无法判断）
- 请根据对话内容及时间分布，判断该客户处于以下哪一阶段，并简要说明判断依据：
潜在客户期:尚未明确表达兴趣，仅停留在初步浏览或泛泛提问阶段，缺乏具体项目或时间计划。
兴趣客户期:对某类项目有明确关注或多轮提问，但仍未表明价格接受度、时间安排或医生偏好。
意向客户期:明确表达需求 + 指定时间/医生/风格倾向 + 对价格/活动敏感，接近预约/付款。
成交客户期:已预约或已支付，有明确手术安排或体验计划。
忠诚客户期:项目后仍保持联系、反馈积极、愿意转介绍或复购，有KOC潜力。
流失换回期:长时间未联系 / 明确表达犹豫、后悔 / 咨询频率显著下降，需重新激活。
eg.
- 当前阶段判断：意向客户期
- 判断依据：用户多次询问“鼻综合价格”和“恢复期”，表达了对刘主任风格的偏好，并计划“7月初安排手术”（时间近，意图明确）。

【2、客户流失风险预警】（实在无法判断写暂无法判断）
- 分析客户是否存在流失风险，识别关键表现（如沟通频次下降、负面情绪表达增加、咨询后长期未回应等）；
- 判断风险等级（低 / 中 / 高），并说明依据；
- 针对不同风险等级，提出具体应对策略，如一对一专属跟进、激活优惠、情感关怀等；
- 可结合客户特征，提出流失干预节点建议，供后续系统打标签或人手安排。
eg.
是否存在流失风险： 是
关键表现： 用户前期咨询频繁，但在提到“价格有点贵”后，已超过7天未回复客服跟进。(2025年6月20日之后未回复)
风险等级判断： 中风险
应对策略：推送限时优惠，降低其价格顾虑；
建议安排顾问一对一回访，提供分期付款方案或体验活动；
控制频率，避免用户产生打扰感。
干预节点建议：
第7天无回复 → 自动发送温和提醒；
第14天无动作 → 转人工介入判断是否彻底流失。

【3、舆情风险监控与应对】（实在无法判断写暂无法判断）
- 判断客户是否存在潜在的负面传播风险，包括口碑不满、社交活跃度高但情绪波动等；
- 如有明显舆情风险，需区分“普通负面”与“重大风险”，并提出对应处理机制（如客服主管介入、公关团队介入等）；
- 同时建议可发布的正面内容（如客户成功案例、服务承诺等）以主动维护品牌声誉。
eg.
是否存在舆情风险： 有
关键表现： 用户在沟通中多次提及“朋友做了感觉很糟糕”、“网上很多人说恢复期很久”，同时本人在小红书有发帖习惯，粉丝约3000人。
风险等级判断： 普通负面
应对策略：建议由客服主管进行情绪安抚，主动提供医生案例图与术后恢复流程说明，增强信任；提供安心保障内容（如“7天内恢复承诺”“不满意可免费修复”政策）。
正面内容建议：推送与其关注项目相关的真实客户好评与术前术后对比图，强化正面印象；
适当推荐门店口碑医生，降低其顾虑。

【4、购买意图识别与精准引导】（没有的写暂无法判断）
- 分析客户当前的购买意向强度（高 / 中 / 低 / 暂不明确），结合其提问内容、情绪状态、表达意愿进行判断；
- 对于高意图客户，建议推进方式（如专属优惠、明确付款流程、强调案例成功）；
- 对于低意图客户，建议通过建立信任、价值共鸣等方式进行关系维系，避免强推；
- 明确是否推荐在短期内发起营销触达。
eg.
购买意向强度： 高
判断依据： 客户多次主动询问术后恢复时间及付款方式，表达希望尽快预约的意愿。
推进建议：提供专属优惠券或套餐价格；明确说明付款流程和分期选项；强调成功案例和客户好评，增强信心。
营销触达建议： 建议近期重点跟进，适时发送促销活动信息。

【5、到店前准备建议】
- 推荐合适的医生/美容师，并简要说明推荐依据（如审美偏好、风格匹配、用户表达信任等）；
- 提出个性化沟通建议：首次到店时的重点介绍内容、如何快速建立信任、是否适合立即制定方案；
- 判断是否建议客户邀约家属/朋友同行，辅助决策。
eg.
推荐医生/美容师： 推荐刘主任，因客户偏好自然韩式风格，且在对话中多次提及对刘主任的信任。
沟通建议： 首次到店重点介绍手术安全和恢复流程，耐心解答客户疑虑，建立信任感。建议避免一次性推介过多项目，循序渐进。
陪同建议： 建议邀请家属或朋友同行，有助于缓解客户紧张情绪，促进决策。

【6、医疗风险与特别注意事项】（没有的写暂无）
- 明确是否存在需额外关注的身体情况（如体质特殊、过敏、术后担忧等）；
- 针对用户对痛感、价格、恢复期等的关注，提出预警项与缓解建议；
- 判断客户是否存在“冲动消费”倾向，并提供干预建议（如先体验、术前教育等）；
- 提出术后护理重点提示，供医生和护理团队参考。
eg。
身体情况关注： 客户无明显过敏史，但提及对恢复期肿胀较为担忧，需重点关注。
关注点及缓解建议： 针对痛感及价格敏感，建议提供术前痛感管理方案及分期付款选项。
冲动消费判断： 客户表达犹豫，建议先体验项目，避免冲动决策。
术后护理提示： 强调术后护理流程和注意事项，建议定期复诊与护理团队跟进。

【7、服务与营销策略推荐】
- 根据用户特征与意图，推荐适合的项目组合，并简要说明理由；
- 判断是否适配当前门店的营销活动（如新人立减、体验卡等）；
- 提出个性化话术关键词建议（如“专业认证”“恢复周期保障”“隐私保护”等），供顾问接待时使用；
- 判断是否适合“先体验后转化”策略，并说明原因。
---------------------------------------------------------------------------------------------------------

"""
    return [{"role": "system", "content": system_msg}] + state["messages"]
def prompt_user_chunk_summary(state: AgentState) -> List[AnyMessage]:
    system_prompt = f"""
你是一名资深医美顾问，以下是AI客服与用户的完整多轮沟通内容：{state["messages"]}

请你基于该对话内容，撰写一份结构化、专业、实用的医美客户信息洞察与接待准备。该报告将用于门店顾问接待前会议或客户管理系统录入，需突出用户特征、沟通风格、项目意图、成交潜力与潜在阻力等关键信息，具备明确的判断结论与执行建议价值。
📌 时间线分析要求：(如果有对话时间或者用户提到的时间）
- 请结合对话时间，评估信息的“新鲜度”，对用户最近表达的信息给予更高权重；
- 若客户长时间未回复、表达犹豫、信息中断，请特别标注“潜在流失预警”；
- 若对时间点表达不明，请使用括号标注“需确认”。

📌 输出格式要求：
- 报告语言风格需正式、专业；
- 各模块内容完整，如无相关信息请标注“暂未提及”或“暂无法判断”，不得遗漏模块；
- 请确保逻辑清晰、语义准确，避免空泛表达，便于CRM系统归档及门店团队落地执行。
- 本次输出有且只需要"医美客户信息洞察与接待准备"部分，请勿将其他内容放进来，且不要放重复内容。
---------------------------------------------------------------------------------------------------------

二、医美客户信息洞察与接待准备报告
【1】用户情况一句话概括（如能提取）
- 用一句话提炼用户核心画像，包括但不限于：城市、性别、主要意向项目、支付状态、时间安排、美学偏好、医生偏好、社交行为、术后担忧等；
- 示例：**“重庆女性杨女士，计划暑期做双眼皮+隆鼻，偏好韩式自然风格，已支付定金，倾向刘主任，沟通谨慎，具KOC潜力。”**

---

【2】身份标签与状态判断
- 初始身份：新客户 / 老客户 / VIP客户 / 暂无法判断
- 当前状态：潜在客户 / 强意向客户（已预约）/ 已转化客户（已支付）/ 暂无法判断
- 推荐标签：如「活跃高意愿」「价格敏感型」「恢复关注型」「社交型传播者」「观望型客户」等

---

【3】基本用户信息
- 姓名 / 性别 / 年龄（如有提及）：
- 所在城市与到店便利性（如是否跨城）：
- 联系方式 / 昵称（如有）：

---

【4】外貌目标与心理动机
- 关注的部位或外貌问题：
- 美学偏好（如韩式、自然风、欧美等）：
- 情绪动机（如提升气质、婚礼计划、自我认可等）：
- 对整形的接受程度与心理准备度：

---

【5】过往历史与服务记录
- 曾接受的项目及恢复反馈：
- 是否有不满经历或修复项目经历：
- 特殊体质或禁忌情况（如麻药过敏、瘢痕体质）：
- 偏好医生 / 投诉记录（如有）：

---

【6】意向项目与对话意图
- 明确表达的项目意向：
- 当前最关注部位及表达原因：
- 对价格、折扣、付款方式的关注程度：
- 风险关注点（如术后肿胀、麻醉方式等）：
- 预约 / 支付 / 下单状态判断：

---

【7】情绪与沟通风格分析
- 当前情绪状态（如焦虑 / 期待 / 观望）：
- 沟通风格（如理性型 / 情绪型 / 决策缓慢型）：
- 决策主导者（本人 / 家属 / 朋友）：
- 与AI/客服互动中的情绪倾向：

---

【8】社交数据与潜在影响力
- 是否活跃于社交平台 / 晒图倾向：
- 是否具备KOC传播潜力：
- 是否表达过邀约朋友或参考他人建议的意愿：

---

【9】他人经历对用户态度的影响
- 明确说明朋友或熟人经历对用户影响（正向 / 负向 / 加强兴趣 / 加深顾虑）；
- 示例： 
  - “朋友术后肿胀严重 → 强化恢复担忧”
  - “闺蜜隆鼻效果好 → 提升组合意愿”

---

【10】用户时间线关键事件
- 明确表达的关键时间节点（如“已预约7月初”，“计划国庆手术”等）：
- 含糊表达请标注“（需确认）”：

---

【11】潜在阻力与应对建议
- 明确客户可能存在的成交阻力（如时间冲突、价格顾虑、术后焦虑等）；
- 针对性服务建议（如：“提供术后案例图册缓解顾虑”，“建议先安排体验项目”）

---------------------------------------------------------------------------------------------------------

"""
    return [{"role": "system", "content": system_prompt}] + state["messages"]




def prompt_ai_dialog_style(state: AgentState) -> List[AnyMessage]:
    system_msg = f"""
你是一名语言风格分析专家，以下是医美AI客服与用户的完整多轮沟通内容摘要：{state["messages"]}

请你基于该对话内容，撰写一份结构化、专业、具培训价值的《AI客服对话风格分析报告》。该报告将用于人工客服接待前的沟通风格对齐和服务一致性培训，需突出AI客服在语气表达、策略运用和节奏把控方面的具体表现，并结合原话术举例支撑判断。
📌 输出要求：
- 报告语言统一采用正式、专业风格；
- 所有维度必须填写，无法判断请写“暂无法提炼”，不可留空；
- 报告将用于团队客服风格统一、AI话术训练与人工接待培训材料归档。
- 本次输出有且只需要"AI客服对话风格分析报告"部分，请勿将其他内容放进来，且不要放重复内容。
---------------------------------------------------------------------------------------------------------
三、AI客服对话风格分析报告

【1】语气与语言风格分析
- 总结AI客服整体的语气特征，如“温和亲切”“同理心强”“简洁专业”“避免施压”等；
- 请引用1-2句具代表性的客服话术作为支持示例；
- 如对话内容不足以判断，请统一填写“暂无法提炼”。

**示例格式：**
- 风格特点：亲切耐心、避免施压
- 典型话术：“没关系，我什么时候都可以等您”、“您可以慢慢考虑，不着急决定”

---

【2】主要沟通策略分析
- 提炼AI客服在此次对话中采用的核心策略，包括但不限于：
  - 引导式提问（循序渐进引出需求）
  - 信息确认（反复确认客户表达）
  - 情绪回应（察觉焦虑并主动安抚）
  - 项目建议（结合意图推荐个性化方案）
- 每种策略建议附带一句实际话术作为例证；
- 若无法识别具体策略，请填写“暂无法提炼”。

**示例格式：**
- 策略类型：信息确认 + 情绪回应
- 示例话术：“您是考虑暑假做项目对吧？”、“听起来您对恢复期有点顾虑，我可以为您详细介绍护理流程。”

---

【3】对话节奏与互动模式分析
- 分析AI客服在交流中的响应节奏与互动逻辑，判断是否具备如下特征：
  - 响应迅速 / 节奏自然
  - 表达结构清晰 / 分层推进
  - 长信息表达后给予停顿空间
- 请引用1段具有代表性的互动片段支撑分析；
- 如节奏信息模糊，请填写“暂无法提炼”。

**示例格式：**
- 节奏特点：响应迅速、表达分层清晰
- 参考片段：“您刚提到鼻综合，我可以为您拆解一下常见的三个术式...” → “另外，根据您说的‘偏自然’风格，我们医生这边也有...”

---------------------------------------------------------------------------------------------------------

"""
    return [{"role": "system", "content": system_msg}] + state["messages"]


user_summary_agent = create_react_agent(
    model=llm,
    tools=[],
    name="user_chunk_summary_agent",
    state_schema=UserSummaryAgentState,
    response_format=UserChunkSummaryResponse,
    prompt=prompt_user_chunk_summary,
)

ai_style_agent = create_react_agent(
    model=llm,
    tools=[],
    name="ai_dialog_style_agent",
    state_schema=AiStyleAgentState,
    response_format=AiDialogStyleResponse,
    prompt=prompt_ai_dialog_style,
)

recommendation_agent = create_react_agent(
    model=llm,
    tools=[],
    name="recommendation_agent",
    state_schema=RecommendationAgentState,
    response_format=RecommendationResponse,
    prompt=prompt_recommendation,
)

industry_name = "医美"
async def run_agent_node(agent, state: AgentState, config: RunnableConfig, role_desc="专家", messages_override=None):
    current_conversation_messages = state.get("messages", [])
    if not current_conversation_messages:
        system_msg = [{"role": "system", "content": f"你是一个{industry_name}领域{role_desc}。"}]
        current_conversation_messages = system_msg

    try:
        agent_response = await asyncio.to_thread(
            agent.invoke,
            {"messages": current_conversation_messages},
            config
        )
        return {
            "structured_response": agent_response.get("structured_response"),
            "error_message": None,
            "messages": agent_response.get("messages", []),
        }
    except Exception as e:
        return {"error_message": str(e)}
#
# # 各节点异步函数
# async def recommendation_node(state: RecommendationAgentState, config):
#     return await run_agent_node(recommendation_agent, state, config, role_desc="对行业非常了解的专家，可给门店建议")
# async def user_summary_node(state: UserSummaryAgentState, config):
#     return await run_agent_node(user_summary_agent, state, config, role_desc="有十年行业经验的专家")
#
# async def ai_style_node(state: AiStyleAgentState, config):
#     return await run_agent_node(ai_style_agent, state, config, role_desc="敏锐洞察客户喜好风格的专家")
#
#
# from typing import List, Dict, Any
#
#
# async def combine_reports_node(state: UserAnalysisAgentState) -> UserAnalysisAgentState:
#     messages = state.get("messages", [])
#     last_three = messages[-3:] if len(messages) >= 3 else messages
#     contents = [msg.content if hasattr(msg, "content") else str(msg) for msg in last_three]
#     combined_text = "\n\n-------\n\n".join(contents)
#
#     # 写入到 structured_response 中，便于 dev 显示
#     state["structured_response"] = combined_text
#     return state
#
# user_analysis_graph = (
#     StateGraph(
#         UserSummaryAgentState,
#         input=ChatReplyAgentStateInput,
#         config_schema=RunnableConfig
#     )
#     .add_node("recommendation", recommendation_node)
#     .add_node("user_summary_node", user_summary_node)
#     .add_node("ai_style_node", ai_style_node)
#     .add_node("combine_reports_node", combine_reports_node)  # 加入合并节点
#     .add_edge(START, "recommendation")
#     .add_edge("recommendation", "user_summary_node")
#     .add_edge("user_summary_node", "ai_style_node")
#     .add_edge("ai_style_node", "combine_reports_node")       # 把合并节点放到最后
#     .add_edge("combine_reports_node", END)
#     .compile()
# )

async def parallel_analysis_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    # 三个 Agent 并行运行
    results = await asyncio.gather(
        run_agent_node(recommendation_agent, state, config, role_desc="对行业非常了解的专家，可给门店建议"),
        run_agent_node(user_summary_agent, state, config, role_desc="有十年行业经验的专家"),
        run_agent_node(ai_style_agent, state, config, role_desc="敏锐洞察客户喜好风格的专家"),
    )

    rec_result, summary_result, style_result = results

    def get_last_message_text(res):
        msgs = res.get("messages", [])
        if msgs and hasattr(msgs[-1], "content"):
            return msgs[-1].content
        elif msgs and isinstance(msgs[-1], dict) and "content" in msgs[-1]:
            return msgs[-1]["content"]
        return ""

    # 取每个 agent 最后一条消息的文本，拼成字符串
    combined_text = (
        "\n\n=== 医美客户风险预警与服务建议 ===\n" + get_last_message_text(rec_result) +
        "\n\n=== 医美客户信息洞察与接待准备报告 ===\n" + get_last_message_text(summary_result) +
        "\n\n=== AI客服对话风格分析报告 ===\n" + get_last_message_text(style_result)
    )
    return {
        "structured_response": combined_text,  # 🔥 结构化输出
        "error_message": None,
        "messages": []
    }

user_analysis_graph = (
    StateGraph(
        UserSummaryAgentState,  # 可以继续用这个 State，AgentState 结构相同即可
        input=ChatReplyAgentStateInput,
        config_schema=RunnableConfig
    )
    .add_node("parallel_analysis_node", parallel_analysis_node)     # 新并行节点
    .add_edge(START, "parallel_analysis_node")                      # 从 START 到并行节点
    .add_edge("parallel_analysis_node", END)                          # 汇总后结束
    .compile()
)
