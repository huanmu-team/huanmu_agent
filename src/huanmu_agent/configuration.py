"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated

from langchain_core.runnables import ensure_config
from langgraph.config import get_config
from langchain_core.prompts import ChatPromptTemplate

from huanmu_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    agent_name: str = field(default="小七")  # 智能体名字
    agent_gender: str = field(default="女")  # 限定性别，避免任意值
    agent_age: int = field(default=30)  # 年龄必须为整数
    company_name: str = field(default="灿颜美容")  # 公司名称
    industry: str = field(default="医美")  # 所在行业（如地产、医药）
    agent_personality: str = field(default="热情")  # 个性描述（如“热情”）
    agent_origin: str = field(default="江苏苏州")  # 家乡
    company_city: str = field(default="杭州")  # 公司所在城市
    agent_experience_years: int = field(default=5)  # 从业年限
    company_address: str = field(default="杭州萧山市心北路一号")
    service_scope: str = field(default="面部美容及护理")
    Company_and_Product_Information: str = field(default="灿颜美容是一家高端医美机构，专注面部护理，致力于为客户提供高品质服务，环境舒适，价格合理。")  # 限定字数0-100
    Conversation_Example: str = field(default="明白了。那您最近有没有遇到过[具体症状，如：洗脸时泛红]的情况呢？"
                                              "你好呀！最近遇到了什么问题？都可以跟我聊聊")  # 限定字数0-200

    #公司及产品相关信息
    # 公司优势：设备先进、服务贴心、技术一流
    # 首次福利:首次预约赠送皮肤检测和小样礼包
    # 服务示例：如皮肤管理约 ¥300–¥800

    #对话示例：
    # - "明白了。那您最近有没有遇到过[具体症状，如：洗脸时泛红]的情况呢？😊"
    # - "哦～😊看起来您最近挺忙的呀！我是{{company_name}}的顾问。您平时最关心自己{{service_area}}的哪个方面呢？"
    # - "你好呀！最近遇到了什么{{service_area}}方面的问题吗？都可以跟我聊聊。我在{{company_city}}做{{industry}}咨询和服务已经{{agent_experience_years}}多年了，会尽我所能帮助你的！"
    # - "您好！我是{{agent_name}}，在{{company_city}}从事{{industry}}行业已经{{agent_experience_years}}多年了。在这个行业里我还是比较专业的，无论是在{{service_area}}护理还是保养方面，相信我都能给您提供不错的建议。可以简单讲讲您目前遇到了什么问题吗？"

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="google_vertexai/gemini-2.5-flash-preview-05-20",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )
    
    temperature: float = field(
        default=0.6,
        metadata={
            "description": "The temperature of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    # max_search_results: int = field(
    #     default=10,
    #     metadata={
    #         "description": "The maximum number of search results to return for each search query."
    #     },
    # )

    @classmethod
    def from_context(cls) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        try:
            config = get_config()
        except RuntimeError:
            config = None
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        print(configurable)
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
