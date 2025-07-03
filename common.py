"""
MAS Cloud Agent - 通用类型定义和数据结构
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, field, asdict
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

@dataclass
class EmotionalState:#其实完全可以考虑将情感状态的数据传给meta，结合一些数学技巧作判断。目前完全是基于写死的搜索空间去做回复。
    """用户情感状态数据类"""
    security_level: float = field(default=0.0)      # 安全感等级 (0-1)
    familiarity_level: float = field(default=0.0)   # 熟悉感等级 (0-1)
    comfort_level: float = field(default=0.0)       # 舒适感等级 (0-1)
    intimacy_level: float = field(default=0.0)      # 亲密感等级 (0-1)
    gain_level: float = field(default=0.0)          # 获得感等级 (0-1)
    recognition_level: float = field(default=0.0)   # 认同感等级 (0-1)
    trust_level: float = field(default=0.0)         # 信任感等级 (0-1)

    # 为向后兼容 pydantic BaseModel 的接口，补充两个辅助方法
    def model_dump(self) -> dict:
        """返回 dataclass 字典表示，以兼容 pydantic 的 model_dump。"""
        return asdict(self)

    def model_dump_json(self) -> str:
        """返回 JSON 字符串，以兼容 model_dump_json 调用。"""
        import json
        return json.dumps(asdict(self), ensure_ascii=False)

class CustomerIntent(BaseModel):
    """客户行为意图分析结果"""
    intent_type: str  # "appointment_request", "price_inquiry", "concern_raised", "general_chat", "ready_to_book"
    confidence: float  # 0.0-1.0 置信度
    extracted_info: Dict[str, Any] = {}  # 提取的结构化信息
    requires_action: List[str] = []  # 需要的后续动作

class AppointmentInfo(BaseModel):
    """预约信息管理"""
    has_time: bool = False
    preferred_time: Optional[str] = None
    has_name: bool = False
    customer_name: Optional[str] = None
    has_phone: bool = False
    customer_phone: Optional[str] = None
    has_address_confirmed: bool = False
    preferred_service: Optional[str] = None
    appointment_status: str = "pending"  # "pending", "confirmed", "info_collecting"

class AgentState(TypedDict):
    """LangGraph 状态定义"""
    # 核心消息处理
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: Optional[str]  # 专门用于接收单次用户输入
    
    # 对话状态
    current_stage: str  # 当前对话阶段
    emotional_state: EmotionalState  # 用户情感状态
    user_profile: dict  # 用户信息，如痛点、兴趣等
    turn_count: int  # 对话轮次计数
    customer_intent_level: Optional[str]  # 客户意向：low, medium, high, fake_high，后期应该会改到五级，但是在graph的meta_design_node里会很难办。
    
    # 运行时控制字段
    internal_monologue: Optional[List[str]]  # 内部独白，用于调试，简化版
    candidate_actions: Optional[List[str]]   # 候选行动
    evaluated_responses: Optional[List[Dict[str, Any]]]  # 评估的回复
    final_response: Optional[str]  # 最终回复
    last_message: Optional[str]  # API输出的最后消息

    # 模型配置
    agent_temperature: Optional[float]  # 生成温度
    node_model: Optional[str]  # 节点使用的模型
    feedback_model: Optional[str]  # 反馈模型
    verbose: Optional[bool]  # 调试开关

    # 新增：行为意图和预约管理
    customer_intent: Optional[CustomerIntent]  # 客户行为意图
    appointment_info: Optional[AppointmentInfo]  # 预约信息

@dataclass
class DebugInfo:
    """用于API输出的调试信息"""
    current_stage: Optional[str] = None
    emotional_state: Optional[Dict[str, Any]] = None
    internal_monologue: Optional[List[str]] = None

@dataclass
class MasAgentOutput:
    """MAS Agent的API输出状态 - 只包含必要字段"""
    last_message: str = field(default="")
    debug_info: Optional[DebugInfo] = field(default=None) 