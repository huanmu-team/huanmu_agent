"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence, Optional, List, Dict, Any
from uuid import UUID

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated
from pydantic import BaseModel

# 导入mas_graph.py的状态定义
@dataclass
class EmotionalState:
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

@dataclass
class DebugInfo:
    """用于API输出的调试信息"""
    current_stage: Optional[str] = None
    emotional_state: Optional[Dict[str, Any]] = None
    internal_monologue: Optional[List[str]] = None

@dataclass
class HumanControlState:#为后面加一些冗余字段6/28
    """管理人工接管状态的数据类"""
    
    is_human_active: bool = False
    """当前是否处于人工接管状态"""
    
    human_operator_id: Optional[str] = None
    """当前接管的人工客服ID"""
    
    transfer_reason: Optional[str] = None
    """转人工的原因，可以是'rescue'(救火),'progress'(推进),'deal'(成交)"""
    
    transfer_time: Optional[str] = None
    """转人工的时间"""
    
    @property
    def is_available(self) -> bool:
        """检查是否可以转人工"""
        return not self.is_human_active
    
    def activate_human(self, operator_id: str, reason: str, time: str) -> None:
        """激活人工接管"""
        self.is_human_active = True
        self.human_operator_id = operator_id
        self.transfer_reason = reason
        self.transfer_time = time
        
    def deactivate_human(self) -> None:
        """取消人工接管"""
        self.is_human_active = False
        self.human_operator_id = None
        self.transfer_reason = None
        self.transfer_time = None

@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """
    
    human_control: HumanControlState = field(default_factory=HumanControlState)
    """人工接管状态控制"""
    
    # 新增：来自mas_graph.py的输入字段
    user_input: Optional[str] = None
    """专门用于接收单次用户输入"""
    
    verbose: bool = False
    """调试模式开关"""


@dataclass(kw_only=True)
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """
    # 原有字段
    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """

    last_message: str = ""
    """API输出的最后消息"""
    
    # 新增：来自mas_graph.py的状态字段
    # 对话状态管理
    current_stage: str = "initial_contact"
    """当前对话阶段：initial_contact, ice_breaking, subtle_expertise, pain_point_mining, solution_visualization, natural_invitation"""
    
    emotional_state: EmotionalState = field(default_factory=EmotionalState)
    """用户情感状态"""
    
    user_profile: Dict[str, Any] = field(default_factory=dict)
    """用户信息，如痛点、兴趣等"""
    
    turn_count: int = 0
    """对话轮次计数"""
    
    customer_intent_level: str = "low"
    """客户意向等级：low, medium, high, fake_high"""
    
    # 运行时控制字段
    internal_monologue: List[str] = field(default_factory=list)
    """内部独白，用于调试"""
    
    candidate_actions: List[str] = field(default_factory=list)
    """候选行动列表"""
    
    evaluated_responses: List[Dict[str, Any]] = field(default_factory=list)
    """评估的回复列表"""
    
    final_response: str = ""
    """最终选择的回复"""
    
    # 模型配置
    agent_temperature: float = 0.5
    """生成温度"""
    
    node_model: str = "openai/gpt-4o-mini-2024-07-18"
    """节点使用的模型"""
    
    feedback_model: str = "openai/gpt-4o-mini-2024-07-18"
    """反馈评估模型"""
    
    # 高级功能：行为意图和预约管理
    customer_intent: Optional[CustomerIntent] = None
    """客户行为意图分析结果"""
    
    appointment_info: Optional[AppointmentInfo] = None
    """预约信息管理"""
    
    # 调试信息
    debug_info: Optional[DebugInfo] = None
    """调试信息，仅在verbose=True时生成"""

@dataclass(kw_only=True)
class SalesAgentStateOutput:
    """Represents the output of the sales agent."""
    
    last_message: str = field(default="")
    """AI的最终回复内容"""
    
    debug_info: Optional[DebugInfo] = field(default=None)
    """调试信息，仅在verbose模式下返回"""

    
    

