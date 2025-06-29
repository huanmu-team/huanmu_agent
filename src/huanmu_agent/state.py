"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Optional
from uuid import UUID

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


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


@dataclass(kw_only=True)
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """
    # run_id: UUID
    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """

    last_message: str = ""
    # Additional attributes can be added here as needed.
    # Common examples include:
    # retrieved_documents: List[Document] = field(default_factory=list)

@dataclass(kw_only=True)
class SalesAgentStateOutput:
    """Represents the output of the sales agent."""
    # run_id: UUID
    last_message: str = field(default= "")

    
    

