from typing import Dict, Type
from .base import BaseBlock
# 只从统一的 conversation_blocks.py 导入所有模块
from .conversation_blocks import (
    GreetingBlock,
    RapportBuildingBlock,
    NeedsAnalysisBlock,
    ValueDisplayBlock,
    StressResponseBlock,
    HumanHandoffBlock,
    PainPointTestBlock,
    ValuePitchBlock,
    ActiveCloseBlock,
    ReverseProbeBlock,
)
# 移除重复的预约管理模块，active_close已覆盖相关功能
from .state_evaluator import evaluate_state
from .intent_analyzer import analyze_customer_intent, update_appointment_info

# 更新 BLOCK_REGISTRY，只包含新框架下的模块
BLOCK_REGISTRY: Dict[str, Type[BaseBlock]] = {
    "greeting": GreetingBlock,
    "rapport_building": RapportBuildingBlock,#建立联系
    "needs_analysis": NeedsAnalysisBlock,#需求分析
    "value_display": ValueDisplayBlock,#价值展示
    "stress_response": StressResponseBlock,#压力应对
    "human_handoff": HumanHandoffBlock,#人工转接
    # 注册新的意图驱动模块
    "pain_point_test": PainPointTestBlock,#
    "value_pitch": ValuePitchBlock,#价值抛投
    "active_close": ActiveCloseBlock,#主动成交
    "reverse_probe": ReverseProbeBlock,#反向试探
# 注释：预约确认功能已由active_close模块完美实现
}

def create_block(action: str, sampler: any, node_model: str) -> BaseBlock:
    """
    根据动作名称创建对应的能力模块实例。
    这是一个工厂函数，用于解耦图逻辑和模块实现。

    Args:
        action (str): 动作名称，对应于 BLOCK_REGISTRY 中的键。
        sampler (any): 一个已经初始化好的模型采样器实例。
        node_model (str): 当前节点使用的模型名称。

    Returns:
        BaseBlock: 一个能力模块的实例，如果动作不存在则返回 None。
    """
    block_class = BLOCK_REGISTRY.get(action)
    if block_class:
        # 修复：直接传递 sampler 和 node_model 参数，与 Block 类构造函数匹配
        return block_class(sampler, node_model)
    return None 