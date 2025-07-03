from typing import List, Dict, Any
from .base import BaseBlock # 导入 BaseBlock
from ..mas_prompts import load_prompt
import json

# 添加通用的JSON解析函数
def _safe_json_parse(response_text: str, fallback_response: str) -> str:
    """
    安全地解析API返回的JSON响应，处理各种异常情况
    """
    try:
        # 处理API返回None的情况
        if response_text is None:
            return fallback_response
        
        # 尝试直接解析JSON
        return json.loads(response_text).get("response", fallback_response)
        
    except (json.JSONDecodeError, TypeError):
        # 如果response_text是None或无效，直接返回fallback
        if response_text is None:
            return fallback_response
            
        # 尝试清理JSON格式（去除markdown包装）
        try:
            clean_text = response_text.strip().lstrip("```json").rstrip("```").strip()
            return json.loads(clean_text).get("response", fallback_response)
        except (json.JSONDecodeError, TypeError, AttributeError):
            # 最终兜底，返回fallback响应
            return fallback_response

# 添加通用的LangChain模型调用函数
def _invoke_langchain_model(sampler, prompt: str, fallback: str) -> str:
    """
    统一的LangChain ChatModel调用函数
    """
    try:
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=prompt)]
        
        # 使用invoke方法调用模型
        model_with_format = sampler.bind(response_format={"type": "json_object"})
        response = model_with_format.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        return _safe_json_parse(response_text, fallback)
        
    except Exception as e:
        print(f"模型调用失败: {e}")
        return fallback

# 添加 _format_messages 函数的导入
def _format_messages(messages: List[Any]) -> str:
    """将 LangChain BaseMessage 对象的列表格式化为单个字符串。"""
    if not messages:
        return "（无历史记录）"
    
    formatted_string = ""
    for message in messages:
        # message.type 在 BaseMessage 对象中是 'human', 'ai', 'system' 等
        role = "客户" if message.type == "human" else "林医生"
        formatted_string += f"{role}: {message.content}\n"
    return formatted_string.strip()

class GreetingBlock(BaseBlock):
    def __init__(self, sampler: Any, node_model: str):
        super().__init__("greeting", sampler, node_model)

    def forward(self, conversation_history: list, temperature: float) -> str:
        prompt_template = load_prompt(self.block_name)
        prompt = prompt_template.format(message_history=_format_messages(conversation_history))
        return _invoke_langchain_model(self.sampler, prompt, "您好！我是林医生，有什么可以帮您？")

class RapportBuildingBlock(BaseBlock):
    def __init__(self, sampler: Any, node_model: str):
        super().__init__("rapport_building", sampler, node_model)

    def forward(self, conversation_history: list, temperature: float) -> str:
        prompt_template = load_prompt(self.block_name)
        prompt = prompt_template.format(message_history=_format_messages(conversation_history))
        return _invoke_langchain_model(self.sampler, prompt, "我们聊点别的吧！")

class NeedsAnalysisBlock(BaseBlock):
    def __init__(self, sampler: Any, node_model: str):
        super().__init__("needs_analysis", sampler, node_model)

    def forward(self, conversation_history: list, temperature: float) -> str:
        prompt_template = load_prompt(self.block_name)
        prompt = prompt_template.format(message_history=_format_messages(conversation_history))
        return _invoke_langchain_model(self.sampler, prompt, "关于您的情况，能再多说一点吗？")

class ValueDisplayBlock(BaseBlock):
    def __init__(self, sampler: Any, node_model: str):
        super().__init__("value_display", sampler, node_model)

    def forward(self, conversation_history: list, temperature: float) -> str:
        prompt_template = load_prompt(self.block_name)
        prompt = prompt_template.format(message_history=_format_messages(conversation_history))
        return _invoke_langchain_model(self.sampler, prompt, "针对您的情况，我们有很多专业的解决方案。")

class StressResponseBlock(BaseBlock):
    def __init__(self, sampler: Any, node_model: str):
        super().__init__("stress_response", sampler, node_model)

    def forward(self, conversation_history: list, temperature: float) -> str:
        prompt_template = load_prompt(self.block_name)
        prompt = prompt_template.format(message_history=_format_messages(conversation_history))
        return _invoke_langchain_model(self.sampler, prompt, "抱歉，我们换个话题吧。")

class HumanHandoffBlock(BaseBlock):
    def __init__(self, sampler: Any, node_model: str):
        super().__init__("human_handoff", sampler, node_model)

    def forward(self, conversation_history: list, temperature: float) -> str:
        prompt_template = load_prompt(self.block_name)
        prompt = prompt_template.format(message_history=_format_messages(conversation_history))
        return _invoke_langchain_model(self.sampler, prompt, "请稍等，正在为您转接人工服务。")

class PainPointTestBlock(BaseBlock):
    def __init__(self, sampler: Any, node_model: str):
        super().__init__("pain_point_test", sampler, node_model)

    def forward(self, conversation_history: list, temperature: float) -> str:
        prompt_template = load_prompt(self.block_name)
        prompt = prompt_template.format(message_history=_format_messages(conversation_history))
        return _invoke_langchain_model(self.sampler, prompt, "我们聊聊您遇到的具体情况吧？")

class ValuePitchBlock(BaseBlock):
    def __init__(self, sampler: Any, node_model: str):
        super().__init__("value_pitch", sampler, node_model)

    def forward(self, conversation_history: list, temperature: float) -> str:
        prompt_template = load_prompt(self.block_name)
        prompt = prompt_template.format(message_history=_format_messages(conversation_history))
        return _invoke_langchain_model(self.sampler, prompt, "关于我们的方案，您最关心哪个方面？")

class ActiveCloseBlock(BaseBlock):
    def __init__(self, sampler: Any, node_model: str):
        super().__init__("active_close", sampler, node_model)

    def forward(self, conversation_history: list, temperature: float) -> str:
        prompt_template = load_prompt(self.block_name)
        prompt = prompt_template.format(message_history=_format_messages(conversation_history))
        return _invoke_langchain_model(self.sampler, prompt, "我们直接进入下一步吧！")

class ReverseProbeBlock(BaseBlock):
    def __init__(self, sampler: Any, node_model: str):
        super().__init__("reverse_probe", sampler, node_model)

    def forward(self, conversation_history: list, temperature: float) -> str:
        prompt_template = load_prompt(self.block_name)
        prompt = prompt_template.format(message_history=_format_messages(conversation_history))
        return _invoke_langchain_model(self.sampler, prompt, "可以多告诉我一些您的具体情况吗？")

 