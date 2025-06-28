"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from langchain_core.runnables import ensure_config
from langgraph.config import get_config
from langchain_core.prompts import ChatPromptTemplate

from huanmu_agent import prompts
from constant import (
    DEEPSEEK_CHAT_MODEL,
    DEEPSEEK_API_ENDPOINT,
    DASHSCOPE_TEXT2IMAGE_MODEL,
    DASHSCOPE_API_ENDPOINT,
)


@dataclass(kw_only=True)
class Configuration:
    system_prompt: str = field(
        default=prompts.BEAUTY_CARE_V1_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

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
    
    # Zilliz Cloud / Milvus configuration
    milvus_uri: str = field(
        default=os.getenv("MILVUS_URI", "https://in03-999fce1882bd070.serverless.gcp-us-west1.cloud.zilliz.com"),
        metadata={
            "description": "The URI for Zilliz Cloud connection. "
            "For cloud service, use the provided cluster endpoint."
        },
    )
    
    milvus_user: str = field(
        default=os.getenv("MILVUS_USER", "db_999fce1882bd070"),
        metadata={
            "description": "Username for Zilliz Cloud authentication."
        },
    )
    
    milvus_password: str = field(
        default=os.getenv("MILVUS_PASSWORD", "Er0^-Y;^7gDT|H+["),
        metadata={
            "description": "Password for Zilliz Cloud authentication."
        },
    )
    
    milvus_collection_name: str = field(
        default=os.getenv("MILVUS_COLLECTION", "product_knowledge"),
        metadata={
            "description": "The name of the Milvus collection to use."
        },
    )
    
    # DashScope API configuration
    dashscope_api_key: str = field(
        default=os.getenv("DASHSCOPE_API_KEY"),
        metadata={
            "description": "API key for DashScope image generation service."
        },
    )
    
    dashscope_model: str = field(
        default=DASHSCOPE_TEXT2IMAGE_MODEL,
        metadata={
            "description": "The DashScope model to use for image generation."
        },
    )
    
    dashscope_api_endpoint: str = field(
        default=DASHSCOPE_API_ENDPOINT,
        metadata={
            "description": "The DashScope API endpoint for image generation."
        },
    )

    # DeepSeek API configuration
    deepseek_api_key: str = field(
        default=os.getenv("DEEPSEEK_API_KEY"),
        metadata={
            "description": "API key for DeepSeek service."
        },
    )
    
    deepseek_model: str = field(
        default=DEEPSEEK_CHAT_MODEL,
        metadata={
            "description": "The DeepSeek model to use for text generation."
        },
    )
    
    deepseek_api_endpoint: str = field(
        default=DEEPSEEK_API_ENDPOINT,
        metadata={
            "description": "The DeepSeek API endpoint for text generation."
        },
    )

    @classmethod
    def from_context(cls) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        try:
            config = get_config()
        except RuntimeError:
            config = None
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})