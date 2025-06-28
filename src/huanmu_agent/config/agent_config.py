from typing import TypedDict

class ModelConfig(TypedDict):
    provider: str
    model_name: str
    temperature: float
