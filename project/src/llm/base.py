from abc import ABC, abstractmethod
from typing import Iterator

from src.llm.models import LLMResponse, Message, StreamChunk


class BaseLLMClient(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(self, messages: list[Message], **kwargs) -> LLMResponse:
        pass

    @abstractmethod
    def stream(self, messages: list[Message], **kwargs) -> Iterator[StreamChunk]:
        pass

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0
