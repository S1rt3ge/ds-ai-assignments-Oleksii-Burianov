"""Base abstract class for LLM clients."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator

from src.llm.models import LLMResponse, Message, StreamChunk


class BaseLLMClient(ABC):
    """Abstract base class for all LLM clients."""

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the LLM client.

        Args:
            model_name: Name of the model to use
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(self, messages: list[Message], **kwargs) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: List of conversation messages
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse containing the generated content and metadata
        """
        pass

    @abstractmethod
    def stream(self, messages: list[Message], **kwargs) -> Iterator[StreamChunk]:
        """
        Stream a response from the LLM.

        Args:
            messages: List of conversation messages
            **kwargs: Additional generation parameters

        Yields:
            StreamChunk objects containing incremental response content
        """
        pass

    @abstractmethod
    async def agenerate(self, messages: list[Message], **kwargs) -> LLMResponse:
        """
        Async version of generate.

        Args:
            messages: List of conversation messages
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse containing the generated content and metadata
        """
        pass

    @abstractmethod
    async def astream(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """
        Async version of stream.

        Args:
            messages: List of conversation messages
            **kwargs: Additional generation parameters

        Yields:
            StreamChunk objects containing incremental response content
        """
        pass

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text.
        Simple heuristic: ~4 characters per token on average.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        return max(1, len(text) // 4)

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for the API call.
        Override this in subclasses for paid APIs.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        return 0.0  # Default: free for local models
