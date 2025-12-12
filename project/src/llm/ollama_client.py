"""Ollama LLM client implementation."""

import time
from typing import Any, AsyncIterator, Iterator

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from src.llm.base import BaseLLMClient
from src.llm.models import LLMResponse, Message, ResponseMetadata, StreamChunk


class OllamaClient(BaseLLMClient):
    """Client for interacting with Ollama local models."""

    SUPPORTED_MODELS = [
        "gemma3:1b",
        "deepseek-r1:1.5b",
        "ministral-3:3b",
    ]

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", **kwargs):
        """
        Initialize Ollama client.

        Args:
            model_name: Name of the Ollama model (e.g., 'gemma3:1b')
            base_url: Base URL for Ollama API
            **kwargs: Additional configuration
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_name}' not in supported models: {self.SUPPORTED_MODELS}. "
                f"Make sure you've pulled the model with 'ollama pull {model_name}'"
            )

        super().__init__(model_name, **kwargs)
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(self, messages: list[Message], **kwargs) -> LLMResponse:
        """
        Generate a response from Ollama.

        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with generated content and metadata
        """
        start_time = time.time()

        # Convert messages to Ollama format
        ollama_messages = [msg.to_dict() for msg in messages]

        # Calculate input tokens
        input_text = " ".join([msg.content for msg in messages])
        input_tokens = self._estimate_tokens(input_text)

        # Extract Ollama-specific options
        # Ollama expects parameters like temperature in an 'options' dict
        options = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            options["num_predict"] = kwargs.pop("max_tokens")  # Ollama uses num_predict
        if "top_p" in kwargs:
            options["top_p"] = kwargs.pop("top_p")
        if "top_k" in kwargs:
            options["top_k"] = kwargs.pop("top_k")

        # Call Ollama API
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=ollama_messages,
                stream=False,
                options=options if options else None,
            )

            # Extract response content
            content = response.get("message", {}).get("content", "")
            output_tokens = self._estimate_tokens(content)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Create metadata
            metadata = ResponseMetadata(
                model_name=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost=self._calculate_cost(input_tokens, output_tokens),
            )

            return LLMResponse(content=content, metadata=metadata, raw_response=response)

        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {str(e)}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def stream(self, messages: list[Message], **kwargs) -> Iterator[StreamChunk]:
        """
        Stream a response from Ollama.

        Args:
            messages: List of conversation messages
            **kwargs: Additional generation parameters

        Yields:
            StreamChunk objects with incremental content
        """
        start_time = time.time()

        # Convert messages to Ollama format
        ollama_messages = [msg.to_dict() for msg in messages]

        # Calculate input tokens
        input_text = " ".join([msg.content for msg in messages])
        input_tokens = self._estimate_tokens(input_text)

        # Extract Ollama-specific options
        # Ollama expects parameters like temperature in an 'options' dict
        options = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            options["num_predict"] = kwargs.pop("max_tokens")  # Ollama uses num_predict
        if "top_p" in kwargs:
            options["top_p"] = kwargs.pop("top_p")
        if "top_k" in kwargs:
            options["top_k"] = kwargs.pop("top_k")

        try:
            response_stream = self.client.chat(
                model=self.model_name,
                messages=ollama_messages,
                stream=True,
                options=options if options else None,
            )

            full_content = ""
            for chunk in response_stream:
                content_delta = chunk.get("message", {}).get("content", "")
                full_content += content_delta

                is_final = chunk.get("done", False)

                if is_final:
                    # Final chunk with metadata
                    output_tokens = self._estimate_tokens(full_content)
                    latency_ms = (time.time() - start_time) * 1000

                    metadata = ResponseMetadata(
                        model_name=self.model_name,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        latency_ms=latency_ms,
                        cost=self._calculate_cost(input_tokens, output_tokens),
                    )

                    yield StreamChunk(content=content_delta, is_final=True, metadata=metadata)
                else:
                    # Intermediate chunk
                    yield StreamChunk(content=content_delta, is_final=False)

        except Exception as e:
            raise RuntimeError(f"Ollama streaming failed: {str(e)}") from e

    async def agenerate(self, messages: list[Message], **kwargs) -> LLMResponse:
        """
        Async generate (not implemented for Ollama - uses sync version).

        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content and metadata
        """
        # Ollama Python client doesn't have async support yet
        # For now, we'll use the sync version
        return self.generate(messages, **kwargs)

    async def astream(self, messages: list[Message], **kwargs) -> AsyncIterator[StreamChunk]:
        """
        Async stream (not implemented for Ollama - uses sync version).

        Args:
            messages: List of conversation messages
            **kwargs: Additional generation parameters

        Yields:
            StreamChunk objects with incremental content
        """
        # Ollama Python client doesn't have async support yet
        # For now, we'll use the sync version wrapped
        for chunk in self.stream(messages, **kwargs):
            yield chunk
