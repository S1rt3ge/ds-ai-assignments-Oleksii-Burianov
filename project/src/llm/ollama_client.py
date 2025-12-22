import time
from typing import Any, Iterator

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from src.llm.base import BaseLLMClient
from src.llm.models import LLMResponse, Message, ResponseMetadata, StreamChunk


class OllamaClient(BaseLLMClient):
    SUPPORTED_MODELS = [
        "gemma3:1b",
        "deepseek-r1:1.5b",
        "ministral-3:3b",
    ]

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", **kwargs):
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
        start_time = time.time()
        ollama_messages = [msg.to_dict() for msg in messages]
        input_text = " ".join([msg.content for msg in messages])
        input_tokens = self._estimate_tokens(input_text)

        options = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            options["num_predict"] = kwargs.pop("max_tokens")
        if "top_p" in kwargs:
            options["top_p"] = kwargs.pop("top_p")
        if "top_k" in kwargs:
            options["top_k"] = kwargs.pop("top_k")

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=ollama_messages,
                stream=False,
                options=options if options else None,
            )

            content = response.get("message", {}).get("content", "")
            output_tokens = self._estimate_tokens(content)
            latency_ms = (time.time() - start_time) * 1000

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
        start_time = time.time()
        ollama_messages = [msg.to_dict() for msg in messages]
        input_text = " ".join([msg.content for msg in messages])
        input_tokens = self._estimate_tokens(input_text)

        options = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            options["num_predict"] = kwargs.pop("max_tokens")
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

            content_parts = []
            for chunk in response_stream:
                content_delta = chunk.get("message", {}).get("content", "")
                content_parts.append(content_delta)

                is_final = chunk.get("done", False)

                if is_final:
                    full_content = "".join(content_parts)
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
                    yield StreamChunk(content=content_delta, is_final=False)

        except Exception as e:
            raise RuntimeError(f"Ollama streaming failed: {str(e)}") from e
