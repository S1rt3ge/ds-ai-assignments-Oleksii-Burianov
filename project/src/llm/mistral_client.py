import os
import time
from typing import Iterator

from mistralai import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential

from src.llm.base import BaseLLMClient
from src.llm.models import LLMResponse, Message, ResponseMetadata, StreamChunk


class MistralClient(BaseLLMClient):
    SUPPORTED_MODELS = ["mistral-medium-latest"]

    PRICING = {
        "mistral-medium-latest": {
            "input": 0.0027,
            "output": 0.0081,
        }
    }

    def __init__(self, model_name: str, api_key: str | None = None, **kwargs):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_name}' not in supported models: {self.SUPPORTED_MODELS}. "
                f"Available models: {', '.join(self.SUPPORTED_MODELS)}"
            )

        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set and no api_key provided")

        self.client = Mistral(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(self, messages: list[Message], **kwargs) -> LLMResponse:
        start_time = time.time()
        mistral_messages = [msg.to_dict() for msg in messages]
        input_text = " ".join([msg.content for msg in messages])
        input_tokens = self._estimate_tokens(input_text)

        try:
            response = self.client.chat.complete(
                model=self.model_name,
                messages=mistral_messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens"),
                top_p=kwargs.get("top_p"),
            )

            if not response.choices or len(response.choices) == 0:
                raise RuntimeError("Mistral AI returned empty response")

            content = response.choices[0].message.content

            usage = getattr(response, "usage", None)
            if usage:
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
            else:
                output_tokens = self._estimate_tokens(content)

            latency_ms = (time.time() - start_time) * 1000
            cost = self._calculate_cost(input_tokens, output_tokens)

            metadata = ResponseMetadata(
                model_name=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost=cost,
            )

            return LLMResponse(content=content, metadata=metadata, raw_response=response)

        except Exception as e:
            raise RuntimeError(f"Mistral AI generation failed: {str(e)}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def stream(self, messages: list[Message], **kwargs) -> Iterator[StreamChunk]:
        start_time = time.time()
        mistral_messages = [msg.to_dict() for msg in messages]
        input_text = " ".join([msg.content for msg in messages])
        input_tokens = self._estimate_tokens(input_text)

        try:
            response_stream = self.client.chat.stream(
                model=self.model_name,
                messages=mistral_messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens"),
                top_p=kwargs.get("top_p"),
            )

            content_parts = []
            for chunk in response_stream:
                if chunk.data.choices:
                    delta = chunk.data.choices[0].delta
                    content_delta = getattr(delta, "content", "") or ""
                    content_parts.append(content_delta)

                    is_final = chunk.data.choices[0].finish_reason is not None

                    if is_final:
                        full_content = "".join(content_parts)
                        output_tokens = self._estimate_tokens(full_content)
                        latency_ms = (time.time() - start_time) * 1000
                        cost = self._calculate_cost(input_tokens, output_tokens)

                        metadata = ResponseMetadata(
                            model_name=self.model_name,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            latency_ms=latency_ms,
                            cost=cost,
                        )

                        yield StreamChunk(content=content_delta, is_final=True, metadata=metadata)
                    else:
                        yield StreamChunk(content=content_delta, is_final=False)

        except Exception as e:
            raise RuntimeError(f"Mistral AI streaming failed: {str(e)}") from e

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = self.PRICING.get(self.model_name, {"input": 0, "output": 0})
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost
