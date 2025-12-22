import os
import time
from typing import Iterator

from cerebras.cloud.sdk import Cerebras
from tenacity import retry, stop_after_attempt, wait_exponential

from src.llm.base import BaseLLMClient
from src.llm.models import LLMResponse, Message, ResponseMetadata, StreamChunk


class CerebrasClient(BaseLLMClient):
    SUPPORTED_MODELS = ["llama-3.3-70b"]

    PRICING = {
        "llama-3.3-70b": {
            "input": 0.0006,
            "output": 0.0006,
        }
    }

    def __init__(self, model_name: str, api_key: str | None = None, **kwargs):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_name}' not in supported models: {self.SUPPORTED_MODELS}. "
                f"Available models: {', '.join(self.SUPPORTED_MODELS)}"
            )

        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        if not self.api_key:
            raise ValueError("CEREBRAS_API_KEY environment variable not set and no api_key provided")

        self.client = Cerebras(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(self, messages: list[Message], **kwargs) -> LLMResponse:
        start_time = time.time()
        cerebras_messages = [msg.to_dict() for msg in messages]
        input_text = " ".join([msg.content for msg in messages])
        input_tokens = self._estimate_tokens(input_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=cerebras_messages,
                temperature=kwargs.get("temperature", 0.7),
                max_completion_tokens=kwargs.get("max_tokens", 1024),
                top_p=kwargs.get("top_p", 1),
                stream=False,
            )

            if not response.choices or len(response.choices) == 0:
                raise RuntimeError("No response choices returned from Cerebras API")

            content = response.choices[0].message.content

            if hasattr(response, "usage") and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
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
            raise RuntimeError(f"Cerebras AI generation failed: {str(e)}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def stream(self, messages: list[Message], **kwargs) -> Iterator[StreamChunk]:
        start_time = time.time()
        cerebras_messages = [msg.to_dict() for msg in messages]
        input_text = " ".join([msg.content for msg in messages])
        input_tokens = self._estimate_tokens(input_text)

        try:
            response_stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=cerebras_messages,
                temperature=kwargs.get("temperature", 0.7),
                max_completion_tokens=kwargs.get("max_tokens", 1024),
                top_p=kwargs.get("top_p", 1),
                stream=True,
            )

            content_parts = []
            for chunk in response_stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    content_delta = getattr(delta, "content", "") or ""
                    content_parts.append(content_delta)

                    finish_reason = chunk.choices[0].finish_reason
                    is_final = finish_reason is not None

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
                        if content_delta:
                            yield StreamChunk(content=content_delta, is_final=False)

        except Exception as e:
            raise RuntimeError(f"Cerebras AI streaming failed: {str(e)}") from e

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = self.PRICING.get(self.model_name, {"input": 0, "output": 0})
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost
