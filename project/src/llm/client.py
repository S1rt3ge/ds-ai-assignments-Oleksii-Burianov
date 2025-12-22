import os
from typing import Literal

from dotenv import load_dotenv

from src.llm.base import BaseLLMClient
from src.llm.cerebras_client import CerebrasClient
from src.llm.mistral_client import MistralClient
from src.llm.ollama_client import OllamaClient
from src.llm.openrouter_client import OpenRouterClient

load_dotenv()

ClientType = Literal["ollama", "mistral", "cerebras", "openrouter"]


def get_llm_client(
    client_type: ClientType = "ollama",
    model_name: str = "gemma3:1b",
    **kwargs,
) -> BaseLLMClient:
    if client_type == "ollama":
        base_url = kwargs.pop("base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        return OllamaClient(model_name=model_name, base_url=base_url, **kwargs)
    elif client_type == "mistral":
        api_key = kwargs.pop("api_key", None)
        return MistralClient(model_name=model_name, api_key=api_key, **kwargs)
    elif client_type == "cerebras":
        api_key = kwargs.pop("api_key", None)
        return CerebrasClient(model_name=model_name, api_key=api_key, **kwargs)
    elif client_type == "openrouter":
        api_key = kwargs.pop("api_key", None)
        return OpenRouterClient(model_name=model_name, api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported client type: {client_type}")


def list_available_models(client_type: ClientType = "ollama") -> list[str]:
    if client_type == "ollama":
        return OllamaClient.SUPPORTED_MODELS
    elif client_type == "mistral":
        return MistralClient.SUPPORTED_MODELS
    elif client_type == "cerebras":
        return CerebrasClient.SUPPORTED_MODELS
    elif client_type == "openrouter":
        return OpenRouterClient.SUPPORTED_MODELS
    else:
        return []
