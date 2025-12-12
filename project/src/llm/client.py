"""LLM client factory and utilities."""

import os
from typing import Literal

from dotenv import load_dotenv

from src.llm.base import BaseLLMClient
from src.llm.ollama_client import OllamaClient

# Load environment variables
load_dotenv()

# Type for supported client types
ClientType = Literal["ollama"]


def get_llm_client(
    client_type: ClientType = "ollama",
    model_name: str = "gemma3:1b",
    **kwargs,
) -> BaseLLMClient:
    """
    Factory function to get an LLM client.

    Args:
        client_type: Type of client ('ollama', future: 'openai', 'anthropic')
        model_name: Name of the model to use
        **kwargs: Additional configuration for the client

    Returns:
        Initialized LLM client

    Raises:
        ValueError: If client_type is not supported
    """
    if client_type == "ollama":
        base_url = kwargs.pop("base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        return OllamaClient(model_name=model_name, base_url=base_url, **kwargs)
    else:
        raise ValueError(f"Unsupported client type: {client_type}")


def list_available_models(client_type: ClientType = "ollama") -> list[str]:
    """
    List available models for a given client type.

    Args:
        client_type: Type of client

    Returns:
        List of available model names
    """
    if client_type == "ollama":
        return OllamaClient.SUPPORTED_MODELS
    else:
        return []
