from src.routing.models import ModelCapabilities, RoutingMode

MODEL_REGISTRY: dict[str, ModelCapabilities] = {
    "ollama/gemma3:1b": ModelCapabilities(
        provider="ollama",
        model_name="gemma3:1b",
        is_local=True,
        avg_latency_ms=150,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        quality_score=5,
        speed_score=10,
        max_context_length=8192,
        supports_streaming=True,
    ),
    "ollama/deepseek-r1:1.5b": ModelCapabilities(
        provider="ollama",
        model_name="deepseek-r1:1.5b",
        is_local=True,
        avg_latency_ms=250,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        quality_score=6,
        speed_score=8,
        max_context_length=4096,
        supports_streaming=True,
    ),
    "ollama/ministral-3:3b": ModelCapabilities(
        provider="ollama",
        model_name="ministral-3:3b",
        is_local=True,
        avg_latency_ms=400,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        quality_score=7,
        speed_score=6,
        max_context_length=8192,
        supports_streaming=True,
    ),
    "cerebras/llama-3.3-70b": ModelCapabilities(
        provider="cerebras",
        model_name="llama-3.3-70b",
        is_local=False,
        avg_latency_ms=400,
        cost_per_1k_input=0.0006,
        cost_per_1k_output=0.0006,
        quality_score=8,
        speed_score=9,
        max_context_length=8192,
        supports_streaming=True,
    ),
    "mistral/mistral-medium-latest": ModelCapabilities(
        provider="mistral",
        model_name="mistral-medium-latest",
        is_local=False,
        avg_latency_ms=1500,
        cost_per_1k_input=0.0027,
        cost_per_1k_output=0.0081,
        quality_score=9,
        speed_score=5,
        max_context_length=32000,
        supports_streaming=True,
    ),
    "openrouter/openai/gpt-4o": ModelCapabilities(
        provider="openrouter",
        model_name="openai/gpt-4o",
        is_local=False,
        avg_latency_ms=2000,
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        quality_score=10,
        speed_score=4,
        max_context_length=128000,
        supports_streaming=True,
    ),
}


class RoutingStrategy:
    def __init__(self, mode: RoutingMode = RoutingMode.BALANCED):
        self.mode = mode

    def select_model(self, complexity_score: int, token_count: int) -> str:
        if self.mode == RoutingMode.BALANCED:
            return self._balanced_selection(complexity_score, token_count)
        elif self.mode == RoutingMode.COST_OPTIMIZED:
            return self._cost_optimized_selection(complexity_score, token_count)
        elif self.mode == RoutingMode.QUALITY_OPTIMIZED:
            return self._quality_optimized_selection(complexity_score, token_count)
        elif self.mode == RoutingMode.ULTRA_FAST:
            return self._ultra_fast_selection(complexity_score, token_count)
        else:
            return self._balanced_selection(complexity_score, token_count)

    def _balanced_selection(self, complexity_score: int, token_count: int) -> str:
        if complexity_score < 40:
            return "ollama/gemma3:1b"
        elif complexity_score < 55:
            return "groq/openai/gpt-oss-120b"
        elif complexity_score < 70:
            return "cerebras/llama-3.3-70b"
        elif complexity_score < 85:
            return "mistral/mistral-medium-latest"
        else:
            return "openrouter/openai/gpt-4o"

    def _cost_optimized_selection(self, complexity_score: int, token_count: int) -> str:
        if complexity_score < 50:
            return "ollama/gemma3:1b"
        elif complexity_score < 70:
            return "ollama/deepseek-r1:1.5b"
        else:
            return "groq/openai/gpt-oss-120b"

    def _quality_optimized_selection(self, complexity_score: int, token_count: int) -> str:
        if complexity_score < 20:
            return "ollama/ministral-3:3b"
        elif complexity_score < 50:
            return "cerebras/llama-3.3-70b"
        elif complexity_score < 80:
            return "mistral/mistral-medium-latest"
        else:
            return "openrouter/openai/gpt-4o"

    def _ultra_fast_selection(self, complexity_score: int, token_count: int) -> str:
        if complexity_score < 30:
            return "ollama/gemma3:1b"
        else:
            return "groq/openai/gpt-oss-120b"

    def get_model_capabilities(self, model_key: str) -> ModelCapabilities:
        if model_key not in MODEL_REGISTRY:
            raise KeyError(f"Model '{model_key}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
        return MODEL_REGISTRY[model_key]

    def list_all_models(self) -> list[str]:
        return list(MODEL_REGISTRY.keys())
