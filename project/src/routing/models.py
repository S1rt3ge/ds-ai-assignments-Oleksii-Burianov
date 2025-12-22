from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    FACTUAL = "factual"
    REASONING = "reasoning"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    UNKNOWN = "unknown"


class RoutingMode(str, Enum):
    BALANCED = "balanced"
    COST_OPTIMIZED = "cost_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    ULTRA_FAST = "ultra_fast"


class QueryAnalysis(BaseModel):
    query: str
    token_count: int
    question_type: QuestionType
    has_complex_keywords: bool
    complexity_score: int = Field(ge=0, le=100, description="Complexity score from 0-100")
    estimated_quality_needed: str = Field(description="low, medium, or high")
    analyzed_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True


class RoutingDecision(BaseModel):
    provider: str = Field(description="Provider name (ollama, mistral, groq, etc.)")
    model_name: str = Field(description="Full model name")
    reason: str = Field(description="Human-readable reason for selection")
    complexity_score: int = Field(ge=0, le=100)
    estimated_cost: float = Field(ge=0, description="Estimated cost in USD")
    estimated_latency_ms: float = Field(ge=0, description="Estimated latency in milliseconds")
    routing_mode: RoutingMode
    factors: dict[str, Any] = Field(default_factory=dict, description="Factors that influenced the decision")
    decided_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True

    def format_decision(self) -> str:
        lines = [
            f"🤖 Auto-Routing Decision",
            f"",
            f"Selected: {self.provider}/{self.model_name}",
            f"Reason: {self.reason}",
            f"",
            f"Factors:",
        ]

        for key, value in self.factors.items():
            formatted_key = key.replace("_", " ").title()
            lines.append(f"  • {formatted_key}: {value}")

        lines.extend(
            [
                f"",
                f"Complexity: {self.complexity_score}/100",
                f"Estimated cost: ${self.estimated_cost:.4f}",
                f"Estimated time: ~{self.estimated_latency_ms:.0f}ms",
            ]
        )

        return "\n".join(lines)


class ModelCapabilities(BaseModel):
    provider: str
    model_name: str
    is_local: bool
    avg_latency_ms: float = Field(description="Average response latency in ms")
    cost_per_1k_input: float = Field(ge=0, description="Cost per 1K input tokens")
    cost_per_1k_output: float = Field(ge=0, description="Cost per 1K output tokens")
    quality_score: int = Field(ge=1, le=10, description="Quality rating 1-10")
    speed_score: int = Field(ge=1, le=10, description="Speed rating 1-10 (10=fastest)")
    max_context_length: int = Field(default=4096, description="Maximum context window")
    supports_streaming: bool = Field(default=True)

    def estimate_cost(self, input_tokens: int, output_tokens: int = 100) -> float:
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output
        return input_cost + output_cost
