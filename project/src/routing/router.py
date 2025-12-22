from src.routing.analyzer import QueryAnalyzer
from src.routing.models import QueryAnalysis, RoutingDecision, RoutingMode
from src.routing.strategies import MODEL_REGISTRY, RoutingStrategy


class QueryRouter:
    def __init__(self, mode: RoutingMode = RoutingMode.BALANCED):
        self.mode = mode
        self.analyzer = QueryAnalyzer()
        self.strategy = RoutingStrategy(mode=mode)

    def route(self, query: str, max_output_tokens: int = 1000) -> tuple[RoutingDecision, QueryAnalysis]:
        analysis = self.analyzer.analyze(query)
        model_key = self.strategy.select_model(
            complexity_score=analysis.complexity_score,
            token_count=analysis.token_count,
        )
        capabilities = MODEL_REGISTRY[model_key]

        total_tokens_needed = analysis.token_count + max_output_tokens
        if total_tokens_needed > capabilities.max_context_length:
            model_key = self._find_model_with_larger_context(
                total_tokens_needed=total_tokens_needed,
                current_model_key=model_key,
            )
            capabilities = MODEL_REGISTRY[model_key]

        decision = self._build_decision(
            model_key=model_key,
            capabilities=capabilities,
            analysis=analysis,
            max_output_tokens=max_output_tokens,
        )

        return decision, analysis

    def _build_decision(
        self,
        model_key: str,
        capabilities: "ModelCapabilities",
        analysis: QueryAnalysis,
        max_output_tokens: int = 1000,
    ) -> RoutingDecision:
        reason = self._generate_reason(
            complexity_score=analysis.complexity_score,
            question_type=analysis.question_type,
            quality_needed=analysis.estimated_quality_needed,
            is_local=capabilities.is_local,
        )

        estimated_cost = capabilities.estimate_cost(
            input_tokens=analysis.token_count,
            output_tokens=max_output_tokens,
        )

        factors = {
            "query_length": self._format_length(analysis.token_count),
            "question_type": analysis.question_type,
            "has_complex_keywords": "Yes" if analysis.has_complex_keywords else "No",
            "quality_needed": analysis.estimated_quality_needed,
            "model_type": "Local" if capabilities.is_local else "Cloud",
            "quality_score": f"{capabilities.quality_score}/10",
            "speed_score": f"{capabilities.speed_score}/10",
        }

        return RoutingDecision(
            provider=capabilities.provider,
            model_name=capabilities.model_name,
            reason=reason,
            complexity_score=analysis.complexity_score,
            estimated_cost=estimated_cost,
            estimated_latency_ms=capabilities.avg_latency_ms,
            routing_mode=self.mode,
            factors=factors,
        )

    def _generate_reason(
        self,
        complexity_score: int,
        question_type: str,
        quality_needed: str,
        is_local: bool,
    ) -> str:
        if complexity_score < 30:
            complexity_desc = "Simple"
        elif complexity_score < 70:
            complexity_desc = "Moderate"
        else:
            complexity_desc = "Complex"

        type_desc = question_type.replace("_", " ").title()

        if self.mode == RoutingMode.BALANCED:
            if is_local:
                return f"{complexity_desc} {type_desc.lower()} query - using fast local model"
            else:
                return f"{complexity_desc} {type_desc.lower()} query requiring {quality_needed} quality"

        elif self.mode == RoutingMode.COST_OPTIMIZED:
            if is_local:
                return f"Cost-optimized: using free local model for {type_desc.lower()} query"
            else:
                return f"Cost-optimized: complexity requires cloud, using cheapest option"

        elif self.mode == RoutingMode.QUALITY_OPTIMIZED:
            return f"Quality-optimized: selected high-quality model for {type_desc.lower()} query"

        elif self.mode == RoutingMode.ULTRA_FAST:
            return f"Ultra-fast mode: selected fastest available model"

        else:
            return f"{complexity_desc} {type_desc.lower()} query"

    def _format_length(self, token_count: int) -> str:
        if token_count < 50:
            return f"Short ({token_count} tokens)"
        elif token_count < 200:
            return f"Medium ({token_count} tokens)"
        else:
            return f"Long ({token_count} tokens)"

    def set_mode(self, mode: RoutingMode) -> None:
        self.mode = mode
        self.strategy = RoutingStrategy(mode=mode)

    def get_available_modes(self) -> list[str]:
        return [mode.value for mode in RoutingMode]

    def explain_mode(self, mode: RoutingMode) -> str:
        explanations = {
            RoutingMode.BALANCED: "Balances cost, speed, and quality for optimal overall performance",
            RoutingMode.COST_OPTIMIZED: "Minimizes cost by preferring local and cheap models",
            RoutingMode.QUALITY_OPTIMIZED: "Maximizes response quality using high-end models",
            RoutingMode.ULTRA_FAST: "Prioritizes speed above all else",
        }
        return explanations.get(mode, "Unknown mode")

    def _find_model_with_larger_context(self, total_tokens_needed: int, current_model_key: str) -> str:
        current_capabilities = MODEL_REGISTRY[current_model_key]

        models_by_context = sorted(
            MODEL_REGISTRY.items(), key=lambda x: x[1].max_context_length, reverse=True
        )

        suitable_models = []
        for model_key, capabilities in models_by_context:
            if capabilities.max_context_length >= total_tokens_needed:
                suitable_models.append((model_key, capabilities))

        if not suitable_models:
            if not models_by_context:
                raise RuntimeError("MODEL_REGISTRY is empty, cannot select fallback model")
            return models_by_context[0][0]

        same_type_models = [
            (key, cap) for key, cap in suitable_models if cap.is_local == current_capabilities.is_local
        ]

        if same_type_models:
            return min(same_type_models, key=lambda x: x[1].max_context_length)[0]
        else:
            return min(suitable_models, key=lambda x: x[1].max_context_length)[0]
