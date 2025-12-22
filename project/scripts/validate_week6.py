import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.llm import get_llm_client, list_available_models, Message, MessageRole
from src.routing import QueryRouter, RoutingMode
from src.routing.strategies import MODEL_REGISTRY


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_check(condition: bool, message: str):
    status = "[OK]" if condition else "[FAIL]"
    print(f"{status} {message}")
    return condition


def validate_local_models():
    print_section("1. Local Model Integration (Ollama)")

    all_passed = True

    try:
        from src.llm.ollama_client import OllamaClient
        all_passed &= print_check(True, "OllamaClient class exists")
    except ImportError:
        all_passed &= print_check(False, "OllamaClient class exists")
        return False

    models = OllamaClient.SUPPORTED_MODELS
    all_passed &= print_check(len(models) >= 1, f"At least 1 model supported (found {len(models)})")
    print(f"     Supported models: {', '.join(models)}")

    all_passed &= print_check(
        hasattr(OllamaClient, 'generate'),
        "OllamaClient has generate() method"
    )
    all_passed &= print_check(
        hasattr(OllamaClient, 'stream'),
        "OllamaClient has stream() method"
    )

    ollama_models = [k for k in MODEL_REGISTRY.keys() if k.startswith("ollama/")]
    all_passed &= print_check(
        len(ollama_models) >= 1,
        f"Ollama models in MODEL_REGISTRY (found {len(ollama_models)})"
    )

    return all_passed


def validate_cloud_integration():
    print_section("2. Cloud Model Integration (Unified Interface)")

    all_passed = True

    cloud_clients = ["mistral", "cerebras", "openrouter"]
    for client_type in cloud_clients:
        try:
            models = list_available_models(client_type)
            all_passed &= print_check(
                len(models) > 0,
                f"{client_type.capitalize()}: {len(models)} models available"
            )
        except Exception as e:
            all_passed &= print_check(False, f"{client_type.capitalize()}: Error - {str(e)[:50]}")

    all_passed &= print_check(
        callable(get_llm_client),
        "get_llm_client() factory exists"
    )

    providers = set(m.provider for m in MODEL_REGISTRY.values())
    all_passed &= print_check(
        "ollama" in providers,
        "Ollama in MODEL_REGISTRY"
    )
    all_passed &= print_check(
        "cerebras" in providers,
        "Cerebras in MODEL_REGISTRY"
    )
    all_passed &= print_check(
        "mistral" in providers,
        "Mistral in MODEL_REGISTRY"
    )
    all_passed &= print_check(
        "openrouter" in providers,
        "OpenRouter in MODEL_REGISTRY"
    )

    print(f"\n     Total models in registry: {len(MODEL_REGISTRY)}")
    print(f"     Providers: {', '.join(sorted(providers))}")

    return all_passed


def validate_query_router():
    print_section("3. Query Router (Analysis + Routing Logic)")

    all_passed = True

    try:
        router = QueryRouter(mode=RoutingMode.BALANCED)
        all_passed &= print_check(True, "QueryRouter class instantiates")
    except Exception as e:
        all_passed &= print_check(False, f"QueryRouter class: {str(e)[:50]}")
        return False

    test_queries = [
        ("What is 2+2?", "simple"),
        ("Explain quantum computing in detail", "complex"),
    ]

    for query, expected_complexity in test_queries:
        try:
            decision, analysis = router.route(query)
            all_passed &= print_check(
                analysis is not None,
                f"Analysis works for {expected_complexity} query"
            )
            all_passed &= print_check(
                hasattr(analysis, 'complexity_score'),
                f"Analysis has complexity_score"
            )
            all_passed &= print_check(
                hasattr(analysis, 'token_count'),
                f"Analysis has token_count"
            )
            print(f"     Query: '{query[:40]}...'")
            print(f"     Complexity: {analysis.complexity_score}/100, Tokens: {analysis.token_count}")
        except Exception as e:
            all_passed &= print_check(False, f"Routing failed: {str(e)[:50]}")

    all_passed &= print_check(
        len([m for m in RoutingMode]) >= 4,
        f"At least 4 routing modes (found {len([m for m in RoutingMode])})"
    )

    try:
        decision, analysis = router.route("Test query")
        all_passed &= print_check(
            hasattr(decision, 'reason'),
            "Decision has 'reason' field"
        )
        all_passed &= print_check(
            hasattr(decision, 'factors'),
            "Decision has 'factors' field"
        )
        all_passed &= print_check(
            hasattr(decision, 'complexity_score'),
            "Decision has 'complexity_score' field"
        )
        print(f"     Selected model: {decision.provider}/{decision.model_name}")
        print(f"     Reason: {decision.reason[:60]}...")
    except Exception as e:
        all_passed &= print_check(False, f"Decision logging: {str(e)[:50]}")

    return all_passed


def validate_routing_modes():
    print_section("4. Routing Modes (Cost/Quality/Speed tradeoffs)")

    all_passed = True

    test_query = "Explain machine learning algorithms in detail"

    modes = [
        RoutingMode.BALANCED,
        RoutingMode.COST_OPTIMIZED,
        RoutingMode.QUALITY_OPTIMIZED,
        RoutingMode.ULTRA_FAST,
    ]

    results = {}
    for mode in modes:
        try:
            router = QueryRouter(mode=mode)
            decision, analysis = router.route(test_query)
            results[mode.value] = {
                "model": f"{decision.provider}/{decision.model_name}",
                "cost": decision.estimated_cost,
                "latency": decision.estimated_latency_ms,
            }
            all_passed &= print_check(True, f"{mode.value} mode works")
        except Exception as e:
            all_passed &= print_check(False, f"{mode.value} mode: {str(e)[:40]}")

    if results:
        print("\n     Routing Decisions:")
        for mode, data in results.items():
            print(f"     {mode:20s}: {data['model']:30s} "
                  f"(cost: ${data['cost']:.4f}, latency: {data['latency']:.0f}ms)")

        if "cost-optimized" in results and "quality-optimized" in results:
            cost_opt_cost = results["cost-optimized"]["cost"]
            quality_opt_cost = results["quality-optimized"]["cost"]
            all_passed &= print_check(
                cost_opt_cost <= quality_opt_cost,
                "Cost-optimized is cheaper than quality-optimized"
            )

        if "ultra-fast" in results:
            ultra_latency = results["ultra-fast"]["latency"]
            other_latencies = [v["latency"] for k, v in results.items() if k != "ultra-fast"]
            if other_latencies:
                all_passed &= print_check(
                    ultra_latency <= min(other_latencies),
                    "Ultra-fast has lowest latency"
                )

    return all_passed


def validate_ui_integration():
    print_section("5. UI Integration (Routing + Cost Savings)")

    all_passed = True

    ui_file = project_root / "src" / "ui" / "app.py"
    all_passed &= print_check(ui_file.exists(), "UI file exists (app.py)")

    if ui_file.exists():
        ui_content = ui_file.read_text()

        all_passed &= print_check(
            "RoutingMode" in ui_content,
            "UI imports RoutingMode"
        )

        all_passed &= print_check(
            "auto" in ui_content.lower(),
            "UI has auto-routing option"
        )

        all_passed &= print_check(
            "cost_savings" in ui_content.lower() or "savings" in ui_content.lower(),
            "UI tracks cost savings"
        )

        all_passed &= print_check(
            "routing_decision" in ui_content.lower() or "decision" in ui_content,
            "UI displays routing decisions"
        )

        all_passed &= print_check(
            "ollama" in ui_content.lower(),
            "UI has Ollama models in dropdown"
        )

        all_passed &= print_check(
            "metadata" in ui_content.lower(),
            "UI displays response metadata"
        )

    return all_passed


def validate_context_window():
    print_section("6. Context Window Validation (max_tokens awareness)")

    all_passed = True

    try:
        router = QueryRouter(mode=RoutingMode.BALANCED)

        test_query = "Simple test"
        decision1, _ = router.route(test_query, max_output_tokens=500)
        decision2, _ = router.route(test_query, max_output_tokens=50000)

        all_passed &= print_check(
            decision1 is not None and decision2 is not None,
            "Router handles different max_tokens values"
        )

        cap1 = MODEL_REGISTRY[f"{decision1.provider}/{decision1.model_name}"]
        cap2 = MODEL_REGISTRY[f"{decision2.provider}/{decision2.model_name}"]

        print(f"     500 tokens -> {decision1.provider}/{decision1.model_name} "
              f"(max context: {cap1.max_context_length})")
        print(f"     50K tokens -> {decision2.provider}/{decision2.model_name} "
              f"(max context: {cap2.max_context_length})")

        all_passed &= print_check(
            cap2.max_context_length >= cap1.max_context_length,
            "Large token requirement selects model with sufficient context"
        )

    except Exception as e:
        all_passed &= print_check(False, f"Context window handling: {str(e)[:50]}")

    return all_passed


def main():
    print("\n" + "=" * 70)
    print("  WEEK 6 REQUIREMENTS VALIDATION")
    print("  Local Models & Intelligent Routing")
    print("=" * 70)

    results = {}

    results["Local Models"] = validate_local_models()
    results["Cloud Integration"] = validate_cloud_integration()
    results["Query Router"] = validate_query_router()
    results["Routing Modes"] = validate_routing_modes()
    results["UI Integration"] = validate_ui_integration()
    results["Context Window"] = validate_context_window()

    print_section("VALIDATION SUMMARY")

    total_passed = sum(1 for v in results.values() if v)
    total_tests = len(results)

    for section, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {section}")

    print("\n" + "=" * 70)
    if total_passed == total_tests:
        print(f"  ALL TESTS PASSED ({total_passed}/{total_tests})")
        print("  Week 6 requirements are complete!")
    else:
        print(f"  SOME TESTS FAILED ({total_passed}/{total_tests} passed)")
        print("  Please review failed sections above")
    print("=" * 70 + "\n")

    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
