import sys
import time
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.llm import get_llm_client
from src.llm.models import Message, MessageRole
from src.routing.strategies import MODEL_REGISTRY

TEST_QUERIES = [
    {
        "query": "What is Python?",
        "complexity": "Simple",
        "expected_tokens": 50,
    },
    {
        "query": "Explain the difference between supervised and unsupervised learning in machine learning.",
        "complexity": "Moderate",
        "expected_tokens": 150,
    },
    {
        "query": "Compare and contrast the transformer architecture with traditional RNNs, discussing attention mechanisms, computational complexity, and practical applications in modern NLP.",
        "complexity": "Complex",
        "expected_tokens": 300,
    },
]


def benchmark_model(provider: str, model_name: str, query: str, max_tokens: int = 500) -> dict[str, Any]:
    try:
        client = get_llm_client(provider, model_name=model_name)
        messages = [Message(role=MessageRole.USER, content=query)]
        start_time = time.time()
        response = client.generate(messages, temperature=0.7, max_tokens=max_tokens)
        end_time = time.time()

        metadata = response.metadata
        actual_latency = (end_time - start_time) * 1000

        return {
            "provider": provider,
            "model_name": model_name,
            "success": True,
            "response_preview": response.content[:100] + "..." if len(response.content) > 100 else response.content,
            "input_tokens": metadata.input_tokens,
            "output_tokens": metadata.output_tokens,
            "total_tokens": metadata.total_tokens,
            "latency_ms": actual_latency,
            "reported_latency_ms": metadata.latency_ms,
            "cost": metadata.cost,
            "tokens_per_second": metadata.output_tokens / (actual_latency / 1000) if actual_latency > 0 else 0,
        }

    except Exception as e:
        return {
            "provider": provider,
            "model_name": model_name,
            "success": False,
            "error": str(e),
            "latency_ms": 0,
            "cost": 0,
            "tokens_per_second": 0,
        }


def print_separator(char: str = "=", length: int = 100):
    print(char * length)


def print_benchmark_header():
    print_separator()
    print("MODEL BENCHMARKING: Speed vs. Quality vs. Cost Comparison")
    print_separator()
    print()


def print_query_header(query_info: dict):
    print(f"\n{'=' * 100}")
    print(f"Query Complexity: {query_info['complexity']}")
    print(f"Query: {query_info['query']}")
    print(f"Expected Output: ~{query_info['expected_tokens']} tokens")
    print(f"{'=' * 100}\n")


def print_result(result: dict):
    model_full_name = f"{result['provider']}/{result['model_name']}"

    if not result["success"]:
        print(f"[FAIL] {model_full_name}")
        print(f"   Error: {result['error']}")
        print()
        return

    model_key = f"{result['provider']}/{result['model_name']}"
    capabilities = MODEL_REGISTRY.get(model_key)

    if capabilities:
        quality_score = capabilities.quality_score
        if quality_score >= 8:
            quality_indicator = "*** High"
        elif quality_score >= 6:
            quality_indicator = "** Medium"
        else:
            quality_indicator = "* Basic"
    else:
        quality_indicator = "? Unknown"

    latency = result["latency_ms"]
    if latency < 500:
        speed_indicator = ">>> Ultra-Fast"
    elif latency < 2000:
        speed_indicator = ">> Fast"
    elif latency < 5000:
        speed_indicator = "> Moderate"
    else:
        speed_indicator = "- Slow"

    cost = result["cost"]
    if cost == 0:
        cost_indicator = "$ Free (Local)"
    elif cost < 0.001:
        cost_indicator = "$$ Very Cheap"
    elif cost < 0.01:
        cost_indicator = "$$$ Cheap"
    else:
        cost_indicator = "$$$$ Expensive"

    print(f"[OK] {model_full_name}")
    print(f"   Quality:       {quality_indicator}")
    print(f"   Speed:         {speed_indicator} ({latency:.0f}ms)")
    print(f"   Cost:          {cost_indicator} (${cost:.6f})")
    print(f"   Throughput:    {result['tokens_per_second']:.1f} tokens/sec")
    print(f"   Tokens:        {result['output_tokens']} output, {result['total_tokens']} total")
    print(f"   Preview:       {result['response_preview']}")
    print()


def print_summary(all_results: list[dict]):
    print(f"\n{'=' * 100}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 100}\n")

    successful = [r for r in all_results if r["success"]]

    if not successful:
        print("No successful benchmark results to summarize.")
        return

    print("[SPEED] Rankings (Fastest to Slowest):")
    speed_sorted = sorted(successful, key=lambda x: x["latency_ms"])
    for i, result in enumerate(speed_sorted[:5], 1):
        model = f"{result['provider']}/{result['model_name']}"
        print(f"   {i}. {model:<40} {result['latency_ms']:>8.0f}ms ({result['tokens_per_second']:.1f} tok/s)")
    print()

    print("[COST] Rankings (Cheapest to Most Expensive):")
    cost_sorted = sorted(successful, key=lambda x: x["cost"])
    for i, result in enumerate(cost_sorted[:5], 1):
        model = f"{result['provider']}/{result['model_name']}"
        cost_label = "FREE" if result["cost"] == 0 else f"${result['cost']:.6f}"
        print(f"   {i}. {model:<40} {cost_label:>12}")
    print()

    print("[QUALITY] Rankings (Best to Worst):")
    quality_sorted = sorted(
        successful,
        key=lambda x: MODEL_REGISTRY.get(f"{x['provider']}/{x['model_name']}", type('obj', (), {'quality_score': 0})).quality_score,
        reverse=True
    )
    for i, result in enumerate(quality_sorted[:5], 1):
        model = f"{result['provider']}/{result['model_name']}"
        model_key = f"{result['provider']}/{result['model_name']}"
        capabilities = MODEL_REGISTRY.get(model_key)
        quality = capabilities.quality_score if capabilities else 0
        print(f"   {i}. {model:<40} {quality}/10")
    print()

    print("[BEST TRADEOFFS]:")
    print(f"   • Best for Speed:   {speed_sorted[0]['provider']}/{speed_sorted[0]['model_name']} ({speed_sorted[0]['latency_ms']:.0f}ms)")
    print(f"   • Best for Cost:    {cost_sorted[0]['provider']}/{cost_sorted[0]['model_name']} (${cost_sorted[0]['cost']:.6f})")

    best_quality_key = f"{quality_sorted[0]['provider']}/{quality_sorted[0]['model_name']}"
    best_quality_score = MODEL_REGISTRY.get(best_quality_key).quality_score
    print(f"   • Best for Quality: {quality_sorted[0]['provider']}/{quality_sorted[0]['model_name']} ({best_quality_score}/10)")
    print()


def run_benchmark():
    print_benchmark_header()

    models_to_test = [
        ("ollama", "gemma3:1b"),
        ("cerebras", "llama-3.3-70b"),
        ("mistral", "mistral-medium-latest"),
    ]

    print("Models being tested:")
    for provider, model_name in models_to_test:
        model_key = f"{provider}/{model_name}"
        capabilities = MODEL_REGISTRY.get(model_key)
        if capabilities:
            print(f"  • {provider}/{model_name} (Quality: {capabilities.quality_score}/10, Speed: {capabilities.speed_score}/10)")
        else:
            print(f"  • {provider}/{model_name}")
    print()

    all_results = []

    for query_info in TEST_QUERIES:
        print_query_header(query_info)

        for provider, model_name in models_to_test:
            print(f"Testing {provider}/{model_name}...")
            result = benchmark_model(
                provider=provider,
                model_name=model_name,
                query=query_info["query"],
                max_tokens=query_info["expected_tokens"] * 2
            )
            print_result(result)
            all_results.append(result)

        print("-" * 100)

    print_summary(all_results)

    print_separator()
    print("Benchmark complete!")
    print_separator()


if __name__ == "__main__":
    run_benchmark()
