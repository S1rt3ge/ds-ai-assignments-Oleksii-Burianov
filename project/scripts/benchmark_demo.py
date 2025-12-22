import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.routing.strategies import MODEL_REGISTRY


def print_separator(char: str = "=", length: int = 100):
    print(char * length)


def demonstrate_model_tradeoffs():
    print_separator()
    print("MODEL COMPARISON: Speed vs. Quality vs. Cost Tradeoffs")
    print_separator()
    print()

    models = []
    for key, capabilities in MODEL_REGISTRY.items():
        provider, model_name = key.split("/", 1)
        models.append({
            "provider": provider,
            "model_name": model_name,
            "quality_score": capabilities.quality_score,
            "speed_score": capabilities.speed_score,
            "cost_per_1k_input": capabilities.cost_per_1k_input,
            "cost_per_1k_output": capabilities.cost_per_1k_output,
            "avg_latency_ms": capabilities.avg_latency_ms,
            "is_local": capabilities.is_local,
        })

    test_input_tokens = 100
    test_output_tokens = 200

    for model in models:
        model["estimated_cost"] = (
            (test_input_tokens / 1000) * model["cost_per_1k_input"]
            + (test_output_tokens / 1000) * model["cost_per_1k_output"]
        )

    print("ALL MODELS (sorted by quality):\n")
    print(f"{'Provider/Model':<40} {'Quality':<12} {'Speed':<12} {'Latency':<15} {'Cost (100/200)':<20} {'Type':<10}")
    print("-" * 110)

    sorted_by_quality = sorted(models, key=lambda x: x["quality_score"], reverse=True)
    for model in sorted_by_quality:
        provider_model = f"{model['provider']}/{model['model_name']}"
        quality_str = f"{model['quality_score']}/10"
        speed_str = f"{model['speed_score']}/10"
        latency_str = f"{model['avg_latency_ms']:.0f}ms"
        cost_str = f"${model['estimated_cost']:.6f}" if model['estimated_cost'] > 0 else "FREE"
        model_type = "Local" if model["is_local"] else "Cloud"

        print(f"{provider_model:<40} {quality_str:<12} {speed_str:<12} {latency_str:<15} {cost_str:<20} {model_type:<10}")

    print()
    print_separator("-")

    print("\n[SPEED] Rankings (Fastest to Slowest):")
    speed_sorted = sorted(models, key=lambda x: x["avg_latency_ms"])
    for i, model in enumerate(speed_sorted, 1):
        print(f"   {i}. {model['provider']}/{model['model_name']:<35} {model['avg_latency_ms']:>8.0f}ms")

    print("\n[COST] Rankings (Cheapest to Most Expensive):")
    cost_sorted = sorted(models, key=lambda x: x["estimated_cost"])
    for i, model in enumerate(cost_sorted, 1):
        cost_label = "FREE" if model["estimated_cost"] == 0 else f"${model['estimated_cost']:.6f}"
        print(f"   {i}. {model['provider']}/{model['model_name']:<35} {cost_label:>12}")

    print("\n[QUALITY] Rankings (Best to Worst):")
    for i, model in enumerate(sorted_by_quality, 1):
        print(f"   {i}. {model['provider']}/{model['model_name']:<35} {model['quality_score']}/10")

    print()
    print_separator("-")

    print("\n[BEST TRADEOFFS]:")
    print(f"   - Best for Speed:   {speed_sorted[0]['provider']}/{speed_sorted[0]['model_name']} ({speed_sorted[0]['avg_latency_ms']:.0f}ms)")
    print(f"   - Best for Cost:    {cost_sorted[0]['provider']}/{cost_sorted[0]['model_name']} (${cost_sorted[0]['estimated_cost']:.6f})")
    print(f"   - Best for Quality: {sorted_by_quality[0]['provider']}/{sorted_by_quality[0]['model_name']} ({sorted_by_quality[0]['quality_score']}/10)")

    print()
    print_separator("-")

    print("\n[KEY INSIGHTS]:")
    print(f"   - LOCAL MODELS: Free to use, fastest response times, but lower quality")
    print(f"     Example: {models[0]['provider']}/{models[0]['model_name']} - {models[0]['avg_latency_ms']:.0f}ms, Quality {models[0]['quality_score']}/10")

    cloud_models = [m for m in models if not m["is_local"]]
    if cloud_models:
        cheapest_cloud = sorted(cloud_models, key=lambda x: x["estimated_cost"])[0]
        print(f"   - CLOUD MODELS (Cheap): Low cost, good balance")
        print(f"     Example: {cheapest_cloud['provider']}/{cheapest_cloud['model_name']} - ${cheapest_cloud['estimated_cost']:.6f}, Quality {cheapest_cloud['quality_score']}/10")

        best_quality = sorted_by_quality[0]
        if not best_quality["is_local"]:
            print(f"   - CLOUD MODELS (Premium): Highest quality, higher cost")
            print(f"     Example: {best_quality['provider']}/{best_quality['model_name']} - ${best_quality['estimated_cost']:.6f}, Quality {best_quality['quality_score']}/10")

    print()
    print_separator("-")

    print("\n[ROUTING RECOMMENDATIONS]:")
    print("   - Simple queries (low complexity)     -> Use local models (FREE, fast)")
    print("   - Moderate queries                    -> Use cheap cloud models (balanced)")
    print("   - Complex queries (high complexity)   -> Use premium cloud models (best quality)")
    print("   - Cost-optimized mode                 -> Prefer local, fallback to cheapest cloud")
    print("   - Quality-optimized mode              -> Use best cloud models")
    print("   - Ultra-fast mode                     -> Use fastest models regardless of quality")

    print()
    print_separator()
    print("Benchmark demonstration complete!")
    print_separator()


if __name__ == "__main__":
    demonstrate_model_tradeoffs()
