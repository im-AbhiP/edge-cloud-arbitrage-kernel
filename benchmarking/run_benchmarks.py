"""
Benchmarking runner.

WHY BENCHMARKING:
Without empirical data, your routing rules are guesses.
With benchmarking, you say: "I tested 8 task types across 4 models
and found that local Mistral handles summaries at comparable quality
with 60% lower latency and zero cost."

That's engineering rigor.
"""

import yaml
import csv
from pathlib import Path
from datetime import datetime
from runtime.models import get_available_models


def load_benchmarks():
    path = Path(__file__).parent / "benchmark_dataset.yaml"
    with open(path) as f:
        return yaml.safe_load(f)["benchmarks"]


def run_benchmarks():
    benchmarks = load_benchmarks()
    models = get_available_models()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "reports"
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / f"benchmark_{timestamp}.csv"
    md_path = output_dir / f"benchmark_{timestamp}.md"

    results = []

    print(f"Running {len(benchmarks)} benchmarks across "
          f"{len(models)} models...\n")

    for bench in benchmarks:
        for model_name, client in models.items():
            print(f"  [{bench['id']}] → {model_name}...",
                  end=" ", flush=True)

            result = client.call(
                prompt=bench["prompt"],
                max_tokens=bench.get("max_tokens", 1024),
                temperature=0.7,
            )

            row = {
                "benchmark_id": bench["id"],
                "benchmark_name": bench["name"],
                "task_type": bench["task_type"],
                "model_name": model_name,
                "tier": result.tier,
                "latency_ms": round(result.latency_ms, 1),
                "prompt_tokens": result.prompt_tokens or 0,
                "completion_tokens": result.completion_tokens or 0,
                "total_tokens": result.total_tokens or 0,
                "success": result.success,
                "error": result.error or "",
                "response_preview": (
                    result.text[:200].replace("\n", " ")
                    if result.text else ""
                ),
            }
            results.append(row)

            status = "✓" if result.success else "✗"
            print(f"{status} {result.latency_ms:.0f}ms")

    # Write CSV
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n📊 CSV: {csv_path}")

    # Write Markdown report
    generate_markdown_report(results, md_path)
    print(f"📝 Report: {md_path}")


def generate_markdown_report(results, md_path):
    """Auto-generate a human-readable benchmark report."""

    lines = [
        "# Benchmark Report",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary by Model",
        "",
        "| Model | Tier | Avg Latency (ms) | Total Tokens | Success Rate |",
        "|-------|------|-------------------|--------------|--------------|",
    ]

    # Aggregate by model
    model_stats = {}
    for r in results:
        mn = r["model_name"]
        if mn not in model_stats:
            model_stats[mn] = {
                "latencies": [], "tokens": 0,
                "successes": 0, "total": 0, "tier": r["tier"]
            }
        model_stats[mn]["latencies"].append(r["latency_ms"])
        model_stats[mn]["tokens"] += r["total_tokens"]
        model_stats[mn]["successes"] += 1 if r["success"] else 0
        model_stats[mn]["total"] += 1

    for mn, stats in model_stats.items():
        avg_lat = sum(stats["latencies"]) / len(stats["latencies"])
        success_rate = stats["successes"] / stats["total"] * 100
        lines.append(
            f"| {mn} | {stats['tier']} | {avg_lat:.0f} | "
            f"{stats['tokens']} | {success_rate:.0f}% |"
        )

    lines.extend(["", "## Detailed Results", "",
                  "| Benchmark | Model | Latency (ms) | Tokens | OK |",
                  "|-----------|-------|-------------|--------|-----|"])

    for r in results:
        ok = "✓" if r["success"] else "✗"
        lines.append(
            f"| {r['benchmark_name']} | {r['model_name']} | "
            f"{r['latency_ms']:.0f} | {r['total_tokens']} | {ok} |"
        )

    with open(md_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_benchmarks()
