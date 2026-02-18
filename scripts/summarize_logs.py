"""Generate ROI summary from logs — your interview artifact."""

from pathlib import Path
from datetime import datetime
from runtime.logging_utils import CallLogger


def generate_roi_report():
    logger = CallLogger()
    stats = logger.get_summary_stats()
    overall = stats["overall"]

    report = f"""# Edge-Cloud AI Arbitrage Kernel — ROI Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Executive Summary
- **Total AI calls:** {overall['total_calls']}
- **Edge (local) calls:** {overall['edge_calls']} ({stats['edge_percentage']:.1f}%)
- **Cloud calls:** {overall['cloud_calls']}
- **Total cloud cost:** ${overall['total_cost']:.4f}
- **Cost if ALL calls were cloud:** ${overall['total_cloud_alternative_cost']:.4f}
- **💰 Cost avoided:** ${stats['cost_avoided']:.4f}
- **Avg latency:** {overall['avg_latency']:.0f}ms
- **Success rate:** {overall['success_rate']:.1f}%

## Per-Model Breakdown
| Model | Tier | Calls | Avg Latency | Cost |
|-------|------|-------|-------------|------|
"""
    for m in stats["per_model"]:
        report += (f"| {m['model_name']} | {m['tier']} | {m['calls']} "
                   f"| {m['avg_latency_ms']:.0f}ms | ${m['total_cost']:.4f} |\n")

    output_path = Path("reports/roi_summary.md")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(report)
    print(f"📊 Report saved to {output_path}")
    print(report)


if __name__ == "__main__":
    generate_roi_report()
