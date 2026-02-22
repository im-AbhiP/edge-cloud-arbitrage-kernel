import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure the images directory exists in the parent folder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Chart 1: Cumulative Cost Avoidance (Edge vs. Cloud)
# ---------------------------------------------------------
def generate_cost_chart():
    # Simulated data: 100 API calls
    calls = np.arange(1, 101)

    # Cloud Cost: Linear scaling (e.g., Gemini Pro)
    cloud_cost_per_call = 0.05
    cumulative_cloud_cost = calls * cloud_cost_per_call

    # Edge Cost: Zero marginal cost
    cumulative_edge_cost = np.zeros_like(calls)

    plt.figure(figsize=(10, 6))
    plt.plot(calls, cumulative_cloud_cost, label='Default Cloud Routing ($)', color='#e74c3c', linewidth=3)
    plt.plot(calls, cumulative_edge_cost, label='Arbitrage Kernel (Local Edge)', color='#2ecc71', linewidth=3)

    # Fill the area between to highlight "Cost Avoided"
    plt.fill_between(calls, cumulative_cloud_cost, cumulative_edge_cost, color='#2ecc71', alpha=0.1,
                     label='Cost Avoided / Arbitrage Yield')

    plt.title('Cumulative Inference Cost: Cloud Default vs. Edge Arbitrage', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Cumulative Inference Calls', fontsize=12)
    plt.ylabel('Cost (USD)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'cost_arbitrage_yield.png')
    plt.savefig(output_path, dpi=300)
    print(f"Generated: {output_path}")
    plt.close()


# ---------------------------------------------------------
# Chart 2: Inference Profiling (Prefill vs. Decode Latency)
# ---------------------------------------------------------
def generate_latency_chart():
    models = ['gpt-oss-20b', 'llama3.1-8b', 'deepseek-r1-8b', 'qwen3-8b', 'deepseek-coder']

    # Simulated latency in milliseconds
    prefill_latency = np.array([1200, 450, 480, 410, 350])
    decode_latency = np.array([4500, 1800, 1950, 1600, 1400])

    plt.figure(figsize=(10, 6))

    # Stacked bar chart
    plt.bar(models, prefill_latency, color='#3498db', label='Prefill Latency (Prompt Ingestion)')
    plt.bar(models, decode_latency, bottom=prefill_latency, color='#2c3e50', label='Decode Latency (Token Generation)')

    plt.title('Edge Model Latency Profiling (M1 Apple Silicon)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Local Edge Models', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'inference_latency_profile.png')
    plt.savefig(output_path, dpi=300)
    print(f"Generated: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Generating architectural charts for README...")
    generate_cost_chart()
    generate_latency_chart()
    print("Done!")