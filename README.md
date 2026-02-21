# 🧠 Edge-Cloud AI Arbitrage Kernel

**A Hybrid Compute Governance Engine for Cost-Aware, Privacy-Aware LLM Orchestration**

[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Built with Ollama](https://img.shields.io/badge/Edge-Ollama-orange.svg)](https://ollama.com/)
[![Cloud: Gemini](https://img.shields.io/badge/Cloud-Gemini%20API-4285F4.svg)](https://ai.google.dev/)

---

## What Is This?

A Python-based runtime that **dynamically allocates AI inference workloads between local edge compute (Ollama on Apple Silicon) and cloud APIs (Google Gemini)**, making routing decisions based on explicit policies for **cost, latency, privacy, and task complexity**.

Every routing decision is logged, auditable, and explainable. Every dollar of cloud spend is tracked against a full-cloud baseline to measure **cost avoidance**. Sensitive data never leaves the device.

> **This is not a chatbot. It is a governance layer for AI compute allocation.**

---

## Why Does This Exist?

Most AI applications send every request to the cloud by default. This is:

- **Expensive** — cloud API costs scale linearly with usage
- **Slow** — network round-trips add latency
- **Risky** — sensitive data leaves your control

The Arbitrage Kernel asks a simple question before every inference call:

> *"Does this task NEED the cloud, or can local hardware handle it?"*

By routing simple, low-stakes, or privacy-sensitive tasks to local models and reserving cloud compute for complex, high-importance tasks, the system achieves:

- **60–70% of workloads processed locally at zero marginal cost**
- **Enterprise-grade privacy enforcement** — HIGH sensitivity data never leaves the device
- **Budget governance** — soft and hard spending caps with automatic downgrade policies
- **Full observability** — every call logged with model, tier, latency, tokens, cost, and routing reason

---

## Key Features

| Feature | Description |
|---|---|
| 🔀 **Explicit Routing Engine** | Rule-based model selection using task type, complexity, importance, privacy level, and budget state. Every decision includes a human-readable reason. |
| 🔒 **Privacy Gating** | `HIGH` sensitivity tasks are forced to edge compute. No exceptions. No overrides. A non-negotiable policy enforced at the routing layer. |
| 💰 **Budget Cap Enforcement** | Soft budget triggers model downgrade warnings. Hard budget blocks all cloud calls. Month-to-date spend tracked automatically. |
| 📊 **Full Observability & Logging** | Every inference call logged to SQLite: model, tier, task type, latency, tokens, actual cost, and hypothetical cloud cost. |
| 🤖 **Multi-Agent Research Council** | Three-agent pipeline (Explainer → Skeptic → Synthesizer) that produces structured JSON output with assumptions, risks, disagreements, and confidence scores. |
| ✅ **Contract Tests** | Pytest-based schema validation ensures council output is always parseable and compliant — production discipline, not prototype behavior. |
| 📈 **Benchmarking Framework** | 8 task scenarios across 6 models (2 edge + 4 cloud), producing CSV data and auto-generated Markdown comparison reports. |
| 🖥️ **Executive Dashboard** | Streamlit dashboard showing cost avoidance, edge/cloud distribution, latency comparison, cumulative spend, and governance events. |

---

Architecture
------------

```text
User Task
    ↓
TaskMetadata (type, complexity, sensitivity, importance)
    ↓
┌──────────────────────────────────────┐
│            ROUTING ENGINE           │
│  ┌───────────┐      ┌──────────────┐│
│  │ Privacy   │      │   Budget     ││
│  │  Policy   │      │   Policy     ││
│  └─────┬─────┘      └──────┬───────┘│
│        └──────────────┬─────────────┘│
│          Task-Based Routing          │
│        (complexity × importance)     │
└──────────┬───────────┬───────────────┘
           ↓           ↓
      ┌────────────┐   ┌────────────────────┐
      │    EDGE    │   │       CLOUD        │
      │   Ollama   │   │     Gemini API     │
      │ Llama 3.1  │   │ 2.5 Flash / Pro    │
      │ DeepSeek R1│   │ 3.0 Pro Preview    │
      └─────┬──────┘   └────────┬───────────┘
            ↓                   ↓
      ┌───────────────────┐
      │   LOGGING LAYER   │
      │ SQLite + Cost     │
      │ Tracking          │
      └────────┬──────────┘
               ↓
      ┌───────────────────┐
      │   ROI DASHBOARD   │
      │ Cost Avoidance    │
      │ Edge/Cloud %      │
      │ Governance Events │
      └───────────────────┘
```



### Routing Decision Hierarchy

Decisions are evaluated **in strict order** — the first matching rule wins:

1. **Forced Override** → Use the explicitly specified model
2. **Privacy Check** → `HIGH` sensitivity → forced to edge (non-negotiable)
3. **Privacy Mode** → `edge_only` mode → everything local
4. **Hard Budget** → Monthly cloud spend ≥ hard cap → forced to edge
5. **Soft Budget** → Monthly cloud spend ≥ soft cap → downgrade or prefer edge
6. **Task Routing** → Match task type + complexity + importance to model capabilities
7. **Default** → Edge (minimize cost)

---

## Model Inventory

### Edge Models (Local — Ollama on Apple Silicon)

| Model | ID | Parameters | Quantization | Strengths |
|---|---|---|---|---|
| **Llama 3.1 8B Instruct** | `ollama/llama3.1-8b` | 8B | Q5_K_M | Strong general-purpose instruction following, balanced quality/speed |
| **DeepSeek R1 8B** | `ollama/deepseek-r1-8b` | 8B | Default | Excellent reasoning and chain-of-thought, strong at code and analysis |

### Cloud Models (Google Gemini API)

| Model | ID | Tier | Best For |
|---|---|---|---|
| **Gemini 2.5 Flash** | `gemini/gemini-2.5-flash` | Standard | Fast general-purpose tasks, good cost/quality balance |
| **Gemini 2.5 Flash Lite** | `gemini/gemini-2.5-flash-lite` | Economy | High-volume, cost-sensitive tasks |
| **Gemini 2.5 Pro** | `gemini/gemini-2.5-pro` | Premium | Complex reasoning, deep research, high-stakes analysis |
| **Gemini 3.0 Pro Preview** | `gemini/gemini-3-pro-preview` | Cutting-edge | Latest capabilities, frontier model performance |

---

## Quick Start

### Prerequisites

- **macOS with Apple Silicon** (M1/M2/M3/M4 — developed on M1 Pro 16GB)
- **Python 3.14+**
- **Ollama** installed and running
- **Google Gemini API key** (free tier)

### 1. Clone the Repository

```bash
git clone https://github.com/im-AbhiP/edge-cloud-arbitrage-kernel.git
cd edge-cloud-arbitrage-kernel
```

### 2. Set Up the Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install httpx python-dotenv pyyaml rich
pip install pytest ruff
pip install streamlit plotly pandas
```

### 3. Install and Start Ollama

```bash
brew install ollama

ollama serve
ollama pull llama3.1:8b-instruct-q5_K_M
ollama pull deepseek-r1:8b
```

### 4. Configure Environment Variables
Create a .env file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key_here
PRIVACY_MODE=hybrid
SOFT_BUDGET_USD=1.00
HARD_BUDGET_USD=5.00
```

Get your free Gemini API key at Google AI Studio.

### 5. Run the Smoke Test

```bash
python test_smoke.py
```

You should see three tests execute:
Test 1: Simple Q&A → routed to edge (Llama 3.1 8B)
Test 2: Deep research → routed to cloud (Gemini 2.5 Pro)
Test 3: Sensitive data → forced to edge regardless of complexity
### Basic: Route a Single Task

```python
from runtime.tasks import TaskMetadata, TaskType, DataSensitivity
from runtime.router import ModelRouter

router = ModelRouter()

# Simple question → routes to local model (free)
result = router.run(
    prompt="What is the difference between TCP and UDP?",
    meta=TaskMetadata(
        task_type=TaskType.QUICK_QA,
        complexity=0.2,
        importance=0.3,
        budget_sensitivity=0.8,
    ),
)

print(result.text)
print(f"Model: {result.model_name} | Cost: $0.00")
```

### Privacy-Enforced Task

```python
# Sensitive data → ALWAYS stays local, no matter what
result = router.run(
    prompt="Analyze this employee's compensation data...",
    meta=TaskMetadata(
        task_type=TaskType.DATA_ANALYSIS,
        complexity=0.9,
        importance=0.9,
        data_sensitivity=DataSensitivity.HIGH,  # Forces edge
    )
)
# Guaranteed: result.tier == "edge"
```

### Multi-Agent Research Council

```python
from council.agents import ResearchCouncil
council = ResearchCouncil() output = council.run( "What are the tradeoffs of edge vs cloud AI inference?" )
print(output["council_output"]["summary"]) print(output["council_output"]["risks"]) print(output["council_output"]["confidence"])
```

Output is structured JSON with assumptions, risks, disagreements.

### Run Benchmarks

```bash
python -m benchmarking.run_benchmarks
```

Runs 8 task scenarios across all 6 available models and generates:
 - `benchmarking/reports/benchmark_YYYYMMDD_HHMMSS.csv`
 - `benchmarking/reports/benchmark_YYYYMMDD_HHMMSS.md`

### Generate ROI Report

```bash
python scripts/summarize_logs.py
```

### Launch the Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Opens an interactive dashboard at http://localhost:8501.
## Project Structure

```text
edge-cloud-arbitrage-kernel/
├── runtime/                   # Core inference engine
│   ├── __init__.py
│   ├── models.py              # Model clients (Ollama, Gemini)
│   ├── tasks.py               # TaskMetadata, TaskType, DataSensitivity
│   ├── router.py              # Routing engine with policy enforcement
│   ├── prompts.py             # Prompt registry loader
│   ├── prompts.yaml           # Prompt templates with expected tokens
│   ├── logging_utils.py       # SQLite logging, cost estimation, ROI stats
│   └── policies.py            # Policy configuration
│
├── council/                   # Multi-agent research pipeline
│   ├── __init__.py
│   ├── agents.py              # Explainer → Skeptic → Synthesizer
│   ├── schemas.py             # JSON schema for structured output
│   └── contract_tests.py      # Pytest schema validation
│
├── benchmarking/              # Empirical evaluation framework
│   ├── __init__.py
│   ├── run_benchmarks.py      # Benchmark runner (8 tasks × 6 models)
│   ├── benchmark_dataset.yaml # Benchmark scenarios
│   └── reports/               # Auto-generated CSV + Markdown reports
│
├── dashboard/
│   └── streamlit_app.py       # Executive ROI dashboard
│
├── scripts/
│   └── summarize_logs.py      # CLI ROI report generator
│
├── analysis/
│   └── analyze_logs.py        # Ad-hoc log analysis
│
├── data/
│   └── logs.db                # SQLite call log database (auto-created)
│
├── reports/                   # Generated ROI summaries
├── test_smoke.py              # End-to-end smoke test
├── .env                       # API keys & config (not committed)
├── .gitignore
└── README.md
```

## How Routing Decisions Work

Every routing decision is transparent and auditable. The router logs not just which model was selected, but **why**.

### Routing Examples
| Scenario                               | Decision                     | Reason                                       |
|----------------------------------------|------------------------------|----------------------------------------------|
| Quick Q&A, low importance              | ollama/llama3.1-8b           | Simple task + cost-conscious → fast local    |
| Deep research, high complexity         | gemini/gemini-2.5-pro        | Complex + important → premium cloud model    |
| Reasoning-heavy, moderate importance   | ollama/deepseek-r1-8b        | Reasoning task → DeepSeek R1 excels locally  |
| High-volume, low-stakes task           | gemini/gemini-2.5-flash-lite | Economy tier → minimize cloud cost           |
| Any task with HIGH data sensitivity    | ollama/llama3.1-8b           | HIGH sensitivity — forced to edge            |
| Any task when budget exceeded          | ollama/llama3.1-8b           | Hard budget exceeded → edge only             |
| Cloud API failure                      | ollama/llama3.1-8b           | Fallback after cloud failure                 |
| Frontier-capability needed             | gemini/gemini-3-pro-preview  | Cutting-edge task → latest model             |


## Cost Model
| Model                        | Tier             | Cost               |
|-----------------------------|------------------|--------------------|
| ollama/llama3.1-8b          | Edge             | $0.00              |
| ollama/deepseek-r1-8b       | Edge             | $0.00              |
| gemini/gemini-2.5-flash-lite| Cloud (Economy)  | Per-token pricing  |
| gemini/gemini-2.5-flash     | Cloud (Standard) | Per-token pricing  |
| gemini/gemini-2.5-pro       | Cloud (Premium)  | Per-token pricing  |
| gemini/gemini-3-pro-preview | Cloud (Frontier) | Per-token pricing  |

For every edge call, the system calculates the hypothetical cloud cost — what it would have cost if sent to Gemini 2.5 Flash. The difference is the cost avoidance metric that powers the ROI dashboard.
### Sample ROI Output

```text
Edge-Cloud AI Arbitrage Kernel — ROI Report

Executive Summary
-----------------
Total AI calls: 46
Edge (local) calls: 31 (67.4%)
Cloud calls: 15 (32.6%)

Total cloud cost: $0.0123
Cost if ALL calls were cloud: $0.0970
💰 Cost avoided: $0.0847

Average latency: 1,847ms
Success rate: 97.8%
```
---

## Testing

### Contract Tests

```bash
pytest council/contract_tests.py -v
```

Validates:
 - ✅ All required keys present (summary, assumptions, risks, disagreements, confidence)
 - ✅ Confidence score is a number in [0.0, 1.0]
 - ✅ assumptions and risks are lists 
 - ✅ Summary is a non-empty string

###Smoke Test

```bash
python test_smoke.py
```

Validates the full pipeline: task metadata → routing → model call → logging → stats.

## Tech Stack
| Component        | Technology                 | Why                                                   |
|------------------|----------------------------|-------------------------------------------------------|
| Language         | Python 3.14                | Latest stable release, performance improvements       |
| Local Inference  | Ollama                     | Best local LLM runtime for Apple Silicon, REST API    |
| Edge Models      | Llama 3.1 8B, DeepSeek R1  | Instruction-following + reasoning coverage            |
| Cloud Inference  | Google Gemini API          | Generous free tier, strong quality, simple REST API   |
| HTTP Client      | httpx                      | Modern, async-capable, no SDK dependency              |
| Database         | SQLite                     | Zero-config, built into Python, SQL-queryable         |
| Config           | YAML + .env                | Human-readable config, secure credential management   |
| Testing          | pytest                     | Industry standard, clean syntax                       |
| Dashboard        | Streamlit + Plotly         | Rapid interactive dashboards with minimal code        |
| Linter           | Ruff                       | Faster than flake8, replaces black + isort            |
| IDE              | PyCharm Professional       | Best-in-class Python IDE with database tools          |

## Dashboard:



The Streamlit dashboard provides five key views:
 - 💰 Cost Avoidance — Dollars saved vs. a full-cloud baseline 
 - 🏠 Edge/Cloud Distribution — Pie chart of compute allocation 
 - ⚡ Latency Comparison — Bar chart of average latency per model 
 - 📈 Cumulative Cost — Line chart comparing hybrid cost vs. cloud-only cost over time 
 - 🔒 Governance Events — Count of privacy and budget enforcement actions

Launch with: 
`streamlit run dashboard/streamlit_app.py`
![Dashboard Screenshot](./reports/dashboard_screenshot.png)

## Roadmap
✅ v1 (Current)
 - Model client abstraction (Ollama + Gemini, 6 models)
 - Explicit routing engine with privacy & budget policies 
 - SQLite logging with cost tracking and ROI computation 
 - Prompt registry with expected token counts 
 - Benchmarking framework (8 scenarios × 6 models)
 - Multi-agent research council with structured JSON output 
 - Contract tests for output schema validation 
 - Executive Streamlit dashboard 
 - CLI ROI report generator

🔜 v2 (Planned — Choose One)
 - Performance-Adaptive Router — Use aggregate log statistics to auto-adjust routing thresholds 
 - Task Decomposition — Split complex tasks: cheap local chunking → expensive cloud synthesis 
 - Confidence-Based Escalation — If council confidence < threshold, re-route to a stronger model

🔮 Future Directions
 - OpenRouter integration for access to 100+ models through a single API 
 - Adaptive routing weights learned from historical performance data 
 - Context budget optimization for long-document tasks 
 - Multi-tenant support with per-user privacy and budget policies 
 - MassGen integration for advanced agent orchestration

## Design Principles
 + Explicit over Magic — Routing rules are readable if-else logic, not opaque ML models. Every decision is explainable in plain English. 
 + Measure Everything — If it's not logged, it didn't happen. Every call captures model, tier, latency, tokens, cost, and routing reason. 
 + Privacy is Non-Negotiable — HIGH sensitivity data never leaves the device. Enforced at the routing layer, not left to the caller. 
 + Ship, Then Improve — v1 uses simple rules. v2 adds data-driven optimization. Complexity is earned, not assumed. 
 + Interview-Ready at Every Commit — Code, tests, documentation, and dashboards are always in a presentable state.

## The 60-Second Explanation
> I built a hybrid AI orchestration kernel that dynamically allocates inference workloads between edge and cloud based on cost, complexity, and privacy constraints.
> The edge tier runs Llama 3.1 and DeepSeek R1 locally on Apple Silicon at zero marginal cost, while the cloud tier leverages four Gemini models from economy to frontier.
> Every routing decision is explainable and auditable. Sensitive data never leaves the device — that's a non-negotiable policy enforced at the routing layer.
> In testing, the system routed approximately 60–70% of workloads to local inference while maintaining comparable output quality.
> I validated this with empirical benchmarks across 8 task types and 6 models, and built an executive dashboard that visualizes cost avoidance, compute distribution, and governance events in real time.


### License
This project is licensed under the MIT License — see the LICENSE file for details.

### Author
Abhishek\
Senior Technical Product Manager @ AMD | Georgia Tech

Building at the intersection of AI infrastructure, edge compute, and hybrid cloud governance.\
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/abhishekhpatil)\
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/im-AbhiP)
