"""
The Routing Engine — the brain of the Arbitrage Kernel.

THIS IS THE CORE INTELLECTUAL PROPERTY.

The routing logic is EXPLICIT RULES, not machine learning.
Why? Because in an interview, you need to explain every decision:

"If the data is sensitive, it stays local. Period.
 If the budget is exceeded, we downgrade or go local.
 If the task is complex and important, we use cloud.
 Otherwise, we default to local to save money."

This is EXPLAINABLE AI GOVERNANCE.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from runtime.tasks import TaskMetadata, TaskType, DataSensitivity
from runtime.models import ModelResult, get_available_models
from runtime.logging_utils import CallLogger
from council.deliberation import CouncilDeliberation

load_dotenv()


class RoutingDecision:
    """
    Captures the routing decision AND the reason why.

    WHY TRACK THE REASON?
    Observability. Every routing decision is auditable.
    When you show the logs, each call says WHY it was routed
    where it was. In an interview: "Every inference decision
    is explainable and auditable."
    """

    def __init__(self, model_name: str, reason: str,
                 privacy_enforced: bool = False,
                 budget_enforced: bool = False):
        self.model_name = model_name
        self.reason = reason
        self.privacy_enforced = privacy_enforced
        self.budget_enforced = budget_enforced

    def __repr__(self):
        return (f"RoutingDecision(model={self.model_name}, "
                f"reason='{self.reason}')")


class ModelRouter:
    """
    The Routing Engine — now powered by the Council of Local LLMs.
    WHAT CHANGED: Previously, routing was rule-based (if complexity > X, use cloud). Now, the 5 local models VOTE on every routing decision.
    The old rule-based logic is kept as a FAST PATH for when you don't want full council deliberation (e.g., during benchmarking).
    TWO MODES:
        council_mode=True → Full council deliberation (slower, smarter)
        council_mode=False → Original rule-based routing (fast, simple)
    Decides which model handles each task based on explicit policies.

    OLD DECISION HIERARCHY (order matters!):
    1. Forced model override → use that exact model
    2. Privacy check → HIGH sensitivity ALWAYS goes edge
    3. Privacy mode → edge_only mode forces everything local
    4. Budget check → over hard limit forces edge
    5. Task-based routing → match task to model capabilities

    Routes tasks using either council deliberation or fast rules.
    """

    # Default model for each scenario
    EDGE_DEFAULT = "ollama/llama3.1-8b"
    EDGE_REASONING = "ollama/deepseek-r1-8b"
    EDGE_CODE = "ollama/deepseek-coder-6.7b"
    EDGE_FAST = "ollama/qwen3-8b"
    EDGE_HEAVY = "ollama/gpt-oss-20b"
    CLOUD_DEFAULT = "gemini/gemini-2.5-flash"
    CLOUD_BUDGET = "gemini/gemini-2.5-flash-lite"
    CLOUD_PREMIUM = "gemini/gemini-2.5-pro"
    CLOUD_LUXURY = "gemini/gemini-3-pro-preview"

    def __init__(self, council_mode: bool = True):
        """
        Parameters:
        - council_mode: If True, uses the council for every routing
          decision. If False, uses fast rule-based routing.
        """
        self.models = get_available_models()
        self.logger = CallLogger()
        self.council_mode = council_mode
        self.privacy_mode = os.getenv("PRIVACY_MODE", "hybrid")
        self.soft_budget = float(os.getenv("SOFT_BUDGET_USD", "1.00"))
        self.hard_budget = float(os.getenv("HARD_BUDGET_USD", "5.00"))

        if council_mode:
            self.council = CouncilDeliberation()

    def run(self, prompt: str, meta: TaskMetadata,
            system_prompt: Optional[str] = None,
            max_tokens: int = 2048,
            temperature: float = 0.7) -> ModelResult:
        """
        Execute a task. In council mode, this triggers full deliberation.
        In fast mode, this uses rule-based routing.
        """

        if self.council_mode:
            return self._run_council(prompt, meta)
        else:
            return self._run_fast(prompt, meta, system_prompt,
                                  max_tokens, temperature)

    def _run_council(self, prompt: str,
                     meta: TaskMetadata) -> ModelResult:
        """Route via council deliberation."""
        meta.user_prompt = prompt
        result = self.council.deliberate(prompt, meta)

        # Wrap deliberation result as a ModelResult for compatibility
        return ModelResult(
            text=result.final_answer,
            model_name=result.selected_model,
            tier="edge" if result.routing_decision == "local" else "cloud",
            latency_ms=result.total_deliberation_ms,
            success=True,
        )

    def _run_fast(self, prompt: str, meta: TaskMetadata,
                  system_prompt: Optional[str] = None,
                  max_tokens: int = 2048,
                  temperature: float = 0.7) -> ModelResult:
        """
        Original rule-based routing (kept for benchmarking
        and when speed > deliberation quality).
        """
        # Privacy enforcement
        if meta.data_sensitivity == DataSensitivity.HIGH:
            model_id = self.EDGE_DEFAULT
        elif self.privacy_mode == "edge_only":
            model_id = self.EDGE_DEFAULT
        # Budget enforcement
        elif self.logger.get_month_to_date_cost() >= self.hard_budget:
            model_id = self.EDGE_DEFAULT
        # Task-based routing
        elif meta.task_type in [TaskType.QUICK_QA, TaskType.SIMPLE_SUMMARY]:
            model_id = self.EDGE_FAST
        elif meta.task_type == TaskType.CODE_REVIEW:
            model_id = self.EDGE_CODE
        elif meta.task_type in [TaskType.DEEP_RESEARCH, TaskType.PLANNING]:
            if meta.complexity > 0.7:
                model_id = self.CLOUD_PREMIUM
            else:
                model_id = self.CLOUD_DEFAULT
        else:
            model_id = self.EDGE_DEFAULT

        client = self.models.get(model_id)
        if client is None:
            client = list(self.models.values())[0]
            model_id = list(self.models.keys())[0]

        result = client.call(prompt, system_prompt=system_prompt,
                             max_tokens=max_tokens, temperature=temperature)

        self.logger.log(
            model_name=result.model_name, tier=result.tier,
            task_type=meta.task_type.value, latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens or 0,
            completion_tokens=result.completion_tokens or 0,
            success=result.success, error=result.error,
        )

        return result

