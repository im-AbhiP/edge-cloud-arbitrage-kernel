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
from runtime.models import BaseModelClient, ModelResult, get_available_models
from runtime.prompts import PromptRegistry
from runtime.logging_utils import CallLogger

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
    Decides which model handles each task based on explicit policies.

    DECISION HIERARCHY (order matters!):
    1. Forced model override → use that exact model
    2. Privacy check → HIGH sensitivity ALWAYS goes edge
    3. Privacy mode → edge_only mode forces everything local
    4. Budget check → over hard limit forces edge
    5. Task-based routing → match task to model capabilities
    """

    # Default model for each scenario
    EDGE_DEFAULT = "ollama/llama3.1-8b"
    EDGE_FAST = "ollama/deepseek-r1-8b"
    CLOUD_DEFAULT = "gemini/gemini-2.5-flash"
    CLOUD_BUDGET = "gemini/gemini-2.5-flash-lite"
    CLOUD_PREMIUM = "gemini/gemini-2.5-pro"
    CLOUD_LUXURY = "gemini/gemini-3-pro-preview"

    def __init__(self):
        self.models = get_available_models()
        self.logger = CallLogger()
        self.prompt_registry = PromptRegistry()

        # Load policy configuration from environment variables
        self.privacy_mode = os.getenv("PRIVACY_MODE", "hybrid")
        self.soft_budget = float(os.getenv("SOFT_BUDGET_USD", "1.00"))
        self.hard_budget = float(os.getenv("HARD_BUDGET_USD", "5.00"))

    def select_model(self, meta: TaskMetadata) -> RoutingDecision:
        """
        Core routing logic. Returns which model to use and WHY.
        """

        # --- RULE 1: Explicit override ---
        if meta.force_model and meta.force_model in self.models:
            return RoutingDecision(
                meta.force_model, "Explicit model override"
            )

        # --- RULE 2: Privacy enforcement (NON-NEGOTIABLE) ---
        if meta.data_sensitivity == DataSensitivity.HIGH:
            return RoutingDecision(
                self.EDGE_DEFAULT,
                "HIGH data sensitivity — forced to edge",
                privacy_enforced=True
            )

        # --- RULE 3: Global privacy mode ---
        if self.privacy_mode == "edge_only":
            return RoutingDecision(
                self.EDGE_DEFAULT,
                "Privacy mode is edge_only — all tasks run locally",
                privacy_enforced=True
            )

        # --- RULE 4: Budget enforcement ---
        mtd_cost = self.logger.get_month_to_date_cost()

        if mtd_cost >= self.hard_budget:
            return RoutingDecision(
                self.EDGE_DEFAULT,
                f"Hard budget exceeded (${mtd_cost:.4f} >= "
                f"${self.hard_budget})",
                budget_enforced=True
            )

        if mtd_cost >= self.soft_budget:
            if meta.importance > 0.8:
                return RoutingDecision(
                    self.CLOUD_DEFAULT,
                    "Soft budget exceeded but high-importance task "
                    "— using cheaper cloud model"
                )
            return RoutingDecision(
                self.EDGE_DEFAULT,
                f"Soft budget exceeded (${mtd_cost:.4f} >= "
                f"${self.soft_budget}) — routing to edge",
                budget_enforced=True
            )

        # --- RULE 5: Task-based routing ---

        # Simple tasks → local (save money)
        if meta.task_type in [TaskType.QUICK_QA, TaskType.SIMPLE_SUMMARY]:
            if meta.budget_sensitivity > 0.5:
                return RoutingDecision(
                    self.EDGE_FAST,
                    "Simple task + cost-conscious → fast local model"
                )
            return RoutingDecision(
                self.EDGE_DEFAULT, "Simple task → local model"
            )

        # Complex tasks → cloud
        if meta.task_type in [TaskType.DEEP_RESEARCH, TaskType.PLANNING]:
            if meta.complexity > 0.7 and meta.importance > 0.7:
                return RoutingDecision(
                    self.CLOUD_PREMIUM,
                    "Complex + important → premium cloud model"
                )
            if meta.complexity > 0.5:
                return RoutingDecision(
                    self.CLOUD_DEFAULT,
                    "Moderately complex → standard cloud model"
                )

        # Code review — cloud is usually better
        if meta.task_type == TaskType.CODE_REVIEW:
            if meta.importance > 0.6:
                return RoutingDecision(
                    self.CLOUD_DEFAULT,
                    "Code review with moderate+ importance → cloud"
                )
            return RoutingDecision(
                self.EDGE_DEFAULT, "Low-importance code review → local"
            )

        # Default: prefer edge to save costs
        return RoutingDecision(
            self.EDGE_DEFAULT, "Default policy → edge to minimize cost"
        )

    def run(self, prompt: str, meta: TaskMetadata,
            system_prompt: Optional[str] = None,
            max_tokens: int = 1024,
            temperature: float = 0.7) -> ModelResult:
        """
        Execute a task: select model → call it → log everything →
        handle failures.
        """

        # Step 1: Decide which model
        decision = self.select_model(meta)
        print(f"[Router] {decision}")

        # Step 2: Get the model client
        client = self.models.get(decision.model_name)
        if client is None:
            # Fallback to any available edge model
            for name, c in self.models.items():
                if c.tier == "edge":
                    client = c
                    decision = RoutingDecision(
                        name, "Fallback — requested model unavailable"
                    )
                    break

        if client is None:
            raise RuntimeError("No models available")

        # Step 3: Make the actual API call
        result = client.call(
            prompt, system_prompt=system_prompt,
            max_tokens=max_tokens, temperature=temperature
        )

        # Step 4: If cloud failed, try edge as fallback
        if not result.success and result.tier == "cloud":
            print(f"[Router] Cloud failed ({result.error}), "
                  f"falling back to edge")
            edge_client = self.models.get(self.EDGE_DEFAULT)
            if edge_client:
                result = edge_client.call(
                    prompt, system_prompt=system_prompt,
                    max_tokens=max_tokens, temperature=temperature
                )
                decision = RoutingDecision(
                    self.EDGE_DEFAULT, "Fallback after cloud failure"
                )

        # Step 5: Log everything
        self.logger.log(
            model_name=result.model_name,
            tier=result.tier,
            task_type=meta.task_type.value,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens or 0,
            completion_tokens=result.completion_tokens or 0,
            success=result.success,
            error=result.error,
            privacy_enforced=decision.privacy_enforced,
            budget_enforced=decision.budget_enforced,
        )

        return result
