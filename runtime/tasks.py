"""
Task metadata system for the Edge-Cloud Arbitrage Kernel.

WHY THIS FILE EXISTS:
Every AI task has characteristics that determine WHERE it should run.
A simple "what's the capital of France?" doesn't need an expensive cloud model.
A complex "analyze this company's 10-K filing" probably does.

By explicitly modeling these characteristics as structured data,
we make routing decisions TRANSPARENT and EXPLAINABLE.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TaskType(Enum):
    """
    Categories of AI tasks. Each has different compute requirements.

    WHAT IS AN ENUM?
    An Enum (enumeration) is a fixed set of named constants.
    Instead of using plain strings like "simple_summary" (where you might
    accidentally type "simpel_summary"), an Enum guarantees only valid
    values can be used. PyCharm will also autocomplete these for you.
    """
    SIMPLE_SUMMARY = "simple_summary"
    DEEP_RESEARCH = "deep_research"
    CODE_REVIEW = "code_review"
    PLANNING = "planning"
    CREATIVE_WRITING = "creative_writing"
    DATA_ANALYSIS = "data_analysis"
    QUICK_QA = "quick_qa"
    LONG_SUMMARIZATION = "long_summarization"


class DataSensitivity(Enum):
    """
    How sensitive is the data in this task?

    HIGH = PII, financial data, health info → MUST stay local (edge)
    MEDIUM = Internal business data → prefer local, allow cloud if needed
    LOW = Public information → can route anywhere

    WHY THIS MATTERS:
    This is your "enterprise governance" signal. In an interview, you say:
    "HIGH sensitivity data NEVER leaves the device. Period.
    That's a non-negotiable policy enforced at the routing layer."
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class TaskMetadata:
    """
    Structured description of an AI task's requirements.

    WHAT IS A DATACLASS?
    A dataclass is a Python shortcut for creating classes that mainly
    hold data. Instead of writing __init__, __repr__, etc. yourself,
    the @dataclass decorator auto-generates them. It's like a struct
    in C or a record in Java.

    HOW IT'S USED:
    - The Router reads this to decide which model handles the task
    - The Logger uses this to tag metrics
    - YOU use this in interviews to explain your routing logic
    """
    task_type: TaskType
    data_sensitivity: DataSensitivity = DataSensitivity.LOW

    # Float scales from 0.0 to 1.0:
    # These are the "knobs" the router uses to make decisions.

    # 0.0 = "take your time", 1.0 = "I need this NOW"
    latency_sensitivity: float = 0.5

    # 0.0 = "spend whatever", 1.0 = "minimize cost at all costs"
    budget_sensitivity: float = 0.5

    # 0.0 = "simple task", 1.0 = "requires deep reasoning"
    complexity: float = 0.5

    # 0.0 = "doesn't matter much", 1.0 = "critical output"
    importance: float = 0.5

    # Estimated token count (from prompt registry). Helps predict cost.
    estimated_tokens: Optional[int] = None

    # Optional override: force a specific model (bypasses all routing)
    force_model: Optional[str] = None

    # Optional: max budget for this specific task
    task_budget_cap_usd: Optional[float] = None

    # NEW: The actual user prompt text. The council needs to see this
    # to make an informed routing decision.
    user_prompt: Optional[str] = None

    def __post_init__(self):
        """
        Validate that all float values are in range.

        WHAT IS __post_init__?
        In a dataclass, __post_init__ runs automatically AFTER the object
        is created. We use it to validate the data — if someone passes
        complexity=5.0, we catch that error immediately rather than
        letting it cause weird routing behavior later.
        """
        for attr in ['latency_sensitivity', 'budget_sensitivity',
                     'complexity', 'importance']:
            val = getattr(self, attr)
            if not 0.0 <= val <= 1.0:
                raise ValueError(
                    f"{attr} must be between 0.0 and 1.0, got {val}"
                )
