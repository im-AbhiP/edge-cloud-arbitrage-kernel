"""
Council of Local LLMs — High-level interface.

This is the file you import when you want to use the council.
It wraps the deliberation engine with convenience methods.

The old Explainer/Skeptic/Synthesizer pattern is replaced by
the democratic council pattern where ALL local models participate
in voting, model selection, answer generation, and iterative review.

USAGE:
    from council.agents import LLMCouncil
    from runtime.tasks import TaskType

    council = LLMCouncil()
    result = council.ask("What is TCP?", task_type=TaskType.QUICK_QA)
    print(result.final_answer)
"""

from typing import Optional, List

from runtime.tasks import TaskMetadata, TaskType, DataSensitivity
from council.deliberation import CouncilDeliberation, DeliberationResult


class LLMCouncil:
    """
    High-level interface to the Council of Local LLMs.

    This class wraps CouncilDeliberation with a simpler API.
    Instead of constructing TaskMetadata objects manually, you
    pass keyword arguments and this class builds them for you.

    TWO MODES:
    - ask()      : Full council (all specified models, up to 3 iterations)
    - ask_fast() : Speed mode (4 small models, 2 iterations, majority approval)

    EXAMPLE:
        council = LLMCouncil()
        result = council.ask(
            "What is the difference between TCP and UDP?",
            task_type=TaskType.QUICK_QA,
            complexity=0.3,
        )
        print(result.final_answer)
        print(result.routing_decision)  # "local" or "cloud"
        print(result.selected_model)    # e.g., "ollama/qwen3-8b"
        print(result.iterations)        # e.g., 2
        print(result.consensus_reached) # True or False
    """

    def __init__(
        self,
        council_models: Optional[List[str]] = None,
        max_iterations: int = 3,
        approval_threshold: float = 1.0,
        skip_review_for_cloud: bool = True,
    ):
        """
        Initialize the council.

        Parameters:
        -----------
        council_models : list of str, optional
            Which local models sit on the council.
            Default: all 5 edge models.
            For faster deliberation, exclude gpt-oss:20b (uses ~14GB RAM):
                council_models=[
                    "ollama/llama3.1-8b",
                    "ollama/deepseek-r1-8b",
                    "ollama/qwen3-8b",
                    "ollama/deepseek-coder-6.7b",
                ]

        max_iterations : int
            Maximum number of review rounds before accepting the
            best answer so far. Default 3.
            - 1 = no revision (accept first answer if approved)
            - 2 = one chance to improve
            - 3 = two chances to improve (recommended default)

        approval_threshold : float
            Fraction of council models that must approve the answer.
            - 1.0 = unanimous (ALL must approve)
            - 0.67 = supermajority (2 of 3, or 4 of 5)
            - 0.5 = simple majority

        skip_review_for_cloud : bool
            If True, when the council votes to send a task to cloud,
            the cloud answer is returned without council review.
            Cloud models are already stronger than local models,
            so reviewing them with local models adds latency
            without much quality benefit.
        """
        self.engine = CouncilDeliberation(
            council_models=council_models,
            max_iterations=max_iterations,
            approval_threshold=approval_threshold,
            skip_review_for_cloud=skip_review_for_cloud,
        )

    def ask(
        self,
        prompt: str,
        task_type: TaskType = TaskType.QUICK_QA,
        complexity: float = 0.5,
        importance: float = 0.5,
        data_sensitivity: DataSensitivity = DataSensitivity.LOW,
        budget_sensitivity: float = 0.5,
    ) -> DeliberationResult:
        """
        Ask the council a question using full deliberation.

        The council will:
        1. Vote on whether to answer locally or send to cloud
        2. If local, vote on which model should answer
        3. Generate an answer with the selected model
        4. Review the answer iteratively until consensus

        Parameters:
        -----------
        prompt : str
            The question or task to send to the council.

        task_type : TaskType
            Category of the task. Affects routing decisions.
            Options: QUICK_QA, SIMPLE_SUMMARY, DEEP_RESEARCH,
                     CODE_REVIEW, PLANNING, CREATIVE_WRITING,
                     DATA_ANALYSIS, LONG_SUMMARIZATION

        complexity : float (0.0 to 1.0)
            How complex the task is.
            0.0 = trivial ("What is 2+2?")
            1.0 = very complex ("Analyze AMD's competitive strategy")

        importance : float (0.0 to 1.0)
            How important getting a high-quality answer is.
            0.0 = doesn't matter much
            1.0 = critical output

        data_sensitivity : DataSensitivity
            How sensitive the data in the prompt is.
            LOW = public info, can go to cloud
            MEDIUM = internal, prefer local
            HIGH = PII/financial, MUST stay local (non-negotiable)

        budget_sensitivity : float (0.0 to 1.0)
            How cost-conscious to be.
            0.0 = spend whatever it takes
            1.0 = minimize cost at all costs

        Returns:
        --------
        DeliberationResult
            Contains: final_answer, routing_decision, selected_model,
            iterations, consensus_reached, routing_votes,
            selection_votes, reviews, total_deliberation_ms
        """
        meta = TaskMetadata(
            task_type=task_type,
            complexity=complexity,
            importance=importance,
            data_sensitivity=data_sensitivity,
            budget_sensitivity=budget_sensitivity,
            user_prompt=prompt,
        )

        return self.engine.deliberate(prompt, meta)

    def ask_fast(
        self,
        prompt: str,
        task_type: TaskType = TaskType.QUICK_QA,
        complexity: float = 0.5,
    ) -> DeliberationResult:
        """
        Fast mode: Uses only the 4 smaller models (excludes gpt-oss:20b)
        and requires only majority approval (3 of 4).

        Use this for quick questions where speed matters more than
        the most thorough deliberation.

        WHY THIS EXISTS:
        Full council with 5 models and 3 iterations can take 10+ minutes
        on a 16GB M1 Pro because of model swapping. Fast mode:
        - Skips gpt-oss:20b (saves ~14GB RAM swap)
        - Only 2 review iterations max
        - 75% approval threshold (3 of 4, not unanimous)
        - Typically completes in 3-5 minutes

        Parameters:
        -----------
        prompt : str
            The question or task.

        task_type : TaskType
            Category of the task.

        complexity : float (0.0 to 1.0)
            How complex the task is.

        Returns:
        --------
        DeliberationResult
        """
        fast_engine = CouncilDeliberation(
            council_models=[
                "ollama/llama3.1-8b",
                "ollama/deepseek-r1-8b",
                "ollama/qwen3-8b",
                "ollama/deepseek-coder-6.7b",
            ],
            max_iterations=2,
            approval_threshold=0.75,
        )

        meta = TaskMetadata(
            task_type=task_type,
            complexity=complexity,
            user_prompt=prompt,
        )

        return fast_engine.deliberate(prompt, meta)
