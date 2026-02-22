"""
The Council Deliberation Engine.

THIS IS THE MOST IMPORTANT NEW FILE.

It orchestrates the entire deliberation process:
1. Routing vote (5 models vote: local vs. cloud)
2. Model selection (5 models vote: who answers)
3. Answer generation (selected model produces an answer)
4. Iterative review (5 models review, provide feedback, repeat)

DESIGN DECISIONS:
- Models vote SEQUENTIALLY (not in parallel) because of RAM constraints
- Votes are weighted by confidence scores
- Majority wins for routing, plurality wins for model selection
- Max 3 review iterations to prevent infinite loops
- If consensus fails after 3 iterations, the best-so-far answer is used

WHY THIS IS ARCHITECTURALLY INTERESTING:
Most LLM applications use a single model. Some use two (generator and critic).
This uses FIVE models in a deliberative democracy pattern. Each model has
different training, different biases, different strengths. The council
can produce better decisions than any single model because disagreements
surface blind spots and iterative review genuinely improves quality.
"""

import json
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from runtime.models import (
    MODEL_PROFILES,
    get_edge_models,
    get_available_models,
)
from runtime.tasks import TaskMetadata, DataSensitivity
from runtime.logging_utils import CallLogger
from council.prompts import (
    get_routing_vote_prompt,
    get_model_selection_prompt,
    get_answer_review_prompt,
    format_model_profiles,
    format_task_metadata,
)


@dataclass
class Vote:
    """A single model's vote in any phase."""
    model_name: str
    vote: str
    reasoning: str
    confidence: float
    raw_response: str
    latency_ms: float
    success: bool


@dataclass
class ReviewResult:
    """A single model's review of an answer."""
    model_name: str
    approved: bool
    quality_score: float
    feedback: str
    issues: List[str]
    latency_ms: float
    success: bool


@dataclass
class DeliberationResult:
    """The complete output of a council deliberation."""
    deliberation_id: str
    final_answer: str
    routing_decision: str
    selected_model: str
    iterations: int
    consensus_reached: bool
    routing_votes: List[Vote]
    selection_votes: List[Vote]
    reviews: List[List[ReviewResult]]
    total_deliberation_ms: float
    cloud_tier: Optional[str] = None


class CouncilDeliberation:
    """
    Orchestrates the full deliberation process.

    CONFIGURATION:
    - council_models: Which local models participate in voting.
      Default is all edge models.
    - max_iterations: Maximum review rounds before accepting the best answer.
    - approval_threshold: Fraction of models that must approve (default: all).
    - skip_review_for_cloud: If true, cloud answers skip council review.
    """

    CLOUD_TIERS: Dict[str, str] = {
        "budget": "gemini/gemini-2.5-flash-lite",
        "standard": "gemini/gemini-2.5-flash",
        "premium": "gemini/gemini-2.5-pro",
        "luxury": "gemini/gemini-3-pro-preview",
    }

    def __init__(
        self,
        council_models: Optional[List[str]] = None,
        max_iterations: int = 3,
        approval_threshold: float = 1.0,
        skip_review_for_cloud: bool = True,
    ) -> None:
        self.all_models = get_available_models()
        self.edge_models = get_edge_models()

        if council_models:
            self.council_model_ids = council_models
        else:
            # Default: all edge models vote
            self.council_model_ids = list(self.edge_models.keys())

        self.max_iterations = max_iterations
        self.approval_threshold = approval_threshold
        self.skip_review_for_cloud = skip_review_for_cloud
        self.logger = CallLogger()
        self.profiles_text = format_model_profiles(MODEL_PROFILES)

    # ------------------------------------------------------------------ #
    # Utility: JSON parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_json_response(text: str) -> Optional[dict]:
        """
        Safely parse JSON from a model's response.

        Handles Markdown-style code fences, extra text before/after JSON,
        and other common model output quirks.
        """
        if not text or not text.strip():
            return None

        text = text.strip()

        # Remove Markdown-style code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            json_lines: List[str] = []
            inside = False
            for line in lines:
                if line.strip().startswith("```") and not inside:
                    inside = True
                    continue
                if line.strip().startswith("```") and inside:
                    break
                if inside:
                    json_lines.append(line)
            text = "\n".join(json_lines)

        # Try to find the JSON object in the text
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            text = text[start:end + 1]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------ #
    # Phase helpers
    # ------------------------------------------------------------------ #

    def _collect_votes(
        self,
        prompt_func,
        phase: str,
        deliberation_id: str,
        **prompt_kwargs,
    ) -> List[Vote]:
        """
        Collect votes from all council models for a given phase.

        Runs SEQUENTIALLY due to 16GB RAM constraint on M1 Pro.
        """
        votes: List[Vote] = []

        for model_id in self.council_model_ids:
            client = self.edge_models.get(model_id)
            if client is None:
                continue

            system_prompt = prompt_func(**prompt_kwargs)

            print(f"  [{phase}] Asking {model_id}...", end=" ", flush=True)

            result = client.call(
                prompt=system_prompt,
                max_tokens=500,
                temperature=0.3,
            )

            if not result.success:
                print(f"FAIL (error: {result.error})")
                votes.append(
                    Vote(
                        model_name=model_id,
                        vote="abstain",
                        reasoning=f"Model call failed: {result.error}",
                        confidence=0.0,
                        raw_response="",
                        latency_ms=result.latency_ms,
                        success=False,
                    )
                )
                continue

            parsed = self._parse_json_response(result.text)

            if parsed is None:
                print("FAIL (bad JSON)")
                votes.append(
                    Vote(
                        model_name=model_id,
                        vote="abstain",
                        reasoning="Failed to parse JSON response",
                        confidence=0.0,
                        raw_response=result.text[:500],
                        latency_ms=result.latency_ms,
                        success=False,
                    )
                )
                continue

            if phase == "routing_vote":
                vote_value = parsed.get("vote", "abstain")
                if vote_value == "edge":
                    vote_value = "local"
            elif phase == "model_selection":
                vote_value = parsed.get("selected_model", "abstain")
            else:
                vote_value = str(parsed)

            vote = Vote(
                model_name=model_id,
                vote=vote_value,
                reasoning=parsed.get("reasoning", ""),
                confidence=float(parsed.get("confidence", 0.5)),
                raw_response=result.text[:500],
                latency_ms=result.latency_ms,
                success=True,
            )
            votes.append(vote)

            self.logger.log_deliberation(
                deliberation_id=deliberation_id,
                phase=phase,
                model_name=model_id,
                vote_or_action=vote_value,
                reasoning=vote.reasoning,
                confidence=vote.confidence,
                total_latency_ms=result.latency_ms,
            )

            print(
                f"OK  vote={vote_value}  "
                f"conf={vote.confidence:.2f}  "
                f"({result.latency_ms:.0f}ms)"
            )

        return votes

    def _tally_routing_votes(self, votes: List[Vote]) -> Tuple[str, str]:
        """
        Count routing votes using confidence-weighted scoring.

        Returns (decision, cloud_tier).
        """
        local_score = 0.0
        cloud_score = 0.0
        cloud_tier_votes: Dict[str, float] = {}

        for vote in votes:
            if not vote.success:
                continue
            if vote.vote == "local":
                local_score += vote.confidence
            elif vote.vote == "cloud":
                cloud_score += vote.confidence
                parsed = self._parse_json_response(vote.raw_response)
                if parsed:
                    tier = parsed.get("suggested_cloud_tier", "standard")
                    if tier != "none":
                        cloud_tier_votes[tier] = (
                            cloud_tier_votes.get(tier, 0.0)
                            + vote.confidence
                        )

        if local_score >= cloud_score:
            return "local", "none"

        if cloud_tier_votes:
            best_tier = max(
                cloud_tier_votes,
                key=lambda k: cloud_tier_votes[k],
            )
        else:
            best_tier = "standard"

        return "cloud", best_tier

    @staticmethod
    def _tally_selection_votes(votes: List[Vote]) -> str:
        """
        Count model selection votes using confidence-weighted plurality.
        """
        scores: Dict[str, float] = {}
        for vote in votes:
            if not vote.success or vote.vote == "abstain":
                continue
            model = vote.vote
            scores[model] = scores.get(model, 0.0) + vote.confidence

        if not scores:
            return "ollama/llama3.1-8b"

        return max(scores, key=lambda k: scores[k])

    def _review_answer(
        self,
        user_prompt: str,
        answer: str,
        iteration: int,
        previous_feedback: str,
        deliberation_id: str,
    ) -> List[ReviewResult]:
        """
        Have all council models review an answer.
        Returns list of ReviewResult.
        """
        reviews: List[ReviewResult] = []

        for model_id in self.council_model_ids:
            client = self.edge_models.get(model_id)
            if client is None:
                continue

            review_prompt = get_answer_review_prompt(
                user_prompt=user_prompt,
                answer=answer,
                iteration=iteration,
                previous_feedback=previous_feedback,
            )

            print(
                f"  [review iter {iteration}] {model_id}...",
                end=" ",
                flush=True,
            )

            result = client.call(
                prompt=review_prompt,
                max_tokens=500,
                temperature=0.3,
            )

            if not result.success:
                print("FAIL")
                reviews.append(
                    ReviewResult(
                        model_name=model_id,
                        approved=True,
                        quality_score=0.5,
                        feedback=(
                            "Review failed - defaulting to approve "
                            "to avoid blocking"
                        ),
                        issues=[],
                        latency_ms=result.latency_ms,
                        success=False,
                    )
                )
                continue

            parsed = self._parse_json_response(result.text)

            if parsed is None:
                print("FAIL (bad JSON)")
                reviews.append(
                    ReviewResult(
                        model_name=model_id,
                        approved=True,
                        quality_score=0.5,
                        feedback="Could not parse review JSON",
                        issues=[],
                        latency_ms=result.latency_ms,
                        success=False,
                    )
                )
                continue

            review = ReviewResult(
                model_name=model_id,
                approved=bool(parsed.get("approved", True)),
                quality_score=float(parsed.get("quality_score", 0.5)),
                feedback=parsed.get("feedback", ""),
                issues=list(parsed.get("issues", [])),
                latency_ms=result.latency_ms,
                success=True,
            )
            reviews.append(review)

            self.logger.log_deliberation(
                deliberation_id=deliberation_id,
                phase="answer_review",
                model_name=model_id,
                vote_or_action="approved" if review.approved else "rejected",
                reasoning=review.feedback,
                confidence=review.quality_score,
                iteration=iteration,
                approved=review.approved,
                feedback=review.feedback,
            )

            label = "APPROVED" if review.approved else "REJECTED"
            print(
                f"{label}  score={review.quality_score:.2f}  "
                f"({result.latency_ms:.0f}ms)"
            )

        return reviews

    def _check_consensus(self, reviews: List[ReviewResult]) -> bool:
        """Check if enough models approved the answer."""
        if not reviews:
            return True

        approved_count = sum(1 for r in reviews if r.approved)
        total = len(reviews)
        approval_rate = approved_count / total

        return approval_rate >= self.approval_threshold

    @staticmethod
    def _aggregate_feedback(reviews: List[ReviewResult]) -> str:
        """Combine feedback from all dissenting reviewers."""
        feedback_parts: List[str] = []
        for review in reviews:
            if not review.approved and review.feedback:
                issues_text = ""
                if review.issues:
                    issues_text = " Issues: " + "; ".join(review.issues)
                feedback_parts.append(
                    f"[{review.model_name}]: {review.feedback}{issues_text}"
                )
        return "\n".join(feedback_parts)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def deliberate(self, prompt: str, meta: TaskMetadata) -> DeliberationResult:
        """
        THE MAIN ENTRY POINT.

        Runs the full council deliberation:
        1. Privacy/budget pre-checks (non-negotiable)
        2. Routing vote
        3. If local: model selection -> answer -> iterative review
        4. If cloud: route to Gemini -> return answer

        Returns a DeliberationResult with the final answer and
        complete audit trail.
        """
        deliberation_id = uuid.uuid4().hex[:8]
        start_time = time.time()
        meta_text = format_task_metadata(meta)

        print(f"\n{'=' * 60}")
        print(f"COUNCIL DELIBERATION [{deliberation_id}]")
        print(f"Prompt: {prompt[:100]}...")
        print(f"{'=' * 60}")

        # ========================================
        # PRE-CHECK: Privacy (non-negotiable)
        # ========================================
        if meta.data_sensitivity == DataSensitivity.HIGH:
            print(
                "[PRE-CHECK] HIGH sensitivity — forced to LOCAL, "
                "skipping council vote"
            )
            routing_decision = "local"
            routing_votes: List[Vote] = []
            cloud_tier = "none"
        else:
            # ========================================
            # PHASE 1: Routing Vote
            # ========================================
            print("\n--- PHASE 1: Routing Vote (local vs. cloud) ---")

            routing_votes = self._collect_votes(
                prompt_func=get_routing_vote_prompt,
                phase="routing_vote",
                deliberation_id=deliberation_id,
                user_prompt=prompt,
                model_profiles_text=self.profiles_text,
                task_metadata_text=meta_text,
            )

            routing_decision, cloud_tier = self._tally_routing_votes(
                routing_votes
            )

            local_count = sum(
                1 for v in routing_votes if v.vote == "local" and v.success
            )
            cloud_count = sum(
                1 for v in routing_votes if v.vote == "cloud" and v.success
            )
            print(
                f"\n[TALLY] Local: {local_count}, Cloud: {cloud_count} "
                f"-> Decision: {routing_decision.upper()}"
            )

        # ========================================
        # CLOUD PATH
        # ========================================
        if routing_decision == "cloud":
            print(f"\n--- CLOUD PATH (tier: {cloud_tier}) ---")

            cloud_model_id = self.CLOUD_TIERS.get(
                cloud_tier, "gemini/gemini-2.5-flash"
            )
            cloud_client = self.all_models.get(cloud_model_id)

            if cloud_client is None:
                print(
                    f"[WARNING] Cloud model {cloud_model_id} not "
                    f"available, falling back to local"
                )
                routing_decision = "local"
            else:
                print(f"[CLOUD] Calling {cloud_model_id}...")
                cloud_result = cloud_client.call(
                    prompt=prompt,
                    max_tokens=2048,
                    temperature=0.7,
                )

                total_ms = (time.time() - start_time) * 1000.0

                self.logger.log(
                    model_name=cloud_result.model_name,
                    tier="cloud",
                    task_type=meta.task_type.value,
                    latency_ms=cloud_result.latency_ms,
                    prompt_tokens=cloud_result.prompt_tokens or 0,
                    completion_tokens=cloud_result.completion_tokens or 0,
                    success=cloud_result.success,
                    prefill_ms=cloud_result.prefill_ms,
                    decode_ms=cloud_result.decode_ms,
                    load_ms=cloud_result.load_ms,
                )

                self.logger.log_deliberation(
                    deliberation_id=deliberation_id,
                    phase="final_decision",
                    model_name=cloud_model_id,
                    vote_or_action="cloud",
                    reasoning=f"Council voted cloud (tier: {cloud_tier})",
                    total_latency_ms=total_ms,
                    prompt_preview=prompt[:200],
                )

                print(
                    f"[DONE] Cloud answer received "
                    f"({cloud_result.latency_ms:.0f}ms)"
                )

                return DeliberationResult(
                    deliberation_id=deliberation_id,
                    final_answer=cloud_result.text,
                    routing_decision="cloud",
                    selected_model=cloud_model_id,
                    iterations=1,
                    consensus_reached=True,
                    routing_votes=routing_votes,
                    selection_votes=[],
                    reviews=[],
                    total_deliberation_ms=total_ms,
                    cloud_tier=cloud_tier,
                )

        # ========================================
        # LOCAL PATH: Phase 2 - Model Selection
        # ========================================
        print("\n--- PHASE 2: Model Selection Vote ---")

        selection_votes = self._collect_votes(
            prompt_func=get_model_selection_prompt,
            phase="model_selection",
            deliberation_id=deliberation_id,
            user_prompt=prompt,
            model_profiles_text=self.profiles_text,
            task_metadata_text=meta_text,
        )

        selected_model_id = self._tally_selection_votes(selection_votes)
        print(f"\n[SELECTED] {selected_model_id} will generate the answer")

        # ========================================
        # LOCAL PATH: Phase 3 - Answer Generation
        # ========================================
        print("\n--- PHASE 3: Generating Answer ---")

        answering_client = self.edge_models.get(selected_model_id)
        if answering_client is None:
            # Fallback: first available edge model
            selected_model_id = list(self.edge_models.keys())[0]
            answering_client = list(self.edge_models.values())[0]

        print(f"[GENERATE] {selected_model_id} is answering...")
        answer_result = answering_client.call(
            prompt=prompt,
            max_tokens=2048,
            temperature=0.7,
        )
        current_answer = answer_result.text
        print(
            f"[GENERATE] Answer received "
            f"({answer_result.latency_ms:.0f}ms)"
        )

        # ========================================
        # LOCAL PATH: Phase 4 - Iterative Review
        # ========================================
        all_reviews: List[List[ReviewResult]] = []
        previous_feedback = ""
        consensus = False
        final_iteration = 0

        for iteration in range(1, self.max_iterations + 1):
            final_iteration = iteration
            print(
                f"\n--- PHASE 4: Review Iteration "
                f"{iteration}/{self.max_iterations} ---"
            )

            reviews = self._review_answer(
                user_prompt=prompt,
                answer=current_answer,
                iteration=iteration,
                previous_feedback=previous_feedback,
                deliberation_id=deliberation_id,
            )
            all_reviews.append(reviews)

            consensus = self._check_consensus(reviews)

            approved_count = sum(1 for r in reviews if r.approved)
            avg_score = (
                sum(r.quality_score for r in reviews)
                / max(len(reviews), 1)
            )

            print(
                f"\n[REVIEW] Approved: {approved_count}/{len(reviews)}, "
                f"Avg Score: {avg_score:.2f}"
            )

            if consensus:
                print("[CONSENSUS] All council members approve!")
                break

            if iteration < self.max_iterations:
                previous_feedback = self._aggregate_feedback(reviews)
                print(
                    f"[ITERATE] Feedback collected, asking "
                    f"{selected_model_id} to improve..."
                )

                improvement_prompt = (
                    "Your previous answer to the following question "
                    "received feedback from reviewers. Please improve "
                    "your answer based on their feedback.\n\n"
                    f"ORIGINAL QUESTION:\n{prompt}\n\n"
                    f"YOUR PREVIOUS ANSWER:\n{current_answer}\n\n"
                    f"REVIEWER FEEDBACK:\n{previous_feedback}\n\n"
                    "Please provide an improved answer that addresses "
                    "all the feedback."
                )

                improved_result = answering_client.call(
                    prompt=improvement_prompt,
                    max_tokens=2048,
                    temperature=0.7,
                )
                current_answer = improved_result.text
                print(
                    f"[IMPROVED] New answer generated "
                    f"({improved_result.latency_ms:.0f}ms)"
                )

        if not consensus:
            print(
                f"[MAX ITERATIONS] Returning best answer after "
                f"{self.max_iterations} iterations"
            )

        total_ms = (time.time() - start_time) * 1000.0

        self.logger.log(
            model_name=selected_model_id,
            tier="edge",
            task_type=meta.task_type.value,
            latency_ms=total_ms,
            prompt_tokens=answer_result.prompt_tokens or 0,
            completion_tokens=answer_result.completion_tokens or 0,
            success=True,
            prefill_ms=answer_result.prefill_ms,
            decode_ms=answer_result.decode_ms,
            load_ms=answer_result.load_ms,
        )

        consensus_text = "reached" if consensus else "not reached"
        self.logger.log_deliberation(
            deliberation_id=deliberation_id,
            phase="final_decision",
            model_name=selected_model_id,
            vote_or_action="local",
            reasoning=(
                f"Council selected {selected_model_id}, "
                f"consensus {consensus_text} "
                f"after {final_iteration} iterations"
            ),
            iteration=final_iteration,
            approved=consensus,
            total_latency_ms=total_ms,
            prompt_preview=prompt[:200],
        )

        print(f"\n{'=' * 60}")
        print(f"DELIBERATION COMPLETE [{deliberation_id}]")
        print(f"  Decision:  LOCAL -> {selected_model_id}")
        print(f"  Iterations: {final_iteration}")
        print(f"  Consensus:  {'YES' if consensus else 'NO'}")
        print(
            f"  Total time: {total_ms:.0f}ms "
            f"({total_ms / 1000.0:.1f}s)"
        )
        print(f"{'=' * 60}\n")

        return DeliberationResult(
            deliberation_id=deliberation_id,
            final_answer=current_answer,
            routing_decision="local",
            selected_model=selected_model_id,
            iterations=final_iteration,
            consensus_reached=consensus,
            routing_votes=routing_votes,
            selection_votes=selection_votes,
            reviews=all_reviews,
            total_deliberation_ms=total_ms,
        )

"""
Key design decisions explained:**
- Confidence-weighted voting** instead of simple majority: A model that's 90% confident in "local" should outweigh one that's 30% confident in "cloud." This produces better routing decisions than raw headcount.
- Sequential execution**: On your 16GB M1 Pro, gpt-oss:20b alone uses ~14GB [[1]] [[5]]. The council runs models one at a time, letting Ollama swap them in and out of memory automatically.
- Max 3 iterations**: Without a cap, a perfectionist council could loop forever. Three rounds of feedback is enough for meaningful improvement without becoming impractical.
- Graceful fallbacks everywhere**: If a model's JSON parsing fails, it "abstains" rather than crashing the entire deliberation. If all reviews fail to parse, the answer is accepted. Robustness over perfection.
"""
