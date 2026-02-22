"""
Contract tests for the Council of Local LLMs.

WHAT ARE CONTRACT TESTS?
Contract tests verify that a system's OUTPUT always conforms to
an agreed-upon structure (the "contract"). In our case, the contract
is: "A council deliberation ALWAYS produces a DeliberationResult
with specific fields, in specific formats, within specific ranges."

WHY DO WE NEED THESE?
Without contract tests, a model update or prompt change could silently
break the council's output — maybe confidence becomes a string instead
of a float, or routing_decision becomes "LOCAL" instead of "local".
These tests catch that immediately.

In interviews, contract tests demonstrate PRODUCTION DISCIPLINE.
You're not just building a prototype — you're building something
that can be relied upon.

HOW TO RUN:
    From PyCharm:
        Right-click this file -> "Run 'pytest in contract_tests'"
        Or click the green arrow next to any individual test function.

    From terminal:
        pytest council/contract_tests.py -v

    The -v flag means "verbose" — it shows each test name and result.

PREREQUISITES:
    - Ollama must be running: ollama serve
    - At least llama3.1:8b and qwen3:8b must be pulled
    - These tests call ACTUAL models, so they take 2-5 minutes

DESIGN CHOICES:
    - We use only 2 models in tests (not all 5) for speed
    - approval_threshold is set to 0.5 so tests don't fail just
      because one model gave a bad review
    - max_iterations is set to 2 to keep test runtime reasonable
    - scope="module" on the fixture means the council runs ONCE
      and all tests share that result (saves ~3 minutes)
"""

import pytest

from council.agents import LLMCouncil
from council.deliberation import CouncilDeliberation, DeliberationResult
from runtime.tasks import TaskMetadata, TaskType, DataSensitivity


# =====================================================================
# FIXTURES
# =====================================================================
# A pytest "fixture" is a function that provides test data.
# When a test function has a parameter with the same name as a fixture,
# pytest automatically calls the fixture and passes the result.
#
# scope="module" means: run this fixture ONCE for the entire file,
# not once per test. Since each deliberation takes 1-3 minutes
# (calling multiple Ollama models), we reuse the result across tests.
# =====================================================================

@pytest.fixture(scope="module")
def council_result():
    """
    Run a small, fast council deliberation and reuse the result
    for all tests in this file.

    WHY THESE SPECIFIC SETTINGS:
    - Only 2 council models (llama3.1 + qwen3) for speed.
      Full 5-model council would take 10+ minutes per test run.
    - max_iterations=2 because we don't need 3 rounds for testing.
    - approval_threshold=0.5 so one model approving is enough.
      We're testing structure, not quality.
    - A simple factual question so models respond quickly
      and are likely to agree.
    """
    council = LLMCouncil(
        council_models=[
            "ollama/llama3.1-8b",
            "ollama/qwen3-8b",
        ],
        max_iterations=2,
        approval_threshold=0.5,
    )

    result = council.ask(
        "What is the difference between TCP and UDP?",
        task_type=TaskType.QUICK_QA,
        complexity=0.3,
    )

    return result


@pytest.fixture(scope="module")
def privacy_result():
    """
    Run a deliberation with HIGH data sensitivity.
    Used to verify that privacy enforcement works correctly.

    WHY A SEPARATE FIXTURE:
    The privacy test needs different input parameters (HIGH sensitivity)
    than the standard tests. We run it separately but still only once
    (scope="module").

    Only 1 council model here since we're testing routing behavior,
    not answer quality. The routing vote is SKIPPED entirely for HIGH
    sensitivity — the system goes straight to local model selection.
    """
    council = LLMCouncil(
        council_models=[
            "ollama/qwen3-8b",
        ],
        max_iterations=1,
        approval_threshold=0.5,
    )

    result = council.ask(
        "Analyze this employee's salary data and bonus structure.",
        task_type=TaskType.DATA_ANALYSIS,
        complexity=0.9,
        data_sensitivity=DataSensitivity.HIGH,
    )

    return result


# =====================================================================
# BASIC STRUCTURE TESTS
# =====================================================================
# These tests verify that the DeliberationResult has the right
# fields with the right types. They're the most fundamental tests —
# if these fail, everything else is broken.
# =====================================================================

class TestDeliberationStructure:
    """Tests that verify the basic structure of deliberation output."""

    def test_result_is_correct_type(self, council_result):
        """
        The deliberation must return a DeliberationResult object,
        not a dict, string, or None.
        """
        assert isinstance(council_result, DeliberationResult), (
            f"Expected DeliberationResult, got {type(council_result)}"
        )

    def test_deliberation_id_exists(self, council_result):
        """
        Every deliberation must have a unique ID.

        WHY THIS MATTERS:
        The ID links all log entries (votes, reviews, decisions)
        for a single deliberation together. Without it, you can't
        trace the decision-making process in the database.
        """
        assert council_result.deliberation_id is not None
        assert isinstance(council_result.deliberation_id, str)
        assert len(council_result.deliberation_id) > 0

    def test_final_answer_is_nonempty_string(self, council_result):
        """
        The deliberation must produce a non-empty answer.
        An empty answer means something went very wrong —
        either no model was called or all calls failed.
        """
        assert council_result.final_answer is not None
        assert isinstance(council_result.final_answer, str)
        assert len(council_result.final_answer) > 10, (
            "Answer should be substantive (more than 10 characters), "
            f"got: '{council_result.final_answer[:50]}'"
        )

    def test_routing_decision_is_valid(self, council_result):
        """
        Routing decision must be exactly 'local' or 'cloud'.
        No other values are acceptable — 'edge', 'LOCAL', 'Cloud',
        or anything else would indicate a parsing bug.
        """
        valid_decisions = ["local", "cloud"]
        assert council_result.routing_decision in valid_decisions, (
            f"Expected 'local' or 'cloud', "
            f"got '{council_result.routing_decision}'"
        )

    def test_selected_model_has_valid_prefix(self, council_result):
        """
        The selected model must start with 'ollama/' (local) or
        'gemini/' (cloud). This ensures the model ID is a real
        model from our registry, not garbage from a parsing error.
        """
        valid_prefixes = ("ollama/", "gemini/")
        assert council_result.selected_model.startswith(valid_prefixes), (
            f"Selected model '{council_result.selected_model}' "
            f"doesn't start with 'ollama/' or 'gemini/'"
        )

    def test_selected_model_is_nonempty(self, council_result):
        """The selected model string must not be empty."""
        assert len(council_result.selected_model) > 7, (
            "Selected model ID is too short to be valid"
        )


# =====================================================================
# ITERATION AND CONSENSUS TESTS
# =====================================================================
# These tests verify that the iterative review process works correctly.
# =====================================================================

class TestIterationAndConsensus:
    """Tests for the review iteration and consensus mechanism."""

    def test_iterations_in_valid_range(self, council_result):
        """
        Iterations must be between 1 and max_iterations (inclusive).
        - 0 iterations would mean no review happened at all
        - More than max_iterations means the loop guard failed
        """
        assert isinstance(council_result.iterations, int)
        assert 1 <= council_result.iterations <= 3, (
            f"Expected 1-3 iterations, got {council_result.iterations}"
        )

    def test_consensus_is_boolean(self, council_result):
        """
        consensus_reached must be a Python boolean (True/False),
        not a string, int, or None.
        """
        assert isinstance(council_result.consensus_reached, bool), (
            f"Expected bool, got {type(council_result.consensus_reached)}"
        )

    def test_reviews_is_list_of_lists(self, council_result):
        """
        reviews is a list of review rounds. Each round is itself
        a list of ReviewResult objects (one per council model).

        Structure: [[review1_model1, review1_model2], [review2_model1, ...]]
        """
        assert isinstance(council_result.reviews, list)
        # If there were any review rounds, each should be a list
        for i, round_reviews in enumerate(council_result.reviews):
            assert isinstance(round_reviews, list), (
                f"Review round {i} should be a list, "
                f"got {type(round_reviews)}"
            )


# =====================================================================
# TIMING TESTS
# =====================================================================

class TestTiming:
    """Tests that verify timing data is captured correctly."""

    def test_total_deliberation_time_is_positive(self, council_result):
        """
        Total deliberation time must be a positive number.
        A zero or negative time would indicate a measurement bug.
        """
        assert isinstance(
            council_result.total_deliberation_ms, (int, float)
        )
        assert council_result.total_deliberation_ms > 0, (
            "Deliberation time should be positive"
        )

    def test_total_time_is_reasonable(self, council_result):
        """
        Deliberation should take between 5 seconds and 30 minutes.
        Less than 5s means models probably weren't called.
        More than 30min means something hung.
        """
        time_seconds = council_result.total_deliberation_ms / 1000
        assert time_seconds > 5, (
            f"Deliberation took only {time_seconds:.1f}s — "
            f"models may not have been called"
        )
        assert time_seconds < 1800, (
            f"Deliberation took {time_seconds:.1f}s — "
            f"possible hang or infinite loop"
        )


# =====================================================================
# VOTE STRUCTURE TESTS
# =====================================================================

class TestVoteStructure:
    """Tests that verify routing and selection votes are well-formed."""

    def test_routing_votes_is_list(self, council_result):
        """Routing votes must be a list (possibly empty for HIGH privacy)."""
        assert isinstance(council_result.routing_votes, list)

    def test_selection_votes_is_list(self, council_result):
        """Selection votes must be a list."""
        assert isinstance(council_result.selection_votes, list)

    def test_routing_votes_have_valid_values(self, council_result):
        """
        Each routing vote must be 'local', 'cloud', or 'abstain'.
        'abstain' happens when a model call fails or JSON parsing fails.
        """
        valid_votes = ["local", "cloud", "abstain"]
        for vote in council_result.routing_votes:
            assert vote.vote in valid_votes, (
                f"Model {vote.model_name} cast invalid routing vote: "
                f"'{vote.vote}'"
            )

    def test_routing_votes_have_confidence_in_range(self, council_result):
        """
        Each vote's confidence must be between 0.0 and 1.0.
        Confidence outside this range indicates a parsing error.
        """
        for vote in council_result.routing_votes:
            assert isinstance(vote.confidence, (int, float)), (
                f"Model {vote.model_name} confidence is not numeric: "
                f"{vote.confidence}"
            )
            assert 0.0 <= vote.confidence <= 1.0, (
                f"Model {vote.model_name} confidence out of range: "
                f"{vote.confidence}"
            )

    def test_each_vote_has_reasoning(self, council_result):
        """
        Successful votes should have non-empty reasoning.
        This is what makes the council EXPLAINABLE — every vote
        comes with a justification.
        """
        for vote in council_result.routing_votes:
            if vote.success:
                assert isinstance(vote.reasoning, str)
                assert len(vote.reasoning) > 0, (
                    f"Model {vote.model_name} voted without reasoning"
                )


# =====================================================================
# PRIVACY ENFORCEMENT TESTS
# =====================================================================
# These are the most important governance tests. Privacy enforcement
# is NON-NEGOTIABLE — HIGH sensitivity data must NEVER go to cloud.
# =====================================================================

class TestPrivacyEnforcement:
    """Tests that verify privacy policies are enforced correctly."""

    def test_high_sensitivity_forces_local(self, privacy_result):
        """
        HIGH data sensitivity MUST route to local.
        This is non-negotiable governance policy.
        The routing vote is skipped entirely — no model gets
        to vote on whether to send sensitive data to the cloud.
        """
        assert privacy_result.routing_decision == "local", (
            f"HIGH sensitivity task was routed to "
            f"'{privacy_result.routing_decision}' instead of 'local'"
        )

    def test_high_sensitivity_skips_routing_vote(self, privacy_result):
        """
        When data sensitivity is HIGH, the routing vote phase
        is skipped entirely. This means routing_votes should be
        an empty list — no models were asked, no votes were cast.
        """
        assert isinstance(privacy_result.routing_votes, list)
        assert len(privacy_result.routing_votes) == 0, (
            f"Expected 0 routing votes for HIGH sensitivity, "
            f"got {len(privacy_result.routing_votes)}"
        )

    def test_high_sensitivity_selects_local_model(self, privacy_result):
        """
        When forced to local by privacy, the selected model
        must be an Ollama model (starts with 'ollama/').
        """
        assert privacy_result.selected_model.startswith("ollama/"), (
            f"HIGH sensitivity selected non-local model: "
            f"'{privacy_result.selected_model}'"
        )

    def test_high_sensitivity_produces_answer(self, privacy_result):
        """
        Even when forced to local, the system must still produce
        a substantive answer. Privacy enforcement should not break
        the answer generation pipeline.
        """
        assert privacy_result.final_answer is not None
        assert len(privacy_result.final_answer) > 10, (
            "Privacy-enforced deliberation produced empty or "
            "trivial answer"
        )


# =====================================================================
# CLOUD PATH TESTS (only run if routing went to cloud)
# =====================================================================

class TestCloudPath:
    """
    Tests specific to the cloud routing path.
    These tests are conditional — they only make assertions
    if the council actually voted for cloud.
    """

    def test_cloud_tier_is_valid_when_cloud(self, council_result):
        """
        If the council voted for cloud, the cloud_tier must be
        one of our defined tiers.
        """
        if council_result.routing_decision == "cloud":
            valid_tiers = [
                "budget", "standard", "premium", "luxury"
            ]
            assert council_result.cloud_tier in valid_tiers, (
                f"Invalid cloud tier: '{council_result.cloud_tier}'"
            )

    def test_cloud_path_has_one_iteration(self, council_result):
        """
        Cloud answers skip the review process, so iterations
        should be exactly 1.
        """
        if council_result.routing_decision == "cloud":
            assert council_result.iterations == 1

    def test_cloud_path_has_consensus(self, council_result):
        """Cloud answers are automatically considered consensus."""
        if council_result.routing_decision == "cloud":
            assert council_result.consensus_reached is True


# =====================================================================
# LOCAL PATH TESTS (only run if routing stayed local)
# =====================================================================

class TestLocalPath:
    """
    Tests specific to the local routing path.
    Conditional on routing_decision being 'local'.
    """

    def test_local_selects_ollama_model(self, council_result):
        """If routed locally, the selected model must be an Ollama model."""
        if council_result.routing_decision == "local":
            assert council_result.selected_model.startswith("ollama/"), (
                f"Local routing selected non-Ollama model: "
                f"'{council_result.selected_model}'"
            )

    def test_local_has_selection_votes(self, council_result):
        """
        If routed locally, there should be model selection votes.
        The council votes on WHICH local model answers.
        """
        if council_result.routing_decision == "local":
            assert len(council_result.selection_votes) > 0, (
                "Local routing should have model selection votes"
            )

    def test_local_has_review_rounds(self, council_result):
        """
        If routed locally, there should be at least one review round.
        The council reviews the generated answer before approving.
        """
        if council_result.routing_decision == "local":
            assert len(council_result.reviews) > 0, (
                "Local routing should have at least one review round"
            )

    def test_local_cloud_tier_is_none(self, council_result):
        """If routed locally, cloud_tier should be None."""
        if council_result.routing_decision == "local":
            assert council_result.cloud_tier is None, (
                f"Local routing should have cloud_tier=None, "
                f"got '{council_result.cloud_tier}'"
            )
