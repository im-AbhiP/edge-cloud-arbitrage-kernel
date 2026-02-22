"""
Smoke test for the Council of Local LLMs.

WHAT IS A SMOKE TEST?
The name comes from electronics: when you first power on a new circuit,
you check if smoke comes out. If it doesn't, the basic functionality
works. A smoke test verifies the most fundamental operations before
you invest time in detailed testing.

WHAT THIS FILE TESTS:
1. A simple question that the council should route LOCALLY
2. A complex question that the council MAY route to CLOUD
3. A sensitive-data question that MUST be forced LOCAL (privacy)

Each test exercises the full deliberation pipeline:
  Council vote -> Model selection -> Answer generation -> Review

PREREQUISITES:
  - Ollama must be running: ollama serve
  - All local models must be pulled (llama3.1, qwen3, deepseek-r1)
  - .env file must have GEMINI_API_KEY set (for cloud path test)

HOW TO RUN:
  In PyCharm: Right-click this file -> "Run 'test_smoke'"
  In terminal: python test_smoke.py

NOTE ON SPEED:
  Each test takes 2-10 minutes because multiple Ollama models are
  called sequentially, with RAM swapping between them on your 16GB
  M1 Pro. This is expected. The smoke test uses only 3 of the 5
  council models to keep total runtime under 30 minutes.
"""

from council.agents import LLMCouncil
from runtime.tasks import TaskType, DataSensitivity


def print_separator(title):
    """Print a visible section separator in the terminal output."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_result(result):
    """
    Print a formatted summary of a deliberation result.

    This helper extracts the key fields from a DeliberationResult
    and prints them in a readable format. Useful both for smoke
    testing and for demos.
    """
    print(f"\n  ROUTING:     {result.routing_decision.upper()}")
    print(f"  MODEL:       {result.selected_model}")
    print(f"  ITERATIONS:  {result.iterations}")
    print(f"  CONSENSUS:   {'YES' if result.consensus_reached else 'NO'}")
    print(f"  TIME:        {result.total_deliberation_ms:.0f}ms "
          f"({result.total_deliberation_ms / 1000:.1f}s)")

    # Print cloud tier if cloud was selected
    if result.routing_decision == "cloud" and result.cloud_tier:
        print(f"  CLOUD TIER:  {result.cloud_tier}")

    # Print vote summary if routing votes exist
    if result.routing_votes:
        local_votes = sum(
            1 for v in result.routing_votes
            if v.vote == "local" and v.success
        )
        cloud_votes = sum(
            1 for v in result.routing_votes
            if v.vote == "cloud" and v.success
        )
        abstain_votes = sum(
            1 for v in result.routing_votes
            if v.vote == "abstain" or not v.success
        )
        print(f"  VOTES:       local={local_votes} "
              f"cloud={cloud_votes} abstain={abstain_votes}")

    # Print truncated answer (first 300 characters)
    answer_preview = result.final_answer[:300]
    if len(result.final_answer) > 300:
        answer_preview += "..."
    print(f"\n  ANSWER:\n  {answer_preview}")


def main():
    """
    Run all three smoke tests.

    WHY THESE SPECIFIC SETTINGS:
    - 3 council models (not all 5) to keep runtime manageable.
      gpt-oss:20b is excluded because it uses ~14GB RAM and would
      cause heavy swapping with the other models.
      deepseek-coder:6.7b is excluded because it's specialized for
      code and not ideal for general voting.
    - max_iterations=2 because smoke tests should finish in
      reasonable time. Full 3-iteration testing is for contract tests.
    - approval_threshold=0.67 means 2 of 3 models must approve.
      This is a reasonable bar for smoke testing.
    """

    # Create the council with 3 models for speed
    council = LLMCouncil(
        council_models=[
            "ollama/llama3.1-8b",
            "ollama/qwen3-8b",
            "ollama/deepseek-r1-8b",
        ],
        max_iterations=2,
        approval_threshold=0.67,
    )

    # ==================================================================
    # TEST 1: Simple Question
    # ==================================================================
    # EXPECTATION: The council should vote LOCAL because this is a
    # trivial factual question that any local model can handle.
    # Sending this to cloud would waste money for zero quality gain.
    # ==================================================================
    print_separator("TEST 1: Simple Question (council should vote LOCAL)")

    result_1 = council.ask(
        prompt="What is the capital of France?",
        task_type=TaskType.QUICK_QA,
        complexity=0.1,
        importance=0.2,
        budget_sensitivity=0.8,
    )

    print_result(result_1)

    # Soft assertion: we EXPECT local, but don't crash if cloud
    # (the council is non-deterministic — models might surprise us)
    if result_1.routing_decision == "local":
        print("\n  [PASS] Council correctly voted LOCAL for simple question")
    else:
        print("\n  [NOTE] Council voted CLOUD for simple question. "
              "This is unexpected but not necessarily wrong. "
              "Check the vote reasoning above.")

    # ==================================================================
    # TEST 2: Complex Question
    # ==================================================================
    # EXPECTATION: The council MAY vote CLOUD because this requires
    # deep domain knowledge, multi-factor analysis, and nuanced
    # reasoning that cloud models handle better than 8B local models.
    # Either outcome is acceptable — the important thing is that
    # the council DELIBERATES and provides reasoning.
    # ==================================================================
    print_separator("TEST 2: Complex Question (council may vote CLOUD)")

    result_2 = council.ask(
        prompt=(
            "Analyze the technical and economic implications of AMD's "
            "MI300X versus NVIDIA H100 for enterprise AI inference "
            "workloads. Consider total cost of ownership, performance "
            "per watt, software ecosystem maturity, and customer "
            "switching costs. Provide specific recommendations for "
            "a mid-size company evaluating both options."
        ),
        task_type=TaskType.DEEP_RESEARCH,
        complexity=0.9,
        importance=0.9,
        budget_sensitivity=0.3,
    )

    print_result(result_2)
    print(f"\n  [INFO] Council voted {result_2.routing_decision.upper()} "
          f"for complex question")

    # ==================================================================
    # TEST 3: Sensitive Data (Privacy Enforcement)
    # ==================================================================
    # EXPECTATION: The system MUST route to LOCAL regardless of what
    # the models might prefer. When data_sensitivity is HIGH, the
    # routing vote is SKIPPED ENTIRELY — this is a non-negotiable
    # governance policy. No model gets to vote on whether sensitive
    # data should leave the device.
    #
    # This is the most important test from a governance perspective.
    # ==================================================================
    print_separator("TEST 3: Sensitive Data (MUST be forced LOCAL)")

    result_3 = council.ask(
        prompt=(
            "Analyze this employee's performance review data and "
            "recommend a compensation adjustment. The employee has "
            "been with the company for 3 years, current salary is "
            "$145,000, and received ratings of Exceeds Expectations "
            "for the last two review cycles."
        ),
        task_type=TaskType.DATA_ANALYSIS,
        complexity=0.7,
        importance=0.8,
        data_sensitivity=DataSensitivity.HIGH,
    )

    print_result(result_3)

    # Hard assertion: this MUST be local. Privacy is non-negotiable.
    if result_3.routing_decision == "local":
        print("\n  [PASS] Privacy enforcement working correctly. "
              "HIGH sensitivity data stayed local.")
    else:
        print("\n  [FAIL] CRITICAL: HIGH sensitivity data was routed "
              "to cloud! Privacy enforcement is broken!")

    # Check that routing votes were skipped (privacy pre-check)
    if len(result_3.routing_votes) == 0:
        print("  [PASS] Routing vote was correctly skipped "
              "(privacy pre-check)")
    else:
        print(f"  [FAIL] Expected 0 routing votes for HIGH sensitivity, "
              f"got {len(result_3.routing_votes)}")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print_separator("SMOKE TEST SUMMARY")

    print(f"\n  Test 1 (Simple):    {result_1.routing_decision.upper()} "
          f"via {result_1.selected_model} "
          f"({result_1.total_deliberation_ms / 1000:.1f}s)")
    print(f"  Test 2 (Complex):   {result_2.routing_decision.upper()} "
          f"via {result_2.selected_model} "
          f"({result_2.total_deliberation_ms / 1000:.1f}s)")
    print(f"  Test 3 (Sensitive): {result_3.routing_decision.upper()} "
          f"via {result_3.selected_model} "
          f"({result_3.total_deliberation_ms / 1000:.1f}s)")

    total_time = (
        result_1.total_deliberation_ms
        + result_2.total_deliberation_ms
        + result_3.total_deliberation_ms
    )
    print(f"\n  Total smoke test time: {total_time / 1000:.1f}s "
          f"({total_time / 60000:.1f} minutes)")

    print("\n  All smoke tests complete. Check data/logs.db for full "
          "deliberation audit trail.")
    print("  View in PyCharm: View -> Tool Windows -> Database -> "
          "drag in data/logs.db")
    print("  Launch dashboard: streamlit run dashboard/streamlit_app.py")


if __name__ == "__main__":
    main()
