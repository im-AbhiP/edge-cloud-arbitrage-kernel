"""
Smoke test — verify the entire runtime stack works end-to-end.

WHAT IS A SMOKE TEST?
The name comes from electronics: when you first power on a new circuit,
you check if smoke comes out. If it doesn't, basic functionality works.
A smoke test verifies the most fundamental operations before you test
anything complex.
"""

from runtime.tasks import TaskMetadata, TaskType, DataSensitivity
from runtime.router import ModelRouter


def main():
    router = ModelRouter()

    # -------- Test 1: Simple question (should route to EDGE) --------
    print("\n" + "=" * 60)
    print("TEST 1: Simple Question (expect EDGE)")
    print("=" * 60)

    meta = TaskMetadata(
        task_type=TaskType.QUICK_QA,
        complexity=0.2,
        importance=0.3,
        budget_sensitivity=0.8,
    )
    result = router.run("What is the capital of France?", meta)

    print(f"  Model used:  {result.model_name}")
    print(f"  Tier:        {result.tier}")
    print(f"  Latency:     {result.latency_ms:.0f}ms")
    print(f"  Response:    {result.text[:200]}")

    # -------- Test 2: Deep research (should route to CLOUD) --------
    print("\n" + "=" * 60)
    print("TEST 2: Deep Research (expect CLOUD)")
    print("=" * 60)

    meta = TaskMetadata(
        task_type=TaskType.DEEP_RESEARCH,
        complexity=0.9,
        importance=0.9,
    )
    result = router.run(
        "Explain the tradeoffs between edge and cloud AI inference "
        "for enterprise applications.",
        meta
    )

    print(f"  Model used:  {result.model_name}")
    print(f"  Tier:        {result.tier}")
    print(f"  Latency:     {result.latency_ms:.0f}ms")
    print(f"  Response:    {result.text[:200]}")

    # -------- Test 3: Sensitive data (should FORCE edge) --------
    print("\n" + "=" * 60)
    print("TEST 3: Sensitive Data (expect EDGE, privacy enforced)")
    print("=" * 60)

    meta = TaskMetadata(
        task_type=TaskType.DEEP_RESEARCH,
        complexity=0.9,
        importance=0.9,
        data_sensitivity=DataSensitivity.HIGH,
    )
    result = router.run(
        "Analyze this employee's salary data...", meta
    )

    print(f"  Model used:  {result.model_name}")
    print(f"  Tier:        {result.tier}")
    print(f"  Latency:     {result.latency_ms:.0f}ms")

    # -------- Summary --------
    print("\n" + "=" * 60)
    print("SUMMARY STATS")
    print("=" * 60)

    stats = router.logger.get_summary_stats()
    print(f"  Total calls:     {stats['overall']['total_calls']}")
    print(f"  Edge calls:      {stats['overall']['edge_calls']}")
    print(f"  Cloud calls:     {stats['overall']['cloud_calls']}")
    print(f"  Total cost:      ${stats['overall']['total_cost']:.6f}")
    print(f"  Cost avoided:    ${stats['cost_avoided']:.6f}")
    print(f"  Edge %_:          {stats['edge_percentage']:.1f}%")


if __name__ == "__main__":
    main()
