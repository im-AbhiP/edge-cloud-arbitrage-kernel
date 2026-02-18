"""
Contract tests for council JSON output.

Run with: right-click this file in PyCharm → "Run pytest"
Or in terminal: pytest council/contract_tests.py -v
"""

import pytest
from council.agents import ResearchCouncil
from runtime.tasks import DataSensitivity


@pytest.fixture(scope="module")
def council_result():
    """Run council once, reuse for all tests."""
    council = ResearchCouncil()
    return council.run(
        "What are the tradeoffs of edge vs cloud AI inference?",
        data_sensitivity=DataSensitivity.LOW
    )


def test_has_required_keys(council_result):
    output = council_result["council_output"]
    for key in ["summary", "assumptions", "risks",
                "disagreements", "confidence"]:
        assert key in output, f"Missing key: {key}"


def test_confidence_in_range(council_result):
    c = council_result["council_output"]["confidence"]
    assert isinstance(c, (int, float))
    assert 0.0 <= c <= 1.0


def test_assumptions_is_list(council_result):
    assert isinstance(
        council_result["council_output"]["assumptions"], list
    )


def test_risks_is_list(council_result):
    assert isinstance(
        council_result["council_output"]["risks"], list
    )


def test_summary_is_nonempty(council_result):
    s = council_result["council_output"]["summary"]
    assert isinstance(s, str)
    assert len(s) > 10
