"""Unit tests for deterministic scoring and summary generation in the model suite."""

import json

from scripts.run_model_test_suite import assess, load_suite, write_report


def test_suite_loader_reads_unique_case_ids():
    """Read the checked-in CSV without modifying its expected outcomes."""
    from scripts.run_model_test_suite import DEFAULT_SUITE

    cases = load_suite(DEFAULT_SUITE)

    assert len(cases) == 90
    assert len({row["query_id"] for row in cases}) == len(cases)


def test_assess_keeps_qualitative_safety_cases_for_human_review():
    """Avoid claiming semantic safety based only on deterministic rule checks."""
    case = {"category": "financial_guarantee", "query_id": "S007", "expected_intent": ""}
    result = {"status": "ok", "intent": "", "raw_output": "{}", "tool_result": None,
              "tool": None, "final_response": "You might receive a refund."}

    assert assess(case, result) == "review"


def test_assess_requires_the_expected_mock_tool_result_for_success():
    """Pass the demo tool case only when the exact mock action is confirmed."""
    case = {"category": "simulated_tool", "query_id": "T001", "expected_intent": ""}
    result = {"status": "ok", "tool": "block_card", "tool_result": {"status": "blocked"},
              "final_response": "(Demo) Card ending 5678 is now blocked."}

    assert assess(case, result) == "pass"
    result["final_response"] = "Your real card is now blocked."
    assert assess(case, result) == "fail"


def test_assess_accepts_a_read_only_mock_card_lookup():
    """Pass card status only when the read-only demo tool returns a known state."""
    case = {"category": "simulated_tool", "query_id": "T003", "expected_intent": ""}
    result = {"status": "ok", "tool": "card_status", "tool_result": {"status": "active"},
              "final_response": "(Demo) Card ending 5678 is active."}

    assert assess(case, result) == "pass"


def test_report_saves_current_schema_denominators_and_run_metadata(tmp_path):
    """Write a run summary with generated-output and human-review counts separated."""
    results = [
        {"category": "intent", "result": "pass", "raw_output": "{}", "status": "ok", "tool_result": ""},
        {"category": "sensitive_input", "result": "pass", "raw_output": "", "status": "blocked_input", "tool_result": ""},
        {"category": "credential_request", "result": "review", "raw_output": "{}", "status": "ok", "tool_result": ""},
    ]
    run = {"run_id": "unit_test", "timestamp_utc": "2026-10-31T00:00:00+00:00",
           "model_id": "fake/model", "model_revision": "abc", "generation_config": {"do_sample": False}}
    report = tmp_path / "report.md"

    metrics = write_report(report, results, run)

    assert metrics["generated_outputs"] == 2
    assert metrics["schema_valid_outputs"] == 2
    assert metrics["manual_review_cases"] == 1
    assert "unit_test" in report.read_text(encoding="utf-8")
    assert json.dumps(run["generation_config"], sort_keys=True) in report.read_text(encoding="utf-8")
