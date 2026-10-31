"""End-to-end checks for generation, validation, guardrails, and mock tools."""

import json

from src.structured.pipeline import handle_query


LABELS = ["cash_withdrawal", "lost_or_stolen_card", "pending_card_payment"]


def structured_output(**overrides):
    """Build a JSON model response with the required structured fields."""
    payload = {
        "intent": "pending_card_payment",
        "response": "I can check the demo transaction status.",
        "action_required": False,
        "action": None,
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_end_to_end_tool_call_is_validated_and_grounded_in_tool_result():
    observed_prompts = []
    action = {
        "tool": "transaction_status",
        "arguments": {"transaction_id": "TXN1002"},
    }

    def generate(prompt):
        """Capture the constructed prompt and return a deterministic tool request."""
        observed_prompts.append(prompt)
        return structured_output(action_required=True, action=action)

    result = handle_query("What is happening with TXN1002?", LABELS, generate)

    assert observed_prompts and "Never ask" in observed_prompts[0]
    assert result["status"] == "ok"
    assert result["tool_result"]["status"] == "pending"
    assert "pending" in result["final_response"]
    assert "Cafe Demo" in result["final_response"]


def test_end_to_end_sensitive_input_stops_before_generation():
    """Prove sensitive input returns before the injected generator is called."""
    def should_not_generate(_prompt):
        """Fail the test if a blocked request reaches generation."""
        raise AssertionError("Sensitive input should be blocked before model inference")

    result = handle_query("My PIN is 4837", LABELS, should_not_generate)

    assert result["status"] == "blocked_input"
    assert result["raw_output"] is None
    assert result["final_response"]


def test_end_to_end_invalid_json_cannot_reach_a_tool():
    """Return a safe invalid-output status without attempting any tool call."""
    result = handle_query("Check the payment", LABELS, lambda _prompt: "not JSON")

    assert result["status"] == "invalid_output"
    assert result["tool"] is None
    assert result["tool_result"] is None


def test_end_to_end_unsafe_response_is_replaced_before_tool_execution():
    """Stop an unsafe response before its otherwise valid action is executed."""
    action = {"tool": "transaction_status", "arguments": {"transaction_id": "TXN1002"}}
    result = handle_query(
        "Check the payment",
        LABELS,
        lambda _prompt: structured_output(
            response="Please send me your PIN.", action_required=True, action=action
        ),
    )

    assert result["status"] == "unsafe_output"
    assert result["tool"] is None
    assert result["tool_result"] is None
