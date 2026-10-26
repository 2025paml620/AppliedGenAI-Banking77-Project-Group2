import json
import pytest
from src.structured.schema import validate_output
from src.structured.pipeline import handle_query
from src.tools.banking_tools import ToolError, run_tool

LABELS = ["cash_withdrawal_not_recognised", "lost_or_stolen_card", "card_payment_not_recognised"]


def out(**kw):
    base = {"intent": "lost_or_stolen_card", "response": "Please use the official app.",
            "action_required": False, "action": None}
    return json.dumps({**base, **kw})


def gen(text):
    return lambda prompt: text


def test_valid_output_with_fence():
    data, errs = validate_output("```json\n" + out() + "\n```", LABELS)
    assert errs == [] and data["intent"] == "lost_or_stolen_card"


@pytest.mark.parametrize("raw", ["not json", "{}", out(intent="made_up"), out(response=""),
                                 out(action_required="yes"), out(action_required=True),
                                 out(action={"tool": "x"}), json.dumps({"intent": "lost_or_stolen_card"})])
def test_invalid_outputs_rejected(raw):
    data, errs = validate_output(raw, LABELS)
    assert data is None and errs


def test_tool_flow_success():
    act = {"tool": "transaction_status", "arguments": {"transaction_id": "TXN1002"}}
    r = handle_query("Where is TXN1002?", LABELS, gen(out(action_required=True, action=act)))
    assert r["status"] == "ok" and "pending" in r["final_response"]


@pytest.mark.parametrize("name,args", [("transaction_status", {"transaction_id": "TXN9999"}),
                                       ("block_card", {"card_last4": "12"}),
                                       ("block_card", {"card_last4": "1234", "extra": 1}),
                                       ("wire_money", {})])
def test_tool_failures(name, args):
    with pytest.raises(ToolError):
        run_tool(name, args)


def test_pipeline_failure_paths():
    assert handle_query("hi", LABELS, gen("garbage"))["status"] == "invalid_output"
    assert handle_query("my pin is 4321", LABELS, gen(out()))["status"] == "blocked_input"
    unsafe = out(response="Please send your PIN to continue.")
    assert handle_query("hi", LABELS, gen(unsafe))["status"] == "unsafe_output"
    bad = {"tool": "block_card", "arguments": {"card_last4": "0000"}}
    assert handle_query("block it", LABELS, gen(out(action_required=True, action=bad)))["status"] == "tool_error"
