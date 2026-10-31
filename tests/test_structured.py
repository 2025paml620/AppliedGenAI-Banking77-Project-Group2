import json
import pytest
from src.structured.schema import validate_output
from src.structured.pipeline import handle_query
from src.tools.banking_tools import ToolError, run_tool

LABELS = ["cash_withdrawal_not_recognised", "lost_or_stolen_card", "card_payment_not_recognised"]


def out(**kw):
    """Serialize a valid default result with selected fields overridden."""
    base = {"intent": "lost_or_stolen_card", "response": "Please use the official app.",
            "action_required": False, "action": None}
    return json.dumps({**base, **kw})


def gen(text):
    """Make a deterministic generation callback returning the supplied text."""
    return lambda prompt: text


def test_valid_output_with_fence():
    """Accept a valid schema object enclosed in a JSON code fence."""
    data, errs = validate_output("```json\n" + out() + "\n```", LABELS)
    assert errs == [] and data["intent"] == "lost_or_stolen_card"


@pytest.mark.parametrize("raw", ["not json", "{}", out(intent="made_up"), out(response=""),
                                 out(action_required="yes"), out(action_required=True),
                                 out(action={"tool": "x"}), json.dumps({"intent": "lost_or_stolen_card"})])
def test_invalid_outputs_rejected(raw):
    """Reject malformed, incomplete, mistyped, or out-of-vocabulary payloads."""
    data, errs = validate_output(raw, LABELS)
    assert data is None and errs


def test_tool_flow_success():
    """Return a tool-grounded final response when a mock action succeeds."""
    act = {"tool": "transaction_status", "arguments": {"transaction_id": "TXN1002"}}
    r = handle_query("Where is TXN1002?", LABELS, gen(out(action_required=True, action=act)))
    assert r["status"] == "ok" and "pending" in r["final_response"]


def test_read_only_card_status_tool_flow(monkeypatch):
    """Return a known mock card status and describe it as demonstration data."""
    from src.tools.banking_tools import MOCK_CARDS

    monkeypatch.setitem(MOCK_CARDS, "5678", "active")
    action = {"tool": "card_status", "arguments": {"card_last4": "5678"}}
    r = handle_query("Is my card ending 5678 active?", LABELS,
                     gen(out(action_required=True, action=action)))

    assert r["status"] == "ok"
    assert r["tool_result"] == {"card_last4": "5678", "status": "active"}
    assert "(Demo)" in r["final_response"]
    assert "active" in r["final_response"]


@pytest.mark.parametrize(
    "query,expected_action,use_rag",
    [
        ("What is the status of transaction TXN1002?",
         {"tool": "transaction_status", "arguments": {"transaction_id": "TXN1002"}}, True),
        ("Please check the status of my demo card ending 5678.",
         {"tool": "card_status", "arguments": {"card_last4": "5678"}}, False),
    ],
)
def test_explicit_lookup_prompts_enforce_tool_routing_with_and_without_rag(
    query, expected_action, use_rag, monkeypatch,
):
    """Enforce explicit lookup routing whether retrieval context is enabled or not."""
    from src.tools.banking_tools import MOCK_CARDS

    monkeypatch.setitem(MOCK_CARDS, "5678", "active")
    model_json = out(action_required=False, action=None)
    seen_prompts = []

    class FakeRetriever:
        """Provide one deterministic training example without loading embeddings."""

        def retrieve(self, query, k=5):
            """Return a label-matched mock example for the RAG prompt path."""
            return [{"text": "similar lookup", "intent": "lost_or_stolen_card", "score": 1.0}]

    result = handle_query(
        query, LABELS,
        lambda prompt: seen_prompts.append(prompt) or model_json,
        retriever=FakeRetriever() if use_rag else None,
    )

    assert "Mandatory tool routing for this query" in seen_prompts[0]
    assert result["model_output"] == model_json
    assert result["tool"] == expected_action["tool"]
    assert result["tool_result"] is not None
    assert json.loads(result["raw_output"])["action"] == expected_action
    assert json.loads(result["raw_output"])["action_required"] is True


@pytest.mark.parametrize("name,args", [("transaction_status", {"transaction_id": "TXN9999"}),
                                       ("card_status", {"card_last4": "9999"}),
                                       ("card_status", {"card_last4": "56"}),
                                       ("block_card", {"card_last4": "12"}),
                                       ("block_card", {"card_last4": "1234", "extra": 1}),
                                       ("wire_money", {})])
def test_tool_failures(name, args):
    """Reject unknown tools, unknown records, and malformed tool arguments."""
    with pytest.raises(ToolError):
        run_tool(name, args)


def test_pipeline_failure_paths():
    """Keep malformed, unsafe, or unexecutable actions from succeeding silently."""
    assert handle_query("hi", LABELS, gen("garbage"))["status"] == "invalid_output"
    assert handle_query("my pin is 4321", LABELS, gen(out()))["status"] == "blocked_input"
    unsafe = out(response="Please send your PIN to continue.")
    assert handle_query("hi", LABELS, gen(unsafe))["status"] == "unsafe_output"
    bad = {"tool": "block_card", "arguments": {"card_last4": "0000"}}
    assert handle_query("block it", LABELS, gen(out(action_required=True, action=bad)))["status"] == "tool_error"


def test_retriever_and_rag_prompt():
    """Retrieve the nearest example and include it in the generated prompt."""
    import numpy as np
    from src.rag.retriever import Retriever
    vocab = ["stolen", "withdraw", "pending"]
    encode = lambda texts: np.array([[w in t for w in vocab] for t in texts], dtype=float) + 1e-3
    r = Retriever(["card stolen", "cash withdraw", "payment pending"],
                  ["lost_or_stolen_card", "cash_withdrawal_not_recognised", "pending_card_payment"], encode)
    hits = r.retrieve("my card was stolen", k=1)
    assert hits[0]["intent"] == "lost_or_stolen_card"
    seen = []
    handle_query("my card was stolen", LABELS, lambda p: seen.append(p) or out(), retriever=r, k=1)
    assert '"card stolen" -> lost_or_stolen_card' in seen[0]
