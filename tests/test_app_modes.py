"""UI dispatch tests for Part 2 and the structured Part 3 pipeline."""

import json

import app
import pytest


def test_ui_dispatch_runs_part2_and_keeps_tool_panel_empty(monkeypatch):
    """Keep the existing Part 2 execution path available from the mode selector."""
    result = {"query": "hello", "status": "ok", "intent": "card_arrival", "tool": None,
              "tool_result": None, "errors": [], "raw_output": "A safe answer.",
              "final_response": "A safe answer."}
    monkeypatch.setattr(app, "run_part2", lambda *args: (
        result, {"input": {"status": "allowed"}}, {"action_required": False, "action": None,
                                                    "tool_result": None, "details": {"path": "part2"}},
    ))

    outputs = app.run_request("hello", app.PROMPTED_MODE, "constrained", "greedy", 0.7, 0.9, "Non-RAG")

    assert outputs[0] == "A safe answer."
    assert outputs[1] == "card_arrival"
    assert outputs[3] == "No mock tool action suggested"
    assert outputs[5]["status"] == "ok"
    assert outputs[6] == {"path": "part2"}


def test_ui_part3_executes_pipeline_and_shows_tool_result(monkeypatch):
    """Expose the parsed action and confirmed mock result after Part 3 execution."""
    monkeypatch.setattr(app, "assistant", lambda: (object(), object()))
    monkeypatch.setattr(app, "allowed_intents", lambda: ("lost_or_stolen_card",))
    raw = json.dumps({
        "intent": "lost_or_stolen_card",
        "response": "I can help with the demo card.",
        "action_required": True,
        "action": {"tool": "block_card", "arguments": {"card_last4": "5678"}},
    })
    monkeypatch.setattr(app, "generate_response", lambda *args: raw)

    outputs = app.run_request(
        "Block my card ending 5678", app.STRUCTURED_MODE, "constrained", "greedy", 0.7, 0.9,
        "Non-RAG",
    )

    assert outputs[1] == "lost_or_stolen_card"
    assert outputs[3] == "block_card - executed against mock data"
    assert outputs[4]["action"]["arguments"] == {"card_last4": "5678"}
    assert outputs[4]["tool_result"]["status"] == "blocked"
    assert outputs[5]["raw_output"] == raw
    assert outputs[5]["final_response"].startswith("(Demo)")


def test_ui_part3_rag_choice_uses_cached_retriever(monkeypatch):
    """When RAG is selected, pass the retrieved training context through pipeline execution."""
    class FakeRetriever:
        """Stand in for the train-only similarity index without model downloads."""

        def retrieve(self, query, k=5):
            """Return a known labeled example so the prompt can be checked."""
            return [{"text": "card was stolen", "intent": "lost_or_stolen_card", "score": 1.0}]

    monkeypatch.setattr(app, "assistant", lambda: (object(), object()))
    monkeypatch.setattr(app, "allowed_intents", lambda: ("lost_or_stolen_card",))
    monkeypatch.setattr(app, "rag_retriever", lambda: FakeRetriever())
    seen_prompts = []
    raw = json.dumps({"intent": "lost_or_stolen_card", "response": "Use official support.",
                      "action_required": False, "action": None})

    def generate(_model, _tokenizer, prompt, _config):
        """Capture the assembled prompt while returning a fixed valid schema object."""
        seen_prompts.append(prompt)
        return raw

    monkeypatch.setattr(app, "generate_response", generate)
    outputs = app.run_request("My card was stolen", app.STRUCTURED_MODE, "constrained",
                              "greedy", 0.7, 0.9, "RAG")

    assert outputs[6]["retrieval"] == "RAG"
    assert seen_prompts
    assert "card was stolen" in seen_prompts[0]


def test_sample_prompt_selection_fills_customer_query():
    """Return the chosen sample unchanged so Gradio can place it in Customer Query."""
    sample = app.SAMPLE_PROMPTS[-1]

    assert app._load_sample_prompt(sample) == sample
    assert app._load_sample_prompt(None) == ""


def test_safety_notice_is_present_and_explicit():
    """Keep the complete educational-only and no-real-actions warning available to the UI."""
    assert app.SAFETY_NOTICE == (
        "Educational demonstration only. Never enter real account details, passwords, PINs, "
        "OTPs or full card numbers. No real banking actions are performed."
    )


@pytest.mark.parametrize("query", [
    "I want to hack the bank server.",
    "Steal money from user 13142",
])
def test_harmful_banking_requests_are_blocked_in_the_ui(query):
    """Block explicit financial abuse prompts before either UI mode loads a model."""
    outputs = app.run_request(query, app.STRUCTURED_MODE, "constrained", "greedy", 0.7, 0.9, "Non-RAG")

    assert outputs[5]["status"] == "blocked_input"
    assert outputs[5]["raw_output"] is None
    assert "harmful_banking_action" in outputs[5]["errors"]
