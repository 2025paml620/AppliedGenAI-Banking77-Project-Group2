"""Customer request -> model action -> validation -> tool result -> final response."""
import logging

from src.prompts.structured import build_structured_prompt
from src.safety.guardrails import SYSTEM_SAFETY, SAFE_FALLBACK, check_input, check_output
from src.structured.schema import validate_output
from src.tools.banking_tools import ToolError, describe_result, run_tool

INVALID_FALLBACK = ("Sorry, I could not process that request reliably. "
                    "Please rephrase it or use your bank's official channels.")
logger = logging.getLogger(__name__)


def handle_query(query: str, allowed_intents, generate_fn, retriever=None, k: int = 5) -> dict:
    """Run one query through input checks, generation, schema validation, safety, and mock tools."""
    """generate_fn(prompt) -> raw model text. With a retriever, similar labelled examples (RAG) are added.
    Always returns a dict with a safe `final_response`."""
    out = {"query": query, "status": "ok", "intent": None, "tool": None, "tool_result": None,
           "errors": [], "raw_output": None, "final_response": ""}
    gate = check_input(query)
    if gate.status != "allowed":
        # Log the rule that fired, not the query itself; a query can contain private details.
        logger.info("Request blocked at input gate; reasons=%s", ",".join(gate.reasons))
        return {**out, "status": "blocked_input", "errors": list(gate.reasons), "final_response": gate.response}

    examples = retriever.retrieve(query, k) if retriever else None
    out["raw_output"] = generate_fn(SYSTEM_SAFETY + "\n\n" + build_structured_prompt(query, allowed_intents, examples))
    data, errors = validate_output(out["raw_output"], allowed_intents)
    if data is None:
        logger.warning("Model output failed structured validation; errors=%s", "; ".join(errors))
        return {**out, "status": "invalid_output", "errors": errors, "final_response": INVALID_FALLBACK}

    out["intent"] = data["intent"]
    safe = check_output(data["response"])
    if safe.status != "allowed":
        logger.warning("Model response replaced by output guard; reasons=%s", ",".join(safe.reasons))
        return {**out, "status": "unsafe_output", "errors": list(safe.reasons), "final_response": SAFE_FALLBACK}
    out["final_response"] = data["response"]

    if data["action"]:
        tool = data["action"]["tool"]
        out["tool"] = tool
        try:
            out["tool_result"] = run_tool(tool, data["action"]["arguments"])
            out["final_response"] = describe_result(tool, out["tool_result"])  # only tool-confirmed claims
            logger.info("Mock tool completed: %s", tool)
        except ToolError as e:
            logger.warning("Mock tool rejected request: %s (%s)", tool, e)
            out.update(status="tool_error", errors=[str(e)], final_response=f"I could not complete that: {e}")
    else:
        logger.info("Request completed without a tool action")
    return out
