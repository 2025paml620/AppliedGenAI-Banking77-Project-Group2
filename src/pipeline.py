"""Integrated query → generation → validation → optional simulated tool flow."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Callable

from src.output.schema import AssistantOutput, parse_output, safe_fallback
from src.prompts.templates import build_structured_prompt
from src.safety.guardrails import check_input, check_output
from src.tools.application_tools import simulate_card_lock


@dataclass(frozen=True)
class PipelineResult:
    output: AssistantOutput
    status: str
    validation_errors: tuple[str, ...] = ()
    tool_result: dict[str, Any] | None = None


def run_pipeline(query: str, generate_fn: Callable[[str], str], context: str | None = None) -> PipelineResult:
    """Run one query. generate_fn accepts a prompt and returns generated text."""
    if not isinstance(query, str) or not query.strip():
        return PipelineResult(safe_fallback(), "invalid_input", ("query must be a non-empty string",))
    input_check = check_input(query)
    if input_check.status != "allowed":
        return PipelineResult(AssistantOutput("card_not_working", input_check.response), input_check.status,
                              input_check.reasons)
    try:
        raw = generate_fn(build_structured_prompt(query, context))
    except Exception as exc:
        return PipelineResult(safe_fallback(), "generation_error", (type(exc).__name__,))
    parsed = parse_output(raw)
    if not parsed.valid:
        return PipelineResult(safe_fallback(), "invalid_model_output", parsed.errors)
    output = parsed.value
    tool_result = None
    if output.action_required:
        action = output.action
        if not isinstance(action, dict) or set(action) != {"tool", "arguments"} or action.get("tool") != "simulate_card_lock":
            return PipelineResult(safe_fallback(output.intent), "invalid_action", ("unsupported tool or action shape",))
        result = simulate_card_lock(action["arguments"])
        tool_result = asdict(result)
        response = result.message if result.ok else "I could not complete the simulated request. Please use your bank's official app or verified support."
        guarded = check_output(response)
        output = AssistantOutput(output.intent, guarded.response, False, None)
        return PipelineResult(output, "tool_succeeded" if result.ok else "tool_failed", (), tool_result)
    guarded = check_output(output.response)
    if guarded.status != "allowed":
        return PipelineResult(AssistantOutput(output.intent, guarded.response), "unsafe_response", guarded.reasons)
    return PipelineResult(output, "ok")
