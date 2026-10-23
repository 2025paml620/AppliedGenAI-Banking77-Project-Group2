"""A deterministic, in-memory card-lock simulation for demonstrations."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    tool: str
    message: str
    data: dict[str, Any]


CARD_TOOL_SCHEMA = {
    "name": "simulate_card_lock",
    "description": "Simulate locking a demo card. This does not contact a bank.",
    "parameters": {"type": "object", "additionalProperties": False,
                    "required": ["card_ref", "reason"],
                    "properties": {"card_ref": {"type": "string", "enum": ["demo-card-001"]},
                                   "reason": {"type": "string", "enum": ["lost", "stolen", "user_request"]}}},
}

_DEMO_CARDS = {"demo-card-001": {"status": "active"}}


def simulate_card_lock(arguments: Any) -> ToolResult:
    """Validate arguments and lock only the fixed mock card in memory."""
    if not isinstance(arguments, dict) or set(arguments) != {"card_ref", "reason"}:
        return ToolResult(False, CARD_TOOL_SCHEMA["name"], "Invalid arguments; expected card_ref and reason.", {})
    card_ref, reason = arguments["card_ref"], arguments["reason"]
    if not isinstance(card_ref, str) or card_ref not in _DEMO_CARDS:
        return ToolResult(False, CARD_TOOL_SCHEMA["name"], "Unknown demo card reference.", {})
    if reason not in {"lost", "stolen", "user_request"}:
        return ToolResult(False, CARD_TOOL_SCHEMA["name"], "Unsupported reason.", {})
    _DEMO_CARDS[card_ref]["status"] = "locked"
    return ToolResult(True, CARD_TOOL_SCHEMA["name"], "Demo card is now locked in the local simulation.",
                      {"card_ref": card_ref, "status": "locked", "simulated": True})


def result_to_dict(result: ToolResult) -> dict[str, Any]:
    return asdict(result)
