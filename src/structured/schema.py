"""JSON schema + validation for model output."""
import json
import re
from typing import Sequence

REQUIRED_KEYS = {"intent", "response", "action_required", "action"}


def extract_json(raw: str):
    """Parse the first {...} block (models often add fences/prose). Returns None if unparseable."""
    match = re.search(r"\{.*\}", str(raw), flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def validate_output(raw: str, allowed_intents: Sequence[str]) -> tuple[dict | None, list[str]]:
    """Return validated contract data, or parsing/schema errors when validation fails."""
    data = extract_json(raw)
    if not isinstance(data, dict):
        return None, ["not a valid JSON object"]
    errors = []
    missing = REQUIRED_KEYS - set(data)
    extra = set(data) - REQUIRED_KEYS
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected keys: {sorted(extra)}")
    if errors:
        return None, errors
    if data["intent"] not in set(allowed_intents):
        errors.append("intent is not an allowed label")
    if not isinstance(data["response"], str) or not data["response"].strip():
        errors.append("response must be a non-empty string")
    if not isinstance(data["action_required"], bool):
        errors.append("action_required must be a boolean")
    action = data["action"]
    if action is not None and not (isinstance(action, dict) and set(action) == {"tool", "arguments"}
                                   and isinstance(action["tool"], str) and isinstance(action["arguments"], dict)):
        errors.append('action must be null or {"tool": str, "arguments": object}')
    elif isinstance(data["action_required"], bool) and data["action_required"] != (action is not None):
        errors.append("action_required must be true exactly when action is set")
    return (None, errors) if errors else (data, [])
