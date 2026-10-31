"""Part 3 prompt template (Jinja2): query -> prompt that asks for one JSON object."""
from pathlib import Path
import json
from typing import Sequence

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_env = Environment(loader=FileSystemLoader(Path(__file__).parent), undefined=StrictUndefined,
                   autoescape=False)

TOOLS_FOR_PROMPT = [
    {"name": "block_card", "arguments": '{"card_last4": "<4 digits>"}',
     "when": "customer wants a lost/stolen card blocked"},
    {"name": "card_status", "arguments": '{"card_last4": "<4 digits>"}',
     "when": "customer asks for the status of a known demo card"},
    {"name": "transaction_status", "arguments": '{"transaction_id": "TXN<4 digits>"}',
     "when": "customer asks about a specific transaction"},
]


def build_structured_prompt(
    query: str,
    allowed_intents: Sequence[str],
    examples: Sequence[dict] | None = None,
    required_action: dict | None = None,
) -> str:
    """Render JSON instructions with labels, optional examples, and any required tool route."""
    if not query or not query.strip():
        raise ValueError("Customer query must not be blank")
    if not allowed_intents:
        raise ValueError("allowed_intents must not be empty")
    return _env.get_template("structured_prompt.j2").render(
        query=query, labels=sorted(set(allowed_intents)), tools=TOOLS_FOR_PROMPT,
        examples=examples,
        required_action_json=json.dumps(required_action, ensure_ascii=False) if required_action else None,
    )
