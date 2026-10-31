"""Deterministic routing hints for explicit identifiers in customer requests."""

import re


TRANSACTION_ID = re.compile(r"\bTXN\d{4}\b", re.IGNORECASE)
CARD_LAST4 = re.compile(r"\bcard\s+(?:ending(?:\s+in)?|last\s*4(?:\s+digits?)?)\s*(\d{4})\b", re.IGNORECASE)


def suggest_action(query: str) -> dict | None:
    """Route explicit transaction/card lookups and lost-card blocks to mock tools."""
    transaction = TRANSACTION_ID.search(query)
    if transaction:
        return {"tool": "transaction_status", "arguments": {"transaction_id": transaction.group(0).upper()}}

    card = CARD_LAST4.search(query)
    if not card:
        return None

    if re.search(r"\b(?:lost|stolen|block)\b", query, re.IGNORECASE):
        return {"tool": "block_card", "arguments": {"card_last4": card.group(1)}}
    if re.search(r"\b(?:status|active|blocked|working|check)\b", query, re.IGNORECASE):
        return {"tool": "card_status", "arguments": {"card_last4": card.group(1)}}
    return None
