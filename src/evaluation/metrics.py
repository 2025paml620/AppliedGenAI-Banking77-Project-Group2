"""Deterministic metrics for intent, schema, and guarded response review."""
from __future__ import annotations
from typing import Iterable
from sklearn.metrics import accuracy_score, f1_score
from src.output.schema import parse_output
from src.safety.guardrails import check_output


def evaluate_records(records: Iterable[dict]) -> dict:
    """Score records with query, reference_intent and raw_output keys.

    Response quality is intentionally a human rubric, not inferred from accuracy.
    """
    rows = list(records)
    if not rows:
        return {"n": 0, "intent_accuracy": None, "macro_f1": None,
                "json_validity_rate": None, "safe_output_rate": None,
                "response_quality": "manual review required"}
    refs, preds = [], []
    valid = safe = 0
    for row in rows:
        refs.append(row.get("reference_intent"))
        result = parse_output(row.get("raw_output", ""))
        if result.valid:
            valid += 1
            preds.append(result.value.intent)
            safe += check_output(result.value.response).status == "allowed"
        else:
            preds.append("__invalid_output__")
    return {"n": len(rows), "intent_accuracy": accuracy_score(refs, preds),
            "macro_f1": f1_score(refs, preds, average="macro", zero_division=0),
            "json_validity_rate": valid / len(rows), "safe_output_rate": safe / len(rows),
            "response_quality": "manual review required"}
