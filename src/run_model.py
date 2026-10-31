"""Run one query through the integrated BANKING77 assistant from the command line."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from src.logging_config import configure_logging
from src.model.llm_baseline import generate_response, load_model
from src.structured.pipeline import handle_query

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def load_allowed_intents(path: Path) -> list[str]:
    """Load the 77 public BANKING77 label names from a CSV intent column."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "intent" not in (reader.fieldnames or []):
            raise ValueError(f"Intent CSV must contain an 'intent' column: {path}")
        intents = sorted({(row.get("intent") or "").strip() for row in reader} - {""})
    if len(intents) != 77:
        raise ValueError(f"Expected 77 BANKING77 intents, found {len(intents)}")
    return intents


def main() -> int:
    """Load the pinned model, run the guarded pipeline, and print its public result."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Synthetic banking support query")
    parser.add_argument("--intents-csv", type=Path, default=ROOT / "data" / "evaluation_sample.csv",
                        help="CSV supplying the public intent vocabulary")
    parser.add_argument("--max-new-tokens", type=int, default=200,
                        help="Greedy decoding token limit")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    intents = load_allowed_intents(args.intents_csv)
    model, tokenizer = load_model()
    generation_config = {"do_sample": False, "max_new_tokens": args.max_new_tokens}
    result = handle_query(
        args.query,
        intents,
        lambda prompt: generate_response(model, tokenizer, prompt, generation_config),
    )
    # Don't echo the query: blocked input could contain the very secret the guard caught.
    public_result = {key: result[key] for key in (
        "status", "intent", "tool", "tool_result", "errors", "final_response"
    )}
    logger.info("Query pipeline completed: status=%s tool=%s", result["status"], result["tool"] or "none")
    print(json.dumps(public_result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"ok", "blocked_input", "invalid_input"} else 1


if __name__ == "__main__":
    sys.exit(main())
