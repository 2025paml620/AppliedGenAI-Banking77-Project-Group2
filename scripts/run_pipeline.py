"""Run one guarded BANKING77 pipeline from the command line."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from scripts.run_model_test_suite import (
    DEFAULT_INTENTS,
    DEFAULT_OUTPUT,
    DEFAULT_SUITE,
    load_allowed_intents,
    load_suite,
    run_cases,
    write_report,
)
from src.logging_config import configure_logging
from src.model.llm_baseline import MODEL_ID, MODEL_REVISION, load_model


def _make_cases(mode: str, query: str | None, suite_path: Path) -> list[dict[str, str]]:
    """Select interactive, demo, or scored-suite cases for the shared runner."""
    if mode == "suite":
        return load_suite(suite_path)
    if mode == "query":
        if not query or not query.strip():
            raise ValueError("--query is required and must not be blank when --mode query is selected")
        return [{
            "query_id": "Q001",
            "category": "ad_hoc",
            "expected_intent": "",
            "query": query,
            "expected_checks": "manual_review",
        }]

    from run_part3 import DEMO_QUERIES

    return [
        {
            "query_id": f"D{index:03d}",
            "category": "demo",
            "expected_intent": "",
            "query": text,
            "expected_checks": "manual_review",
        }
        for index, text in enumerate(DEMO_QUERIES, start=1)
    ]


def _save_run(
    output_dir: Path,
    results: list[dict],
    run: dict,
) -> Path:
    """Write the result table, readable summary, and complete run metadata."""
    run_dir = output_dir / run["run_id"]
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    run["metrics"] = write_report(run_dir / "report.md", results, run)
    (run_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    """Execute selected cases through handle_query and save a reproducible run."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("query", "demo", "suite"), default="suite",
                        help="Run one query, the predefined demo, or the scored 90-case suite")
    parser.add_argument("--query", help="Customer query used with --mode query")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE, help="Scored synthetic query CSV")
    parser.add_argument("--intents-csv", type=Path, default=DEFAULT_INTENTS,
                        help="CSV supplying the public BANKING77 intent vocabulary")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Parent directory for timestamped run artifacts")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Greedy decoding token limit")
    parser.add_argument("--limit", type=int, help="Run only the first N cases (suite or demo mode)")
    parser.add_argument("--rag", action="store_true", help="Retrieve similar labeled training examples")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--max-new-tokens and --limit must be positive integers")
    if args.mode != "query" and args.query:
        parser.error("--query can only be used with --mode query")

    try:
        cases = _make_cases(args.mode, args.query, args.suite)
    except ValueError as exc:
        parser.error(str(exc))
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        parser.error("No cases selected")

    intents = load_allowed_intents(args.intents_csv)
    retriever = None
    if args.rag:
        from src.rag.retriever import build_retriever

        retriever = build_retriever()
    model, tokenizer = load_model()
    config = {"do_sample": False, "max_new_tokens": args.max_new_tokens}
    run_id = f"{args.mode}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}"
    run = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "dataset": str(args.intents_csv),
        "prompt_variant": "structured_json",
        "retrieval": "RAG" if retriever is not None else "Non-RAG",
        "generation_config": config,
        "case_ids": [case["query_id"] for case in cases],
        "sample_ids": [case["query_id"] for case in cases],
    }
    results = run_cases(cases, intents, model, tokenizer, args.max_new_tokens, retriever=retriever)
    for record in results:
        record.update(
            run_id=run_id,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            prompt_variant=run["prompt_variant"],
            retrieval=run["retrieval"],
            decoding=json.dumps(config, sort_keys=True),
        )

    run_dir = _save_run(args.output_dir, results, run)
    print(f"Results: {run_dir}")
    for record in results:
        print(f"{record['query_id']} [{record['status']}/{record['result']}]: {record['final_response']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
