"""Append-only JSONL experiment records for integrated runs."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


def log_run(record: dict[str, Any], path: str | Path = "experiments/experiment_runs.jsonl") -> dict[str, Any]:
    required = {"experiment_type", "model_id", "prompt_variant", "sample_ids", "metrics"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"missing experiment fields: {', '.join(sorted(missing))}")
    entry = {"run_id": str(uuid4()), "timestamp_utc": datetime.now(timezone.utc).isoformat(), **record}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    return entry
