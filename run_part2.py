"""Run BANKING77 E1/E2/E3 with conservative input/output safety guardrails.
Usage: python run_part2.py --evaluation_csv data/evaluation_sample.csv --limit 3
Do not supply ground-truth intent labels to the model.
"""
import argparse
import json
import random
from pathlib import Path
import pandas as pd
import torch
from src.model.llm_baseline import load_model, generate_response, MODEL_ID
from src.model.decoding import DECODING
from src.prompts.templates import build_prompt
from src.safety.guardrails import guarded_generate, redact_sensitive


def main():
    """Run the configured baseline, prompt, and decoding comparison experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_csv", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--diverse_intents", action="store_true", help="Select one example per intent for a diverse smoke test")
    args = parser.parse_args()
    df = pd.read_csv(args.evaluation_csv)
    if "text" not in df.columns:
        raise ValueError("Evaluation CSV must contain 'text' column")
    if "sample_id" not in df.columns:
        df["sample_id"] = [f"eval_{i}" for i in range(len(df))]
    if args.diverse_intents and "intent" in df.columns:
        df = df.groupby("intent", sort=True).head(1)
    df = df.head(args.limit)
    random.seed(42)
    torch.manual_seed(42)
    model, tokenizer = load_model()
    out = Path("experiments")
    out.mkdir(exist_ok=True)
    baseline_rows, prompt_rows, decoding_rows, logs = [], [], [], []

    def run_one(q, sid, experiment, variant, decoding, params, repetition=None):
        """Generate and record one query under a specific experiment configuration."""
        prompt = (f"Answer this banking customer question clearly and safely:\nCustomer: {q}\nAssistant:"
                  if variant == "baseline" else build_prompt(q, variant))
        result = guarded_generate(
            q, prompt,
            lambda safe_prompt: generate_response(model, tokenizer, safe_prompt, params)
        )
        record = {
            "sample_id": sid,
            "query": redact_sensitive(q),
            "prompt_variant": variant,
            "decoding": decoding,
            "response": result.response,
            "safety_status": result.status,
            "safety_reasons": ";".join(result.reasons),
        }
        if repetition is not None:
            record["repetition"] = repetition
            record["parameters"] = json.dumps(params)
        logs.append({"experiment": experiment, "model_id": MODEL_ID, **record})
        return record

    for _, row in df.iterrows():
        q, sid = str(row["text"]), str(row["sample_id"])
        baseline_rows.append(run_one(q, sid, "E1", "baseline", "greedy", DECODING["greedy"]))
        for variant in ("zero_shot", "few_shot", "role_based", "constrained"):
            prompt_rows.append(run_one(q, sid, "E2", variant, "greedy", DECODING["greedy"]))
        for decoding, params in DECODING.items():
            for repetition in range(1, 4):
                decoding_rows.append(run_one(q, sid, "E3", "constrained", decoding, params, repetition))

    pd.DataFrame(baseline_rows).to_csv(out / "baseline_outputs.csv", index=False)
    pd.DataFrame(prompt_rows).to_csv(out / "prompt_comparison.csv", index=False)
    pd.DataFrame(decoding_rows).to_csv(out / "decoding_comparison.csv", index=False)
    with (out / "experiment_runs.jsonl").open("w", encoding="utf-8") as f:
        for record in logs:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = pd.Series([r["safety_status"] for r in logs]).value_counts().to_dict()
    (out / "safety_summary.json").write_text(json.dumps({"status_counts": counts}, indent=2), encoding="utf-8")
    print(f"Saved {len(baseline_rows)} baseline rows, {len(prompt_rows)} prompt rows and {len(decoding_rows)} decoding rows")
    print("Safety statuses:", counts)


if __name__ == "__main__":
    main()
