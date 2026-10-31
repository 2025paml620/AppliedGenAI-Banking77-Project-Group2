"""Evaluate Qwen zero-shot intent classification on held-out BANKING77 examples.

From repository root:
  python -m scripts.evaluate_intents --evaluation_csv data/evaluation_sample.csv --limit 10
  python -m scripts.evaluate_intents --evaluation_csv data/evaluation_sample.csv

Ground-truth intent is used for scoring ONLY, never for model prompting.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.model.intent_classifier import predict_intent
from src.model.llm_baseline import MODEL_ID, load_model


def load_evaluation_data(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load and validate evaluation rows, returning data and the public label set."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"sample_id": "string", "text": "string", "intent": "string"})
    required = {"sample_id", "text", "intent"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Evaluation CSV is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Evaluation CSV is empty")
    if df[list(required)].isna().any().any():
        raise ValueError("Evaluation CSV contains null sample IDs, queries, or intents")
    if df["text"].str.strip().eq("").any() or df["intent"].str.strip().eq("").any():
        raise ValueError("Evaluation CSV contains blank queries or intent labels")
    if df["sample_id"].duplicated().any():
        raise ValueError("Evaluation CSV contains duplicate sample IDs")

    # Derive the label vocabulary from the FULL evaluation file, before --limit.
    # This is permissible for a closed-set task: only label NAMES, not row answers,
    # are provided to Qwen. The vocabulary should contain all 77 BANKING77 labels.
    allowed_intents = sorted(df["intent"].astype(str).unique().tolist())
    if len(allowed_intents) != 77:
        print(f"WARNING: Found {len(allowed_intents)} labels, expected 77 BANKING77 intents.", flush=True)
    return df, allowed_intents


def evaluate(args: argparse.Namespace) -> None:
    """Run closed-set inference over selected rows and save predictions and metrics."""
    df, allowed_intents = load_evaluation_data(Path(args.evaluation_csv))

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        if args.diverse_intents:
            # One example per intent first, then further examples if limit > 77.
            df = df.sort_values(["intent", "sample_id"])
            df = df.assign(_round=df.groupby("intent").cumcount())
            df = df.sort_values(["_round", "intent", "sample_id"]).drop(columns="_round")
        df = df.head(args.limit).copy()

    print(f"Model: {MODEL_ID}", flush=True)
    print(f"Evaluating {len(df)} queries; label vocabulary: {len(allowed_intents)} intents", flush=True)
    model, tokenizer = load_model()  # Load Qwen ONCE, not for every query.

    rows = []
    for index, (_, row) in enumerate(df.iterrows(), start=1):
        query = str(row["text"])
        result = predict_intent(model, tokenizer, query, allowed_intents)
        actual = str(row["intent"])
        prediction = result["predicted_intent"]
        record = {
            "sample_id": str(row["sample_id"]),
            "query": query,
            "actual_intent": actual,
            "predicted_intent": prediction or "",  # invalid => empty, not a correct prediction
            "raw_prediction": result["raw_prediction"],
            "valid_prediction": result["valid_prediction"],
            "correct": prediction == actual,
        }
        rows.append(record)
        print(f"[{index}/{len(df)}] {record['sample_id']}: "
              f"predicted={prediction or '[INVALID]'} actual={actual} "
              f"correct={record['correct']}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results_path = out_dir / "intent_predictions.csv"
    results.to_csv(results_path, index=False, encoding="utf-8")

    total = len(results)
    valid = int(results["valid_prediction"].sum())
    correct = int(results["correct"].sum())
    metrics = {
        "model_id": MODEL_ID,
        "evaluation_csv": str(args.evaluation_csv),
        "num_allowed_intents": len(allowed_intents),
        "evaluated_samples": total,
        "distinct_actual_intents": int(results["actual_intent"].nunique()),
        "valid_predictions": valid,
        "invalid_predictions": total - valid,
        "correct_predictions": correct,
        "accuracy": round(correct / total, 6),
        "valid_prediction_rate": round(valid / total, 6),
        "accuracy_on_valid_only": round(correct / valid, 6) if valid else None,
        "method": "zero-shot closed-set classification; greedy decoding; no fine-tuning",
    }
    metrics_path = out_dir / "intent_evaluation.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    errors_path = out_dir / "intent_errors.csv"
    results.loc[~results["correct"]].to_csv(errors_path, index=False, encoding="utf-8")

    print("\n--- Intent evaluation ---")
    print(f"Accuracy (invalid counted as wrong): {correct}/{total} = {metrics['accuracy']:.2%}")
    print(f"Valid label rate: {valid}/{total} = {metrics['valid_prediction_rate']:.2%}")
    print(f"Saved: {results_path}, {metrics_path}, {errors_path}")


def main() -> None:
    """Parse evaluation options and start the intent evaluation run."""
    parser = argparse.ArgumentParser(description="Evaluate Qwen intent classification on BANKING77")
    parser.add_argument("--evaluation_csv", default="data/evaluation_sample.csv")
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of queries to evaluate; omit for the full CSV")
    parser.add_argument("--diverse_intents", action="store_true",
                        help="When using --limit, prefer distinct intents first")
    parser.add_argument("--output_dir", default="experiments")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
