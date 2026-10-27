"""Compare intent accuracy of the Part 3 pipeline without vs with RAG (same test examples).

  python -m scripts.evaluate_rag --limit 77 --k 5
Ground-truth labels are used for scoring only. Retrieval corpus = train split.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from run_part3 import load_intents
from src.model.llm_baseline import generate_response, load_model
from src.rag.retriever import build_retriever
from src.structured.pipeline import handle_query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_csv", default="data/evaluation_sample.csv")
    parser.add_argument("--limit", type=int, default=77)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output_dir", default="experiments")
    args = parser.parse_args()

    df = pd.read_csv(args.evaluation_csv)
    intents = load_intents(args.evaluation_csv)
    # Evenly spread over the (label-sorted) file so a small limit still covers many intents.
    df = df.iloc[:: max(1, len(df) // args.limit)].head(args.limit)

    model, tokenizer = load_model()
    retriever = build_retriever()
    gen = lambda prompt: generate_response(model, tokenizer, prompt, {"do_sample": False, "max_new_tokens": 200})

    rows = []
    for _, r in df.iterrows():
        row = {"sample_id": r["sample_id"], "query": r["text"], "actual": r["intent"],
               "retrieval_top1": retriever.retrieve(r["text"], 1)[0]["intent"]}
        for name, ret in (("no_rag", None), ("rag", retriever)):
            res = handle_query(r["text"], intents, gen, retriever=ret, k=args.k)
            row[f"{name}_pred"], row[f"{name}_status"] = res["intent"], res["status"]
        rows.append(row)
    out = pd.DataFrame(rows)

    metrics = {"n": len(out), "k": args.k,
               "accuracy_no_rag": float((out["no_rag_pred"] == out["actual"]).mean()),
               "accuracy_rag": float((out["rag_pred"] == out["actual"]).mean()),
               "invalid_rate_no_rag": float((out["no_rag_status"] == "invalid_output").mean()),
               "invalid_rate_rag": float((out["rag_status"] == "invalid_output").mean()),
               "retrieval_only_top1_accuracy": float((out["retrieval_top1"] == out["actual"]).mean())}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    out.to_csv(out_dir / "rag_comparison.csv", index=False)
    (out_dir / "rag_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
