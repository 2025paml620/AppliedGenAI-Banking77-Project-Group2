"""Standalone P2.5 verification; does not overwrite existing E1/E2/E3 results.
Run from project root: python -m scripts.run_p2_5 --limit 3
"""
import argparse
import json
import re
from pathlib import Path
import pandas as pd
import torch
from src.model.llm_baseline import load_model, generate_response, MODEL_ID
from src.prompts.templates import build_zero_shot_intent_response_prompt
from src.safety.guardrails import guarded_generate, redact_sensitive


def parse_output(text):
    """Return (object, strict_json, normalized_json); do not conceal formatting errors."""
    try:
        return json.loads(text), True, True
    except (ValueError, TypeError):
        pass
    match = re.fullmatch(r"\s*```(?:json)?\s*([\s\S]*?)\s*```\s*", text, re.I)
    if match:
        try:
            return json.loads(match.group(1)), False, True
        except (ValueError, TypeError):
            pass
    return None, False, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_csv', default='data/evaluation_sample.csv')
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--diverse_intents', action='store_true', help='Select one query per intent')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    df = pd.read_csv(args.evaluation_csv)
    if not {'text', 'intent'}.issubset(df.columns):
        raise ValueError('Evaluation CSV must contain text and intent columns')
    # The entire label vocabulary is public task metadata, not per-row ground truth.
    allowed = sorted(df['intent'].dropna().astype(str).unique().tolist())
    if len(allowed) != 77:
        raise ValueError(f'Expected 77 intent labels, got {len(allowed)}')
    if args.diverse_intents:
        df = df.groupby('intent', sort=True).head(1)
    torch.manual_seed(42)
    model, tokenizer = load_model()
    records = []
    for index, row in df.head(args.limit).iterrows():
        query = str(row['text'])
        prompt = build_zero_shot_intent_response_prompt(query, allowed)
        params = {'do_sample': False, 'max_new_tokens': 140}
        result = guarded_generate(
            query, prompt,
            lambda safe_prompt: generate_response(model, tokenizer, safe_prompt, params),
        )
        predicted, response, valid = None, None, False
        strict_json, normalized_json = False, False
        if result.status == 'allowed':
            obj, strict_json, normalized_json = parse_output(result.response)
            if isinstance(obj, dict):
                predicted = obj.get('intent')
                response = obj.get('response')
                valid = (predicted in allowed and isinstance(response, str)
                         and bool(response.strip()))
        records.append({
            'sample_id': str(row.get('sample_id', index)),
            'query': redact_sensitive(query),
            'prompt': redact_sensitive(prompt),
            'model_id': MODEL_ID,
            'decoding_parameters': json.dumps(params),
            'raw_output': redact_sensitive(result.response),
            'predicted_intent': predicted,
            'customer_response': redact_sensitive(response) if isinstance(response, str) else '',
            'valid_structured_output': valid,
            'strict_json_valid': strict_json,
            'normalized_json_valid': normalized_json,
            'safety_status': result.status,
            'safety_reasons': ';'.join(result.reasons),
            # Ground truth is only used after generation for evaluation.
            'ground_truth_intent': str(row['intent']),
            'intent_correct': valid and predicted == str(row['intent']),
        })
    path = Path('experiments/p2_5_zero_shot.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)
    print(f'Saved {len(records)} rows to {path}')
    print(f'Valid structured outputs: {sum(r["valid_structured_output"] for r in records)}/{len(records)}')
    print(f'Correct intents: {sum(r["intent_correct"] for r in records)}/{len(records)}')
    print(f'Strict JSON outputs: {sum(r["strict_json_valid"] for r in records)}/{len(records)}')


if __name__ == '__main__':
    main()
