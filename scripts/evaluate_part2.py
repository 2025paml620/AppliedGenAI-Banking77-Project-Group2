"""Evaluate BANKING77 Part 2 (P2.12) from saved CSVs; no model inference.

Run from the repository root:
    python -m scripts.evaluate_part2 --config config/evaluation_config.yaml

Requires: PyYAML (pip install pyyaml). Uses only Python standard library otherwise.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit('PyYAML is required: python -m pip install pyyaml') from exc


def read_csv(path: Path):
    if not path.exists():
        print(f'[SKIP] Missing file: {path}')
        return []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def pick(row, *names):
    lookup = {str(k).strip().lower(): v for k, v in row.items()}
    for name in names:
        value = lookup.get(name.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ''


def evaluate_intents(rows, cfg, out):
    if not rows:
        return {'status': 'not_evaluated', 'reason': 'intent predictions CSV unavailable or empty'}
    truth_keys = ('actual_intent', 'true_intent', 'intent', 'ground_truth', 'label')
    pred_keys = ('predicted_intent', 'prediction', 'pred_intent')
    parsed = []
    for row in rows:
        truth = pick(row, *truth_keys)
        pred = pick(row, *pred_keys)
        if not truth:
            continue
        parsed.append({'sample_id': pick(row, 'sample_id', 'id'),
                       'query': pick(row, 'query', 'text'),
                       'actual_intent': truth, 'predicted_intent': pred,
                       'correct': truth == pred, 'raw_prediction': pick(row, 'raw_prediction')})
    if not parsed:
        return {'status': 'not_evaluated', 'reason': 'no ground-truth intent column found'}
    labels = cfg.get('intent', {}).get('labels') or sorted({r['actual_intent'] for r in parsed})
    label_set = set(labels)
    n = len(parsed)
    correct = sum(r['correct'] for r in parsed)
    invalid = sum(r['predicted_intent'] not in label_set for r in parsed)
    per_class = []
    for label in labels:
        tp = sum(r['actual_intent'] == label and r['predicted_intent'] == label for r in parsed)
        fp = sum(r['actual_intent'] != label and r['predicted_intent'] == label for r in parsed)
        fn = sum(r['actual_intent'] == label and r['predicted_intent'] != label for r in parsed)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({'intent': label, 'support': tp + fn, 'precision': round(precision, 6),
                          'recall': round(recall, 6), 'f1': round(f1, 6)})
    errors = [r for r in parsed if not r['correct']]
    write_csv(out / 'incorrect_intents.csv', errors,
              ['sample_id', 'query', 'actual_intent', 'predicted_intent', 'correct', 'raw_prediction'])
    write_csv(out / 'intent_per_class.csv', per_class, ['intent', 'support', 'precision', 'recall', 'f1'])
    return {'status': 'evaluated', 'samples': n, 'correct': correct, 'incorrect': n - correct,
            'accuracy': round(correct / n, 6),
            'macro_f1': round(sum(r['f1'] for r in per_class) / len(per_class), 6) if per_class else None,
            'macro_f1_class_count': len(per_class), 'invalid_predictions': invalid,
            'invalid_prediction_rate': round(invalid / n, 6),
            'note': 'Macro-F1 is over configured labels or labels present as ground truth.'}


def infer_columns(rows):
    """Extract query, strategy, response and repetition from common CSV schemas."""
    result = []
    for index, row in enumerate(rows):
        query = pick(row, 'query', 'text', 'customer_query', 'user_query', 'input_text', 'input')
        strategy = pick(row, 'decoding_strategy', 'decoding', 'strategy', 'decoding_config',
                        'prompt_strategy', 'prompt_type', 'template', 'prompt_variant', 'method')
        response = pick(row, 'response', 'generated_response', 'model_response',
                        'output', 'generated_text', 'answer', 'baseline_response')
        repetition = pick(row, 'repetition', 'repeat', 'run', 'run_id', 'trial', 'iteration')
        result.append({'row_index': index, 'query': query, 'strategy': strategy,
                       'response': response, 'repetition': repetition, 'source': row})
    return result


def evaluate_variability(rows, cfg, out):
    if not rows:
        return {'status': 'not_evaluated', 'reason': 'decoding comparison CSV unavailable or empty'}
    normalized = infer_columns(rows)
    if not any(r['response'] for r in normalized):
        return {'status': 'not_evaluated', 'reason': 'response column not recognized; update infer_columns() aliases'}
    groups = defaultdict(list)
    for row in normalized:
        if row['response'] and row['query']:
            response = row['response']
            if cfg.get('variability', {}).get('normalize_whitespace', True):
                response = ' '.join(response.split())
            groups[(row['query'], row['strategy'] or '(unspecified)')].append(response)
    comparisons = []
    for (query, strategy), responses in sorted(groups.items()):
        if len(responses) < 2:
            continue
        unique = len(set(responses))
        comparisons.append({'query': query, 'strategy': strategy, 'repetitions': len(responses),
                            'unique_responses': unique, 'unique_response_rate': round(unique / len(responses), 6),
                            'all_repetitions_identical': unique == 1})
    if not comparisons:
        return {'status': 'not_evaluated', 'reason': 'no query + strategy groups with at least two responses; check column aliases'}
    write_csv(out / 'variability_by_query.csv', comparisons,
              ['query', 'strategy', 'repetitions', 'unique_responses', 'unique_response_rate', 'all_repetitions_identical'])
    by_strategy = defaultdict(list)
    for row in comparisons:
        by_strategy[row['strategy']].append(row)
    summary = []
    for strategy, group in sorted(by_strategy.items()):
        summary.append({'strategy': strategy, 'query_groups': len(group),
                        'mean_unique_response_rate': round(sum(r['unique_response_rate'] for r in group) / len(group), 6),
                        'fully_consistent_groups': sum(r['all_repetitions_identical'] for r in group)})
    write_csv(out / 'variability_by_strategy.csv', summary,
              ['strategy', 'query_groups', 'mean_unique_response_rate', 'fully_consistent_groups'])
    return {'status': 'evaluated', 'compared_query_strategy_groups': len(comparisons),
            'by_strategy': summary,
            'note': 'Exact textual consistency only; semantic similarity is not measured.'}


def extract_json(text):
    text = text.strip()
    # Accept a complete JSON object or a fenced JSON object; reject explanatory prose.
    match = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', text, flags=re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    try:
        value = json.loads(text)
        return value, None
    except (json.JSONDecodeError, ValueError) as exc:
        return None, str(exc)


def evaluate_formatting(prompt_rows, decoding_rows, intent_rows, cfg, out):
    settings = cfg.get('formatting', {})
    if not settings.get('constrained_prompt_requires_json', False):
        return {'status': 'not_evaluated', 'reason': 'JSON not declared mandatory in evaluation_config.yaml'}
    keywords = [s.lower() for s in settings.get('constrained_strategy_keywords', ['constrained', 'json'])]
    required = settings.get('required_fields', [])
    allowed_actions = set(settings.get('allowed_actions', []))
    allowed_intents = {pick(r, 'actual_intent', 'true_intent', 'intent') for r in intent_rows}
    allowed_intents.discard('')
    details = []
    for source_name, source_rows in [('prompts', prompt_rows), ('decoding', decoding_rows)]:
        for row in infer_columns(source_rows):
            source = row['source']
            # Decoding rows may carry both a prompt strategy and a decoding strategy.
            prompt_type = pick(source, 'prompt_strategy', 'prompt_type', 'template', 'prompt_variant', 'prompt_style')
            strategy_text = ' '.join([prompt_type, row['strategy']]).lower()
            if not any(word in strategy_text for word in keywords):
                continue
            if not row['response']:
                details.append({'source': source_name, 'query': row['query'], 'strategy': strategy_text,
                                'valid_json': False, 'required_fields_present': False,
                                'schema_valid': False, 'error': 'empty or unrecognized response'})
                continue
            obj, error = extract_json(row['response'])
            is_json = error is None and isinstance(obj, dict)
            present = is_json and all(k in obj for k in required)
            schema_valid = bool(present)
            errors = []
            if not is_json:
                errors.append('not a JSON object')
            elif not present:
                errors.append('missing required fields')
            if present:
                if not all(isinstance(obj[k], str) for k in required):
                    schema_valid = False
                    errors.append('required fields must be strings')
                if 'action_required' in obj and allowed_actions and obj['action_required'] not in allowed_actions:
                    schema_valid = False
                    errors.append('invalid action_required')
                if (settings.get('allowed_intents_from_intent_predictions', True)
                        and allowed_intents and 'intent' in obj and obj['intent'] not in allowed_intents):
                    schema_valid = False
                    errors.append('intent not in allowed label vocabulary')
            details.append({'source': source_name, 'query': row['query'], 'strategy': strategy_text,
                            'valid_json': bool(is_json), 'required_fields_present': bool(present),
                            'schema_valid': bool(schema_valid), 'error': '; '.join(errors)})
    if not details:
        return {'status': 'not_evaluated', 'reason': 'no constrained/JSON response rows found; check strategy columns and keywords'}
    write_csv(out / 'formatting_checks.csv', details,
              ['source', 'query', 'strategy', 'valid_json', 'required_fields_present', 'schema_valid', 'error'])
    n = len(details)
    return {'status': 'evaluated', 'checked_responses': n,
            'valid_json': sum(r['valid_json'] for r in details),
            'required_fields_present': sum(r['required_fields_present'] for r in details),
            'schema_valid': sum(r['schema_valid'] for r in details),
            'json_validity_rate': round(sum(r['valid_json'] for r in details) / n, 6),
            'schema_compliance_rate': round(sum(r['schema_valid'] for r in details) / n, 6)}


def evaluate_quality(baseline_rows, prompt_rows, manual_rows, cfg, out):
    settings = cfg.get('response_quality', {})
    criteria = settings.get('criteria', ['relevance', 'clarity', 'helpfulness', 'safety', 'formatting'])
    minimum, maximum = settings.get('score_min', 1), settings.get('score_max', 5)
    template = []
    for source_name, rows in [('baseline', baseline_rows), ('prompts', prompt_rows)]:
        for item in infer_columns(rows):
            if item['response']:
                template.append({'source': source_name, 'query': item['query'],
                                 'strategy': item['strategy'] or source_name, 'response': item['response'],
                                 **{key: '' for key in criteria}, 'review_notes': ''})
    if template and not (cfg['inputs'].get('manual_scores') and Path(cfg['inputs']['manual_scores']).exists()):
        write_csv(out / 'manual_review_template.csv', template,
                  ['source', 'query', 'strategy', 'response', *criteria, 'review_notes'])
    if not manual_rows:
        return {'status': 'awaiting_manual_review', 'review_candidates': len(template),
                'note': 'Fill scores in manual_response_scores.csv and rerun. Blank cells are not scored.'}
    scored = defaultdict(list)
    invalid_scores = 0
    for row in manual_rows:
        for criterion in criteria:
            raw = pick(row, criterion)
            if not raw:
                continue
            try:
                value = float(raw)
                if not math.isfinite(value) or not minimum <= value <= maximum:
                    raise ValueError('out of range')
                scored[criterion].append(value)
            except ValueError:
                invalid_scores += 1
    return {'status': 'evaluated' if scored else 'awaiting_manual_review',
            'review_rows': len(manual_rows), 'invalid_score_cells': invalid_scores,
            'averages': {k: round(sum(v) / len(v), 3) for k, v in scored.items()},
            'score_counts': {k: len(v) for k, v in scored.items()},
            'note': 'Averages are based only on nonblank valid human-entered scores.'}


def main():
    parser = argparse.ArgumentParser(description='Evaluate BANKING77 Part 2 CSV outputs (P2.12).')
    parser.add_argument('--config', default='config/evaluation_config.yaml')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # Assumes config/ is one level below the repository root.
    root = config_path.parent.parent
    input_paths = {key: root / value for key, value in cfg['inputs'].items()}
    cfg['inputs'] = {key: str(path) for key, path in input_paths.items()}
    out = root / cfg.get('output_dir', 'experiments/part2_evaluation')
    out.mkdir(parents=True, exist_ok=True)
    data = {key: read_csv(path) for key, path in input_paths.items()}
    results = {
        'intent_correctness': evaluate_intents(data.get('intent_predictions', []), cfg, out),
        'response_quality': evaluate_quality(data.get('baseline', []), data.get('prompts', []),
                                             data.get('manual_scores', []), cfg, out),
        'variability': evaluate_variability(data.get('decoding', []), cfg, out),
        'formatting': evaluate_formatting(data.get('prompts', []), data.get('decoding', []),
                                          data.get('intent_predictions', []), cfg, out),
    }
    (out / 'evaluation_summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    summary = []
    for criterion, metrics in results.items():
        for metric, value in metrics.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            summary.append({'criterion': criterion, 'metric': metric, 'value': value})
    write_csv(out / 'evaluation_summary.csv', summary, ['criterion', 'metric', 'value'])
    print('\nP2.12 evaluation complete:')
    for criterion, metrics in results.items():
        print(f'  {criterion}: {metrics["status"]}')
    print(f'Outputs: {out}')
    print('Summary:', out / 'evaluation_summary.json')


if __name__ == '__main__':
    main()
