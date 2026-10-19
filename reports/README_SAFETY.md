# BANKING77 Part 2: Safety guardrails

Copy `run_part2.py` to the repository root and copy `src/safety/` into the existing `src/` folder. This update preserves the existing `src/model/` and `src/prompts/` modules.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_part2.py --evaluation_csv data/evaluation_sample.csv --limit 3 --diverse_intents
```

The runner creates `experiments/baseline_outputs.csv`, `prompt_comparison.csv`, `decoding_comparison.csv`, `experiment_runs.jsonl`, and `safety_summary.json`. It overwrites existing experiment outputs: back them up first.

Safety checks include explicit credential patterns, potential prompt-injection instructions, unverified banking action claims, and financial guarantees. Unsafe model outputs are replaced with a conservative fallback; `safety_status` and `safety_reasons` are included in experiment outputs. Input text is redacted in logs where obvious labeled secrets or long card numbers are detected. Do not use real banking data in this educational project.

## Limitations

These are lightweight regular-expression checks, **not** comprehensive PII detection, prompt-injection prevention, or a security guarantee. They can miss paraphrases and incorrectly flag benign statements. A real banking deployment requires stronger privacy controls, policy-based evaluation, verified tool authorization, audit design, and human review. The `zero_shot` comparison is now safety-prefixed for all variants, so compare runs using the same guardrail configuration and document that constraint.
