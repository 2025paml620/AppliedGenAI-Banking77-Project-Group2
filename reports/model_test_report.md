# Model Test Run Report

The automated suite exercises the agreed `src.structured.pipeline.handle_query` contract. It records run metadata, case-level outputs, and metrics under `experiments/model_test_runs/` without overwriting earlier runs.

Run from the repository root:

```powershell
python -m scripts.run_pipeline --mode suite
```

For a short smoke run, add `--limit 5`. The [model test query suite](model_test_queries.csv) contains one example for each of the 77 Banking77 intents plus safety, input-validation, ambiguity, and mock-tool cases. These are representative prompts, not a proof of performance for every wording.

The runner loads the allowed intent names from `data/evaluation_sample.csv`. A query row's `expected_intent` is used only after inference for scoring; it is never provided to the model. Model ID, pinned revision, generation settings, run ID, and case IDs are saved in each run's `run.json`.

## Report template

| Field | Value |
|---|---|
| Run ID and timestamp | _Fill from `run.json`_ |
| Tester | _Fill in_ |
| Code revision | _Fill in commit or branch_ |
| Model ID and revision | `Qwen/Qwen2.5-1.5B-Instruct` / `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Python, PyTorch, and Transformers versions | _Fill from `run.json`_ |
| Hardware | _Fill in CPU/GPU and memory_ |
| Generation settings | Greedy; see `run.json` for token limit |

Use the generated `report.md` for exact denominators and `results.csv` for case-by-case evidence. Intent accuracy is exact label matching over generated, structurally accepted outputs. Empty and blocked inputs are excluded because they never reached generation. JSON/schema validity is structural only; it does not establish semantic correctness.

Cases involving free-form advice, ambiguity, credential guidance, unverified actions, and financial guarantees require human review. The regular-expression guardrails cannot establish semantic safety. Tool actions run against fixed local mock data only. Never enter real credentials or private customer data into the suite.

## Findings to complete after each run

### Intent classification

_Summarize weak and strong intent areas. List each mismatch using query ID, predicted intent, and expected intent._

### Response quality and safety

_Review outputs for secret solicitation, unsupported action claims, guarantees, invented policies, relevance, and clarity. Record query IDs and short descriptions; do not copy real secrets._

### Structured output and tools

_Review invalid JSON, wrong fields, unsupported actions, tool arguments, mock results, and whether replies clearly identify simulated actions._

### Limitations

_Record sample size, hardware, model/dependency revisions, decoding settings, and any manual-review exclusions. The 77 intent queries provide one example per class, not a statistically representative benchmark._
