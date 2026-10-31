# Contributing

## Working agreement

- Work on a task branch and keep changes focused. Do not commit model weights, credentials, private customer data, or generated cache files.
- Keep the public Banking77 test sample fixed. Never pass its `intent` or `label` into that sample's model prompt.
- Use mock tools only. Never enter real banking credentials or account data.
- Record experiment settings and outputs under `experiments/`; document interface changes in the report and README.
- Before handing work back, summarize files changed, commands run, results, and known limitations. Each member should verify their contribution statement in `reports/contributions.md`.
- Give every function a short docstring stating its purpose; add a longer one when its inputs, side effects, or safety boundaries need explanation.

## Shared contracts

- Dataset row: `sample_id`, `source_index`, `text`, `label`, `intent` (the final two are evaluation-only references).
- Pipeline: `handle_query(query, allowed_intents, generate_fn, retriever=None, k=5)`; `generate_fn(prompt)` returns raw model text.
- Structured model object: exactly `intent` (allowed label), `response` (non-empty string), `action_required` (boolean), `action` (null or `{tool, arguments}`).
- Tools: `block_card(card_last4)` and `transaction_status(transaction_id)`, both backed by mock data.
- Pipeline result: `query`, `status`, `intent`, `tool`, `tool_result`, `errors`, `raw_output`, `final_response`.

## Reproducible changes

Run commands from the repository root in the documented Python environment. Record model and dataset revisions, prompt variant, effective decoding configuration, seed, sample IDs, outputs, and metric definitions. Do not overwrite reference datasets or prior runs without preserving provenance.
