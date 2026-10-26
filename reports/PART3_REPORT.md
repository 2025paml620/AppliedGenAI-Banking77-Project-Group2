# Part 3 — Structured Output, Prompt Template and Tool Interface

Run: `python run_part3.py` (writes `experiments/part3_results.csv`). Tests: `pytest tests/test_structured.py`.

## Files
| Task | File |
|------|------|
| P3.1–3.2 Prompt contract + template | `src/prompts/structured.py` (`build_structured_prompt`) |
| P3.3–3.4 JSON schema + validation | `src/structured/schema.py` (`validate_output`) |
| P3.5–3.7 Tools + schemas (mock data) | `src/tools/banking_tools.py` |
| P3.8–3.9 Flow + failure handling | `src/structured/pipeline.py` (`handle_query`) |
| P3.11 Valid/invalid examples | `tests/test_structured.py`, `run_part3.py` |

## Contract
Input: customer query + the 77 allowed labels. Output: one JSON object, exactly these keys:

| Field | Type | Rule |
|-------|------|------|
| `intent` | string | one of the 77 BANKING77 labels |
| `response` | string | non-empty, passes output guardrails |
| `action_required` | bool | true iff `action` is not null |
| `action` | null or `{"tool", "arguments"}` | tool in `block_card`, `transaction_status` |

Tools: `block_card(card_last4: 4 digits)`, `transaction_status(transaction_id: TXN + 4 digits)`.
Arguments must match the required set exactly; unknown cards/transactions raise `ToolError`.

## Flow
Query → input guardrail → model JSON → schema validation → output guardrail → tool (mock) → final response.
The final text for a tool action is built from the tool result, never from model text, so the
assistant only claims "blocked" when the mock tool confirms it.

## Failure handling
| Case | Status | Result |
|------|--------|--------|
| Credentials/injection in query | `blocked_input` | fixed fallback, model not called |
| Non-JSON / wrong keys / bad label | `invalid_output` | fixed fallback (no retry) |
| Unsafe `response` text | `unsafe_output` | safe fallback |
| Bad arguments / unknown id / unknown tool | `tool_error` | explanatory message |

## P3.10 Adaptation strategies
| Strategy | Fit here | Cost / limits |
|----------|----------|---------------|
| Prompting (used) | No training, instant iteration; schema is enforced by validation | 1.5B model may break format or pick wrong label; long 77-label prompt |
| RAG | Retrieve similar labelled queries / bank policy text to narrow labels and ground answers | Needs index + retriever; adds latency; retrieved examples must exclude the evaluation row |
| Fine-tuning / PEFT (LoRA) | Best for reliable JSON and label accuracy on 77 classes | Needs training data/GPU, versioning; risk of overfitting; not done in this project |

## RAG (implemented)
`src/rag/retriever.py` embeds the BANKING77 **train** split (all-MiniLM-L6-v2) and returns the top-k most similar
labelled queries; they are listed in the prompt as guidance (`handle_query(..., retriever=r, k=5)`, `run_part3.py --rag`).
Evaluation uses the test split, so no leakage. Compare with `python -m scripts.evaluate_rag` -> `experiments/rag_metrics.json`
(accuracy with/without RAG, invalid-JSON rate, and retrieval-only top-1 accuracy as a reference).

## Reliability notes
- Validation rejects malformed output before any action; tools re-validate arguments independently.
- Tools are mock-only, so no real transactions are possible.
- Limits: regex guardrails are not a complete defence; intent accuracy is not measured here (see Part 2 evaluation); the model may omit an action even when one applies.
