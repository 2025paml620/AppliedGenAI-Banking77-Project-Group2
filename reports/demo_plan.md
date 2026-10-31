# 8–10 minute demonstration plan

| Time | Segment | Show | Speaker cue |
|---|---|---|---|
| 0:00–1:00 | Problem and boundaries | `reports/problem_definition.md`; educational-only banner | Explain the user problem and that the prototype has no real banking access. |
| 1:00–2:00 | Data and model | Fixed 231-row sample, pinned Qwen model/revision | Explain intent labels, held-out evaluation, and model limitations. |
| 2:00–3:30 | Prompt and decoding experiments | `experiments/prompt_comparison.csv`, `decoding_comparison.csv` | State the evaluated sample sizes and settings; do not imply prompt results measure safety by themselves. |
| 3:30–5:00 | Structured output | `run_part3.py` or integrated notebook; show schema and a valid result | Trace query → safety gate → generation → validation. |
| 5:00–6:30 | Tool flow | Mock `block_card` / transaction status demo | Show tool arguments, mock result, and response grounded in that result. Mention state mutation is local mock state. |
| 6:30–8:00 | Evaluation and failures | Comparison summary and failure-analysis CSV | Cover intent correctness, JSON validity, response quality, and safety/failure behavior, including limitations. |
| 8:00–9:00 | Responsible use and reproducibility | Safety discussion and setup commands | Explain privacy controls, regex limitations, dependency/model download needs, and clean-run caveat. |
| 9:00–10:00 | Questions / handover | contribution record and remaining checklist | Identify any open items honestly. |

Keep a recorded fallback demonstration ready because first-time model loading downloads a large checkpoint and may exceed demo time. Use synthetic queries only. Confirm the final notebook and live environment before presenting.
