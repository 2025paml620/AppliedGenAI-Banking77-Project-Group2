# Consolidated experiment comparison

This index consolidates the existing Part 2 evidence. The source files preserve row-level results; the summary below reports only metrics that are currently recorded and does not imply an end-to-end evaluation of Part 3.

| Experiment | Evidence | Coverage / recorded result | Caveat |
|---|---|---|---|
| Baseline outputs | `baseline_outputs.csv` | Per-query generated responses and guardrail status | Historical run metadata may not include every pinned revision/config detail. |
| Prompt comparison | `prompt_comparison.csv` | 4 prompt variants; 41 data rows (header excluded) | Different variants may have unequal query coverage; inspect IDs before claiming paired comparison. |
| Decoding comparison | `decoding_comparison.csv` | 105 data rows; includes repetition and parameters | Preserve actual effective settings when summarizing. |
| Intent classification | `intent_evaluation.json`, `intent_predictions.csv` | 77 samples; 35 correct (45.45%); 75 valid outputs (97.40%); accuracy among valid outputs 46.67% | Small balanced sample; not full official test-set performance. |
| Zero-shot verification | `p2_5_zero_shot.csv` | README records 2/3 correct and strict JSON 0/3 | Three examples only. |
| Response review | `part2_evaluation/manual_review_completed_by_prompt.csv` | Human-scored sample | Apply and report its rubric; do not combine subjective scores with intent accuracy. |
| Integrated structured pipeline | `run_part3.py`, `experiments/part3_results.csv` (generated on execution) | Includes JSON validation and mock tool outcomes | No checked-in end-to-end metric summary was found at this handover; run it and preserve the results before asserting integrated metrics. |

## Interpretation

Intent correctness, JSON validity, response quality, and safety/failure behavior are separate criteria. Existing artifacts support intent and selected manual response-quality analysis. P2.5 documents strict-JSON failure, and the logs contain an unsafe credential instruction missed by the safety rule. The checked-in evidence does not yet provide a consistent, same-run four-criterion table for the integrated pipeline. Use the executable notebook to generate that table on the fixed evaluation sample, then update this document with run ID, sample size, and measured values.
