# Part 4 cross-check and handover

## Evidence map

| Requirement | Current evidence | Status / action before final submission |
|---|---|---|
| P4.1 Common repository | `README.md`, `requirements.txt`, `config/`, `src/`, `CONTRIBUTING.md` | In place; verify lock file matches the supported environment. |
| P4.2 Interfaces | `CONTRIBUTING.md`, `reports/PART3_REPORT.md`, `src/structured/schema.py` | Contracts consolidated; owners should confirm shared interfaces. |
| P4.3 Integrated pipeline | `src/structured/pipeline.py`, `scripts/run_pipeline.py`, Part 3 notebook, `app.py` | Model-to-validation-to-mock-tool path is available through the unified query/demo/suite CLI and Gradio app. The app also retains the prompt-only mode and RAG/non-RAG selector. |
| P4.4 Run logging | `experiments/experiment_runs.jsonl`, `config/experiment_config.yaml` | Existing logs; audit historical run IDs/revisions/settings for completeness. |
| P4.5 Consolidated results | `experiments/comparison_summary.md` and source CSV/JSON artifacts | Index and current metrics supplied; produce paired/full comparison after coverage audit. |
| P4.6 Four-criteria evaluation | Intent metrics, manual review, JSON checks, safety artifacts | Evidence exists across different runs. No single integrated four-criterion run is checked in; execute notebook and report denominators. |
| P4.7 Three failures | `experiments/failure_analysis.csv` | Three representative failures and mitigations documented; verify source row details before final report. |
| P4.8 Responsible use | `reports/README_SAFETY.md`, problem definition, Part 3 report | Expand report with hallucination, unsupported claims, privacy, leakage, and reliability controls (see review below). |
| P4.9 Reproducibility | pinned model/dataset revisions, requirements, protocol | Not verified in this checkout: `.venv` interpreter is inaccessible here. A group member must run the clean setup and notebook in a working Python 3.12 environment and record outcome. |
| P4.10 Report consolidation | `reports/PART2_REPORT.md`, `PART3_REPORT.md`, this map | Merge or explicitly cross-reference into the agreed final report and map each course requirement. |
| P4.11 Demo | `reports/demo_plan.md` | Time-boxed 9–10 minute plan supplied; rehearse and verify live model readiness. |
| P4.12 Contributions | `reports/contributions.md` | Members must fill and confirm individual rows. |
| P4.13 Final checks | this handover | Complete the remaining checks below with the group. |

## Responsible-use discussion points

- **Hallucination / unsupported claims:** generated model text can invent bank policies, phone numbers, timelines, or action completion. Constrain the assistant to general guidance; state that policy is bank-specific; only report tool-confirmed state; check outputs and retain an abstention path.
- **Privacy / data leakage:** never submit credentials or real account data. Block obvious credentials, redact sensitive values in logs, keep evaluation data public and synthetic, and avoid logging unredacted prompts. Regex detection can miss secrets; it is not a privacy guarantee.
- **Prompt injection:** customer text is untrusted data and the prompt says so; pattern checks catch only known strings. Schema validation limits executable output, and tools validate exact argument sets, but neither proves intent or authorization.
- **Reliability:** small balanced sample, model stochasticity, invalid JSON, intent errors, mock-only tools, dependency drift, and hardware/model download variability all limit conclusions. Report denominators and revisions; use human/official-channel fallback for uncertain or high-impact cases.

## Final group checklist

- [ ] Confirm each member's contract and contribution statement.
- [ ] Run from a clean Python 3.12 environment; install dependencies; execute notebook top to bottom and capture outputs.
- [ ] Run the integrated evaluation on the agreed sample and save a run ID, configuration, sample IDs, raw outputs, metrics, and failures.
- [ ] Reconcile prompt/decoding row coverage and compare only matched sample IDs.
- [ ] Confirm the presentation/report meets page/length requirements and includes links/references.
- [ ] Check repository status and remove local-only files; do not commit `.idea/`, caches, credentials, or model weights.
