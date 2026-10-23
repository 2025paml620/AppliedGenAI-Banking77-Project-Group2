# Integrated demonstration plan (8–10 minutes)

1. **Scope and data (1 min):** Explain Banking77, the fixed 231-row balanced evaluation sample, and that labels are scoring-only.
2. **Model and experiments (2 min):** Identify Qwen2.5-1.5B-Instruct and show recorded baseline, four prompt strategies, and decoding settings/results. Populate this section only after comparable runs complete.
3. **Structured response (1 min):** Show `src/output/schema.py`, one valid JSON response, and invalid JSON/unknown-intent fallback.
4. **Tool flow (2 min):** Submit a lost-card query; show proposed action, strict argument validation, local `demo-card-001` simulation result and final wording. Show an unknown-reference error.
5. **Evaluation and failures (2 min):** Report intent accuracy, macro-F1, JSON validity, and safety rate for the same sample; review at least three cases from `reports/failure_analysis.csv`. Response quality is reviewed separately by people.
6. **Responsible use and limitations (1 min):** No real banking connection, no credentials, no real transactions; discuss unsupported claims, prompt injection, sample size and output variability.

Before presenting, replace pending experiment references with actual run IDs and results. Do not present a planned metric as measured evidence.
