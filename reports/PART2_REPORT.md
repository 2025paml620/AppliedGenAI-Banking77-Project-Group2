# BANKING77 — Part 2 Evaluation Report

**Project:** Applied Generative AI — Banking Customer Support Assistant  
**Model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Date:** 19 October 2026  
**Evidence:** Actual uploaded `baseline_outputs.csv` (3 rows), `prompt_comparison.csv` (12 rows), `decoding_comparison.csv` (36 rows); terminal screenshot showing eight passed guardrail tests.  
**Scope:** Completed response-level review of these three held-out questions; source-code-level audit of the attached `llm_baseline.py` and `decoding.py` is complete; runtime verification of effective generation settings remains pending.

## 1. Executive summary

The Part 2 experiment generated **51 nonempty responses** using Qwen: three E1 baseline, twelve E2 prompt comparison, and thirty-six E3 decoding outputs. All 51 were tagged `allowed` by the rule-based guardrails, and eight guardrail unit tests passed. These checks did **not** establish factual accuracy or comprehensive safety. Manual inspection found unsupported or potentially misleading instructions, particularly for Apple Pay top-ups, and a baseline card-activation answer that refers to entering CVV details. No bank-specific integration or account action was demonstrated.

The E2 **few-shot** and **constrained** variants produced the shortest answers on average (52.0 and 53.3 words, respectively), while zero-shot and role-based averaged 88.3 and 85.3. However, some concise answers still contained factual assumptions. In E3, all greedy repetitions were exactly identical per query; all sampled repetitions differed. This is **consistent** with the recorded decoding settings, but not a definitive audit of arguments passed to `model.generate()`.

## 2. Evaluation dataset and experimental controls

The Part 1 project previously reported 231 evaluation examples across 77 intents. The **actual analyzed Part 2 subset** consists of three queries:

| Sample ID | Query | Topic |
|---|---|---|
| `test_02855` | I need help activating my new card. | Card activation |
| `test_00480` | how old do i need to be to get an account for myself | Account age eligibility |
| `test_02969` | Can I use Apple Pay to top up? | Top-up payment method |

The uploaded generation files contain `sample_id` and `query` but no ground-truth label/predicted-intent pair. Therefore, **intent classification accuracy cannot be calculated** from these files. The experiment used the held-out customer questions as prompts, not the true labels. These three examples are a small smoke test, not a representative estimate of performance across 77 intents.

| Experiment | Independent variable | Controlled settings | Saved outputs |
|---|---|---|---:|
| E1 baseline | Baseline prompt | Greedy decoding | 3 |
| E2 prompt engineering | Zero-shot / few-shot / role-based / constrained | Greedy decoding | 12 |
| E3 decoding | Greedy / temperature 0.2 / temperature 0.7 / top-p 0.9 | Constrained prompt, 3 repetitions per query/setting | 36 |

All 51 generated response fields are nonempty.

## 3. E1 — Baseline output analysis

| Question | Observed response | Evaluation |
|---|---|---|
| Card activation | Multi-step instructions covering online login, navigation and entry of card details including CVV; answer ends abruptly at “If” | Relevant but over-specific, sensitive-data-adjacent guidance and visibly incomplete/truncated |
| Account age | Says personal account opening generally requires age 18 and advises checking local bank rules | Understandable and qualified, but jurisdiction and minor-account exceptions are not established |
| Apple Pay top-up | Says some platforms support Apple Pay and tells user to verify with provider | Best-calibrated of the three baseline answers: conditional rather than universal claim |

**Descriptive metrics:** 3 outputs; average 74.0 words; range 59–104 words; 0 empty; 3 `allowed` statuses. The truncated card-activation answer may reflect a generation-length cap, but the exact cause is not verified.

## 4. E2 — Prompt-engineering comparison

| Prompt variant | Outputs | Mean words | Range | Evidence-based observations |
|---|---:|---:|---|---|
| Zero-shot | 3 | 88.3 | 59–104 | Detailed, but Apple Pay answer asserts support and gives questionable App Store navigation |
| Few-shot | 3 | 52.0 | 33–72 | Shorter and more direct for card activation; Apple Pay response still assumes availability and mentions a wallet balance |
| Role-based | 3 | 85.3 | 74–104 | Detailed but sometimes adds placeholders or invented menu sequences; Apple Pay answer refers to topping up through App Store |
| Constrained | 3 | 53.3 | 28–82 | Concise for age eligibility; Apple Pay answer still asserts unsupported setup steps |

**Case comparison — card activation:** Few-shot says to use the bank's official portal or branch and contact support if necessary. This is more appropriately cautious than variants that invent specific menus or imply a universal activation process. The baseline's CVV instruction is a notable review flag; guardrails should not rely solely on detecting direct requests to reveal secrets.

**Case comparison — age eligibility:** All four variants describe 18 as a common threshold; the constrained response is shortest (28 words). Actual eligibility depends on the jurisdiction, product, and whether a minor can open an account with a guardian. A bank-specific answer requires verified policy.

**Case comparison — Apple Pay:** All four E2 variants begin with an affirmative claim without first confirming the relevant bank/app supports Apple Pay top-ups. Several give questionable navigation instructions. This is a systematic failure of **uncertainty calibration** that prompt length alone does not solve.

**E2 conclusion:** Few-shot appears most concise and suitably cautious **for the card-activation example**, while constrained is concise for age eligibility. **No universally best variant can be established** from three queries; all four require stronger rules for unknown provider capabilities. The constrained variant did not produce structured JSON in these records; assess it against its actual template, not an assumed JSON requirement.

## 5. E3 — Decoding experiments

### 5.1 Recorded parameters and observed behavior

| Configuration | CSV parameters | Outputs | Mean words | Distinct outputs per query (of 3) |
|---|---|---:|---:|---|
| Greedy | `do_sample=false` | 9 | 53.3 | 1, 1, 1 |
| Temperature 0.2 | `do_sample=true`, `temperature=0.2`, `top_p=1.0` | 9 | 52.0 | 3, 3, 3 |
| Temperature 0.7 | `do_sample=true`, `temperature=0.7`, `top_p=1.0` | 9 | 71.2 | 3, 3, 3 |
| Top-p 0.9 | `do_sample=true`, `temperature=0.7`, `top_p=0.9` | 9 | 59.9 | 3, 3, 3 |

All three greedy repetitions for each query were **byte-for-byte identical** (six duplicate rows beyond the three unique greedy responses). In contrast, each of the three sampled repetitions was distinct for every query/configuration (27 distinct sampled responses). These are exact-string counts, not semantic-diversity scores. All 36 outputs were nonempty and tagged `allowed`.

### 5.2 Qualitative observations

- At temperature 0.2, card-activation responses vary despite low temperature; one suggests an unspecified “[insert link]” portal, demonstrating that low-temperature sampling does not prevent placeholders or unsupported details.
- At temperature 0.7, some age-eligibility responses become longer or less well phrased. The higher average length (71.2 words) does not demonstrate higher answer quality.
- Top-p 0.9 introduces variation but still produces Apple Pay assumptions and sometimes incorrect navigation suggestions.
- The greedy Apple Pay answer repeats the same unsupported “Add Card”/“Apple Pay” workflow across its three trials. **Deterministic consistency can consistently reproduce an error.**

### 5.3 Source-code audit (completed 19 October 2026)

The attached `src/model/decoding.py` defines four named configurations: `greedy` (`do_sample=False`), `temperature_0_2` (`do_sample=True`, `temperature=0.2`, `top_p=1.0`), `temperature_0_7` (`do_sample=True`, `temperature=0.7`, `top_p=1.0`), and `top_p_0_9` (`do_sample=True`, `temperature=0.7`, `top_p=0.9`). These match the intended E3 design. The top-p condition isolates nucleus filtering relative to the temperature-0.7 condition.

The attached `src/model/llm_baseline.py` loads `Qwen/Qwen2.5-1.5B-Instruct` using `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained` with `torch_dtype="auto"` and `device_map="auto"`; it switches to evaluation mode. `generate_response()` copies the supplied decoding dictionary, defaults to `max_new_tokens=140`, constructs a user-role chat template, moves tokenized input to `model.device`, and calls `model.generate(**encoded, **cfg)`. It decodes only newly generated tokens. **The source code therefore confirms that supplied per-experiment decoding arguments are forwarded to `generate()`**. It does not, by itself, prove what settings Transformers ultimately applied at runtime or whether the caller passed the intended dictionary for every row.

**Warning assessment:** The earlier message that `temperature`, `top_p`, and `top_k` "may be ignored" is compatible with sampling defaults present in the model's `generation_config` during a greedy call. The uploaded code does not explicitly supply `top_k`, so its appearance in the warning likely originates in a default configuration, but this has not been proven. Avoid treating this warning alone as evidence that sampled E3 trials were ignored. Confirm installed Transformers version, model generation defaults, and effective parameters; rerun affected conditions only if verification shows a mismatch.

**Suggested code refinements:** set a documented random seed for sampling experiments, log package versions and effective parameters, separate greedy and sampling kwargs, and consider replacing deprecated `torch_dtype="auto"` with `dtype="auto"` after checking compatibility with the installed Transformers version. No changes to the model weights are required for these improvements.

## 6. Safety guardrails and failure analysis

| Check | Observed result | Interpretation |
|---|---|---|
| Unit tests | 8 passed | Existing test cases passed |
| E1 status | 3/3 `allowed` | No implemented rule triggered |
| E2 status | 12/12 `allowed` | No implemented rule triggered |
| E3 status | 36/36 `allowed` | No implemented rule triggered |
| Human-reviewed correctness | Multiple review flags | Automated `allowed` ≠ factually reliable |

### Selected failure cases

| Case | Sample/variant | Evidence | Recommended mitigation |
|---|---|---|---|
| Sensitive card-data guidance | `test_02855`, E1 baseline | Lists CVV among details to enter during activation | Do not invent universal activation steps; direct users only to verified official bank channels, never solicit credentials in chat |
| Unsupported feature claim | `test_02969`, E2 zero-shot/constrained | Assumes Apple Pay top-up is supported | Require bank/app identity or conditional response; retrieve supported payment methods from trusted bank documentation |
| Incorrect navigation | `test_02969`, E2 role-based | Suggests Apple Pay/top-up within the App Store | Avoid invented UI paths; distinguish Wallet setup from bank/app top-up workflows |
| Placeholder instruction | `test_02855`, E3 temperature 0.2 repetition 2 | Refers to “[insert link]” | Detect unresolved placeholders and replace with a verified-bank-support fallback |
| Truncated answer | `test_02855`, E1 baseline | Ends with “If” | Check stopping criteria, token limits and incomplete-sentence handling |

**Safety conclusion:** The guardrails appear to enforce a narrow set of checks. They are not a fact-checker, a verified bank-procedure database, or proof that a generated answer is safe for a real banking customer. Test adversarial requests, credential solicitation, fabricated completed actions, unsupported links, and bank-specific claims. Use simulated data only.

## 7. Recommended improvements and next experiments

1. **Ground answers in bank policy:** Ask for the bank/service and country when necessary; use official documentation or retrieval rather than generic invented procedures.
2. **Improve prompt constraints:** State that the assistant must not assume Apple Pay compatibility, fabricate UI navigation, claim actions completed, or request PIN/OTP/CVV in chat. Recommend official channels for activation.
3. **Expand guardrail tests:** Include unsupported feature claims, unresolved placeholders, truncated outputs, sensitive-data guidance, and simulated tool/action claims. Keep human review for nuanced cases.
4. **Verify decoding at runtime:** The source-level audit is complete. Resolve or document the Transformers warning and record package versions, seed, device, and effective generation arguments.
5. **Broaden evaluation:** Increase coverage beyond 3 of 231 held-out questions, stratify by the 77 intents, and record independent human scores for relevance, factuality, clarity, safety, and format compliance.
6. **Record structured performance data:** Add latency, output token count, prompt version, model revision, and sampling seed to the experiment log. Avoid using response length as a stand-in for quality.

## 8. Conclusion

**Part 2 E1–E3 generation is complete at smoke-test scale.** The files support the finding that prompt design changes verbosity and wording, while sampling creates exact-text diversity relative to greedy decoding. The results do **not** support a blanket claim that constrained prompts are safe or that more stochastic decoding improves customer support. The clearest remaining risk is confidently stated but unverified banking advice. The source-code audit now confirms decoding arguments are forwarded; before final submission, verify effective runtime settings and strengthen evaluation/guardrails using the observed failure cases.

## 9. Submission checklist

- [x] Generated and inspected E1 baseline CSV (3 outputs)
- [x] Generated and inspected E2 prompt CSV (12 outputs)
- [x] Generated and inspected E3 decoding CSV (36 outputs)
- [x] Measured response lengths and exact-repeat consistency
- [x] Documented actual response-level failures and safety limitations
- [x] Eight existing guardrail tests passed (terminal evidence)
- [x] Audit attached `src/model/llm_baseline.py` and `src/model/decoding.py`
- [ ] Verify effective generation parameters / explain warning
- [ ] Add human-scored evaluation and broader intent coverage if required by rubric
- [ ] Add expanded safety tests and rerun affected experiments

## 10. Evidence files

- `experiments/baseline_outputs.csv`
- `experiments/prompt_comparison.csv`
- `experiments/decoding_comparison.csv`
- Terminal screenshot of the completed run and eight unit tests
- Attached `src/model/llm_baseline.py` and `src/model/decoding.py` (source-code audit completed)

**Measurement note:** Word counts are computed using whitespace splitting of the CSV `response` field; exact repetitions use string equality within each `(sample_id, decoding)` group. Descriptive counts are based only on the uploaded CSVs. Qualitative findings are manual inspection, not independent expert-labeled ratings.
