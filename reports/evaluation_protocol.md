# Banking77 Evaluation Protocol

## 1. Dataset

- Dataset: PolyAI Banking77
- Training split: 10,003 examples
- Test split: 3,080 examples
- Total examples: 13,083
- Number of intent classes: 77
- Input field: `text`
- Ground-truth field: `intent` (human-readable intent name derived from the dataset's `ClassLabel` feature)

The dataset is loaded from the public Hugging Face repository `PolyAI/banking77`. The current implementation uses `datasets==3.6.0` and `trust_remote_code=True`.

**Security and reproducibility:** `trust_remote_code=True` allows Python code supplied by the dataset repository to execute. Review the repository code before running it. For the final submission, record the reviewed dataset revision and relevant dependency versions.

## 2. Evaluation Sample

- Source: Official Banking77 test split
- Samples per intent: 3
- Total evaluation examples: 231
- Number of intents represented: 77
- Random seed: 42
- Artifact: `data/evaluation_sample.csv`
- Selection: Group the test split by integer label and sample three examples per intent using a fixed random seed.
- Output fields: `sample_id`, `source_index`, `text`, `label`, and `intent`.

The sample was verified to contain 231 rows, 77 unique intents, three examples per intent, and no missing values.

This is a fixed development evaluation sample. Results on this sample must not be presented as equivalent to results on the complete 3,080-example test split. Equal representation across intents also differs from the natural frequency of customer queries in a real banking environment.

## 3. Data Separation and Leakage Prevention

- Use the official training split for few-shot demonstrations.
- Use the fixed evaluation sample for model evaluation.
- Do not include the correct `label` or `intent` of an evaluation example in that example's prompt.
- Keep ground-truth labels available for scoring and error analysis only.
- Use the same evaluation examples across comparable baseline, prompt, and decoding experiments.
- Do not select or remove evaluation examples based on model performance.
- Do not silently regenerate the evaluation sample during experiments.

The ground-truth label columns in `data/evaluation_sample.csv` are reference annotations. They must not be passed to the model when it predicts the intent for the corresponding query.

## 4. Evaluation Metrics

Metrics are reported separately because they measure different properties of the application.

### 4.1 Intent accuracy

Accuracy is the proportion of evaluated examples for which the predicted intent exactly matches the ground-truth intent.

`Accuracy = correct intent predictions / total evaluated examples`

### 4.2 Macro-F1

Macro-F1 is the unweighted mean of the F1 scores across the 77 intent classes. Each class contributes equally to the average.

Accuracy and Macro-F1 will use the same ground-truth labels and prediction set. Report the evaluation sample size alongside the results.

### 4.3 JSON validity rate

JSON validity rate is the proportion of generated outputs that both parse as JSON and conform to the agreed output schema.

The schema will define required fields, field types, the allowed intent values, and how an absent suggested action is represented. JSON that parses successfully but violates the schema counts as invalid.

`JSON validity rate = valid schema-conforming outputs / total outputs evaluated`

### 4.4 Response quality

Response quality is assessed separately from intent accuracy. The review considers:

- Relevance to the customer's query.
- Consistency with the predicted intent.
- Clarity and helpfulness.
- Unsupported claims or invented banking policies.

The team must document the review rubric, scoring scale (if numerical scores are used), and examples reviewed. Response quality must not be inferred from intent accuracy alone.

### 4.5 Safety and error analysis

Record observed failures, including:

- Incorrect intent predictions.
- Irrelevant, misleading, or unsupported responses.
- Unsupported banking claims or invented policies.
- Unsafe or inappropriate suggested actions.
- Invalid or incomplete structured outputs.
- Incorrect tool selection or invalid tool arguments.

Document at least three representative failure cases, their likely causes, and proposed mitigations. A review of selected cases does not prove that all unsafe behaviour has been detected.

### 4.6 Tool-call correctness

For scenarios where a tool is appropriate, assess whether the selected tool, arguments, and simulated result match the intended operation and satisfy the interface requirements.

No real banking operation will be executed. A successful simulated tool call does not establish real-world authorization or transaction capability.

## 5. Comparison Protocol

1. Use the same 231 evaluation examples for all comparable prompt and decoding experiments.
2. Supply each example's query and only permitted context to the model; do not supply its correct evaluation label.
3. Use training-split examples for few-shot demonstrations.
4. Apply the same prediction-normalization and scoring rules across comparable variants.
5. Record the experiment ID, model identifier/revision, prompt version, decoding parameters, and random seed where applicable.
6. Preserve raw model outputs, predictions, and evaluation results.
7. Compare metrics on the same examples and inspect representative successes and failures.
8. Change one experimental factor at a time where possible. If multiple factors change, document the differences.
9. Do not silently regenerate or selectively change the evaluation sample based on model performance.

Planned prompt comparisons include zero-shot, few-shot, role/context-based, and constraint-based prompts. Planned decoding comparisons include greedy decoding and sampling configurations using temperature and at least one of top-k or top-p. Record the actual settings used for each run.

## 6. Reference Evaluation File

The reference file is `data/evaluation_sample.csv`.

| Field | Meaning |
|---|---|
| `sample_id` | Identifier derived from the original test-row index |
| `source_index` | Original row position in the loaded test split |
| `text` | Customer query |
| `label` | Ground-truth integer label |
| `intent` | Human-readable ground-truth intent |

The CSV is the shared reference artifact for comparable experiments. Preserve it and avoid overwriting it casually. Commit it only if permitted by the dataset licence and course requirements.

For strict reproducibility, record the dataset revision and relevant dependency versions. The seed alone does not guarantee identical examples if the dataset contents, row ordering, or sampling-library behaviour changes.

## 7. Reproducibility

Run the sample-preparation script from the project root:

```powershell
# Regenerate the balanced evaluation sample using the project's preparation pipeline.
python -m scripts.prepare_evaluation_data
```

For each experiment, save the raw outputs and record the model identifier/revision, prompt version, decoding settings, random seed where applicable, evaluation sample, metrics, and observed failures.

Do not silently replace the committed reference CSV during routine experiments. If the sample must be changed, document the reason, create a new version, and communicate the change to the team.

## 8. Limitations

- The sample contains only three examples per intent, so estimates may be sensitive to the selected examples.
- The balanced sample does not reflect the natural frequency of intents in a real banking environment.
- Intent accuracy does not establish that a generated response is relevant, factually correct, helpful, or safe.
- JSON validity establishes structural conformance, not semantic correctness.
- Automated metrics do not fully capture response quality or responsible behaviour.
- Reviewing selected failure cases cannot prove that all unsafe behaviour has been detected.
- Simulated tool calls do not demonstrate real-world authorization or transaction capability.
- Model outputs can vary with decoding parameters, random seeds, model revisions, and software versions.

Every reported result must state whether it was measured on the 231-example subset or the full official test split.

## 9. Related Documentation

- [Main Project README](../README.md)
- [Problem Definition](problem_definition.md)
- [Dataset Notes](../data/DATASET_NOTES.md)
- [Dataset Preparation Script](../scripts/prepare_evaluation_data.py)
- [Dataset Utilities](../src/data/dataset.py)
