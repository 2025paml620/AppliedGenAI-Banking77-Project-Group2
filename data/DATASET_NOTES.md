# Banking77 Dataset — Inspection and Preparation Notes

## 1. Purpose

This document records the Banking77 dataset characteristics, preprocessing decisions, evaluation-sample construction, and data-leakage controls for the Applied Generative AI group project.

The reusable implementation is in `src/data/dataset.py`, and the sample-generation script is `scripts/prepare_evaluation_data.py`.

## 2. Dataset Source

- Dataset name: Banking77
- Hugging Face repository: [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)
- Task: Classify natural-language banking queries into banking-related intents.
- Number of intent classes: 77

The dataset is loaded using the Hugging Face `datasets` library. The current implementation uses `datasets==3.6.0` and `trust_remote_code=True`.

**Security and reproducibility:** `trust_remote_code=True` allows Python code from the dataset repository to execute. Review the repository code before running it. For the final submission, record the reviewed dataset revision and dependency versions.

## 3. Dataset Structure

The loaded dataset contains two official splits:

| Split | Examples |
|---|---:|
| Training | 10,003 |
| Test | 3,080 |
| Total | 13,083 |

The dataset has two source features:

| Feature | Type | Description |
|---|---|---|
| `text` | String | Natural-language customer query |
| `label` | ClassLabel | Integer identifier for the correct intent |

The `ClassLabel` feature provides the mapping from integer labels to their human-readable intent names. For example, label ID `11` maps to `card_arrival`.

During local inspection, all 77 intent classes were present in both official splits.

## 4. Class Distribution

### Training split

The training split contains 10,003 examples across 77 intents.

- Minimum examples for an intent: 35
- Maximum examples for an intent: 187
- Mean examples per intent: approximately 129.91

The training distribution is not uniform.

### Test split

The test split contains 3,080 examples.

- Examples per intent: 40
- Number of intents: 77

The official test split is balanced across the 77 classes.

These counts were observed during local dataset inspection. The complete intent-count table is available in the data-exploration notebook.

## 5. Representative Queries

Examples observed during inspection include:

| Customer query | Correct intent |
|---|---|
| `I am still waiting on my card?` | `card_arrival` |

This example illustrates how a natural-language query maps to a dataset intent. It does not imply that every query about a card has the same intent.

## 6. Evaluation Sample

### Selection procedure

The evaluation sample is generated from the official test split, not the training split.

| Setting | Value |
|---|---|
| Source split | `test` |
| Samples per intent | 3 |
| Number of intents | 77 |
| Total samples | 231 |
| Random seed | 42 |
| Output file | `data/evaluation_sample.csv` |

The preparation script groups test examples by integer label and selects three examples from each group using pandas sampling with `random_state=42`. The resulting rows are sorted by label and original source index.

### Output columns

| Column | Meaning |
|---|---|
| `sample_id` | Identifier derived from the original test-row index |
| `source_index` | Original row position in the loaded test split |
| `text` | Customer query |
| `label` | Ground-truth integer intent ID |
| `intent` | Human-readable ground-truth intent |

The generated CSV was verified locally to contain 231 rows, 77 unique intents, three examples per intent, and no missing values.

### Why use a subset?

The 231-example subset is a manageable, balanced sample for initial prompt and decoding experiments. It reduces inference time while ensuring every intent is represented.

It is a development evaluation sample, not a substitute for evaluating on the complete 3,080-example test split. Conclusions should reflect this limitation.

## 7. Preprocessing Decisions

The current preparation pipeline makes the following decisions:

1. Load the official train and test splits.
2. Convert the test split into a pandas DataFrame.
3. Preserve each test row's original index as `source_index`.
4. Map integer labels to their human-readable intent names using the dataset's `ClassLabel` metadata.
5. Select three examples per intent with a fixed random seed of 42.
6. Sort the selected examples by label and source index.
7. Create stable sample identifiers based on the original test-row index.
8. Save the required fields to `data/evaluation_sample.csv`.

No text normalization, stop-word removal, stemming, or other query-text transformations are performed by this sampling pipeline.

No examples are intentionally excluded beyond selecting the fixed subset. The saved CSV contains the original query text and its ground-truth label.

## 8. Evaluation Leakage Prevention

The following rules apply to all model experiments:

- Use training-split examples for few-shot demonstrations.
- Do not use the correct label of an evaluation query in that query's prompt.
- Keep ground-truth labels available for scoring and analysis.
- Use the same fixed evaluation sample for comparable prompt and decoding experiments.
- Do not select evaluation examples based on model performance.
- Do not silently regenerate the sample between experiments.

The ground-truth `label` and `intent` columns are evaluation references. They are not inputs to the model when predicting the intent for the corresponding query.

## 9. Reproducibility

### Dataset and sampling configuration

- **Dataset:** `PolyAI/banking77`
- **Pinned revision:** `90d4e2ee5521c04fc1488f065b8b083658768c57`
- **Source split:** Official `test` split
- **Sampling strategy:** 3 examples per intent, using random seed `42`
- **Preserved evaluation artifact:** `data/evaluation_sample.csv`

The dataset revision is explicitly pinned in `src/data/dataset.py` so future runs use the same repository snapshot rather than silently loading a newer revision.

### Environment and dependencies

- **Python:** `3.12.9`
- **datasets:** `3.6.0`
- **pandas:** `3.0.6`
- **numpy:** `2.5.4`
- **Full dependency snapshot:** `requirements-lock.txt`

The lock file records the installed package versions used in the current environment. Use a compatible Python environment when recreating the setup.

### Regenerate and verify the evaluation sample

Run the preparation script from the project root:

```powershell
# Regenerate and save the balanced evaluation sample using the pinned dataset revision.
python -m scripts.prepare_evaluation_data
```

To verify reproducibility without overwriting the preserved CSV, run:

```powershell
# Regenerate the sample in memory and compare it with the preserved evaluation CSV.
python -m scripts.verify_evaluation_sample
```

The verification script compares the regenerated sample with the preserved CSV, including the selected examples and reference labels. The check passed in the current environment: 231 rows, 77 intents, and 3 examples per intent.

### Reproducibility considerations

A fixed random seed alone does not guarantee identical samples if the dataset contents, ordering, or sampling-library behaviour changes. Pinning the dataset revision and recording dependency versions helps reduce these differences.

The preserved `data/evaluation_sample.csv` should be shared with the team so all experiments use the same evaluation examples and reference labels. Do not overwrite this reference file when performing the verification check.

The successful verification establishes reproducibility with the pinned revision and current environment. It does not independently establish which revision was originally used to create the CSV.

## 10. Limitations

- Three examples per intent provide limited evidence about performance within each class.
- Balanced evaluation sampling does not reproduce the natural frequency of intents in a real banking support environment.
- Correct intent classification does not establish that the generated customer response is correct, helpful, or safe.
- Results on this subset should not be presented as results on the complete official test split.
- The original dataset labels are used as reference annotations; generated responses require separate evaluation.

## 11. Related Files

- [Project README](../README.md)
- [Problem Definition](../reports/problem_definition.md)
- [Evaluation Protocol](../reports/evaluation_protocol.md)
- [Data Exploration Notebook](../notebooks/data_exploration.ipynb)
- [Dataset Utilities](../src/data/dataset.py)
- [Sample Preparation Script](../scripts/prepare_evaluation_data.py)