# Applied Gen AI Group Project — Banking77

## 1. Project Overview

This project is part of the BITS Pilani Applied Generative AI course.

We are building an educational banking customer-support assistant using a pretrained causal language model. The application will identify a customer's intent, generate a relevant and safe response, return structured output, and demonstrate a simulated tool/function call.

The project investigates how prompt design, context, and decoding controls influence application behaviour; it also documents potential retrieval and lightweight adaptation options.

Our implementation follows a learning-by-doing approach: understand the necessary concepts, implement the workflow, run controlled experiments, analyse failures, and document the results.

## 2. Quick Navigation

Use this section as the main entry point for navigating the project.

| Resource | Description |
|---|---|
| [Project Overview](#1-project-overview) | Understand the problem and project goals |
| [Project Decisions](#3-project-decisions) | Review the dataset, model, and development environment |
| [Dataset and Evaluation](#4-dataset-and-evaluation-sample) | Understand Banking77 and the evaluation sample |
| [Evaluation Protocol](reports/evaluation_protocol.md) | Sampling strategy, metrics, reproducibility, and evaluation rules |
| [Data Exploration Notebook](notebooks/data_exploration.ipynb) | Inspect the dataset and independently verify data |
| [Dataset Utilities](src/data/dataset.py) | Reusable dataset loading and preparation functions |
| [Evaluation Preparation Script](scripts/prepare_evaluation_data.py) | Generate the reproducible evaluation sample |
| [Requirements](requirements.txt) | Review Python dependencies |
| [Current Progress](#12-current-progress) | Check completed and pending work |
| [Problem Definition](reports/problem_definition.md) | Target user, input/output contract, and safety boundaries |
| [Dataset Notes](data/DATASET_NOTES.md) | Dataset statistics, label mapping, preprocessing, evaluation sampling, and leakage prevention |
| [Part 2 — Qwen Experiments](#8-part-2--qwen-llm-baseline-prompt-engineering-and-decoding) | Baseline, prompting, decoding, safety, and results |
| [P2.2 — Model Configuration](#822-p22--model-configuration-and-reproducibility) | Recorded model revision, tokenizer, software, and device |
| [P2.3 — Model Architecture](#823-p23--model-architecture-and-autoregressive-generation) | Tokenization, attention, language-model head, and generation |
| [P2.5 — Zero-Shot Verification](#814-p25--zero-shot-intent-and-response-verification) | Three-intent experiment and observed formatting failures |
| [P2.13 — Adaptation Options](#815-p213--prompting-vs-rag-vs-peft) | Technical comparison and project recommendation |
| [Part 2 Completion Matrix](#816-part-2-requirement-completion-matrix) | Verified evidence and remaining actions |
| [Part 2 — P2.12 Evaluation](#813-intent-classification-and-model-evaluation-p212) | Intent correctness, manual response quality by prompt, variability, and formatting |
| [Gradio Demo](app.py) | Interactive educational banking assistant |
| [Model Test Suite](reports/model_test_report.md) | Synthetic cases for both execution paths |
| [Integrated Group Notebook](notebooks/group_project.ipynb) | End-to-end evaluation and run artifacts |
| [Part 4 Handover](reports/PART4_HANDOVER.md) | Requirement evidence, gaps, and final checks |
| [Run Guide](reports/RUN_GUIDE.md) | Individual commands for tests, demos, evaluation, and failure review |
| [Experiment Comparison](experiments/comparison_summary.md) | Consolidated experiment evidence and caveats |
| [Failure Analysis](experiments/failure_analysis.csv) | Representative failures and mitigations |
| [Demo Plan](reports/demo_plan.md) | 8–10 minute demonstration outline |
| [Contribution Record](reports/contributions.md) | Member confirmation template |
| [Contribution Guidance](CONTRIBUTING.md) | Shared interface and repository conventions |
| [Model Test Suite](reports/model_test_report.md) | Run current pipeline cases and review outputs |

### Project workstreams

The project is divided into four parts. The detailed responsibilities are listed in [Team Responsibilities](#10-team-responsibilities).

- **Part 1:** Problem definition, dataset preparation, and evaluation design.
- **Part 2:** Pretrained LLM baseline, prompt engineering, and decoding experiments.
- **Part 3:** Prompt templates, structured output, tool/function interface, and adaptation analysis.
- **Part 4:** Integration, experiment consolidation, safety analysis, and final delivery.

As implementation progresses, links to the relevant experiment logs, notebooks, scripts, and reports should be added here.

## 3. Project Decisions

| Item | Decision |
|---|---|
| Dataset | Banking77 |
| Dataset repository | `PolyAI/banking77` |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Development environment | VS Code |
| Python version | Python 3.12 (verified Part 2 runtime: 3.12.10) |
| Environment management | Python `venv` |
| Initial dataset library | `datasets==3.6.0` |
| Notebook environment | Jupyter notebook |
| Version control | Git |

The selected model has been evaluated in initial Part 2 prompt, decoding, and intent-classification experiments.

## 4. Dataset and Evaluation Sample

### 4.1 Dataset overview

Banking77 contains natural-language banking customer queries labelled with one of 77 banking-related intents.

The dataset loaded successfully with the following official splits:

| Split | Number of examples |
|---|---:|
| Training | 10,003 |
| Test | 3,080 |
| Total | 13,083 |

The available fields are:

- `text`: the customer's query.
- `label`: the integer ID corresponding to an intent.
- `intent`: the human-readable intent name derived from the dataset's label mapping.

For example, label ID `11` corresponds to `card_arrival`.

### 4.2 Evaluation sample

We created and verified a fixed evaluation sample from the official test split.

| Property | Value |
|---|---|
| Source | Official Banking77 test split |
| Evaluation examples | 231 |
| Intent classes | 77 |
| Examples per intent | 3 |
| Random seed | 42 |
| Saved artifact | `data/evaluation_sample.csv` |
| Missing values | None found during verification |

The sample is balanced across all 77 intents and is intended for initial model and prompt experiments.

The sample was generated using the reusable dataset utilities and the script in `scripts/prepare_evaluation_data.py`.

### 4.3 Evaluation rules

- Use the training split for few-shot demonstrations.
- Use the fixed evaluation sample for comparable experiments.
- Do not include the correct intent labels of evaluation examples in prompts.
- Use the same evaluation examples when comparing prompt strategies and decoding settings.
- Record model identifiers, prompt versions, decoding parameters, and random seeds.
- Interpret results in light of the small evaluation sample.

The 231-example sample is a development evaluation set, not a replacement for the complete 3,080-example test split.

See the [Evaluation Protocol](reports/evaluation_protocol.md) for the detailed procedure and metrics.

## 5. Project Scope and Deliverables

The planned application will:

1. Identify a customer's intent from a text query.
2. Generate a concise, safe customer-facing response.
3. Return a machine-readable structured output.
4. Demonstrate at least one simulated banking tool/function.
5. Compare prompt variants and decoding strategies.
6. Evaluate intent correctness, response quality, structured-output validity, and safety-related failures.
7. Discuss prompting, retrieval-augmented generation (RAG), and fine-tuning/parameter-efficient fine-tuning (PEFT).

The project deliverables include a technical report, executable implementation, experiment records, evaluation and failure analysis, a presentation/demo, and individual contribution statements.

This is an educational prototype, not a production banking service. It will not access real bank accounts, process financial transactions, or use real customer credentials.

## 6. Development Setup

### 6.0 Run the pipeline checks

After installing dependencies, run the deterministic end-to-end checks (they use a stub model response and mock tools, so they do not download model weights):

```powershell
python -m pytest tests
```

The end-to-end checks cover safe prompt construction, structured validation, mock tool execution, invalid JSON rejection, and input/output guardrail failures. Set `BANKING77_LOG_LEVEL=DEBUG` to increase console logging; customer query text and tool arguments are intentionally excluded from application logs.

Use the unified command `python -m scripts.run_pipeline --mode suite` for the 90-case model-backed evaluation. Select `--mode query` or `--mode demo` for one query or the eight Part 3 demonstration prompts; add `--rag` to any mode. All runs use the same structured pipeline and save timestamped CSV, Markdown, and JSON artifacts under `experiments/model_test_runs/`. Use `--mode suite --limit 5` for a short smoke run. See [RUN_GUIDE.md](reports/RUN_GUIDE.md) for commands and mode details.

### 6.1 Prerequisites

- Python 3.12
- Git
- Visual Studio Code
- Python and Jupyter extensions for VS Code

### 6.2 Create and activate the virtual environment

From the project root, run:

```powershell
# Create and activate the project's isolated Python environment.
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use a temporary process-scoped execution-policy adjustment or invoke the environment's Python executable directly.

### 6.3 Install dependencies

```powershell
# Install the dependencies pinned or listed in requirements.txt.
python -m pip install -r requirements.txt
```

Select the project's `.venv` interpreter as the Python environment and notebook kernel in VS Code.

### 6.4 Run the evaluation preparation script

Run this command from the project root:

```powershell
# Generate the balanced evaluation sample and save it under data/.
python -m scripts.prepare_evaluation_data
```

The expected summary is 231 examples, 77 unique intents, and 3 examples per intent.

### 6.5 Notebook working directory

The notebook in `notebooks/` may use a different working directory from the project root.

For example, when the notebook's working directory is the `notebooks/` folder, load the evaluation CSV with:

```python
# Load the evaluation CSV from the project root when running inside notebooks/.
import pandas as pd

evaluation_df = pd.read_csv("../data/evaluation_sample.csv")
```

The Python script should continue to be run from the project root using the command above.

### 6.6 Launch the interactive assistant

From the project root, run `python app.py`. Choose **Prompted response** for intent classification with prompt and decoding controls, or **Structured response + tools** for validated JSON and simulated tools (`card_status`, `block_card`, and `transaction_status`). Select **RAG** or **Non-RAG** under **Knowledge retrieval**; RAG retrieves examples from the Banking77 training split. Choose a sample prompt to fill **Customer Query**. A prominent warning identifies the educational-only boundary, while input guardrails block obvious credential exposure, prompt injection, and direct requests to hack banking systems or steal customer funds. The result panel shows the complete response object, and the highlighted tool panel shows the suggested action, arguments, and mock result when applicable.

## 7. Loading Banking77 and Security

The dataset was loaded from Hugging Face using:

```python
# Load the public Banking77 dataset with its custom loading script enabled.
from datasets import load_dataset

dataset = load_dataset(
    "PolyAI/banking77",
    trust_remote_code=True,
)
```

**Security note:** `trust_remote_code=True` permits execution of Python code supplied by the dataset repository. Review and trust the repository code before running it. For the final reproducible submission, record the reviewed dataset revision and the package versions used.

The Hugging Face cache stores downloaded files locally. An access token is not currently required for this public dataset, although authenticated access may provide higher rate limits.

## 8. Part 2 — Qwen LLM Baseline, Prompt Engineering and Decoding

### 8.1 Objective

Part 2 implements a pretrained causal language model for the BANKING77 banking customer-support assistant.

The objectives are to:

- Establish a baseline using Qwen2.5-1.5B-Instruct.
- Compare zero-shot, few-shot, role-based and constraint-based prompts.
- Compare greedy decoding, temperature sampling and top-p sampling.
- Record generated responses and experiment configurations.
- Evaluate response relevance, clarity, safety and consistency.
- Provide reusable LLM generation functions for subsequent project stages.

### 8.2 Model Selection

**Model:** `Qwen/Qwen2.5-1.5B-Instruct`

**Architecture:** Decoder-only Transformer (causal language model)

**Framework:** Hugging Face Transformers and PyTorch

**Adaptation strategy:** Prompt engineering without model fine-tuning.

The pretrained Qwen model generates banking support responses using autoregressive token generation.

### 8.2.1 P2.2 — Model Configuration and Reproducibility

The model and tokenizer configuration were captured using `scripts/record_model_config.py` and saved to `experiments/model_configuration.json`. The configuration below reflects the verified Windows CPU environment on 24 October 2026.

| Property | Recorded value |
|---|---|
| Hugging Face model ID | `Qwen/Qwen2.5-1.5B-Instruct` |
| Model revision (commit SHA) | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Model class | `Qwen2ForCausalLM` |
| Tokenizer class | `Qwen2TokenizerFast` |
| Decoder layers | 28 |
| Hidden dimension | 1,536 |
| Query attention heads | 12 |
| Key/value attention heads | 2 |
| Model vocabulary size | 151,936 |
| Tokenizer vocabulary size | 151,643 |
| Operating system | Windows 11 |
| Python | 3.12.10 |
| Execution device | CPU (CUDA unavailable) |
| PyTorch | `2.14.1+cpu` |
| Transformers | `4.57.6` |
| Hugging Face Hub | `0.36.2` |
| Tokenizers | `0.22.2` |
| Default generation length | `max_new_tokens=140` |

**Recorded decoding configurations:** Greedy (`do_sample=False`); temperature 0.2 (`do_sample=True`, `temperature=0.2`, `top_p=1.0`); temperature 0.7 (`do_sample=True`, `temperature=0.7`, `top_p=1.0`); and top-p 0.9 (`do_sample=True`, `temperature=0.7`, `top_p=0.9`).

The model loader in `src/model/llm_baseline.py` was updated to pin the recorded revision for future runs. The revised loader was tested successfully: the model ID and revision were printed, and the model and tokenizer loaded without errors. **Historical experiments are not automatically proven to have used this revision**, because earlier runs did not pin it. Actual generation parameters can also be overridden per call, so each experiment should retain its effective settings in its run log.

**Reproduce the configuration record:**

```powershell
.\.venv\Scripts\python.exe -m scripts.record_model_config
```

### 8.2.2 P2.3 — Model Architecture and Autoregressive Generation

The BANKING77 assistant uses `Qwen/Qwen2.5-1.5B-Instruct`, a **decoder-only Transformer** that predicts the next token from the preceding context. Its selected configuration has 28 decoder layers, a hidden size of 1,536, 12 query attention heads, and 2 key/value heads (grouped-query attention).

#### Inference flow in the banking application

```text
Customer query (e.g., "Where is my replacement card?")
                  |
                  v
        Prompt / chat template
                  |
                  v
         Qwen2TokenizerFast
                  |
                  v
              Token IDs
                  |
                  v
           Token embeddings
                  |
                  v
    28 Transformer decoder layers
    - Causal self-attention
    - Grouped-query attention
    - Feed-forward transformations
    - Normalization and residual paths
                  |
                  v
         Language-model head
                  |
                  v
         Next-token scores
                  |
                  v
      Greedy or sampling decoder
                  |
                  v
       Next generated token
                  |
                  +----> Repeat until stop condition
                  |
                  v
       Decode generated tokens
                  |
                  v
   Predicted intent or support reply
                  |
                  v
       Output/safety validation
```

#### Component explanations

1. **Tokenization.** `Qwen2TokenizerFast` breaks the prompt into subword tokens and converts them into integer IDs. The application uses the tokenizer's chat template to format a customer query as a model input.
2. **Embeddings.** Learned token embeddings convert integer IDs into dense vectors of size 1,536 so the Transformer can process them.
3. **Transformer attention.** Causal self-attention allows each position to use preceding context without looking ahead to future positions. Grouped-query attention shares key/value heads across multiple query heads, reducing key/value-cache requirements. Feed-forward layers, normalization, and residual connections further transform token representations.
4. **Language-model head.** The final hidden state is projected into vocabulary scores (logits), which are converted into next-token probabilities for generation.
5. **Autoregressive generation.** The model selects one next token, appends it to the available context, and repeats until an end token or the configured token limit is reached. It does not generate the whole response in a single step.
6. **Decoding.** Greedy decoding selects the highest-scoring token at each step. Temperature sampling changes the sharpness of the probability distribution; top-p sampling restricts candidates to a probability-mass subset before sampling.

#### Connection to BANKING77 experiments

In **E1**, the model produces baseline customer-support replies. In **E2**, prompt instructions (zero-shot, few-shot, role-based, and constrained) change the context supplied to the same model. In **E3**, the prompt and model remain fixed while the decoding strategy changes. Separate intent-classification prompts ask the model to produce one of the 77 BANKING77 intent labels. The model's output is decoded to text and checked by application-level validation and safety guardrails; the model architecture itself does not guarantee safe or correct banking advice.

**Evidence:** `src/model/llm_baseline.py`, `scripts/record_model_config.py`, and `experiments/model_configuration.json`.

### 8.3 Dataset Integration

Part 2 reuses the evaluation dataset generated during Part 1.

**Dataset location:** `data/evaluation_sample.csv`

**Evaluation records:** 231

**Number of banking intents:** 77

**CSV columns:**

- `sample_id`: Unique evaluation sample identifier
- `source_index`: Original dataset row index
- `text`: Banking customer query
- `label`: Numeric ground-truth intent label
- `intent`: Ground-truth intent name

Ground-truth labels are retained for evaluation and must not be supplied to the model during intent-prediction experiments.

### 8.4 Part 2 Source Files

| File | Description |
|---|---|
| `src/model/llm_baseline.py` | Qwen model loading and response generation |
| `scripts/record_model_config.py` | Records model architecture, tokenizer, and runtime metadata |
| `experiments/model_configuration.json` | Captured model and runtime configuration |
| `src/model/decoding.py` | Decoding strategies and parameters |
| `src/prompts/templates.py` | Reusable prompt templates |
| `run_part2.py` | Prompt and decoding experiment runner |
| `scripts/record_model_config.py` | Save model, tokenizer, runtime, and decoding configuration |
| `scripts/run_p2_5.py` | Independent zero-shot intent-and-response verification |
| `notebooks/BANKING77_Qwen_Reference_Colab.ipynb` | Reference notebook for Colab |

### 8.5 Environment Setup

Use the project's Python 3.12 virtual environment.

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verify important libraries:

```powershell
.\.venv\Scripts\python.exe -c "import torch, transformers, pandas, accelerate; print('Dependencies OK')"
```

### 8.6 Run Part 2 Experiments

From the project root:

```powershell
.\.venv\Scripts\python.exe run_part2.py --evaluation_csv data/evaluation_sample.csv --limit 3
```

After a successful smoke test:

```powershell
.\.venv\Scripts\python.exe run_part2.py --evaluation_csv data/evaluation_sample.csv --limit 10
```

A compatible GPU is recommended. Google Colab with a T4 GPU can be used if local execution is too slow.

### 8.7 Experiments

**E1 — Baseline:** Generate at least three banking support responses using a simple prompt and the pretrained Qwen model.

**E2 — Prompt comparison:** Compare zero-shot, few-shot, role-based and constraint-based prompting using identical evaluation queries.

**E3 — Decoding comparison:** Compare greedy decoding, temperature sampling and top-p sampling while keeping the model and prompt fixed.

### 8.8 Expected Experiment Outputs

The experiment runner generates:

```text
experiments/
    baseline_outputs.csv
    prompt_comparison.csv
    decoding_comparison.csv
    experiment_runs.jsonl
```

The baseline experiment and evaluation analysis must also be documented for final submission.

### 8.9 Evaluation

Responses will be reviewed for:

1. Relevance to the customer query.
2. Safety and avoidance of sensitive credential requests.
3. Clarity and usefulness.
4. Consistency across repeated generations.
5. Compliance with prompt constraints.

Initial experimental findings, including the completed manual review, are documented in Section 8.13.

### 8.10 Responsible Use

This is an educational demonstration, not a production banking service.

The assistant must not request real banking passwords, PINs or OTPs. It must not claim that a transaction or card-blocking operation has occurred without verified confirmation.

### 8.11 Execution Status

- Part 1 dataset loading: Completed.
- Part 1 evaluation sample: Generated.
- Part 2 baseline and prompt/decoding experiments: Completed for the initial three-query run.
- Zero-shot intent classification: Evaluated on 77 queries (one per intent).
- Automated P2.12 evaluation: Implemented.
- Human response-quality review: Completed for 15 responses (three per prompting strategy, including baseline); see Section 8.13.6.
- Full 231-query intent evaluation: Pending.
- P2.5 three-intent verification: Completed (2/3 correct; strict JSON 0/3).
- P2.13 adaptation-options discussion: Documented; no RAG or PEFT implementation claimed.

### 8.12 Safety Guardrails

The BANKING77 Qwen assistant implements input validation, safety-aware prompting, and post-generation output validation.

The guardrails aim to prevent disclosure or solicitation of sensitive banking credentials, unsupported claims of completed transactions, and unsafe financial guarantees.

Unsafe responses are replaced with a predefined safe fallback. Guardrail decisions are recorded using `safety_status` and `safety_reasons` fields.

The implementation is rule-based and intended for educational evaluation. It does not guarantee detection of all unsafe outputs, sensitive information, or prompt-injection attempts.

All E1, E2, and E3 experiments use the same safety policy to support consistent comparison.

### 8.13 Intent Classification and Model Evaluation (P2.12)

#### Overview

This project extends the Qwen2.5-1.5B-Instruct baseline with intent classification and an automated evaluation framework for the BANKING77 dataset.

The implementation evaluates the model across four dimensions:

1. **Intent Correctness:** Accuracy, macro-F1, and incorrect intent predictions.
2. **Response Quality:** Human evaluation of generated banking support responses.
3. **Response Variability:** Consistency and diversity across decoding strategies.
4. **Formatting Failures:** Validation of predicted intent labels and structured responses.

The evaluation is performed using saved model outputs, allowing metrics to be recalculated without loading or rerunning Qwen.

#### 8.13.1 Intent Classification

The intent classification module uses the pretrained Qwen2.5-1.5B-Instruct model to identify customer banking intents.

The BANKING77 dataset contains 77 banking intent categories.

**Implementation files:**

```text
src/evaluation/intent_evaluator.py
scripts/evaluate_intents.py
tests/test_intent_evaluator.py
```

**Classification workflow:**

```text
BANKING77 Evaluation Dataset
            |
            v
Load Customer Queries
            |
            v
Construct Intent Classification Prompt
            |
            v
Qwen2.5-1.5B-Instruct
            |
            v
Generate Predicted Intent
            |
            v
Validate Against 77 Intent Labels
            |
            v
Compare with Ground Truth
            |
            v
Calculate Accuracy and Macro-F1
            |
            v
Save Predictions and Metrics
```

##### Run Intent Classification

Run the following command from the project root:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_intents --evaluation_csv data/evaluation_sample.csv --limit 10 --diverse_intents
```

For evaluating one query from each of the 77 banking intents:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_intents --evaluation_csv data/evaluation_sample.csv --limit 77 --diverse_intents
```

For evaluating the complete 231-query evaluation sample:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_intents --evaluation_csv data/evaluation_sample.csv --limit 231
```

The classification process uses greedy decoding to produce consistent predictions.

##### Generated Outputs

```text
experiments/
├── intent_predictions.csv
├── intent_evaluation.json
└── intent_errors.csv
```

**intent_predictions.csv** contains customer queries, ground-truth intents, model-predicted intents, and prediction correctness.

**intent_evaluation.json** records accuracy and valid-label statistics from the intent-classification run. The P2.12 evaluator separately calculates macro-F1 and per-intent metrics in `experiments/part2_evaluation/`.

#### 8.13.2 Intent Classification Results

The initial evaluation was performed using 77 customer queries, representing one query per BANKING77 intent.

| Metric | Result |
|---|---:|
| Total evaluated queries | 77 |
| Correct predictions | 35 |
| Incorrect predictions | 42 |
| Accuracy | 45.45% |
| Macro-F1 | 36.80% |
| Valid intent predictions | 75 |
| Invalid intent predictions | 2 |
| Invalid prediction rate | 2.60% |

**Observations:**

- The model correctly classified 35 of the 77 evaluated customer queries.
- Some incorrect predictions occurred between closely related banking intents.
- Two predictions did not match the permitted BANKING77 intent labels.
- Macro-F1 was lower than accuracy, indicating uneven classification performance across intent categories.
- The results establish an initial zero-shot intent classification baseline.

**Limitation:** Only one query per intent was evaluated. A larger evaluation using the complete held-out sample is recommended before drawing broader conclusions.

#### 8.13.3 Automated Model Evaluation (P2.12)

An automated evaluation framework has been implemented to analyze the results of the Part 2 experiments.

**Implementation files:**

```text
config/evaluation_config.yaml
scripts/evaluate_part2.py
```

The evaluation configuration is maintained separately from the evaluation logic.

This allows evaluation criteria and validation requirements to be updated without modifying the main Python implementation.

##### Evaluation Criteria

| Criterion | Evidence | Evaluation Method |
|---|---|---|
| Intent correctness | Accuracy, macro-F1, incorrect predictions | Automated |
| Response quality | Manual scores and response examples | Human review |
| Variability | Repetition consistency across decoding strategies | Automated |
| Formatting failures | JSON validity and required-field checks | Automated |

##### Run Automated Evaluation

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_part2 --config config/evaluation_config.yaml
```

The evaluation script reads previously generated experiment files:

```text
experiments/
├── baseline_outputs.csv
├── prompt_comparison.csv
├── decoding_comparison.csv
├── intent_predictions.csv
└── manual_response_scores.csv   # Optional
```

The Qwen model is not loaded during this evaluation.

#### 8.13.4 Evaluation Outputs

The evaluation results are saved in:

```text
experiments/
└── part2_evaluation/
    ├── evaluation_summary.csv
    ├── evaluation_summary.json
    ├── incorrect_intents.csv
    ├── intent_per_class.csv
    ├── variability_by_query.csv
    ├── variability_by_strategy.csv
    ├── formatting_checks.csv
    ├── manual_review_template.csv
    ├── manual_review_completed_by_prompt.csv
    └── manual_review_summary_by_prompt.csv
```

**Output descriptions:**

| File | Description |
|---|---|
| evaluation_summary.csv | Consolidated P2.12 evaluation metrics |
| evaluation_summary.json | Machine-readable evaluation summary |
| incorrect_intents.csv | Incorrect intent predictions for error analysis |
| intent_per_class.csv | Precision, recall, and F1 by intent |
| variability_by_query.csv | Response consistency for individual queries |
| variability_by_strategy.csv | Aggregated consistency for each decoding strategy |
| formatting_checks.csv | Structured-output validation results |
| manual_review_template.csv | Template for human response-quality assessment |
| manual_review_completed_by_prompt.csv | 15 reviewed responses, individual ratings, comments, and corrected prompting-strategy labels |
| manual_review_summary_by_prompt.csv | Aggregated mean scores for each prompting strategy |

#### 8.13.5 Response Variability Analysis

Response variability is evaluated using the saved decoding experiment outputs.

The experiments compare:

- Greedy decoding
- Temperature = 0.2
- Temperature = 0.7
- Top-p = 0.9

Each decoding configuration is repeated three times per customer query.

**Observed results:**

| Decoding Strategy | Query Groups | Mean Unique Response Rate | Fully Consistent Groups |
|---|---:|---:|---:|
| Greedy | 3 | 0.3333 | 3 |
| Temperature 0.2 | 3 | 1.0000 | 0 |
| Temperature 0.7 | 3 | 1.0000 | 0 |
| Top-p 0.9 | 3 | 1.0000 | 0 |

**Interpretation:**

Greedy decoding generated identical responses across all three repetitions for each query.

Sampling-based decoding produced different responses across repetitions.

A mean unique response rate of 0.3333 for greedy decoding represents one unique response across three repetitions, not a 33.33% accuracy score.

These results demonstrate that greedy decoding provides greater textual consistency, while sampling introduces response variability.

#### 8.13.6 Response Quality Evaluation — Completed Manual Review

A manual review was completed for **15 generated banking-support responses**, using three responses each from baseline, zero-shot, few-shot, role-based, and constrained prompting. The review used existing experiment outputs; no additional model generation was needed. The reviewed dataset is small, so the results are exploratory rather than definitive.

Each response was rated on a **1–5 scale** (higher is better) for the following dimensions:

| Dimension | Description |
|---|---|
| Relevance | Whether the response addresses the customer query |
| Clarity | Whether the response is understandable |
| Helpfulness | Whether useful guidance is provided |
| Safety | Whether the response avoids unsafe or unsupported advice |
| Formatting | Whether the response follows the requested structure |

The overall quality score for each response is the arithmetic mean of its five dimension scores. The per-strategy averages below are calculated from the three reviewed responses in each group. The initial review template grouped all prompt variants under `prompts`; these labels were corrected by matching the reviewed responses to the original `prompt_comparison.csv` records. Existing human scores were preserved.

**Manual evaluation results (mean score out of 5):**

| Prompt strategy | Responses | Relevance | Clarity | Helpfulness | Safety | Formatting | Overall quality |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 3 | 3.67 | 3.33 | 2.67 | 4.00 | 2.33 | **3.20** |
| Zero-shot | 3 | 4.00 | 4.00 | 3.00 | 4.00 | 4.33 | **3.87** |
| Few-shot | 3 | 4.00 | 4.00 | 3.00 | 4.00 | 2.33 | **3.47** |
| Role-based | 3 | 4.00 | 4.00 | 3.00 | 4.00 | 4.33 | **3.87** |
| Constrained | 3 | 4.00 | 4.00 | 3.00 | 4.00 | 3.00 | **3.60** |

**Findings and interpretation:**

- **Zero-shot and role-based prompting** tied for the highest observed overall quality (**3.87/5**); baseline was lowest (**3.20/5**).
- The most visible difference between prompt variants was **formatting**: zero-shot and role-based averaged 4.33/5, while few-shot averaged 2.33/5 in this reviewed sample.
- **Helpfulness** remained modest (3.00/5 for each prompt variant), suggesting that clearer formatting does not necessarily imply more actionable or reliable banking guidance.
- Reviewer comments identify potentially inaccurate bank-specific instructions, unsupported procedural assumptions, and an unresolved placeholder. Safety ratings of 4.00/5 should not be interpreted as proof that every response is safe or factually correct.
- With only **three responses per strategy** and subjective ratings, differences are descriptive and cannot establish that one prompt is generally superior.

**Saved review evidence:**

```text
experiments/part2_evaluation/manual_review_completed_by_prompt.csv
experiments/part2_evaluation/manual_review_summary_by_prompt.csv
```

The first file preserves each query, response, prompting-strategy label, five human scores, and review notes. The second contains the aggregated scores reproduced above. These manual results complement, rather than replace, the automated intent, variability, and formatting evaluations. The automated evaluator's optional input `experiments/manual_response_scores.csv` is a separate ingestion convention; the two review artifacts above are the documented source for this table and should not be described as automatically ingested without verifying the expected schema and rerunning the evaluator.

#### 8.13.7 Formatting Validation

The evaluation framework checks model outputs for formatting compliance.

For intent classification, predictions are validated against the 77 allowed BANKING77 labels.

For structured response generation, formatting validation can include:

- JSON syntax validity
- Presence of required fields
- Validation of allowed field values
- Detection of missing or malformed fields

The required fields and formatting expectations are configured in:

```text
config/evaluation_config.yaml
```

JSON validation should only be enabled for experiments where JSON output is explicitly required.

#### 8.13.8 Key Findings

The Part 2 evaluation produced the following findings:

**Intent correctness:** The zero-shot Qwen baseline achieved 45.45% accuracy on the 77-query evaluation subset. Classification errors were observed between semantically similar banking intents.

**Response quality:** Manual review of 15 responses found overall scores ranging from 3.20/5 (baseline) to 3.87/5 (zero-shot and role-based). Helpfulness and factual reliability remain improvement areas; see Section 8.13.6.

**Response variability:** Greedy decoding demonstrated complete repetition consistency in the tested queries. Sampling strategies produced more varied responses.

**Formatting compliance:** Intent-label validation identified two invalid predictions. Structured-response validation is supported through configurable formatting rules.

#### 8.13.9 Future Improvements

Potential improvements include:

1. Improving intent prompts with descriptions of commonly confused BANKING77 categories.
2. Introducing few-shot intent classification examples without using held-out evaluation answers.
3. Evaluating the complete 231-query dataset.
4. Strengthening constrained response generation using explicit output schemas.
5. Expanding manual response-quality review beyond three responses per strategy and considering independent reviewers.
6. Comparing updated model results against the initial baseline.

#### 8.13.10 Reproducibility

The evaluation framework separates model inference from metric calculation.

This design supports:

- Repeated evaluation without rerunning the LLM.
- Consistent comparison of prompt and decoding strategies.
- Configurable evaluation criteria.
- Saved evidence for reporting and error analysis.
- Reproducible metric calculations from experiment outputs.

**Overall:** The project now supports both zero-shot banking intent classification and a reusable automated evaluation pipeline aligned with the four P2.12 assessment criteria.

### 8.14 P2.5 — Zero-Shot Intent and Response Verification

The P2.5 verification uses `scripts/run_p2_5.py` with `build_zero_shot_intent_response_prompt()` from `src/prompts/templates.py`. The prompt provides the **77 permitted intent names** as the task vocabulary but **no labelled demonstration examples and no per-query ground-truth intent**. It asks Qwen to produce an `intent` and a customer-facing `response` in JSON. Ground truth is used only after generation to calculate correctness.

**Reproduce the diverse-intent smoke test** (run from the project root):

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p2_5 --limit 3 --diverse_intents
```

**Evidence:** `experiments/p2_5_zero_shot.csv` (three evaluated queries). The console confirmed that Qwen loaded and the CSV was saved.

| Ground-truth intent | Predicted intent | Correct | Valid intent-and-response structure |
|---|---|---|---|
| `activate_my_card` | `activate_my_card` | Yes | Yes |
| `age_limit` | `age_limit` | Yes | Yes |
| `apple_pay_or_google_pay` | `top_up_by_apple_pay` | No | No (intent does not match the allowed label vocabulary in this run) |

| Metric | Result |
|---|---:|
| Queries tested | 3 |
| Correct intents | 2/3 (66.67%) |
| Valid intent-and-response structures | 2/3 (66.67%) |
| JSON parsed after removing Markdown fences | 3/3 (100%) |
| Strict JSON without Markdown fences | 0/3 (0%) |

**Interpretation:** This experiment demonstrates zero-shot intent identification and response generation, not overall classification performance. The sample is too small for general performance claims. The model wrapped all three JSON responses in Markdown fences; normalized parsing recovered them, but strict formatting still failed. The third prediction was a different banking label from the reference intent. These are observed errors to retain for P2.12, not results to hide or automatically correct. Banking advice generated by the model still requires human quality and safety review.

### 8.15 P2.13 — Prompting vs RAG vs PEFT

Part 2 contributes the model-side technical comparison needed to decide whether to retain prompt-only inference or consider retrieval and lightweight adaptation. **Only prompt engineering is implemented and experimentally evaluated in this project stage.** RAG and PEFT are proposed alternatives, not completed experiments.

#### A. Prompt engineering (implemented)

Prompt engineering changes the **instructions and context** supplied to the frozen pretrained Qwen model. It does not update model weights. The project compares zero-shot, few-shot, role/context-based, and constrained prompts, along with greedy and sampling-based generation.

- **Strengths:** Minimal setup, no training dataset required for weight updates, inexpensive to iterate on, and easy to compare on the fixed BANKING77 evaluation sample.
- **Limitations:** Prompt wording can affect predictions; the model can confuse similar intents, generate unsupported banking details, or violate strict output formatting.
- **Evidence:** `run_part2.py`, `src/prompts/templates.py`, `experiments/prompt_comparison.csv`, and `experiments/p2_5_zero_shot.csv`.

#### B. Retrieval-augmented generation (RAG; proposed)

RAG retrieves relevant passages from a maintained knowledge base (for example, approved bank FAQs, card-support policies, or account-opening eligibility rules) and places them into the model's prompt before response generation. A typical pipeline is **customer query → retrieval → relevant passages → Qwen prompt → generated answer → source/format checks**.

- **Strengths:** Can ground responses in current, bank-approved information without retraining the language model; source passages can support auditing.
- **Limitations:** Requires curated documents, indexing/retrieval, access controls, retrieval evaluation, and protection against untrusted retrieved text. Retrieval does not guarantee factual or safe output.
- **BANKING77 relevance:** More useful for accurate support answers than for the narrow task of choosing one of 77 fixed intent labels. It may reduce unsupported claims about bank-specific policies if trusted documents are available.

#### C. Parameter-efficient fine-tuning (PEFT/LoRA; proposed)

PEFT adapts a pretrained model by training a small subset of additional parameters instead of updating all model weights. With LoRA, trainable low-rank matrices are introduced into selected layers while most original model parameters remain frozen.

- **Strengths:** May improve consistency on recurring domain-specific tasks and reduce reliance on lengthy prompts, using fewer trainable parameters than full fine-tuning.
- **Limitations:** Requires suitable labelled training data, compute resources, experiment tracking, overfitting checks, and held-out evaluation. Fine-tuning does not by itself provide up-to-date banking policies or guarantee valid JSON.
- **BANKING77 relevance:** A candidate if prompt-only intent accuracy remains inadequate after systematic evaluation; training must use the official training split and preserve the held-out evaluation set.

#### Comparison and recommendation

| Criterion | Prompt engineering | RAG | PEFT / LoRA |
|---|---|---|---|
| Model weights updated? | No | No | Small trainable adaptation |
| Extra resources | Prompts and evaluation examples | Trusted documents and retrieval pipeline | Training data and compute |
| Main benefit | Rapid behavior control and baseline comparison | Grounding answers in source material | Task/domain adaptation |
| Main risk | Sensitivity to instructions and formatting errors | Poor retrieval or untrusted content | Overfitting, training cost, stale learned facts |
| Current project status | **Implemented** | Not implemented | Not implemented |

**Recommendation:** Keep prompt engineering as the Part 2 baseline because it meets the controlled-comparison objective with the available pretrained Qwen model. Consider RAG later if the application needs verifiable, bank-specific policy answers. Consider PEFT only if larger held-out evaluations show that prompt-only intent classification remains inadequate and adequate training resources are available. Any future approach should be measured against the same evaluation protocol, including correctness, quality, safety, and formatting.

### 8.16 Part 2 Requirement Completion Matrix

This matrix distinguishes **implemented**, **documented**, and **still needing evidence**; it does not treat a script's existence as proof of a successful experiment.

| ID | Requirement | Status | Evidence / remaining work |
|---|---|---|---|
| P2.1 | Select and load model | Complete | Qwen model and tokenizer loaded successfully |
| P2.2 | Record configuration | Complete for future runs | `experiments/model_configuration.json`; pinned revision and successful loader test; historical revision unverified |
| P2.3 | Explain architecture | Documented | Section 8.2.2 |
| P2.4 | Baseline with three outputs | Complete for initial experiment | `experiments/baseline_outputs.csv` |
| P2.5 | Zero-shot intent and response | Complete for initial verification | Section 8.14 and `experiments/p2_5_zero_shot.csv` |
| P2.6 | Few-shot examples and selection | Partially documented | Existing two demonstrations; their selection/source has not been verified as training-only |
| P2.7 | Role/context prompt | Implemented | `src/prompts/templates.py`; E2 comparison |
| P2.8 | Constrained prompt/output | Implemented; compliance unverified | E2 constrained prompt; verify exact schema and strict-format success before claiming compliance |
| P2.9 | Greedy decoding | Complete for initial experiment | E3 logs and consistency analysis |
| P2.10 | Temperature and top-p sampling | Complete for initial experiment | E3 comparison of temperature 0.2/0.7 and top-p 0.9 |
| P2.11 | Full experiment evidence | Needs audit | Logs exist; verify exact prompts, effective parameters, run identifiers, and model revision in each record |
| P2.12 | Four-part analysis | Complete for initial evaluation | Automated intent, variability, and formatting analysis plus 15 manually reviewed responses; see Section 8.13.6 |
| P2.13 | Prompting/RAG/PEFT discussion | Documented | Section 8.15; no implementation of RAG/PEFT required by this item |

**Minimal remaining actions for Part 2:** (1) save the completed manual review and by-prompt summary artifacts under `experiments/part2_evaluation/`; (2) audit P2.11 logs without changing working inference code unless fields are truly missing; (3) briefly document the origin and selection rationale of the existing few-shot demonstrations; and (4) preserve strict JSON failures in the report instead of reporting normalized JSON as strict compliance.

## 9. Evaluation Protocol and Experiment Tracking

The project will use a consistent evaluation procedure so that results from different experiments can be compared fairly.

### Evaluation metrics

Planned metrics include:

- **Accuracy:** Proportion of examples assigned the correct intent.
- **Macro-F1:** Average F1 score across the 77 intent classes.
- **JSON validity rate:** Proportion of outputs conforming to the required JSON format.
- **Tool-call correctness:** Proportion of tool invocations that match the intended action and satisfy the tool's input requirements.
- **Response quality and safety:** Review of relevance, unsupported claims, unsafe behaviour, and other observed failures.

Not every metric applies to every experiment. For example, JSON validity and tool-call correctness apply to the structured-output and tool-use stages.

### Experiment records

Each experiment should record, where applicable:

- Experiment ID and objective.
- Model identifier and version/revision.
- Prompt strategy and prompt version.
- Decoding parameters.
- Evaluation sample and random seed.
- Raw model outputs and predictions.
- Metric results.
- Observed failures and conclusions.

Detailed sampling, evaluation, and reproducibility rules are maintained in the [Evaluation Protocol](reports/evaluation_protocol.md).

Initial Part 2 results are reported in Section 8; the 77-query intent baseline is preliminary.

## 10. Project Structure

```text
AppliedGenAI-Banking77-Project-Group2/
├── README.md
├── app.py
├── run_part2.py
├── requirements.txt
├── .gitignore
├── config/
│   └── evaluation_config.yaml
├── data/
│   ├── evaluation_sample.csv
│   └── DATASET_NOTES.md
├── notebooks/
│   └── data_exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── model/
│   │   ├── llm_baseline.py
│   │   └── intent_classifier.py
│   ├── prompts/
│   │   └── templates.py
│   ├── safety/
│   │   └── guardrails.py
│   └── evaluation/
│       └── intent_evaluator.py
├── scripts/
│   ├── prepare_evaluation_data.py
│   ├── evaluate_intents.py
│   ├── evaluate_part2.py
│   └── record_model_config.py
├── experiments/
│   ├── model_configuration.json
│   └── part2_evaluation/
└── reports/
    └── evaluation_protocol.md
```

This structure separates reusable project logic, execution scripts, exploratory notebooks, generated data, experiment records, and documentation.

- `config/`: Shared configuration for model and experiment settings.
- `data/`: Prepared evaluation data and related dataset documentation.
- `notebooks/`: Exploratory analysis and independent verification.
- `src/`: Reusable Python modules.
- `scripts/`: Executable scripts for repeatable project tasks.
- `experiments/`: Experiment logs, outputs, and comparison results.
- `reports/`: Evaluation protocols, technical analysis, and report material.

Add new files to this structure as the implementation progresses. Update this section and the Quick Navigation table whenever important resources are added.

## 11. Team Responsibilities

- **Part 1 —** Problem definition, dataset preparation and evaluation design.
- **Part 2 —** Pretrained LLM baseline, prompt engineering and decoding experiments.
- **Part 3 —** Prompt templates, structured output, tool/function interface and adaptation analysis.
- **Part 4 —** Integration, experiment consolidation, safety analysis and final delivery.

The team will agree on shared interfaces and evaluation conventions before integrating components.

## 12. Current Progress

### Environment and data preparation

- [x] Python and Git verified.
- [x] Project virtual environment created.
- [x] Initial Part 1 dependencies installed and checked.
- [x] VS Code notebook kernel configured.
- [x] Banking77 loaded and official splits inspected.
- [x] Intent counts inspected for the official splits.
- [x] Balanced evaluation sample generated and saved.
- [x] Evaluation CSV verified: 231 rows, 77 intents, 3 examples per intent, and no missing values.
- [x] Problem definition and input/output contract documented in `reports/problem_definition.md`.
- [x] Dataset inspection and preprocessing notes documented in `data/DATASET_NOTES.md`.
- [x] Evaluation protocol documented in `reports/evaluation_protocol.md`.
- [x] Evaluation metrics, comparison rules, leakage prevention, reference-file fields, and limitations specified.
- [ ] Evaluation protocol reviewed and approved by the team.
- [x] Dataset revision and dependency versions recorded for strict reproducibility.
- [x] Controlled regeneration check confirms the generated sample matches the preserved CSV.

### Model and application development

- [x] Pretrained LLM baseline implemented and evaluated on initial examples.
- [x] P2.2 model configuration recorded, revision pinned for future runs, and loader verified.
- [x] P2.3 model architecture and BANKING77 inference flow documented.
- [x] Zero-shot, few-shot, role-based, and constrained prompts compared in initial experiments.
- [x] Greedy and sampling-based decoding experiments completed for three queries.
- [x] Constrained JSON schema and validation added; runtime compliance requires continued testing.
- [x] Simulated tool/function flow implemented in the Part 3 pipeline and app.
- [x] P2.13 prompting, RAG, and PEFT options documented; train-split RAG retrieval implemented, PEFT not implemented.
- [x] Initial intent metrics, variability, and formatting evaluation automated; manual review completed for 15 responses.
- [x] P2.5 three-intent zero-shot verification recorded (2/3 correct; strict JSON 0/3).
- [ ] P2.11 experiment evidence audited for exact prompts, effective parameters, and run IDs.
- [x] P2.12 human response-quality scoring completed for initial 15-response sample.
- [ ] Responsible-use risks and controls documented.
- [ ] Final report, presentation, demo, and contribution statements completed.

## 13. References and Related Documentation

- [Banking77 dataset on Hugging Face](https://huggingface.co/datasets/PolyAI/banking77)
- [Qwen2.5 model family on Hugging Face](https://huggingface.co/Qwen)
- [Problem Definition](reports/problem_definition.md)
- [Dataset Notes](data/DATASET_NOTES.md)
- [Evaluation Protocol](reports/evaluation_protocol.md)
- [Data Exploration Notebook](notebooks/data_exploration.ipynb)
- [Dataset Preparation Module](src/data/dataset.py)
- [Evaluation Preparation Script](scripts/prepare_evaluation_data.py)

External links are provided for reference. Model implementation details and experimental results will be documented as the corresponding work is completed.
