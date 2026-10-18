# Applied Gen AI Group Project — Banking77

## 1. Project Overview

This project is part of the BITS Pilani Applied Generative AI course.

We are building an educational banking customer-support assistant using a pretrained causal language model. The application will identify a customer's intent, generate a relevant and safe response, return structured output, and demonstrate a simulated tool/function call.

The project will investigate how prompt design, context, decoding controls, and lightweight adaptation choices influence application behaviour.

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
| [Current Progress](#11-current-progress) | Check completed and pending work |
| [Problem Definition](reports/problem_definition.md) | Target user, input/output contract, and safety boundaries |
| [Dataset Notes](data/DATASET_NOTES.md) | Dataset statistics, label mapping, preprocessing, evaluation sampling, and leakage prevention |

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
| Python version | Python 3.12.9 |
| Environment management | Python `venv` |
| Initial dataset library | `datasets==3.6.0` |
| Notebook environment | Jupyter notebook |
| Version control | Git |

The base model has been selected for Part 2 but has not yet been evaluated as part of the application.

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

## 8. Evaluation Protocol and Experiment Tracking

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

No model performance or accuracy results have been established yet.

## 9. Project Structure

```text
AppliedGenAI-Banking77-Project-Group2/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
├── data/
│   ├── evaluation_sample.csv
│   └── DATASET_NOTES.md
├── notebooks/
│   └── data_exploration.ipynb
├── src/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       └── dataset.py
├── scripts/
│   └── prepare_evaluation_data.py
├── experiments/
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

## 10. Team Responsibilities

- **Part 1 —** Problem definition, dataset preparation and evaluation design.
- **Part 2 —** Pretrained LLM baseline, prompt engineering and decoding experiments.
- **Part 3 —** Prompt templates, structured output, tool/function interface and adaptation analysis.
- **Part 4 —** Integration, experiment consolidation, safety analysis and final delivery.

The team will agree on shared interfaces and evaluation conventions before integrating components.

## 11. Current Progress

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

- [ ] Pretrained LLM baseline implemented and evaluated.
- [ ] Zero-shot, few-shot, role/context-based, and constraint-based prompts compared.
- [ ] Greedy and sampling-based decoding experiments completed.
- [ ] Structured output schema and validation implemented.
- [ ] Simulated tool/function flow implemented.
- [ ] Prompting, RAG, and fine-tuning/PEFT approaches compared.
- [ ] Evaluation metrics and failure cases consolidated.
- [ ] Responsible-use risks and controls documented.
- [ ] Final report, presentation, demo, and contribution statements completed.

## 12. References and Related Documentation

- [Banking77 dataset on Hugging Face](https://huggingface.co/datasets/PolyAI/banking77)
- [Qwen2.5 model family on Hugging Face](https://huggingface.co/Qwen)
- [Problem Definition](reports/problem_definition.md)
- [Dataset Notes](data/DATASET_NOTES.md)
- [Evaluation Protocol](reports/evaluation_protocol.md)
- [Data Exploration Notebook](notebooks/data_exploration.ipynb)
- [Dataset Preparation Module](src/data/dataset.py)
- [Evaluation Preparation Script](scripts/prepare_evaluation_data.py)

External links are provided for reference. Model implementation details and experimental results will be documented as the corresponding work is completed.