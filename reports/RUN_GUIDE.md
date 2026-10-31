# How to run and evaluate the project

Run commands from the repository root with the project environment active. These workflows serve different purposes: tests check code behavior, model-backed commands evaluate the real Qwen model, the notebook runs a larger integrated evaluation, and the failure CSV documents already observed problems.

## Environment setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` is the normal install path. `requirements-lock.txt` records a resolved set of exact versions for recreating a recorded environment; the lock can be platform-specific, so use it when matching that environment is the goal.

## 1. Run unit tests

Unit tests check small functions and contracts with fixed inputs or mocked dependencies. They do not download or invoke the Qwen model.

Run all tests:

```powershell
python -m pytest tests
```

Run focused unit groups:

```powershell
python -m pytest tests/test_structured.py tests/test_guardrails.py
python -m pytest tests/test_data_and_prompts.py tests/test_model_intent_classifier.py
python -m pytest tests/test_intent_evaluator.py tests/test_decoding.py tests/test_model_test_suite.py
```

Run the app dispatch/UI glue checks separately:

```powershell
python -m pytest tests/test_app_modes.py
```

## 2. Run deterministic end-to-end tests

These pass fixed model responses into the real `handle_query` pipeline. They exercise input safety, prompt construction, schema validation, output safety, deterministic tool routing, mock tool execution, and fallback behavior. They do not call Qwen or require model downloads.

```powershell
python -m pytest tests/test_end_to_end.py
```

The distinction is scope: unit tests isolate individual rules and helpers; these end-to-end tests connect multiple real components together while stubbing only model generation.

### Optional browser E2E with pytest-playwright

The separate `browser_tests/` folder checks the real Gradio UI in Chromium. These tests use the Playwright `page` fixture and are intentionally outside `tests/`, so `python -m pytest tests` remains the Python-only suite.

Install the optional pytest plugin and its browser once:

```powershell
python -m pip install -r requirements.txt
python -m playwright install chromium
```

Start the Gradio app in one terminal:

```powershell
python app.py
```

Run only the browser E2E file from a second terminal:

```powershell
python -m pytest browser_tests/test_end_to_end.py
```

The test uses `http://127.0.0.1:7860` by default. If Gradio reports another port, set its URL before running pytest:

```powershell
$env:APP_BASE_URL = "http://127.0.0.1:7861"
python -m pytest browser_tests/test_end_to_end.py
```

The browser suite checks harmful and sensitive-input blocking, blank and overlength query validation, sample-prompt loading, and mode-specific control visibility. These cases do not load or call Qwen. See the [official Playwright Python pytest guide](https://playwright.dev/python/docs/intro) for plugin setup and browser options.

## 3. Run the integrated model pipeline from the command line

Use one command with a mode to choose its input set. Every mode uses the same `handle_query` pipeline: input safety, prompt creation, model generation, JSON validation, output safety, deterministic routing, and mock tool execution.

Run one customer query:

```powershell
python -m scripts.run_pipeline --mode query --query "What is the status of transaction TXN1002?"
```

Run the predefined eight Part 3 demo scenarios:

```powershell
python -m scripts.run_pipeline --mode demo
```

Run the 90-case synthetic test suite (77 intent cases plus safety, input-validation, ambiguity, and mock-tool cases):

```powershell
python -m scripts.run_pipeline --mode suite
```

For a quick suite smoke run, limit it to five cases:

```powershell
python -m scripts.run_pipeline --mode suite --limit 5
```

Add retrieval from labeled Banking77 training examples to any mode with `--rag`, for example:

```powershell
python -m scripts.run_pipeline --mode demo --rag
```

Each run creates a unique folder under `experiments/model_test_runs/` with `results.csv`, `report.md`, and `run.json`. Suite cases receive automatic exact-intent and deterministic safety/tool checks; ad-hoc and demo cases are marked for human review. Model files, and the embedding model and train split for RAG, may download on first use. Tools use local mock data only.

`run_part3.py` and `scripts.run_model_test_suite` remain compatibility entry points; new runs should use `scripts.run_pipeline` so input mode, RAG, logging, and artifacts follow one interface.

## 4. Run the integrated project notebook

Open `notebooks/group_project.ipynb` in Jupyter or VS Code and execute all cells from the repository root. The notebook runs the structured pipeline across the fixed evaluation sample, writes per-query output and metrics files to `experiments/`, and includes deterministic examples for successful tools and failure handling.

To open it from the terminal:

```powershell
python -m jupyter lab notebooks/group_project.ipynb
```

The notebook defaults to all 231 evaluation rows. For a quick notebook smoke run, change `LIMIT = len(DATA)` to `LIMIT = 5` before running the model cell. Restore the full limit for reported evaluation. The notebook can take a long time and requires the pinned Qwen model to be available locally or downloadable.

The notebook is a reproducible integrated evaluation/demo artifact; it is not a replacement for the unit or deterministic end-to-end test suites.

## 5. Use the failure analysis

`experiments/failure_analysis.csv` is a review artifact, not an executable test file. It records three representative observed problems with the input, expected behavior, actual output, impact, and mitigation. Use it to discuss model/guardrail weaknesses and compare new runs against known failure patterns. Update it only when a failure is supported by a saved run or an explicitly documented test case.

## Which workflow should I run?

| Goal                                                           | Command/artifact | Calls Qwen? | Main output |
|----------------------------------------------------------------|---|---:|---|
| Check helper behavior and contracts                            | `python -m pytest tests` | No | Pass/fail test results |
| Check the connected pipeline with controlled outputs           | `python -m pytest tests/test_end_to_end.py` | No | Pass/fail integration results |
| Check the rendered app in Chromium                              | `python -m pytest browser_tests/test_end_to_end.py` | No | Browser assertions for UI and safety behavior |
| Run one real query                                             | `python -m scripts.run_pipeline --mode query --query "..."` | Yes | Timestamped artifacts under `experiments/model_test_runs/` |
| Run demo scenarios                                             | `python -m scripts.run_pipeline --mode demo [--rag]` | Yes | Timestamped artifacts under `experiments/model_test_runs/` |
| Evaluate the 90-case model suite                               | `python -m scripts.run_pipeline --mode suite [--limit 5]` | Yes | Timestamped run folder under `experiments/model_test_runs/` |
| Run the integrated fixed-sample evaluation                     | `notebooks/group_project.ipynb` | Yes | Per-run CSV and JSON metrics under `experiments/` |
| Review previously documented weaknesses                        | `experiments/failure_analysis.csv` | No | Failure analysis records |

For definitions and interpretation limits, see [Evaluation Protocol](evaluation_protocol.md), [Part 3 Report](PART3_REPORT.md), and [Part 4 Handover](PART4_HANDOVER.md).
