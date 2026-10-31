# Applied Generative AI Project Submission

## Group Information

**Group name:** Group 2  
**Project:** Banking77 Banking Support Assistant

| Part | Group member |
|---|---|
| Part 1 | Avinash |
| Part 2 | Shubhra |
| Part 3 | Karthik |
| Part 4 | Abijith |

## Project Links

- **GitHub repository:** [Applied GenAI Banking77 Project Group 2](https://github.com/2025paml620/AppliedGenAI-Banking77-Project-Group2)
- **Demo video:** Add the demo recording URL before submission.
- **Run guide:** [Project setup, execution, and testing instructions](RUN_GUIDE.md)

## Running the Project

Install the Python dependencies using the setup steps in the [Run Guide](RUN_GUIDE.md). From the repository root, launch the interactive Gradio application with:

```powershell
python app.py
```

The integrated command-line pipeline supports a single query, the demonstration scenarios, and the synthetic evaluation suite:

```powershell
python -m scripts.run_pipeline --mode query --query "What is the status of transaction TXN1002?"
python -m scripts.run_pipeline --mode demo
python -m scripts.run_pipeline --mode suite
```

Run the Python unit and deterministic end-to-end tests with:

```powershell
python -m pytest tests
```

The separate browser test setup and command are documented in the [Run Guide](RUN_GUIDE.md).
