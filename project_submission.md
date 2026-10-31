# Applied Generative AI Project Submission

## Group Information

**Group name:** Group 2  
**Project:** Banking77 Banking Support Assistant

| Part | Group member                                       |
|---|----------------------------------------------------|
| Part 1 | AVINASH KUMAR BARNWAL (2025paml613@wilp.bits-pilani.ac.in) |
| Part 2 | SHUBHRA SHRIVASTAVA (2025paml620@wilp.bits-pilani.ac.in)         |
| Part 3 | KARTHIKEYAN B (2025paml541@wilp.bits-pilani.ac.in) |
| Part 4 | Abijith M G (2025paml617@wilp.bits-pilani.ac.in) |

Note: Avinash by mistake highlighted that part 3 is done by Abijith in his demo video part. 

## Project Links

- **GitHub repository:** [Applied GenAI Banking77 Project Group 2](https://github.com/2025paml620/AppliedGenAI-Banking77-Project-Group2)
- **Project Explanation + Demo video:** [genai_proj_grp2](https://drive.google.com/file/d/1xltgGpVBlfDbFZixaiITlztdqqsdOskP)
- **Project Demo video:** [demo_group2](https://drive.google.com/file/d/1J6B9AtT8MAjOcswHD8abhMIPpQM21KoC/view?usp=sharing)
- **Run guide:** [Project setup, execution, and testing instructions](https://github.com/2025paml620/AppliedGenAI-Banking77-Project-Group2/blob/main/RUN_GUIDE.md)

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
