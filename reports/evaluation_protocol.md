# Banking77 Evaluation Protocol

## 1. Dataset

- Dataset: PolyAI Banking77
- Training split: 10,003 examples
- Test split: 3,080 examples
- Number of intent classes: 77
- Input field: `text`
- Ground-truth field: `intent`

## 2. Evaluation Sample

- Source: Official Banking77 test split
- Samples per intent: 3
- Total evaluation examples: 231
- Random seed: 42
- Artifact: `data/evaluation_sample.csv`
- Selection: Stratified by intent, with an equal number of examples per class.

This is a fixed development evaluation sample. Results on this sample will not be treated as equivalent to results on the complete test split.

## 3. Data Separation

- Use the training split for few-shot demonstrations.
- Use the fixed evaluation sample for model evaluation.
- Do not include the correct intent labels of evaluation examples in prompts.
- Use the same evaluation examples across prompt and decoding experiments.

## 4. Evaluation Metrics

1. Accuracy: Proportion of examples assigned the correct intent.
2. Macro-F1: Average F1 score across the 77 intent classes.
3. JSON validity rate: Proportion of outputs that conform to the required JSON format.
4. Tool-call correctness: Proportion of tool invocations that match the intended action and satisfy the tool's input requirements.

Additional metrics may be recorded where relevant to a specific experiment.

## 5. Reproducibility

- Preserve the generated evaluation CSV.
- Record the model identifier, prompt version, decoding settings, and random seed for each experiment.
- Save raw model outputs and evaluation results.
- Do not silently regenerate the evaluation sample during experiments.

## 6. Limitations

The evaluation sample contains three examples per intent. Although this gives equal representation to all classes, it is relatively small. Results may vary with the selected examples and should be interpreted accordingly.