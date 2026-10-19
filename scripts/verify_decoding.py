
from src.model.llm_baseline import load_model, generate_response
from src.model.decoding import DECODING
from transformers import GenerationConfig

model, tokenizer = load_model()

query = "How can I activate my new debit card?"

for name, config in DECODING.items():
    print(f"\nTesting: {name}", flush=True)

    do_sample = config["do_sample"]

    effective = GenerationConfig(
        do_sample=do_sample,
        max_new_tokens=80,
        pad_token_id=tokenizer.eos_token_id,
        **({
            "temperature": config["temperature"],
            "top_p": config["top_p"]
        } if do_sample else {})
    )

    print("Effective configuration:", effective.to_dict())

    response = generate_response(
        model,
        tokenizer,
        query,
        {**config, "max_new_tokens": 80}
    )

    print("Response:", response, flush=True)
```

From the repository root, execute:

```powershell
.\.venv\Scripts\python.exe -m scripts.verify_decoding
```

<text color="secondary" size="sm">If Python cannot find the `scripts` module, add an empty `scripts/__init__.py` file.</text>

**Success criteria:** All four configurations generate responses, sampling parameters appear in the printed configuration, and the invalid-generation-flags warning no longer appears. If the warning persists, inspect the installed Transformers version and the model's generation configuration before rerunning E3.

## Step 3 — Expand the BANKING77 evaluation

Your Part 1 evaluation dataset contains 231 samples across 77 intents, with three examples per intent.

The first Part 2 run used only three customer queries. That's suitable for a smoke test but too small for a convincing comparison across banking support scenarios.

### Recommended evaluation sizes

| Evaluation | Queries | E1 outputs | E2 outputs | E3 outputs |
|---|---:|---:|---:|---:|
| Initial smoke test | 3 | 3 | 12 | 36 |
| Intermediate validation | 10 | 10 | 40 | 120 |
| Broader evaluation | 77 | 77 | 308 | 924 |
| Full held-out sample | 231 | 231 | 924 | 2,772 |

Because your laptop previously took considerable time to generate responses, **start with 10 diverse intents**.

Run:

```powershell
.\.venv\Scripts\python.exe run_part2.py --evaluation_csv data/evaluation_sample.csv --limit 10 --diverse_intents
```

This will generate 170 responses if your runner still uses four E2 strategies and four E3 strategies with three repetitions.

Back up your existing `experiments/` files first, because your current runner may overwrite them.

For a broader final evaluation, use one query per intent:

```powershell
.\.venv\Scripts\python.exe run_part2.py --evaluation_csv data/evaluation_sample.csv --limit 77 --diverse_intents
```

This is optional if your assignment does not require evaluating all 77 intents. A smaller, well-documented evaluation is better than reporting unsupported conclusions from a large run.

## Step 4 — Document the limitations in `PART2_REPORT.md`

Open:

`docs/PART2_REPORT.md`

Add the following section near the end of the report.

<WritingBlock id="68241" variant="document">## Evaluation Limitations and Future Improvements

### 1. Limited Evaluation Coverage

The initial Part 2 evaluation used three customer queries from the BANKING77 evaluation dataset.

Although the experiments generated 51 responses across E1, E2, and E3, the results represent only a small subset of the 77 banking intents.

Consequently, the initial findings cannot be generalized to all banking customer-support scenarios.

**Improvement:** Extend the evaluation to a representative set of banking intents, using a reproducible sampling strategy.

### 2. Decoding Configuration Verification

The implementation supports greedy decoding, temperature-based sampling, and nucleus (top-p) sampling.

During the initial execution, the Transformers library displayed a warning indicating that some generation parameters might be ignored.

The generation implementation has been reviewed, and explicit generation configuration is recommended to reduce ambiguity.

**Improvement:** Verify the effective runtime configuration and repeat affected decoding experiments if necessary.

### 3. Safety Guardrail Limitations

The initial experiment generated 51 responses, all of which were marked as allowed by the implemented rule-based guardrails.

However, an allowed status does not guarantee factual correctness or complete safety.

Manual inspection identified potentially misleading banking instructions that were not rejected by the automated checks.

**Improvement:** Introduce human-reviewed safety test cases, stronger validation of unsupported banking claims, and clearer escalation guidance.

### 4. Evaluation Metrics

Response length and consistency provide useful descriptive measurements but are insufficient to establish overall response quality.

**Improvement:** Evaluate relevance, factual accuracy, clarity, safety, and instruction compliance using a consistent scoring rubric.

### 5. Hardware and Runtime Constraints

The project uses Qwen2.5-1.5B-Instruct, which may have substantial inference latency on CPU-only systems.

**Improvement:** Use GPU acceleration, optimize inference settings, and cache experiment outputs to avoid unnecessary repeated generations.

### Conclusion

Part 2 successfully demonstrates baseline response generation, prompt engineering, decoding comparisons, and rule-based safety guardrails using Qwen2.5-1.5B-Instruct.

The initial results establish a working experimental pipeline. Broader evaluation, runtime decoding verification, and improved safety assessment are recommended before drawing general conclusions about the best generation strategy.</WritingBlock>

## Step 5 — Final submission checklist

{@body const checks=[["Decoding configuration","Update llm_baseline.py and run the four-strategy verification."],["Unit tests","Run all safety and decoding tests."],["Evaluation coverage","Run a diverse sample and record the exact query count."],["Result analysis","Review relevance, clarity, factual reliability and safety."],["Final report","Add measured results and evaluation limitations."],["GitHub submission","Commit source code, tests, reports and appropriate experiment artifacts."]]}
{@body const [completed,setCompleted]=DIL.useState([])}
<box border radius="xl" padding={3} gap={3}>
  <row align="center" justify="between">
    **Submission readiness**
    <text color="secondary" weight="medium" tabularNums>{completed.length} / {checks.length}</text>
  </row>
  <box background="surface-tertiary" height="7px" radius="full" clip>
    <box background="surface-inverted" width={`${completed.length/checks.length*100}%`} height="100%" radius="full"/>
  </box>
  {#each checks as item,i}
    <checkbox key={i} checked={completed.includes(i)} onChange={v=>setCompleted(old=>v?[...old,i]:old.filter(x=>x!==i))} lineThrough={false}>
      <box gap={1}>
        **{item[0]}**
        <text color="secondary" size="xs">{item[1]}</text>
      </box>
    </checkbox>
  {/each}
</box>

## Step 6 — Commit the completed work to GitHub

Once the tests and final report are verified:

```powershell
git status

git add src/model/llm_baseline.py src/model/decoding.py tests/ scripts/ docs/PART2_REPORT.md

git commit -m "Complete BANKING77 Part 2 decoding verification and evaluation"

git push origin HEAD
```

Review `git status` before committing your experiment files, and ensure no real banking credentials or sensitive customer data are included.

**My recommendation:** Complete Step 1 and Step 2 first. You can then run the 10-intent evaluation and update the report without spending time generating a much larger dataset unnecessarily.

Once the decoding warning is resolved and the results are documented accurately, your Part 2 work will be in a much stronger position for final submission.