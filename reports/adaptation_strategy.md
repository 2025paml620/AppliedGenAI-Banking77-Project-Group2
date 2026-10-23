# Adaptation strategy comparison

## Decision for this prototype

Use prompt engineering with the selected pretrained Qwen2.5-1.5B-Instruct model. The Banking77 task is intent classification plus a short general support response. The assignment needs a reproducible baseline and prompt comparisons; prompting permits those comparisons without changing model weights or introducing retrieval infrastructure. This is a project choice, not a claim that prompting is best for a deployed bank.

| Approach | What it adds | Benefits here | Costs and risks | Fit |
|---|---|---|---|---|
| Prompting | Instructions, examples, context and output constraints at inference time | Fast iteration; easy to inspect; no training set transformation or model artifact | Output may ignore instructions; prompts consume context; JSON must be validated; no guarantee of policy correctness | Selected for the demonstration |
| RAG | Retrieves relevant approved documents and adds them to the prompt | Could ground responses in current, bank-approved procedures and cite source passages | Requires a trusted corpus, chunking, indexing, retrieval evaluation, access controls and freshness monitoring; retrieval can miss or select wrong material | Useful if verified policy documents become available; none are part of this prototype |
| Fine-tuning / PEFT | Updates all or selected model parameters on curated examples | Can improve task-specific style or label patterns and reduce repeated instruction text | Needs representative, quality-controlled training data, compute, held-out evaluation and ongoing drift/safety review; does not reliably add current facts | Not selected because the project is scoped around prompt experiments and has no approved bank data |

No approach by itself authorizes financial actions or guarantees safe responses. The application therefore validates schema and tool arguments, uses mock-only tools, applies guardrails and falls back on invalid outputs. Automated intent scores do not measure response quality; responses still need human review.
