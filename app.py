"""BANKING77 educational Gradio app. Run from repository root: python app.py."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from src.model.llm_baseline import load_model, generate_response
from src.model.intent_classifier import predict_intent
from src.prompts.templates import build_prompt
from src.safety import guardrails

PROMPT_VARIANTS = ["zero_shot", "few_shot", "role_based", "constrained"]
MAX_QUESTION_LENGTH = 1500
EVALUATION_CSV = Path(__file__).resolve().parent / "data" / "evaluation_sample.csv"
SAFE_FALLBACK = (
    "I can't safely provide a response to that request. "
    "Please contact your bank using its official app or support channel."
)


@lru_cache(maxsize=1)
def assistant():
    """Reuse the same Qwen model for classification and response generation."""
    return load_model()


@lru_cache(maxsize=1)
def allowed_intents() -> tuple[str, ...]:
    """Read the BANKING77 label vocabulary, never an individual row's true label."""
    if not EVALUATION_CSV.is_file():
        raise FileNotFoundError(f"Missing intent vocabulary source: {EVALUATION_CSV}")
    labels = pd.read_csv(EVALUATION_CSV, usecols=["intent"])["intent"].dropna()
    vocabulary = tuple(sorted(set(labels.astype(str).str.strip()) - {""}))
    if len(vocabulary) != 77:
        raise ValueError(f"Expected 77 BANKING77 labels, found {len(vocabulary)}")
    return vocabulary


def _safety_check(text: str, stage: str) -> tuple[dict[str, Any], str]:
    """Use the exact SafetyResult contract in src/safety/guardrails.py."""
    checker = guardrails.check_input if stage == "input" else guardrails.check_output
    result = checker(text)
    return (
        {
            "status": result.status,
            "allowed": result.status == "allowed",
            "reasons": list(result.reasons),
        },
        result.response,
    )


def _parse_json_response(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return {"valid_json": False, "schema_valid": False, "error": str(exc)}
    required = ("intent", "response", "action_required")
    missing = [key for key in required if not isinstance(payload, dict) or key not in payload]
    wrong_types = [key for key in required if isinstance(payload, dict) and key in payload
                   and not isinstance(payload[key], str)]
    allowed_actions = {"none", "contact_bank", "block_card_request"}
    action_valid = isinstance(payload, dict) and payload.get("action_required") in allowed_actions
    schema_valid = isinstance(payload, dict) and not missing and not wrong_types and action_valid
    return {"valid_json": True, "schema_valid": schema_valid, "missing_fields": missing,
            "invalid_field_types": wrong_types, "valid_action": action_valid,
            "parsed_output": payload}


def run(question: str, variant: str, decoding: str, temperature: float, top_p: float):
    """Return response, predicted intent, safety checks, and generation details."""
    details: dict[str, Any] = {"tool_action": "None executed (educational demo)"}
    question = (question or "").strip()
    if not question or len(question) > MAX_QUESTION_LENGTH:
        return "Enter a question between 1 and 1500 characters.", "Not evaluated", details, details
    if variant not in PROMPT_VARIANTS or decoding not in {"greedy", "sampling"}:
        return "Invalid generation settings.", "Not evaluated", details, details

    input_safety, input_fallback = _safety_check(question, "input")
    details["input_safety"] = input_safety
    if not input_safety["allowed"]:
        return input_fallback, "Not evaluated", {"input": input_safety}, details

    try:
        model, tokenizer = assistant()
        prediction = predict_intent(model, tokenizer, question, allowed_intents())
        intent = prediction.get("predicted_intent")
        details["intent_classification"] = {
            **prediction,
            "raw_prediction": guardrails.redact_sensitive(
                str(prediction.get("raw_prediction", ""))
            ),
        }

        prompt = build_prompt(question, prompt_variant=variant, predicted_intent=intent)
        # Apply the same system-safety preface used by guarded_generate().
        prompt = guardrails.SYSTEM_SAFETY + "\n\n" + prompt
        config = {"do_sample": decoding == "sampling",
                  "max_new_tokens": 220 if variant == "constrained" else 140}
        if decoding == "sampling":
            config.update({"temperature": float(temperature), "top_p": float(top_p)})
        details.update({"prompt_strategy": variant, "decoding": decoding,
                        "generation_config": config})
        raw = str(generate_response(model, tokenizer, prompt, config)).strip()
        details["raw_response"] = guardrails.redact_sensitive(raw)
        response_text = raw
        if variant == "constrained":
            validation = _parse_json_response(raw)
            # Avoid echoing unreviewed model text through the details panel.
            details["format_validation"] = {
                k: v for k, v in validation.items() if k != "parsed_output"
            }
            if not validation["schema_valid"]:
                details["output_safety"] = {"status": "blocked", "allowed": False,
                                            "reason": "Constrained response failed JSON/schema validation"}
                return guardrails.SAFE_FALLBACK, intent or "Invalid / unknown", {
                    "input": input_safety, "output": details["output_safety"]}, details
            payload = validation["parsed_output"]
            response_text = payload["response"]
            details["json_intent_matches_classifier"] = payload["intent"] == (intent or "unknown")

        # Validate the user-facing response, not only the raw JSON wrapper.
        output_safety, safe_response = _safety_check(response_text, "output")
        details["output_safety"] = output_safety
        response_text = safe_response
        return response_text, intent or "Invalid / unknown", {
            "input": input_safety, "output": output_safety}, details
    except Exception as exc:
        details["error"] = type(exc).__name__
        return "Unable to process this request. Check the terminal for details.", "Unavailable", {
            "input": input_safety, "output": {"status": "not_run"}}, details


with gr.Blocks(title="BANKING77 | GenAI Support") as demo:
    gr.Markdown(
        "# BANKING77 — Generative AI Banking Assistant\n"
        "Educational demonstration only. Never enter real account details, passwords, PINs, "
        "OTPs or full card numbers. No real banking actions are performed."
    )
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(label="Customer question", lines=3,
                                  placeholder="My card payment was declined. What should I do?")
            variant = gr.Dropdown(PROMPT_VARIANTS, value="constrained", label="Prompt strategy")
            decoding = gr.Radio(["greedy", "sampling"], value="greedy", label="Decoding")
            temperature = gr.Slider(0.1, 1.5, value=0.7, label="Temperature (sampling)")
            top_p = gr.Slider(0.1, 1.0, value=0.9, label="Top-p (sampling)")
            submit = gr.Button("Generate response", variant="primary")
        with gr.Column():
            answer = gr.Textbox(label="Assistant response", lines=6)
            predicted_intent = gr.Textbox(label="Predicted BANKING77 intent")
            safety = gr.JSON(label="Input / output safety validation")
            details = gr.JSON(label="Classification / JSON validation / generation details")
    gr.Examples(examples=[["My card payment was declined. What should I do?"],
                          ["How do I change my card PIN?"],
                          ["How can I check my card delivery status?"],
                          ["Please help me report a lost card."]], inputs=[question])
    submit.click(run, [question, variant, decoding, temperature, top_p],
                 [answer, predicted_intent, safety, details], concurrency_limit=1)


if __name__ == "__main__":
    demo.launch()
