"""BANKING77 educational Gradio app with Part 2 and Part 3 execution paths."""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from src.logging_config import configure_logging
from src.model.llm_baseline import generate_response, load_model
from src.model.intent_classifier import predict_intent
from src.prompts.templates import build_prompt
from src.rag.retriever import build_retriever
from src.safety import guardrails
from src.structured.pipeline import handle_query
from src.structured.schema import validate_output


configure_logging()
logger = logging.getLogger(__name__)

PROMPT_VARIANTS = ["zero_shot", "few_shot", "role_based", "constrained"]
PROMPTED_MODE = "Prompted response"
STRUCTURED_MODE = "Structured response + tools"
SAMPLE_PROMPTS = [
    "My card payment was declined. What should I do?",
    "How do I change my card PIN?",
    "How can I check my card delivery status?",
    "My card ending 5678 was stolen. Please block it.",
    "What is the status of transaction TXN1002?",
]
MAX_QUESTION_LENGTH = 1500
SAFETY_NOTICE = (
    "Educational demonstration only. Never enter real account details, passwords, PINs, "
    "OTPs or full card numbers. No real banking actions are performed."
)
EVALUATION_CSV = Path(__file__).resolve().parent / "data" / "evaluation_sample.csv"
SAFE_FALLBACK = (
    "I can't safely provide a response to that request. "
    "Please contact your bank using its official app or support channel."
)


@lru_cache(maxsize=1)
def assistant():
    """Load and cache the shared Qwen model/tokenizer pair for both app modes."""
    return load_model()


@lru_cache(maxsize=1)
def allowed_intents() -> tuple[str, ...]:
    """Return the 77 public labels without using a query row's expected label as context."""
    if not EVALUATION_CSV.is_file():
        raise FileNotFoundError(f"Missing intent vocabulary source: {EVALUATION_CSV}")
    labels = pd.read_csv(EVALUATION_CSV, usecols=["intent"])["intent"].dropna()
    vocabulary = tuple(sorted(set(labels.astype(str).str.strip()) - {""}))
    if len(vocabulary) != 77:
        raise ValueError(f"Expected 77 BANKING77 labels, found {len(vocabulary)}")
    return vocabulary


@lru_cache(maxsize=1)
def rag_retriever():
    """Build and cache a retriever backed only by the official Banking77 train split."""
    return build_retriever()


def _result(
    query: str, status: str, intent: str | None, errors: list[str], raw_output: str | None,
    final_response: str, tool: str | None = None, tool_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the stable response fields shared by both UI execution modes."""
    return {
        "query": guardrails.redact_sensitive(query),
        "status": status,
        "intent": intent,
        "tool": tool,
        "tool_result": tool_result,
        "errors": errors,
        "raw_output": guardrails.redact_sensitive(raw_output) if raw_output is not None else None,
        "final_response": final_response,
    }


def _ui_outputs(
    result: dict[str, Any], safety_details: dict[str, Any],
    tool_details: dict[str, Any] | None = None, run_details: dict[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any], str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Shape a pipeline result for the answer, safety, tool, and JSON UI panels."""
    action = (tool_details or {}).get("action")
    if action:
        if (tool_details or {}).get("tool_result"):
            tool_label = f"{action['tool']} - executed against mock data"
        elif result["status"] == "tool_error":
            tool_label = f"{action['tool']} - rejected"
        else:
            tool_label = f"{action['tool']} - suggested, not executed"
    else:
        tool_label = "No mock tool action suggested"
    return (
        result["final_response"],
        result["intent"] or "Not classified",
        safety_details,
        tool_label,
        tool_details or {"action_required": False, "action": None, "tool_result": None},
        result,
        run_details or {},
    )


def run_part2(
    question: str, variant: str, decoding: str, temperature: float, top_p: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Execute the Part 2 closed-set intent classification and response generation flow."""
    query = (question or "").strip()
    if not query or len(query) > MAX_QUESTION_LENGTH:
        return (
            _result(query, "invalid_input", None, ["query_length_out_of_range"], None,
                    "Enter a question between 1 and 1500 characters."),
            {"input": {"status": "not_run"}, "output": {"status": "not_run"}},
            {"action_required": False, "action": None, "tool_result": None},
        )

    input_check = guardrails.check_input(query)
    if input_check.status != "allowed":
        return (
            _result(query, "blocked_input", None, list(input_check.reasons), None, input_check.response),
            {"input": {"status": input_check.status, "allowed": False,
                       "reasons": list(input_check.reasons)}, "output": {"status": "not_run"}},
            {"action_required": False, "action": None, "tool_result": None},
        )
    if variant not in PROMPT_VARIANTS or decoding not in {"greedy", "sampling"}:
        return (
            _result(query, "invalid_input", None, ["invalid_generation_settings"], None,
                    "Invalid generation settings."),
            {"input": {"status": "allowed", "allowed": True, "reasons": []},
             "output": {"status": "not_run"}},
            {"action_required": False, "action": None, "tool_result": None},
        )

    model, tokenizer = assistant()
    prediction = predict_intent(model, tokenizer, query, allowed_intents())
    intent = prediction.get("predicted_intent")
    prompt = guardrails.SYSTEM_SAFETY + "\n\n" + build_prompt(
        query, prompt_variant=variant, predicted_intent=intent,
    )
    config: dict[str, Any] = {
        "do_sample": decoding == "sampling",
        "max_new_tokens": 220 if variant == "constrained" else 140,
    }
    if decoding == "sampling":
        config.update({"temperature": float(temperature), "top_p": float(top_p)})
    raw = str(generate_response(model, tokenizer, prompt, config)).strip()
    output_check = guardrails.check_output(raw)
    status = "ok" if output_check.status == "allowed" else "unsafe_output"
    errors = [] if status == "ok" else list(output_check.reasons)
    if not prediction.get("valid_prediction"):
        errors.append("invalid_intent_prediction")
    response = raw if status == "ok" else output_check.response
    result = _result(query, status, intent, errors, raw, response)
    safety = {
        "input": {"status": "allowed", "allowed": True, "reasons": []},
        "output": {"status": output_check.status, "allowed": output_check.status == "allowed",
                   "reasons": list(output_check.reasons)},
    }
    details = {
        "execution_mode": PROMPTED_MODE,
        "prompt_strategy": variant,
        "decoding": decoding,
        "generation_config": config,
        "intent_classification": {
            **prediction,
            "raw_prediction": guardrails.redact_sensitive(str(prediction.get("raw_prediction", ""))),
        },
    }
    return result, safety, {"action_required": False, "action": None, "tool_result": None, "details": details}


def run_part3(question: str, decoding: str, temperature: float, top_p: float, use_rag: bool):
    """Execute the Part 3 JSON validation and mock-tool pipeline, optionally with train-only RAG."""
    query = (question or "").strip()
    if not query or len(query) > MAX_QUESTION_LENGTH:
        result = _result(query, "invalid_input", None, ["query_length_out_of_range"], None,
                         "Enter a question between 1 and 1500 characters.")
        return result, {"input": {"status": "not_run"}, "output": {"status": "not_run"}}, {}, {
            "execution_mode": STRUCTURED_MODE, "retrieval": "RAG" if use_rag else "Non-RAG",
        }
    if decoding not in {"greedy", "sampling"}:
        result = _result(query, "invalid_input", None, ["invalid_decoding_setting"], None,
                         "Invalid decoding setting.")
        return result, {}, {}, {"execution_mode": STRUCTURED_MODE}

    input_check = guardrails.check_input(query)
    if input_check.status != "allowed":
        result = _result(query, "blocked_input", None, list(input_check.reasons), None, input_check.response)
        return result, {"input": {"status": input_check.status, "allowed": False,
                                  "reasons": list(input_check.reasons)}, "output": {"status": "not_run"}}, {}, {
            "execution_mode": STRUCTURED_MODE, "retrieval": "RAG" if use_rag else "Non-RAG",
        }

    model, tokenizer = assistant()
    config: dict[str, Any] = {"do_sample": decoding == "sampling", "max_new_tokens": 200}
    if decoding == "sampling":
        config.update({"temperature": float(temperature), "top_p": float(top_p)})
    retriever = rag_retriever() if use_rag else None
    result = handle_query(
        query,
        allowed_intents(),
        lambda prompt: generate_response(model, tokenizer, prompt, config),
        retriever=retriever,
    )
    # Keep the original completion for diagnostics; show the routed JSON in the response object.
    model_output = result.get("model_output")
    parsed, schema_errors = validate_output(result["raw_output"] or "", allowed_intents())
    result = _result(
        result["query"], result["status"], result["intent"], result["errors"],
        result["raw_output"], result["final_response"], result["tool"], result["tool_result"],
    )
    action_required = bool(parsed and parsed["action_required"])
    action = parsed["action"] if parsed else None
    tool_details = {
        "action_required": action_required,
        "action": action,
        "tool_result": result["tool_result"],
        "execution_status": result["status"],
        "errors": result["errors"] or schema_errors,
        "notice": "Any tool shown here runs against local mock data only.",
    }
    safety = {
        "input": {"status": "allowed", "allowed": True, "reasons": []},
        "output": {"status": result["status"],
                   "allowed": result["status"] not in {"unsafe_output", "invalid_output"},
                   "reasons": result["errors"]},
    }
    details = {
        "execution_mode": STRUCTURED_MODE,
        "retrieval": "RAG" if use_rag else "Non-RAG",
        "decoding": decoding,
        "generation_config": config,
        "json_schema_valid": parsed is not None,
    }
    if model_output is not None and model_output != result["raw_output"]:
        details["original_model_output"] = guardrails.redact_sensitive(str(model_output))
    return result, safety, tool_details, details


def run_request(
    question: str, mode: str, variant: str, decoding: str, temperature: float,
    top_p: float, retrieval: str,
):
    """Dispatch the UI request to Part 2 or Part 3 and format all seven UI outputs."""
    try:
        if mode == STRUCTURED_MODE:
            result, safety, tool, details = run_part3(
                question, decoding, temperature, top_p, retrieval == "RAG",
            )
        else:
            result, safety, tool = run_part2(question, variant, decoding, temperature, top_p)
            details = tool.pop("details", {})
        logger.info("Assistant request completed: mode=%s status=%s tool=%s",
                    mode, result["status"], result["tool"] or "none")
        return _ui_outputs(result, safety, tool, details)
    except Exception as exc:
        logger.exception("Assistant request failed (%s)", type(exc).__name__)
        query = (question or "").strip()
        result = _result(query, "pipeline_error", None, [type(exc).__name__], None, SAFE_FALLBACK)
        return _ui_outputs(
            result,
            {"input": {"status": "not_run"}, "output": {"status": "not_run"}},
            {"action_required": False, "action": None, "tool_result": None},
            {"execution_mode": mode, "error": type(exc).__name__},
        )


def _mode_visibility(mode: str):
    """Show controls that apply to the selected Part 2 or Part 3 path."""
    part3 = mode == STRUCTURED_MODE
    return gr.update(visible=not part3), gr.update(visible=part3)


def _decoding_visibility(decoding: str):
    """Show sampling controls only when sampling is the selected decoding mode."""
    return gr.update(visible=decoding == "sampling")


def _load_sample_prompt(sample_prompt: str | None) -> str:
    """Copy the selected sample into the customer query field."""
    return sample_prompt or ""


with gr.Blocks(
    title="BANKING77 | GenAI Support",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green", neutral_hue="slate"),
    css="""
    :root {
        color-scheme: light !important;
        --body-background-fill: #f8fafc !important;
        --background-fill-primary: #ffffff !important;
        --background-fill-secondary: #f8fafc !important;
        --block-background-fill: #ffffff !important;
        --body-text-color: #172033 !important;
        --body-text-color-subdued: #475569 !important;
        --block-label-text-color: #334155 !important;
        --block-label-text-color-subdued: #475569 !important;
        --input-background-fill: #ffffff !important;
        --input-background-fill-hover: #f8fafc !important;
        --input-border-color: #cbd5e1 !important;
        --input-placeholder-color: #64748b !important;
    }
    body, .gradio-container, button, input, textarea, select {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    }
    .gradio-container {
        max-width: 1360px !important;
        margin: 0 auto !important;
        background: #f8fafc !important;
        color: #172033 !important;
    }
    .app-title h1 {
        color: #102a43 !important;
        font-size: 2.05rem !important;
        font-weight: 750 !important;
        letter-spacing: -0.025em !important;
        line-height: 1.2 !important;
    }
    .section-heading h2 {
        color: #17324d !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        line-height: 1.3 !important;
        margin: 0 0 14px !important;
    }
    .gradio-container p {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
        color: #334155 !important;
        font-size: 1rem !important;
        line-height: 1.55 !important;
    }
    .safety-banner {
        border: 1px solid #b42318 !important;
        border-left: 7px solid #b42318 !important;
        border-radius: 12px !important;
        background-color: #fff1f0 !important;
        color: #7a271a !important;
        padding: 14px 18px !important;
        margin: 10px 0 20px !important;
        font-size: 1.02rem !important;
        line-height: 1.55 !important;
        font-weight: 650 !important;
    }
    .safety-banner * {
        color: #7a271a !important;
        background-color: transparent !important;
    }
    .safety-banner p {
        margin: 0 !important;
    }
    .input-panel, .output-panel {
        border: 1px solid #d8e1ec !important;
        border-radius: 16px !important;
        padding: 20px !important;
        background: #ffffff !important;
        box-shadow: 0 4px 16px rgba(15, 23, 42, .045) !important;
    }
    .input-panel label, .output-panel label {
        color: #334155 !important;
        font-size: .94rem !important;
        font-weight: 620 !important;
    }
    .input-panel .info,
    .input-panel small,
    .input-panel [data-testid="info"],
    .input-panel .description {
        color: #475569 !important;
        opacity: 1 !important;
    }
    .input-panel input,
    .input-panel textarea,
    .input-panel select,
    .input-panel [role="listbox"] {
        background-color: #ffffff !important;
        color: #172033 !important;
        border-color: #cbd5e1 !important;
    }
    .input-panel input::placeholder,
    .input-panel textarea::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }
    .input-panel [role="radiogroup"] label,
    .input-panel [role="radio"] {
        color: #334155 !important;
    }
    .input-panel [role="option"] {
        background-color: #ffffff !important;
        color: #172033 !important;
    }
    .generate-button {
        min-height: 44px !important;
        width: 100% !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
        font-weight: 650 !important;
    }
    .tool-action-panel {
        border: 2px solid #087e5b !important;
        border-left: 7px solid #087e5b !important;
        border-radius: 14px !important;
        background: linear-gradient(135deg, #e9f8f1, #eef5ff) !important;
        box-shadow: 0 5px 18px rgba(8, 126, 91, .12) !important;
    }
    .tool-action-panel > div:first-child {
        color: #075e45 !important;
        font-weight: 700 !important;
        font-size: 1.06rem !important;
    }
    .tool-action-panel [data-testid="textbox"] input {
        font-weight: 650 !important;
    }
    """,
) as demo:
    gr.Markdown(
        "# BANKING77 | Banking Support Assistant\n"
        "Choose a response mode, try a sample request, and inspect the result.",
        elem_classes=["app-title"],
    )
    gr.Markdown(f"**SAFETY NOTICE**  \n{SAFETY_NOTICE}", elem_classes=["safety-banner"])
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=400, elem_classes=["input-panel"]):
            gr.Markdown("## Customer request", elem_classes=["section-heading"])
            mode = gr.Radio(
                [PROMPTED_MODE, STRUCTURED_MODE],
                value=PROMPTED_MODE, label="Execution mode",
            )
            sample_prompt = gr.Dropdown(
                choices=SAMPLE_PROMPTS,
                value=None,
                label="Try a sample prompt",
                info="Choose one to fill the Customer Query field.",
            )
            question = gr.Textbox(
                label="Customer Query", lines=3,
                placeholder="My card payment was declined. What should I do?",
            )
            with gr.Column(visible=True) as part2_options:
                variant = gr.Dropdown(PROMPT_VARIANTS, value="constrained", label="Prompt strategy")
            with gr.Column(visible=False) as part3_options:
                retrieval = gr.Radio(["Non-RAG", "RAG"], value="Non-RAG", label="Knowledge retrieval")
            decoding = gr.Radio(["greedy", "sampling"], value="greedy", label="Decoding")
            with gr.Column(visible=False) as sampling_options:
                temperature = gr.Slider(0.1, 1.5, value=0.7, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, label="Top-p")
            submit = gr.Button("Generate response", variant="primary", size="md",
                               elem_classes=["generate-button"])
        with gr.Column(scale=1, min_width=400, elem_classes=["output-panel"]):
            gr.Markdown("## Assistant response", elem_classes=["section-heading"])
            answer = gr.Textbox(label="Response", lines=5, interactive=False)
            predicted_intent = gr.Textbox(label="Predicted BANKING77 intent")
            safety = gr.JSON(label="Input / output safety validation")
            with gr.Accordion("Suggested tool and execution details", open=True,
                              elem_classes=["tool-action-panel"]):
                tool_action = gr.Textbox(label="Tool suggestion")
                tool_details = gr.JSON(label="Tool arguments, result, and status")
            result_json = gr.JSON(label="Complete response object")
            details = gr.JSON(label="Mode, decoding, retrieval, and validation details")
    mode.change(_mode_visibility, inputs=[mode], outputs=[part2_options, part3_options])
    decoding.change(_decoding_visibility, inputs=[decoding], outputs=[sampling_options])
    sample_prompt.change(_load_sample_prompt, inputs=[sample_prompt], outputs=[question])
    submit.click(
        run_request,
        [question, mode, variant, decoding, temperature, top_p, retrieval],
        [answer, predicted_intent, safety, tool_action, tool_details, result_json, details],
        concurrency_limit=1,
    )


if __name__ == "__main__":
    demo.launch()
