
"""Qwen2.5-1.5B-Instruct model loading and generation."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model(model_id: str = MODEL_ID):
    print("Loading Qwen model...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )

    model.eval()

    print("Qwen model loaded successfully.", flush=True)
    return model, tokenizer


@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    decoding_config: dict | None = None
) -> str:

    cfg = dict(decoding_config or {"do_sample": False})

    do_sample = cfg.get("do_sample", False)
    max_new_tokens = cfg.get("max_new_tokens", 140)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        generation_kwargs["temperature"] = cfg.get(
            "temperature", 0.7
        )
        generation_kwargs["top_p"] = cfg.get(
            "top_p", 1.0
        )

    # Build a clean configuration instead of inheriting
    # potentially conflicting sampling defaults.
    generation_config = GenerationConfig(
        **generation_kwargs
    )

    messages = [
        {"role": "user", "content": prompt}
    ]

    chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    encoded = tokenizer(
        chat,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        **encoded,
        generation_config=generation_config
    )

    generated_tokens = output[0][
        encoded["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

