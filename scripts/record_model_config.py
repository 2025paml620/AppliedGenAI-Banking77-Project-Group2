import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
import huggingface_hub
import tokenizers


from src.model.llm_baseline import MODEL_ID
from src.model.decoding import DECODING
from huggingface_hub import model_info


def main():
    """Collect model, tokenizer, runtime, and generation metadata for reproducibility."""
    output_dir = Path("experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoConfig, AutoTokenizer

    print("Reading model and tokenizer configuration...")

    model_config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model_revision = model_info(MODEL_ID).sha

    print("Model ID:", MODEL_ID)
    print("Current Hub revision:", model_revision)

    cuda_available = torch.cuda.is_available()

    config = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),

        "model": {
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "model_type": model_config.model_type,
        "architecture": model_config.architectures,
        "hidden_size": getattr(model_config, "hidden_size", None),
        "num_hidden_layers": getattr(
            model_config, "num_hidden_layers", None
        ),
        "num_attention_heads": getattr(
            model_config, "num_attention_heads", None
        ),
        "num_key_value_heads": getattr(
            model_config, "num_key_value_heads", None
        ),
        "vocab_size": getattr(model_config, "vocab_size", None),
        },
        

        "tokenizer": {
            "tokenizer_class": tokenizer.__class__.__name__,
            "vocab_size": tokenizer.vocab_size,
            "pad_token": tokenizer.pad_token,
            "eos_token": tokenizer.eos_token,
            "padding_side": tokenizer.padding_side,
        },

        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "device": (
                torch.cuda.get_device_name(0)
                if cuda_available
                else "CPU"
            ),
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "torch_num_threads": torch.get_num_threads(),
        },

        "libraries": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "tokenizers": tokenizers.__version__,
        },

        "generation": {
            "decoding_strategies": DECODING,
            "default_max_new_tokens": 140,
        },
    }

    output_file = output_dir / "model_configuration.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"\nModel configuration saved: {output_file}")
    print(json.dumps(config, indent=2, default=str))


if __name__ == "__main__":
    main()
