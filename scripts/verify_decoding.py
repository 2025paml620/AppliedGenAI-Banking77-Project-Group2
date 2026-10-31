"""Run the four configured generation strategies and show their effective settings."""

import logging

from transformers import GenerationConfig

from src.logging_config import configure_logging
from src.model.decoding import DECODING
from src.model.llm_baseline import generate_response, load_model

logger = logging.getLogger(__name__)


def main() -> None:
    """Load the shared model and compare each configured decoding strategy."""
    configure_logging()
    model, tokenizer = load_model()
    query = "How can I activate my new debit card?"
    logger.info("Starting decoding verification for %d strategies", len(DECODING))

    for name, config in DECODING.items():
        logger.info("Generating a verification response with strategy %s", name)
        print(f"\nTesting: {name}", flush=True)
        effective = GenerationConfig(
            do_sample=config["do_sample"],
            max_new_tokens=80,
            pad_token_id=tokenizer.eos_token_id,
            **({
                "temperature": config["temperature"],
                "top_p": config["top_p"],
            } if config["do_sample"] else {}),
        )
        logger.info("Prepared effective generation settings for strategy %s", name)
        print("Effective configuration:", effective.to_dict())
        response = generate_response(
            model,
            tokenizer,
            query,
            {**config, "max_new_tokens": 80},
        )
        print("Response:", response, flush=True)
        logger.info("Completed decoding strategy %s", name)


if __name__ == "__main__":
    main()
