
"""Prepare and save the reproducible Banking77 evaluation sample."""

import logging

from src.data.dataset import (
    load_banking77,
    create_evaluation_sample,
    save_evaluation_sample,
)
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main():
    """Regenerate the fixed, balanced sample from the pinned Banking77 test split."""
    configure_logging()
    logger.info("Loading the pinned Banking77 test split")
    # Load Banking77 and create a balanced sample from the official test split.
    dataset = load_banking77()
    evaluation_sample = create_evaluation_sample(
        dataset,
        samples_per_intent=3,
        random_seed=42,
    )

    # Save the evaluation sample so all project experiments use the same data.
    output_path = save_evaluation_sample(evaluation_sample)
    logger.info("Saved %d examples across %d intents to %s", len(evaluation_sample), evaluation_sample["intent"].nunique(), output_path)

    # Print summary checks to verify the generated artifact.
    print(f"Evaluation samples: {len(evaluation_sample)}")
    print(f"Unique intents: {evaluation_sample['intent'].nunique()}")
    print(
        "Examples per intent:",
        sorted(evaluation_sample["intent"].value_counts().unique()),
    )
    print(f"Saved evaluation sample to: {output_path}")


if __name__ == "__main__":
    main()
