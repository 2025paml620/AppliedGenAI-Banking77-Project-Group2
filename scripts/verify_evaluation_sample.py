"""Verify that regenerating the evaluation sample matches the saved CSV."""
"""Check that the pinned Banking77 sample still matches its preserved CSV."""

import logging

import pandas as pd

from src.data.dataset import load_banking77, create_evaluation_sample
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Regenerate the fixed sample in memory and compare it to the checked-in CSV."""
    configure_logging()
    logger.info("Loading the pinned Banking77 test split for sample reproducibility check")
    dataset = load_banking77()
    regenerated = create_evaluation_sample(dataset, samples_per_intent=3, random_seed=42)
    preserved = pd.read_csv("data/evaluation_sample.csv")

    pd.testing.assert_frame_equal(
        regenerated.reset_index(drop=True),
        preserved[regenerated.columns].reset_index(drop=True),
        check_dtype=False,
    )

    counts = sorted(preserved["intent"].value_counts().unique())
    print("PASS: Regenerated sample matches the preserved CSV.")
    print(f"Rows verified: {len(preserved)}")
    print(f"Unique intents: {preserved['intent'].nunique()}")
    print("Examples per intent:", counts)
    logger.info("Verified %d rows across %d intents; examples_per_intent=%s",
                len(preserved), preserved["intent"].nunique(), counts)


if __name__ == "__main__":
    main()
