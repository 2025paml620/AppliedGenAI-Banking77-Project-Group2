
"""Reusable utilities for loading and preparing the Banking77 dataset."""

from pathlib import Path
import pandas as pd
from datasets import DatasetDict, load_dataset

# Identify the dataset repository and pin a specific revision for reproducibility.
DATASET_ID = "PolyAI/banking77"
DATASET_REVISION = "90d4e2ee5521c04fc1488f065b8b083658768c57"


def load_banking77() -> DatasetDict:
    """Load Banking77 from the explicitly pinned repository revision."""
    # Pin the revision so future runs do not silently use a newer dataset version.
    return load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        trust_remote_code=True,
        )


def create_evaluation_sample(
    dataset: DatasetDict,
    samples_per_intent: int = 3,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Create a reproducible, balanced sample from the official test split."""

    if samples_per_intent < 1:
        raise ValueError("samples_per_intent must be at least 1.")

    # Convert the official test split to a DataFrame and preserve source row IDs.
    test_df = dataset["test"].to_pandas().reset_index(names="source_index")

    # Map integer labels to their human-readable intent names.
    label_names = dataset["test"].features["label"].names
    test_df["intent"] = test_df["label"].map(
        lambda label_id: label_names[label_id]
    )

    # Verify that every intent has enough examples for the requested sample.
    intent_counts = test_df["label"].value_counts()
    insufficient = intent_counts[intent_counts < samples_per_intent]

    if not insufficient.empty:
        raise ValueError(
            "Some intents do not have enough examples in the test split: "
            f"{insufficient.to_dict()}"
        )

    # Select the same number of examples per intent using a fixed random seed.
    sample_df = (
        test_df.groupby("label", group_keys=False)
        .sample(n=samples_per_intent, random_state=random_seed)
        .sort_values(["label", "source_index"])
        .reset_index(drop=True)
    )

    # Create stable identifiers that refer to the original test-split rows.
    sample_df["sample_id"] = sample_df["source_index"].map(
        lambda index: f"test_{index:05d}"
    )

    # Return only the fields required for evaluation and reproducibility.
    return sample_df[
        ["sample_id", "source_index", "text", "label", "intent"]
    ]


def save_evaluation_sample(
    sample_df: pd.DataFrame,
    output_path: str = "data/evaluation_sample.csv",
) -> Path:
    """Save the evaluation sample as a CSV file."""

    # Create the destination directory if it does not already exist.
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save the sample without adding an extra pandas index column.
    sample_df.to_csv(path, index=False, encoding="utf-8")

    return path
