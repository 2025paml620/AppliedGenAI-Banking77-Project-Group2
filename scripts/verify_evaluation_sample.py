"""Verify that regenerating the evaluation sample matches the saved CSV."""
# Regenerates the sample in memory and checks it against the saved CSV without modifying the CSV.

# Import the existing dataset preparation functions and comparison utilities.
import pandas as pd

from src.data.dataset import load_banking77, create_evaluation_sample

# Load the official dataset and regenerate the sample without saving over the CSV.
dataset = load_banking77()
regenerated = create_evaluation_sample(
    dataset,
    samples_per_intent=3,
    random_seed=42,
)

# Read the preserved reference file for an exact comparison.
preserved = pd.read_csv("data/evaluation_sample.csv")

# Compare the same columns in the same order, including every selected query and label.
pd.testing.assert_frame_equal(
    regenerated.reset_index(drop=True),
    preserved[regenerated.columns].reset_index(drop=True),
    check_dtype=False,
)

# Report success only if the regenerated sample matches the preserved file.
print("PASS: Regenerated sample matches the preserved CSV.")
print(f"Rows verified: {len(preserved)}")
print(f"Unique intents: {preserved['intent'].nunique()}")
print("Examples per intent:", sorted(preserved["intent"].value_counts().unique()))