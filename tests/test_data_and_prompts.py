"""Unit tests for deterministic sample preparation and prompt selection."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.data.dataset import create_evaluation_sample, save_evaluation_sample
from src.prompts.templates import build_prompt, build_zero_shot_intent_response_prompt
from src.prompts.structured import build_structured_prompt


class FakeSplit:
    """Provide the small Hugging Face split interface used by the sampler."""

    features = {"label": SimpleNamespace(names=["card_arrival", "cash_withdrawal"])}

    def __init__(self, frame):
        """Keep a private frame so test sampling cannot mutate its source."""
        self.frame = frame

    def to_pandas(self):
        """Return a copy like the actual dataset conversion does."""
        return self.frame.copy()


def fake_dataset():
    """Make two labels with enough source rows for a balanced sample."""
    frame = pd.DataFrame({"text": ["a", "b", "c", "d", "e", "f"],
                          "label": [0, 0, 0, 1, 1, 1]})
    return {"test": FakeSplit(frame)}


def test_evaluation_sample_is_balanced_stable_and_keeps_expected_columns():
    """Check the sample contract, equal class counts, and repeatability."""
    first = create_evaluation_sample(fake_dataset(), samples_per_intent=2, random_seed=17)
    second = create_evaluation_sample(fake_dataset(), samples_per_intent=2, random_seed=17)

    assert first.equals(second)
    assert list(first.columns) == ["sample_id", "source_index", "text", "label", "intent"]
    assert first["intent"].value_counts().to_dict() == {"card_arrival": 2, "cash_withdrawal": 2}
    assert first["sample_id"].str.fullmatch(r"test_\d{5}").all()


def test_evaluation_sample_rejects_unavailable_per_class_count():
    """Fail clearly if any class cannot supply the requested sample size."""
    with pytest.raises(ValueError, match="not have enough examples"):
        create_evaluation_sample(fake_dataset(), samples_per_intent=4)


def test_save_evaluation_sample_creates_parent_and_csv(tmp_path):
    """Write the selected sample without an accidental pandas index column."""
    frame = create_evaluation_sample(fake_dataset(), samples_per_intent=1)
    destination = tmp_path / "nested" / "sample.csv"

    assert save_evaluation_sample(frame, destination) == destination
    loaded = pd.read_csv(destination)
    assert list(loaded.columns) == list(frame.columns)
    assert len(loaded) == 2


@pytest.mark.parametrize("variant", ["zero_shot", "few_shot", "role_based", "constrained"])
def test_prompt_builder_supports_every_variant(variant):
    """Return a prompt with the customer question for each named strategy."""
    prompt = build_prompt("My card is delayed", prompt_variant=variant, predicted_intent="card_arrival")

    assert "My card is delayed" in prompt
    assert "Predicted intent" in prompt


def test_prompt_builders_reject_unknown_variants_and_missing_contract_data():
    """Reject unsupported prompt names and empty intent vocabularies."""
    with pytest.raises(ValueError, match="Unknown prompt variant"):
        build_prompt("Help", prompt_variant="unknown")
    with pytest.raises(ValueError, match="allowed_intents"):
        build_zero_shot_intent_response_prompt("Help", [])
    with pytest.raises(ValueError, match="must not be blank"):
        build_structured_prompt("  ", ["card_arrival"])


def test_structured_prompt_includes_only_label_names_and_tool_contract():
    """Check the rendered prompt advertises labels and available mock actions."""
    prompt = build_structured_prompt("My card was stolen", ["lost_or_stolen_card"])

    assert "lost_or_stolen_card" in prompt
    assert "block_card" in prompt
    assert "card_status" in prompt
    assert "transaction_status" in prompt
