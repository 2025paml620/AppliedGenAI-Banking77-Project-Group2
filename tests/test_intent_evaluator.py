import unittest
import tempfile
from pathlib import Path
import pandas as pd
from src.evaluation.intent_evaluator import (
    build_intent_prompt, classify_query, calculate_metrics, parse_predicted_intent,
    validate_intents, save_results,
)


class IntentEvaluationTests(unittest.TestCase):
    def setUp(self):
        """Share a minimal allowed-label vocabulary across evaluator tests."""
        self.labels = ["activate_my_card", "cash_withdrawal_pending"]

    def test_prompt_excludes_ground_truth(self):
        """Include the query and public labels in the classifier prompt."""
        prompt = build_intent_prompt("Please activate my card", self.labels)
        self.assertIn("Allowed labels", prompt)
        self.assertIn("Please activate my card", prompt)

    def test_exact_label(self):
        """Accept an exact output from the allowed label list."""
        self.assertEqual(parse_predicted_intent("activate_my_card", self.labels), "activate_my_card")

    def test_explanation_is_invalid(self):
        """Reject a label surrounded by explanatory text."""
        self.assertIsNone(parse_predicted_intent("Intent: activate_my_card because...", self.labels))

    def test_classification(self):
        """Pass the prompt to an injected generator and return its parsed label."""
        prediction, raw = classify_query("Hello", self.labels, lambda prompt: "activate_my_card")
        self.assertEqual(prediction, "activate_my_card")

    def test_metrics_invalid_counts_as_wrong(self):
        """Count a blank prediction as incorrect and invalid."""
        df = pd.DataFrame({"intent": self.labels, "predicted_intent": ["activate_my_card", ""]})
        result = calculate_metrics(df, self.labels)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["invalid_predictions"], 1)

    def test_validate_intents_sorts_and_rejects_duplicates(self):
        """Normalize label order and reject duplicate or blank vocabulary entries."""
        self.assertEqual(validate_intents([" z ", "a"]), ["a", "z"])
        with self.assertRaises(ValueError):
            validate_intents(["a", "a"])
        with self.assertRaises(ValueError):
            validate_intents([" "])

    def test_metrics_reject_empty_or_out_of_vocabulary_truth(self):
        """Reject scoring inputs with no rows or labels outside the declared classes."""
        with self.assertRaises(ValueError):
            calculate_metrics(pd.DataFrame(columns=["intent", "predicted_intent"]), self.labels)
        invalid_truth = pd.DataFrame({"intent": ["not_a_label"], "predicted_intent": ["activate_my_card"]})
        with self.assertRaises(ValueError):
            calculate_metrics(invalid_truth, self.labels)

    def test_save_results_writes_predictions_and_metrics(self):
        """Persist both row-level predictions and the metric summary."""
        frame = pd.DataFrame({"intent": ["activate_my_card"], "predicted_intent": ["activate_my_card"]})
        metrics = calculate_metrics(frame, self.labels)
        with tempfile.TemporaryDirectory() as temporary:
            save_results(frame, metrics, temporary)
            self.assertTrue((Path(temporary) / "intent_predictions.csv").is_file())
            self.assertTrue((Path(temporary) / "intent_metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
