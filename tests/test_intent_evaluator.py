import unittest
import pandas as pd
from src.evaluation.intent_evaluator import (
    build_intent_prompt, classify_query, calculate_metrics, parse_predicted_intent
)


class IntentEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.labels = ["activate_my_card", "cash_withdrawal_pending"]

    def test_prompt_excludes_ground_truth(self):
        prompt = build_intent_prompt("Please activate my card", self.labels)
        self.assertIn("Allowed labels", prompt)
        self.assertIn("Please activate my card", prompt)

    def test_exact_label(self):
        self.assertEqual(parse_predicted_intent("activate_my_card", self.labels), "activate_my_card")

    def test_explanation_is_invalid(self):
        self.assertIsNone(parse_predicted_intent("Intent: activate_my_card because...", self.labels))

    def test_classification(self):
        prediction, raw = classify_query("Hello", self.labels, lambda prompt: "activate_my_card")
        self.assertEqual(prediction, "activate_my_card")

    def test_metrics_invalid_counts_as_wrong(self):
        df = pd.DataFrame({"intent": self.labels, "predicted_intent": ["activate_my_card", ""]})
        result = calculate_metrics(df, self.labels)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["invalid_predictions"], 1)


if __name__ == "__main__":
    unittest.main()
