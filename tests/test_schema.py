import unittest
from src.output.schema import INTENTS, parse_output, validate_output


class SchemaTests(unittest.TestCase):
    def test_valid_contract(self):
        result = validate_output({"intent": "card_arrival", "response": "Check the official app.",
                                  "action_required": False, "action": None})
        self.assertTrue(result.valid)

    def test_reject_unknown_intent(self):
        self.assertFalse(validate_output({"intent": "made_up", "response": "Help", "action_required": False,
                                          "action": None}).valid)

    def test_reject_unverified_action_claim(self):
        result = validate_output({"intent": "card_arrival", "response": "I have blocked your card.",
                                  "action_required": False, "action": None})
        self.assertFalse(result.valid)

    def test_reject_wrong_action_semantics(self):
        self.assertFalse(validate_output({"intent": "lost_or_stolen_card", "response": "I can help.",
                                          "action_required": False, "action": {}}).valid)

    def test_parse_fenced_json(self):
        result = parse_output('```json\n{"intent":"card_arrival","response":"Use the official app.","action_required":false,"action":null}\n```')
        self.assertTrue(result.valid)

    def test_all_77_labels_present(self):
        self.assertEqual(len(INTENTS), 77)


if __name__ == "__main__": unittest.main()
