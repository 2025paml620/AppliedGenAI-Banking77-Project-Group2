import unittest
from src.tools.application_tools import simulate_card_lock


class ToolTests(unittest.TestCase):
    def test_valid_simulation(self):
        result = simulate_card_lock({"card_ref": "demo-card-001", "reason": "lost"})
        self.assertTrue(result.ok)
        self.assertTrue(result.data["simulated"])

    def test_unknown_reference(self):
        self.assertFalse(simulate_card_lock({"card_ref": "real-card-123", "reason": "lost"}).ok)

    def test_invalid_arguments(self):
        self.assertFalse(simulate_card_lock({"card_ref": "demo-card-001", "reason": "lost", "pin": "1234"}).ok)


if __name__ == "__main__": unittest.main()
