import json
import unittest
from src.pipeline import run_pipeline


class PipelineTests(unittest.TestCase):
    def test_invalid_generation_falls_back(self):
        result = run_pipeline("Where is my card?", lambda _: "not json")
        self.assertEqual(result.status, "invalid_model_output")
        self.assertTrue(result.output.response)

    def test_sensitive_input_stops_before_generation(self):
        def unexpected(_): raise AssertionError("generation should not run")
        self.assertEqual(run_pipeline("My PIN is 1234", unexpected).status, "blocked_input")

    def test_simulated_tool_flow(self):
        payload = {"intent": "lost_or_stolen_card", "response": "I can help.", "action_required": True,
                   "action": {"tool": "simulate_card_lock", "arguments": {"card_ref": "demo-card-001", "reason": "stolen"}}}
        result = run_pipeline("My card was stolen", lambda _: json.dumps(payload))
        self.assertEqual(result.status, "tool_succeeded")
        self.assertTrue(result.tool_result["data"]["simulated"])
        self.assertNotIn("have blocked", result.output.response)


if __name__ == "__main__": unittest.main()
