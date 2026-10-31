
"""Unit tests for supported text-generation decoding configurations."""

import unittest
from src.model.decoding import DECODING


class DecodingTests(unittest.TestCase):

    def test_greedy(self):
        """Keep greedy decoding configured without sampling."""
        cfg = DECODING["greedy"]
        self.assertFalse(cfg["do_sample"])

    def test_sampling(self):
        """Give every sampling strategy valid positive probability settings."""
        for name, cfg in DECODING.items():
            if name == "greedy":
                continue

            with self.subTest(name=name):
                self.assertTrue(cfg["do_sample"])
                self.assertGreater(cfg["temperature"], 0)
                self.assertLessEqual(cfg["temperature"], 1)
                self.assertGreater(cfg["top_p"], 0)
                self.assertLessEqual(cfg["top_p"], 1)


if __name__ == "__main__":
    unittest.main()
