import unittest

from pydantic.v1 import ValidationError

from agentchat.settings import MemoryConfig


class MemoryConfigTest(unittest.TestCase):
    def test_defaults(self):
        config = MemoryConfig()
        self.assertEqual(config.recent_history_count, 6)
        self.assertEqual(config.semantic_memory_limit, 5)
        self.assertEqual(config.memory_min_score, 0.2)
        self.assertEqual(config.context_token_budget, 2000)

    def test_invalid_values_are_rejected(self):
        invalid_values = (
            {"recent_history_count": 0},
            {"semantic_memory_limit": 0},
            {"memory_min_score": -0.1},
            {"memory_min_score": 1.1},
            {"context_token_budget": 0},
            {"context_token_budget": "2000"},
        )
        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValidationError):
                    MemoryConfig(**values)


if __name__ == "__main__":
    unittest.main()
