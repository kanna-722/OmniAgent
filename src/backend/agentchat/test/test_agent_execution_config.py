import unittest

from pydantic.v1 import ValidationError

from agentchat.settings import AgentExecutionConfig


class AgentExecutionConfigTest(unittest.TestCase):
    def test_default_recursion_limit_is_25(self):
        self.assertEqual(AgentExecutionConfig().recursion_limit, 25)

    def test_positive_recursion_limit_is_accepted(self):
        self.assertEqual(AgentExecutionConfig(recursion_limit=10).recursion_limit, 10)

    def test_non_positive_recursion_limit_is_rejected(self):
        for invalid_value in (0, -1):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises(ValidationError):
                    AgentExecutionConfig(recursion_limit=invalid_value)

    def test_non_integer_recursion_limit_is_rejected(self):
        for invalid_value in ("25", 2.5, True):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises(ValidationError):
                    AgentExecutionConfig(recursion_limit=invalid_value)


if __name__ == "__main__":
    unittest.main()
