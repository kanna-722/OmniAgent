import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agentchat.core.agents.execution_summary_skill import ExecutionSummarySkill


class FakeSummaryModel:
    def __init__(self, content: str):
        self.content = content
        self.call_count = 0
        self.bind_tools_call_count = 0
        self.last_input = None
        self.last_config = None

    def bind_tools(self, tools):
        self.bind_tools_call_count += 1
        raise AssertionError("总结模型不应绑定工具")

    async def ainvoke(self, messages, config=None):
        self.call_count += 1
        self.last_input = messages
        self.last_config = config
        return AIMessage(content=self.content)


class ExecutionSummarySkillTest(unittest.IsolatedAsyncioTestCase):
    async def test_summary_model_is_invoked_exactly_once_without_tools(self):
        expected = (
            "## 已完成工作\n已查询天气。\n"
            "## 已获得结果\n北京晴。\n"
            "## 未完成事项\n暂无。\n"
            "## 建议下一步\n结束任务。"
        )
        model = FakeSummaryModel(expected)
        skill = ExecutionSummarySkill(model)

        result = await skill.ainvoke(
            original_messages=[HumanMessage(content="查询北京天气")],
            execution_messages=[
                HumanMessage(content="查询北京天气"),
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "weather",
                        "args": {"city": "北京"},
                        "id": "call-1",
                        "type": "tool_call",
                    }],
                ),
                ToolMessage(
                    content="北京晴",
                    name="weather",
                    tool_call_id="call-1",
                ),
            ],
            callbacks=["usage-callback"],
        )

        self.assertEqual(result, expected)
        self.assertEqual(model.call_count, 1)
        self.assertEqual(model.last_config, {"callbacks": ["usage-callback"]})
        self.assertEqual(model.bind_tools_call_count, 0)
        self.assertIn("助手请求工具 weather", model.last_input[1].content)
        self.assertIn("工具 weather：北京晴", model.last_input[1].content)

    async def test_system_messages_are_not_copied_into_summary_transcript(self):
        model = FakeSummaryModel(
            "## 已完成工作\n暂无。\n"
            "## 已获得结果\n暂无。\n"
            "## 未完成事项\n暂无。\n"
            "## 建议下一步\n暂无。"
        )
        skill = ExecutionSummarySkill(model)

        await skill.ainvoke(
            original_messages=[HumanMessage(content="原始问题")],
            execution_messages=[
                SystemMessage(content="内部系统提示"),
                HumanMessage(content="原始问题"),
            ],
        )

        transcript = model.last_input[1].content
        self.assertIn("用户：原始问题", transcript)
        self.assertNotIn("内部系统提示", transcript)

    async def test_empty_summary_is_rejected(self):
        skill = ExecutionSummarySkill(FakeSummaryModel("  "))

        with self.assertRaisesRegex(ValueError, "空内容"):
            await skill.ainvoke(
                original_messages=[HumanMessage(content="原始问题")],
                execution_messages=[],
            )

    async def test_summary_without_required_sections_is_rejected(self):
        skill = ExecutionSummarySkill(FakeSummaryModel("只有一段普通总结"))

        with self.assertRaisesRegex(ValueError, "缺少必要章节"):
            await skill.ainvoke(
                original_messages=[HumanMessage(content="原始问题")],
                execution_messages=[],
            )


if __name__ == "__main__":
    unittest.main()
