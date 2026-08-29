import unittest
from dataclasses import dataclass

from agentchat.services.memory.context import (
    build_budgeted_memory_context,
    build_memory_write_kwargs,
    estimate_context_tokens,
    format_memory_context,
    merge_memory_results,
    retrieve_mixed_memories,
)


@dataclass
class FakeMessage:
    type: str
    content: str


class FakeMemoryClient:
    def __init__(self, primary=None, legacy=None, primary_error=None, legacy_error=None):
        self.primary = primary or []
        self.legacy = legacy or []
        self.primary_error = primary_error
        self.legacy_error = legacy_error
        self.calls = []

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        if "run_id" in kwargs:
            if self.legacy_error:
                raise self.legacy_error
            return {"results": self.legacy}
        if self.primary_error:
            raise self.primary_error
        return {"results": self.primary}


class MemoryContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scope = {
            "user_id": "user-1",
            "agent_id": "agent-1",
            "dialog_id": "dialog-1",
            "recent_messages": [],
            "limit": 5,
            "min_score": 0.2,
        }

    def test_filters_wrong_user_agent_dialog_and_low_score(self):
        primary = [
            {"id": "ok", "memory": "正确记忆", "score": 0.9, "user_id": "user-1", "agent_id": "agent-1"},
            {"id": "wrong-user", "memory": "串用户", "score": 0.9, "user_id": "user-2", "agent_id": "agent-1"},
            {"id": "wrong-agent", "memory": "串 Agent", "score": 0.9, "user_id": "user-1", "agent_id": "agent-2"},
            {"id": "low", "memory": "低相关", "score": 0.19, "user_id": "user-1", "agent_id": "agent-1"},
        ]
        legacy = [
            {"id": "legacy", "memory": "旧记忆", "score": 0.8, "run_id": "dialog-1"},
            {"id": "wrong-dialog", "memory": "其他会话旧记忆", "score": 0.9, "run_id": "dialog-2"},
        ]

        result = merge_memory_results(primary, legacy, **self.scope)

        self.assertEqual([item["id"] for item in result], ["ok", "legacy"])

    def test_deduplicates_by_id_and_content_preferring_best_result(self):
        primary = [
            {"id": "same-id", "memory": "旧内容", "score": 0.4, "user_id": "user-1", "agent_id": "agent-1"},
            {"id": "content-a", "memory": "喜欢 Python", "score": 0.7, "user_id": "user-1", "agent_id": "agent-1"},
        ]
        legacy = [
            {"id": "same-id", "memory": "新内容", "score": 0.8, "run_id": "dialog-1"},
            {"id": "content-b", "memory": "  喜欢   python  ", "score": 0.9, "run_id": "dialog-1"},
        ]

        result = merge_memory_results(primary, legacy, **self.scope)

        self.assertEqual([item["id"] for item in result], ["content-b", "same-id"])
        self.assertEqual(result[1]["memory"], "新内容")

    def test_equal_score_duplicate_prefers_newer_memory(self):
        primary = [{
            "id": "old",
            "memory": "住在杭州",
            "score": 0.8,
            "updated_at": "2026-01-01T00:00:00",
            "user_id": "user-1",
            "agent_id": "agent-1",
        }]
        legacy = [{
            "id": "new",
            "memory": "住在杭州",
            "score": 0.8,
            "updated_at": "2026-02-01T00:00:00",
            "run_id": "dialog-1",
        }]

        result = merge_memory_results(primary, legacy, **self.scope)

        self.assertEqual([item["id"] for item in result], ["new"])

    def test_recent_user_message_wins_over_duplicate_long_term_memory(self):
        self.scope["recent_messages"] = [
            FakeMessage("ai", "你使用什么语言？"),
            FakeMessage("human", "我主要使用 Python"),
        ]
        primary = [{
            "id": "duplicate",
            "memory": "主要使用 Python",
            "score": 0.9,
            "user_id": "user-1",
            "agent_id": "agent-1",
        }]

        result = merge_memory_results(primary, [], **self.scope)

        self.assertEqual(result, [])

    async def test_dual_read_uses_new_and_legacy_scopes(self):
        client = FakeMemoryClient(
            primary=[{"id": "new", "memory": "新范围", "score": 0.9, "user_id": "user-1", "agent_id": "agent-1"}],
            legacy=[{"id": "old", "memory": "旧范围", "score": 0.8, "run_id": "dialog-1"}],
        )

        result = await retrieve_mixed_memories(client, query="问题", **self.scope)

        self.assertEqual({item["id"] for item in result}, {"new", "old"})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0], {
            "query": "问题", "user_id": "user-1", "agent_id": "agent-1",
            "limit": 10, "threshold": 0.2,
        })
        self.assertEqual(client.calls[1], {
            "query": "问题", "run_id": "dialog-1", "limit": 10, "threshold": 0.2,
        })

    async def test_one_failed_search_does_not_discard_other_results(self):
        client = FakeMemoryClient(
            legacy=[{"id": "old", "memory": "可用旧记忆", "score": 0.8, "run_id": "dialog-1"}],
            primary_error=RuntimeError("primary unavailable"),
        )

        result = await retrieve_mixed_memories(client, query="问题", **self.scope)

        self.assertEqual([item["id"] for item in result], ["old"])

    def test_context_sections_keep_recent_message_order(self):
        messages = [FakeMessage("human", f"用户消息{i}") for i in range(3, 9)]
        memories = [{"memory": "长期偏好"}]

        context = format_memory_context(messages, memories)

        positions = [context.index(f"用户消息{i}") for i in range(3, 9)]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("[近期对话]", context)
        self.assertIn("[长期记忆]", context)
        self.assertIn("[使用规则]", context)
        self.assertGreater(estimate_context_tokens(context), 0)

    def test_limit_keeps_highest_scores(self):
        primary = [
            {"id": str(index), "memory": f"事实{index}", "score": index / 10, "user_id": "user-1", "agent_id": "agent-1"}
            for index in range(1, 10)
        ]
        self.scope["limit"] = 5

        result = merge_memory_results(primary, [], **self.scope)

        self.assertEqual([item["id"] for item in result], ["9", "8", "7", "6", "5"])

    def test_memory_write_uses_only_user_input_and_user_agent_scope(self):
        kwargs = build_memory_write_kwargs(
            user_input="我不吃香菜",
            user_id="user-1",
            agent_id="agent-1",
            dialog_id="dialog-1",
        )

        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "我不吃香菜"}])
        self.assertEqual(kwargs["user_id"], "user-1")
        self.assertEqual(kwargs["agent_id"], "agent-1")
        self.assertNotIn("run_id", kwargs)
        self.assertEqual(kwargs["metadata"], {"dialog_id": "dialog-1", "run_id": "dialog-1"})

    def test_token_budget_keeps_recent_history_and_trims_long_term_memory(self):
        messages = [FakeMessage("human", "最近消息")]
        memories = [
            {"memory": "A" * 20, "score": 0.9},
            {"memory": "B" * 20, "score": 0.8},
            {"memory": "C" * 20, "score": 0.7},
        ]

        result = build_budgeted_memory_context(
            messages,
            memories,
            token_budget=170,
            token_counter=len,
        )

        self.assertIn("最近消息", result.context)
        self.assertLess(len(result.memories), len(memories))
        self.assertEqual(result.token_count, len(result.context))
        self.assertEqual(result.token_count_mode, "model_tokenizer")
        self.assertLessEqual(result.token_count, 170)

    def test_oversized_memory_does_not_block_later_smaller_memory(self):
        memories = [
            {"memory": "X" * 1000, "score": 0.9},
            {"memory": "短记忆", "score": 0.8},
        ]

        result = build_budgeted_memory_context(
            [],
            memories,
            token_budget=170,
            token_counter=len,
        )

        self.assertEqual(result.memories, [{"memory": "短记忆", "score": 0.8}])

    def test_recent_history_is_never_trimmed_when_it_exceeds_budget(self):
        messages = [FakeMessage("human", "必须保留" * 100)]

        result = build_budgeted_memory_context(
            messages,
            [{"memory": "长期记忆", "score": 0.9}],
            token_budget=100,
            token_counter=len,
        )

        self.assertIn("必须保留", result.context)
        self.assertEqual(result.memories, [])
        self.assertTrue(result.budget_exceeded_by_recent_history)

    def test_tokenizer_failure_falls_back_to_character_estimate(self):
        def broken_counter(_text):
            raise RuntimeError("tokenizer unavailable")

        result = build_budgeted_memory_context(
            [],
            [],
            token_budget=2000,
            token_counter=broken_counter,
        )

        self.assertEqual(result.token_count_mode, "character_estimate")
        self.assertEqual(result.token_count, estimate_context_tokens(result.context))
if __name__ == "__main__":
    unittest.main()
