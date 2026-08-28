import asyncio
import logging
import math
import re
from typing import Any, Iterable, Sequence


logger = logging.getLogger(__name__)


def normalize_memory_text(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text).casefold()


def _safe_score(memory: dict) -> float:
    try:
        return float(memory.get("score", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _is_newer(left: dict, right: dict) -> bool:
    return str(left.get("updated_at") or left.get("created_at") or "") > str(
        right.get("updated_at") or right.get("created_at") or ""
    )


def _prefer_memory(left: dict, right: dict) -> dict:
    left_score = _safe_score(left)
    right_score = _safe_score(right)
    if left_score != right_score:
        return left if left_score > right_score else right
    return left if _is_newer(left, right) else right


def merge_memory_results(
    primary_results: Iterable[dict],
    legacy_results: Iterable[dict],
    *,
    user_id: str,
    agent_id: str,
    dialog_id: str,
    recent_messages: Sequence[Any],
    limit: int,
    min_score: float,
) -> list[dict]:
    """合并新旧范围记忆，并执行隔离、阈值和内容去重。"""

    scoped_memories = []
    for memory in primary_results or []:
        if memory.get("user_id") != user_id or memory.get("agent_id") != agent_id:
            continue
        scoped_memories.append(dict(memory))

    for memory in legacy_results or []:
        if memory.get("run_id") != dialog_id:
            continue
        if memory.get("user_id") not in (None, "", user_id):
            continue
        if memory.get("agent_id") not in (None, "", agent_id):
            continue
        scoped_memories.append(dict(memory))

    by_id = {}
    without_id = []
    for memory in scoped_memories:
        if _safe_score(memory) < min_score:
            continue
        memory_id = str(memory.get("id") or "").strip()
        if not memory_id:
            without_id.append(memory)
            continue
        if memory_id in by_id:
            by_id[memory_id] = _prefer_memory(by_id[memory_id], memory)
        else:
            by_id[memory_id] = memory

    by_content = {}
    for memory in [*by_id.values(), *without_id]:
        normalized = normalize_memory_text(memory.get("memory"))
        if not normalized:
            continue
        if normalized in by_content:
            by_content[normalized] = _prefer_memory(by_content[normalized], memory)
        else:
            by_content[normalized] = memory

    recent_user_texts = []
    for message in recent_messages:
        message_type = str(getattr(message, "type", "")).casefold()
        if message_type not in {"human", "user"}:
            continue
        normalized = normalize_memory_text(getattr(message, "content", ""))
        if normalized:
            recent_user_texts.append(normalized)

    deduplicated = []
    for normalized, memory in by_content.items():
        if any(normalized == recent or normalized in recent for recent in recent_user_texts):
            continue
        deduplicated.append(memory)

    deduplicated.sort(
        key=lambda memory: str(
            memory.get("updated_at") or memory.get("created_at") or ""
        ),
        reverse=True,
    )
    deduplicated.sort(key=_safe_score, reverse=True)
    return deduplicated[:max(0, limit)]


async def retrieve_mixed_memories(
    memory_client,
    *,
    query: str,
    user_id: str,
    agent_id: str,
    dialog_id: str,
    recent_messages: Sequence[Any],
    limit: int = 5,
    min_score: float = 0.2,
) -> list[dict]:
    """并行读取 user+agent 新范围与当前 dialog 的旧 run_id 范围。"""

    search_limit = max(limit * 2, limit)
    primary_task = memory_client.search(
        query=query,
        user_id=user_id,
        agent_id=agent_id,
        limit=search_limit,
        threshold=min_score,
    )
    legacy_task = memory_client.search(
        query=query,
        run_id=dialog_id,
        limit=search_limit,
        threshold=min_score,
    )
    primary_response, legacy_response = await asyncio.gather(
        primary_task,
        legacy_task,
        return_exceptions=True,
    )
    if isinstance(primary_response, Exception):
        logger.warning("Primary memory search failed: %s", primary_response)
    if isinstance(legacy_response, Exception):
        logger.warning("Legacy memory search failed: %s", legacy_response)
    primary_results = (
        primary_response.get("results", [])
        if isinstance(primary_response, dict)
        else []
    )
    legacy_results = (
        legacy_response.get("results", [])
        if isinstance(legacy_response, dict)
        else []
    )
    return merge_memory_results(
        primary_results,
        legacy_results,
        user_id=user_id,
        agent_id=agent_id,
        dialog_id=dialog_id,
        recent_messages=recent_messages,
        limit=limit,
        min_score=min_score,
    )


def format_memory_context(
    recent_messages: Sequence[Any],
    long_term_memories: Sequence[dict],
) -> str:
    recent_lines = []
    for message in recent_messages:
        message_type = str(getattr(message, "type", "")).casefold()
        role = "用户" if message_type in {"human", "user"} else "助手"
        content = str(getattr(message, "content", "") or "").strip()
        if content:
            recent_lines.append(f"- {role}：{content}")

    memory_lines = [
        f"- {str(memory.get('memory') or '').strip()}"
        for memory in long_term_memories
        if str(memory.get("memory") or "").strip()
    ]
    recent_section = "\n".join(recent_lines) or "- 暂无近期对话"
    memory_section = "\n".join(memory_lines) or "- 暂无相关长期记忆"
    return f"""[近期对话]
{recent_section}

[长期记忆]
{memory_section}

[使用规则]
- 近期用户明确表达优先于长期记忆。
- 当前消息与长期记忆冲突时，以当前消息为准。
- 长期记忆仅作为回答上下文，不得向用户暴露内部 ID、分数或检索过程。"""


def estimate_context_tokens(context: str) -> int:
    """无额外 tokenizer 依赖的保守上下文估算，仅用于日志和对比测试。"""
    return math.ceil(len(context) / 4)


def build_memory_write_kwargs(
    *,
    user_input: str,
    user_id: str,
    agent_id: str,
    dialog_id: str,
) -> dict:
    """构造长期记忆写入参数，确保事实来源只有用户原始输入。"""
    return {
        "messages": [{"role": "user", "content": user_input}],
        "user_id": user_id,
        "agent_id": agent_id,
        # dialog_id/run_id 仅用于来源追踪，不参与新记忆的查询范围。
        "metadata": {"dialog_id": dialog_id, "run_id": dialog_id},
    }
