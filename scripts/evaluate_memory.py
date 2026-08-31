"""Run the reproducible Memory Before/After and full-chain evaluation.

Use the project's existing environment; this script does not install dependencies.
It creates uniquely scoped temporary MySQL/Chroma data and removes it in finally blocks.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "src" / "backend"
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)


def _content(response) -> str:
    return str(getattr(response, "content", response) or "")


def _json_content(response) -> dict:
    from agentchat.services.memory.utils import remove_code_blocks

    return json.loads(remove_code_blocks(_content(response)))


def _recall(results: list[dict], required_terms: tuple[str, ...]) -> int:
    text = "\n".join(str(item.get("memory", "")) for item in results).casefold()
    return int(all(term.casefold() in text for term in required_terms))


class DeterministicEmbedding:
    """Offline character n-gram embedding for integration plumbing tests only."""

    def __init__(self, dimensions: int = 1024):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        normalized = "".join(str(text).casefold().split())
        features = [*normalized, *(normalized[i : i + 2] for i in range(len(normalized) - 1))]
        vector = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


async def evaluate(
    output_path: Path | None,
    llm_id: str | None = None,
    use_deepseek_env: bool = False,
    deterministic_embedding: bool = False,
) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage

    from agentchat.settings import app_settings, initialize_app_settings

    await initialize_app_settings("agentchat/config.yaml")

    # Imports below intentionally happen after settings initialization.
    from agentchat.api.services.history import HistoryService
    from agentchat.database import engine
    from agentchat.database.models.history import HistoryTable
    from agentchat.services.memory.client import memory_client
    from agentchat.services.memory.context import build_budgeted_memory_context
    from agentchat.services.memory.vector_stores.chroma import ChromaDB
    from agentchat.services.memory.prompts import (
        FACT_RETRIEVAL_PROMPT,
        get_update_memory_messages,
    )
    from agentchat.core.models.manager import ModelManager
    from agentchat.core.models.embedding import EmbeddingModel
    from sqlmodel import Session

    run_tag = uuid4().hex
    user_id = f"memory-eval-user-{run_tag}"
    other_user_id = f"memory-eval-other-user-{run_tag}"
    agent_id = f"memory-eval-agent-{run_tag}"
    other_agent_id = f"memory-eval-other-agent-{run_tag}"
    dialog_a = f"memory-eval-dialog-a-{run_tag}"
    dialog_b = f"memory-eval-dialog-b-{run_tag}"

    if use_deepseek_env:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not configured")
        model = ModelManager.get_user_model(
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            api_key=api_key,
        )
        memory_client.llm = model
    elif llm_id:
        from agentchat.api.services.llm import LLMService

        model_config = await LLMService.get_llm_by_id(llm_id)
        if not model_config:
            raise ValueError(f"LLM not found: {llm_id}")
        model = ModelManager.get_user_model(**model_config)
        # Reuse the selected provider credential for the configured embedding model.
        memory_client.llm = model
        memory_client.embedding_model = EmbeddingModel(
            model=app_settings.multi_models.embedding.model_name,
            base_url=model_config["base_url"],
            api_key=model_config["api_key"],
        )
    else:
        model = ModelManager.get_conversation_model()

    if deterministic_embedding:
        memory_client.embedding_model = DeterministicEmbedding()

    # Never reuse the application's default collection: different embedding models may
    # have different dimensions, and evaluation data must not touch user memories.
    evaluation_vector_store = ChromaDB(
        collection_name=f"memory_eval_{run_tag}",
    )
    memory_client.vector_store = evaluation_vector_store
    metrics = {
        "run_tag": run_tag,
        "timestamp": datetime.now().isoformat(),
        "model": getattr(model, "model_name", None) or getattr(model, "model", None),
        "embedding_model": app_settings.multi_models.embedding.model_name,
        "embedding_mode": "deterministic_integration_stub" if deterministic_embedding else "configured_provider",
    }

    extraction_cases = [
        ("你好", False),
        ("Python 的 GIL 是什么？", False),
        ("我不吃香菜", True),
        ("我下个月准备应聘 AI 应用开发岗位", True),
    ]
    extraction_results = []
    for text, should_extract in extraction_cases:
        started = time.perf_counter()
        response = await asyncio.to_thread(
            model.invoke,
            [
                SystemMessage(content=FACT_RETRIEVAL_PROMPT),
                HumanMessage(content=f"Input:\n{text}"),
            ],
            response_format={"type": "json_object"},
        )
        facts = _json_content(response).get("facts", [])
        extraction_results.append(
            {
                "input": text,
                "expected_non_empty": should_extract,
                "facts": facts,
                "passed": bool(facts) is should_extract,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
    metrics["fact_extraction"] = {
        "passed": sum(item["passed"] for item in extraction_results),
        "total": len(extraction_results),
        "cases": extraction_results,
    }

    action_prompt = get_update_memory_messages(
        [{"id": "0", "text": "我喜欢喝咖啡"}],
        ["我现在不喝咖啡了"],
    )
    action_response = await asyncio.to_thread(
        model.invoke,
        [{"role": "user", "content": action_prompt}],
        response_format={"type": "json_object"},
    )
    actions = _json_content(action_response).get("memory", [])
    update_passed = any(
        item.get("id") == "0"
        and item.get("event") == "UPDATE"
        and "不" in str(item.get("text", ""))
        for item in actions
    )
    metrics["preference_update_decision"] = {
        "passed": update_passed,
        "actions": actions,
    }

    history_dialog_id = f"memory-eval-history-{run_tag}"
    try:
        base_time = datetime(2026, 1, 1, 0, 0, 0)
        with Session(engine) as session:
            for index in range(8):
                session.add(
                    HistoryTable(
                        content=f"message-{index}",
                        role="user" if index % 2 == 0 else "assistant",
                        events=[],
                        dialog_id=history_dialog_id,
                        create_time=base_time + timedelta(seconds=index),
                        update_time=base_time + timedelta(seconds=index),
                    )
                )
            session.commit()
        recent = await HistoryService.select_history(history_dialog_id, top_k=6)
        recent_contents = [message.content for message in recent]
        metrics["mysql_recent_history"] = {
            "passed": recent_contents == [f"message-{index}" for index in range(2, 8)],
            "count": len(recent_contents),
            "contents": recent_contents,
        }
    finally:
        from agentchat.database.dao.history import HistoryDao

        await HistoryDao.delete_history_by_dialog_id(history_dialog_id)

    try:
        greeting_result = await memory_client.add(
            messages=[{"role": "user", "content": "你好"}],
            user_id=user_id,
            agent_id=agent_id,
            metadata={"dialog_id": dialog_a, "run_id": dialog_a},
        )
        knowledge_result = await memory_client.add(
            messages=[{"role": "user", "content": "Python 的 GIL 是什么？"}],
            user_id=user_id,
            agent_id=agent_id,
            metadata={"dialog_id": dialog_a, "run_id": dialog_a},
        )
        fact_cases = [
            ("我喜欢喝咖啡", "我的饮品偏好", ("咖啡",)),
            ("我是后端开发工程师", "我的职业是什么", ("后端",)),
            ("我下个月准备应聘 AI 应用开发岗位", "我的求职计划", ("AI", "应用开发")),
        ]
        initial_actions = []
        for statement, _query, _terms in fact_cases:
            add_result = await memory_client.add(
                messages=[{"role": "user", "content": statement}],
                user_id=user_id,
                agent_id=agent_id,
                metadata={"dialog_id": dialog_a, "run_id": dialog_a},
            )
            initial_actions.extend(add_result.get("results", []))

        before_fact_hits = 0
        after_fact_hits = 0
        before_results = []
        after_results = []
        for _statement, query, terms in fact_cases:
            before_response = await memory_client.search(
                query,
                run_id=dialog_b,
                limit=5,
                threshold=0.2,
            )
            after_response = await memory_client.search(
                query,
                user_id=user_id,
                agent_id=agent_id,
                limit=5,
                threshold=0.2,
            )
            before_case_results = before_response.get("results", [])
            after_case_results = after_response.get("results", [])
            before_fact_hits += _recall(before_case_results, terms)
            after_fact_hits += _recall(after_case_results, terms)
            before_results.extend(before_case_results)
            after_results.extend(after_case_results)
        wrong_user = await memory_client.search(
            "我的饮品偏好",
            user_id=other_user_id,
            agent_id=agent_id,
            limit=5,
            threshold=0.2,
        )
        wrong_agent = await memory_client.search(
            "我的饮品偏好",
            user_id=user_id,
            agent_id=other_agent_id,
            limit=5,
            threshold=0.2,
        )

        correction_result = await memory_client.add(
            messages=[{"role": "user", "content": "我现在不喝咖啡了"}],
            user_id=user_id,
            agent_id=agent_id,
            metadata={"dialog_id": dialog_b, "run_id": dialog_b},
        )
        corrected_search = await memory_client.search(
            "我的咖啡偏好",
            user_id=user_id,
            agent_id=agent_id,
            limit=5,
            threshold=0.2,
        )

        corrected_results = corrected_search.get("results", [])
        false_memory_count = len(greeting_result.get("results", [])) + len(
            knowledge_result.get("results", [])
        )
        stale_count = sum(
            "喜欢" in item.get("memory", "") and "不" not in item.get("memory", "")
            for item in corrected_results
        )
        metrics["full_chain"] = {
            "initial_actions": initial_actions,
            "correction_actions": correction_result.get("results", []),
            "fact_case_count": len(fact_cases),
            "before_fact_recall": before_fact_hits / len(fact_cases),
            "after_fact_recall": after_fact_hits / len(fact_cases),
            "before_cross_dialog_recall": before_fact_hits / len(fact_cases),
            "after_cross_dialog_recall": after_fact_hits / len(fact_cases),
            "false_memory_count": false_memory_count,
            "user_isolation_errors": len(wrong_user.get("results", [])),
            "agent_isolation_errors": len(wrong_agent.get("results", [])),
            "stale_memory_count": stale_count,
            "corrected_memories": corrected_results,
        }

        fake_recent = [HumanMessage(content=f"recent-{index}") for index in range(6)]
        unique_after_results = {
            item.get("id"): item for item in after_results if item.get("id")
        }
        budget_candidates = [
            {"memory": item.get("memory", ""), "score": item.get("score", 0)}
            for item in unique_after_results.values()
        ]
        budget_result = build_budgeted_memory_context(
            fake_recent,
            budget_candidates,
            token_budget=app_settings.memory.context_token_budget,
            token_counter=model.get_num_tokens,
        )
        metrics["context_budget"] = {
            "budget": app_settings.memory.context_token_budget,
            "token_count": budget_result.token_count,
            "token_count_mode": budget_result.token_count_mode,
            "candidate_memories": len(budget_candidates),
            "selected_memories": len(budget_result.memories),
            "within_budget": budget_result.token_count <= app_settings.memory.context_token_budget,
        }
    finally:
        try:
            await memory_client.delete_all(user_id=user_id, agent_id=agent_id)
        finally:
            evaluation_vector_store.delete_col()

    metrics["acceptance"] = {
        "fact_extraction": metrics["fact_extraction"]["passed"] == metrics["fact_extraction"]["total"],
        "preference_update": metrics["preference_update_decision"]["passed"],
        "recent_history": metrics["mysql_recent_history"]["passed"],
        "cross_dialog_improved": (
            metrics["full_chain"]["after_cross_dialog_recall"]
            > metrics["full_chain"]["before_cross_dialog_recall"]
        ),
        "no_false_memories": metrics["full_chain"]["false_memory_count"] == 0,
        "isolation": (
            metrics["full_chain"]["user_isolation_errors"] == 0
            and metrics["full_chain"]["agent_isolation_errors"] == 0
        ),
        "no_stale_memory": metrics["full_chain"]["stale_memory_count"] == 0,
        "context_budget": metrics["context_budget"]["within_budget"],
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--llm-id")
    parser.add_argument("--use-deepseek-env", action="store_true")
    parser.add_argument("--deterministic-embedding", action="store_true")
    args = parser.parse_args()
    output_path = args.output
    if output_path and not output_path.is_absolute():
        output_path = ROOT / output_path
    result = asyncio.run(
        evaluate(
            output_path,
            args.llm_id,
            args.use_deepseek_env,
            args.deterministic_embedding,
        )
    )
    # ASCII stdout avoids Windows conda-run encoding failures; the result file remains UTF-8 Chinese.
    print(json.dumps(result, ensure_ascii=True, indent=2))
    if not all(result["acceptance"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
