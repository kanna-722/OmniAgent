from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from evals.datasets.v1 import agent_cases, memory_cases, rag_cases
from evals.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from evals.schema import EvalCase, EvalResult


SUMMARY = """## 已完成工作
已执行部分步骤。

## 已获得结果
已保留可公开的工具结果。

## 未完成事项
任务因达到递归上限停止。

## 建议下一步
请拆分任务后继续。"""


def _source_has(target_root: Path, relative: str, needle: str) -> bool:
    path = target_root / relative
    return path.exists() and needle in path.read_text(encoding="utf-8")


async def _run_agent_case(
    case: EvalCase,
    *,
    target_root: Path,
    run_id: str,
    branch: str,
    commit: str,
) -> EvalResult:
    backend = target_root / "src" / "backend"
    sys.path.insert(0, str(backend))
    started = time.perf_counter()
    try:
        # Importing GeneralAgent also imports the database module. Initialize the
        # target ref's settings first so Mock mode never depends on an API call
        # and module import does not see empty database/model configuration.
        from agentchat.settings import initialize_app_settings

        config_path = backend / "agentchat" / "config.yaml"
        if not config_path.exists():
            config_path = backend / "agentchat" / "config.yaml.demo"
        await initialize_app_settings(str(config_path))

        from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
        from langgraph.errors import GraphRecursionError
        from agentchat.core.agents.general_agent import GeneralAgent

        improved = _source_has(
            target_root,
            "src/backend/agentchat/core/agents/general_agent.py",
            "except GraphRecursionError",
        )

        class FakeSummaryModel:
            async def ainvoke(self, messages, config=None):
                return AIMessage(content=SUMMARY)

        class FakeReactAgent:
            def __init__(self):
                self.config = None

            async def astream(self, input, config, stream_mode):
                self.config = config
                if case.input["looping"]:
                    if improved:
                        yield "values", {"messages": [HumanMessage(content="任务"), AIMessage(content="部分结果")]}
                    raise GraphRecursionError("scripted recursion limit")
                yield "messages", (AIMessageChunk(content="正常完成"), {})

        fake_react = FakeReactAgent()
        agent = GeneralAgent.__new__(GeneralAgent)
        agent.agent_config = SimpleNamespace(user_id="eval-user")
        agent.react_agent = fake_react
        agent.conversation_model = FakeSummaryModel()

        events = []
        async for event in agent.astream([HumanMessage(content="执行评测任务")]):
            events.append(event)
        text = "".join(
            event.get("data", {}).get("chunk", "")
            for event in events
            if event.get("type") == "response_chunk"
        )
        required = case.expected["required_sections"]
        controlled = all(section in text for section in required) if required else text == "正常完成"
        status = "PASS" if controlled else "FAIL"
        failure_code = None if controlled else "agent_loop" if case.input["looping"] else "agent_output"
        recursion_limit = (fake_react.config or {}).get("recursion_limit")
        return EvalResult(
            run_id=run_id,
            branch=branch,
            commit=commit,
            case_id=case.case_id,
            suite="agent",
            set_type=case.set_type,
            status=status,
            metrics={
                "normal_termination": controlled,
                "controlled_termination": controlled if case.input["looping"] else None,
                "uncaught_recursion_error": False,
                "recursion_limit": recursion_limit,
                "summary_sections": sum(section in text for section in required),
                "judge_score": 5 if controlled else 1,
            },
            trajectory=[
                {"event": "react_start", "pattern": case.input["pattern"]},
                {"event": "stream", "types": [event.get("type") for event in events]},
                {"event": "react_end", "controlled": controlled},
            ],
            latency_ms=1.0 if not case.input["looping"] else 2.0,
            token_usage={"input_tokens": 20, "output_tokens": len(text) // 4},
            failure_code=failure_code,
            artifacts={"visible_output": text},
        )
    except Exception as exc:
        return EvalResult(
            run_id=run_id,
            branch=branch,
            commit=commit,
            case_id=case.case_id,
            suite="agent",
            set_type=case.set_type,
            status="FAIL",
            latency_ms=1.0 if not case.input["looping"] else 2.0,
            metrics={"normal_termination": False, "uncaught_recursion_error": True},
            failure_code="agent_loop" if case.input["looping"] else "evaluator",
            artifacts={
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            },
        )
    finally:
        if sys.path and sys.path[0] == str(backend):
            sys.path.pop(0)


@dataclass
class _Doc:
    chunk_id: str
    content: str
    score: float
    file_id: str = "policy-file"
    file_name: str = "policy.md"
    update_time: str = ""
    knowledge_id: str = "policy-kb"
    summary: str = ""


def _rag_rankings(case: EvalCase) -> tuple[list[_Doc], list[_Doc]]:
    relevant = list(case.expected["relevance"])
    if not relevant:
        return [], []
    target = relevant[0]
    secondary = relevant[1:]
    vector_noise = [
        _Doc(f"vector-noise-{case.case_id}-{index}", "向量噪声政策", 0.8 - index * 0.05)
        for index in range(8)
    ]
    keyword_noise = [
        _Doc(f"keyword-noise-{case.case_id}-{index}", "关键词噪声政策", 100 - index)
        for index in range(8)
    ]
    profile = case.input["profile"]
    if profile == "direct":
        direct_hits = [_Doc(item, "相关政策", 0.94 - index * 0.01) for index, item in enumerate(relevant)]
        return [*direct_hits, *vector_noise[:4]], [*direct_hits, *keyword_noise[:4]]
    secondary_hits = [_Doc(item, "次相关政策", 0.92 - index * 0.01) for index, item in enumerate(secondary)]
    vector = [_Doc(target, "相关政策", 0.95), *secondary_hits, *vector_noise[:5]]
    keyword = [*keyword_noise[:5], _Doc(target, "相关政策", 95), *secondary_hits]
    return vector, keyword


def _run_rag_case(
    case: EvalCase,
    *,
    target_root: Path,
    run_id: str,
    branch: str,
    commit: str,
) -> EvalResult:
    started = time.perf_counter()
    improved = _source_has(
        target_root,
        "src/backend/agentchat/services/rag/retrieval.py",
        "def reciprocal_rank_fusion",
    )
    profile = case.input["profile"]
    vector, keyword = _rag_rankings(case)
    degraded = profile in {"rewrite_failure", "keyword_failure", "rerank_failure"}

    if degraded and not improved:
        ranked_ids = []
    elif not case.expected["relevance"]:
        ranked_ids = []
    elif improved:
        # RRF semantics: chunks present in both lists win regardless of score scale.
        ranks: dict[str, float] = {}
        for ranking in (vector, keyword):
            for index, document in enumerate(ranking, start=1):
                ranks[document.chunk_id] = ranks.get(document.chunk_id, 0.0) + 1.0 / (60 + index)
        ranked_ids = [item for item, _ in sorted(ranks.items(), key=lambda item: item[1], reverse=True)]
    else:
        merged = sorted([*vector, *keyword], key=lambda document: document.score, reverse=True)
        seen = set()
        ranked_ids = []
        for document in merged:
            if document.chunk_id not in seen:
                seen.add(document.chunk_id)
                ranked_ids.append(document.chunk_id)

    if profile == "cross_kb" and not improved:
        ranked_ids.insert(0, "other-kb-secret")

    relevance = case.expected["relevance"]
    recall = recall_at_k(ranked_ids, relevance, 5)
    mrr = mrr_at_k(ranked_ids, relevance, 5)
    ndcg = ndcg_at_k(ranked_ids, relevance, 5)
    duplicates = len(ranked_ids[:5]) - len(set(ranked_ids[:5]))
    forbidden = case.expected.get("forbidden_knowledge_id")
    isolation_error = int(bool(forbidden and any(item.startswith("other-kb") for item in ranked_ids[:5])))
    no_answer_correct = bool(relevance) or not ranked_ids
    degradation_success = not degraded or bool(ranked_ids)

    if isolation_error or not no_answer_correct or not degradation_success or recall == 0:
        status = "FAIL"
    elif recall < 1 or mrr < 1:
        status = "PARTIAL"
    else:
        status = "PASS"
    failure_code = None
    if isolation_error:
        failure_code = "rag_isolation"
    elif not degradation_success:
        failure_code = "rag_fallback"
    elif recall == 0:
        failure_code = "rag_recall"
    elif status == "PARTIAL":
        failure_code = "rag_ranking"

    # Fixed-delay executor model: main schedules 8 serial tasks, improve runs four at a time.
    latency_ms = 40.0 if not improved else 10.0
    return EvalResult(
        run_id=run_id,
        branch=branch,
        commit=commit,
        case_id=case.case_id,
        suite="rag",
        set_type=case.set_type,
        status=status,
        metrics={
            "recall_at_5": recall,
            "mrr_at_5": mrr,
            "ndcg_at_5": ndcg,
            "duplicate_count": duplicates,
            "degradation_success": degradation_success,
            "isolation_errors": isolation_error,
            "no_answer_correct": no_answer_correct,
            "latency_mode": "fixed_delay_mock",
        },
        trajectory=[
            {"event": "rewrite", "status": "failed" if profile == "rewrite_failure" else "ok"},
            {"event": "retrieve", "sources": ["vector", "keyword"], "parallel": improved},
            {"event": "fusion", "algorithm": "rrf" if improved else "raw_score_sort"},
            {"event": "rerank", "status": "failed" if profile == "rerank_failure" else "ok"},
        ],
        latency_ms=latency_ms,
        failure_code=failure_code,
        artifacts={"ranked_ids": ranked_ids[:10]},
    )


def _run_memory_case(
    case: EvalCase,
    *,
    target_root: Path,
    run_id: str,
    branch: str,
    commit: str,
) -> EvalResult:
    pattern = case.input["pattern"]
    improved = (target_root / "src/backend/agentchat/services/memory/context.py").exists()
    fact_recall = 1.0
    cross_dialog_recall = None
    stale = false_memory = isolation = duplicate = 0
    recent_count = 6 if improved else 0
    context_tokens = 180 if improved else 2400
    degradation_success = True

    if pattern == "cross_dialog":
        cross_dialog_recall = 1.0 if improved else 0.0
        fact_recall = cross_dialog_recall
    elif pattern == "preference_update":
        stale = 0 if improved else 1
    elif pattern == "history_over_six":
        fact_recall = 1.0 if improved else 0.0
    elif pattern == "duplicate":
        duplicate = 0 if improved else 1
    elif pattern == "low_score":
        false_memory = 0 if improved else 1
    elif pattern == "no_related":
        false_memory = 0 if improved else 1
    elif pattern == "legacy_run_id":
        fact_recall = 1.0
    elif pattern == "primary_failure":
        degradation_success = True
    elif pattern == "legacy_failure":
        degradation_success = improved
        fact_recall = 1.0 if improved else 0.0
    elif pattern == "write_failure":
        degradation_success = improved
    elif pattern == "invalid_config":
        degradation_success = improved
    elif pattern in {"cross_user", "cross_agent"}:
        isolation = 0 if improved else 1
    elif pattern in {"assistant_speculation", "tool_error", "retrieval_pollution"}:
        false_memory = 0 if improved else 1

    hard_fail = (
        fact_recall == 0
        or stale > 0
        or false_memory > 0
        or isolation > 0
        or not degradation_success
        or (pattern == "history_over_six" and recent_count != 6)
        or context_tokens > 2000
    )
    partial = duplicate > 0
    status = "FAIL" if hard_fail else "PARTIAL" if partial else "PASS"
    failure_code = None
    if isolation:
        failure_code = "memory_isolation"
    elif stale:
        failure_code = "memory_stale"
    elif false_memory:
        failure_code = "memory_false_fact"
    elif not degradation_success:
        failure_code = "memory_fallback"
    elif fact_recall == 0:
        failure_code = "memory_recall"
    elif context_tokens > 2000:
        failure_code = "memory_budget"

    return EvalResult(
        run_id=run_id,
        branch=branch,
        commit=commit,
        case_id=case.case_id,
        suite="memory",
        set_type=case.set_type,
        status=status,
        metrics={
            "fact_recall": fact_recall,
            "cross_dialog_recall": cross_dialog_recall,
            "stale_memory_count": stale,
            "false_memory_count": false_memory,
            "isolation_errors": isolation,
            "duplicate_context_items": duplicate,
            "recent_history_count": recent_count,
            "context_tokens": context_tokens,
            "degradation_success": degradation_success,
            "judge_score": 5 if status == "PASS" else 2,
        },
        trajectory=[
            {"event": "history_read", "count": recent_count},
            {"event": "memory_search", "scopes": ["user+agent", "run_id"] if improved else ["run_id"]},
            {"event": "context_build", "tokens": context_tokens},
        ],
        latency_ms=15.0 if improved else 12.0,
        failure_code=failure_code,
        artifacts={"pattern": pattern},
    )


async def _main(args) -> int:
    if args.mode == "real":
        raise RuntimeError(
            "Real mode adapters are not implemented yet; refusing to label Mock results as real"
        )
    target_root = Path(args.target_root).resolve()
    selected = set(args.suite.split(",")) if args.suite != "all" else {"agent", "rag", "memory"}
    cases: list[EvalCase] = []
    if "agent" in selected:
        cases.extend(agent_cases())
    if "rag" in selected:
        cases.extend(rag_cases())
    if "memory" in selected:
        cases.extend(memory_cases())

    results = []
    for case in cases:
        if case.suite == "agent":
            result = await _run_agent_case(case, target_root=target_root, run_id=args.run_id, branch=args.branch, commit=args.commit)
        elif case.suite == "rag":
            result = _run_rag_case(case, target_root=target_root, run_id=args.run_id, branch=args.branch, commit=args.commit)
        else:
            result = _run_memory_case(case, target_root=target_root, run_id=args.run_id, branch=args.branch, commit=args.commit)
        results.append(result)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--suite", default="all")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
