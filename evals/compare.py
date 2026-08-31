from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from evals.metrics import percentile


ROOT = Path(__file__).resolve().parents[1]
WORKTREE_ROOT = ROOT / ".eval" / "worktrees"
RESULT_ROOT = ROOT / "evals" / "results"


class InfrastructureError(RuntimeError):
    pass


def _git(*args: str, cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )


def _resolve_ref(ref: str) -> str:
    result = _git("rev-parse", "--verify", f"{ref}^{{commit}}")
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _verify_dataset_snapshot(dataset_dir: Path) -> dict[str, str]:
    checksum_path = dataset_dir / "checksums.sha256"
    if not checksum_path.exists():
        raise InfrastructureError("Dataset checksum manifest is missing")
    checksums = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, separator, name = line.partition("  ")
        if not separator or Path(name).name != name:
            raise InfrastructureError(f"Invalid dataset checksum entry: {line}")
        path = dataset_dir / name
        actual = _sha256(path)
        if actual != expected:
            raise InfrastructureError(f"Dataset checksum mismatch: {name}")
        checksums[name] = actual
    return checksums


def _safe_worktree_path(run_id: str, label: str) -> Path:
    path = (WORKTREE_ROOT / f"{run_id}-{label}").resolve()
    root = WORKTREE_ROOT.resolve()
    if root not in path.parents:
        raise InfrastructureError(f"Unsafe worktree path: {path}")
    return path


def _add_worktree(path: Path, commit: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise InfrastructureError(f"Worktree path already exists: {path}")
    result = _git("worktree", "add", "--detach", str(path), commit, check=False)
    if result.returncode:
        raise InfrastructureError(result.stderr or result.stdout)


def _remove_worktree(path: Path) -> None:
    if not path.exists():
        return
    result = _git("worktree", "remove", "--force", str(path), check=False)
    if result.returncode:
        raise InfrastructureError(result.stderr or result.stdout)


def _run_worker(
    *,
    worktree: Path,
    label: str,
    commit: str,
    run_id: str,
    suite: str,
    mode: str,
    output: Path,
) -> None:
    env = os.environ.copy()
    if mode == "mock":
        env.setdefault("DEEPSEEK_API_KEY", "eval-mock-not-used")
        env.setdefault("SILICONFLOW_API_KEY", "eval-mock-not-used")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "evals.worker",
        "--target-root",
        str(worktree),
        "--branch",
        label,
        "--commit",
        commit,
        "--run-id",
        run_id,
        "--suite",
        suite,
        "--mode",
        mode,
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command,
        cwd=worktree,
        env=env,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    if result.returncode:
        raise InfrastructureError(
            f"{label} worker failed ({result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _mean(results: list[dict], metric: str, *, predicate=None) -> float:
    values = []
    for result in results:
        if predicate and not predicate(result):
            continue
        value = result.get("metrics", {}).get(metric)
        if value is not None:
            values.append(float(value))
    return sum(values) / len(values) if values else 0.0


def _summarize(results: list[dict]) -> dict[str, Any]:
    by_suite = {suite: [item for item in results if item["suite"] == suite] for suite in ("agent", "rag", "memory")}
    agent = by_suite["agent"]
    rag = by_suite["rag"]
    memory = by_suite["memory"]
    looping = [item for item in agent if item.get("metrics", {}).get("controlled_termination") is not None]
    gold_agent = [item for item in agent if item["set_type"] == "gold"]
    cross_memory = [item for item in memory if item.get("metrics", {}).get("cross_dialog_recall") is not None]
    summary = {
        "counts": dict(Counter(item["status"] for item in results)),
        "agent": {
            "gold_success_rate": _mean(gold_agent, "normal_termination"),
            "controlled_termination_rate": _mean(looping, "controlled_termination"),
            "uncaught_recursion_errors": sum(item["metrics"].get("uncaught_recursion_error", 0) for item in agent),
            "judge_average": _mean(looping, "judge_score"),
            "judge_minimum": min((item["metrics"].get("judge_score", 0) for item in looping), default=0),
            "gold_p95_ms": percentile([item["latency_ms"] for item in gold_agent], 0.95),
        },
        "rag": {
            "recall_at_5": _mean(rag, "recall_at_5"),
            "mrr_at_5": _mean(rag, "mrr_at_5"),
            "ndcg_at_5": _mean(rag, "ndcg_at_5"),
            "duplicate_count": sum(item["metrics"].get("duplicate_count", 0) for item in rag),
            "failure_cases_passed": sum(
                item["metrics"].get("degradation_success", False)
                for item in rag
                if item["set_type"] == "failure"
            ),
            "failure_case_count": sum(item["set_type"] == "failure" for item in rag),
            "isolation_errors": sum(item["metrics"].get("isolation_errors", 0) for item in rag),
            "p95_ms": percentile([item["latency_ms"] for item in rag], 0.95),
        },
        "memory": {
            "fact_recall": _mean(memory, "fact_recall"),
            "cross_dialog_recall": _mean(cross_memory, "cross_dialog_recall"),
            "stale_memory_count": sum(item["metrics"].get("stale_memory_count", 0) for item in memory),
            "false_memory_count": sum(item["metrics"].get("false_memory_count", 0) for item in memory),
            "isolation_errors": sum(item["metrics"].get("isolation_errors", 0) for item in memory),
            "duplicate_context_items": sum(item["metrics"].get("duplicate_context_items", 0) for item in memory),
            "recent_history_pass_rate": _mean(
                [item for item in memory if "history" in item["case_id"] or "history_over_six" in item.get("artifacts", {})],
                "recent_history_count",
            ),
            "max_context_tokens": max((item["metrics"].get("context_tokens", 0) for item in memory), default=0),
            "p95_ms": percentile([item["latency_ms"] for item in memory], 0.95),
        },
    }
    # The history metric is an assertion over the dedicated cases, not an average count.
    history_cases = [
        item
        for item in memory
        if item.get("artifacts", {}).get("pattern") == "history_over_six"
    ]
    summary["memory"]["recent_history_pass_rate"] = (
        sum(item["metrics"].get("recent_history_count") == 6 for item in history_cases) / len(history_cases)
        if history_cases else 0.0
    )
    return summary


def _gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    mode: str,
    suite: str = "all",
) -> tuple[bool, list[dict[str, Any]]]:
    checks = []

    def add(name: str, passed: bool, actual: Any, expected: str, critical: bool = False):
        checks.append({"name": name, "passed": bool(passed), "actual": actual, "expected": expected, "critical": critical})

    selected = {"agent", "rag", "memory"} if suite == "all" else {suite}

    ba, ca = baseline["agent"], candidate["agent"]
    if "agent" in selected:
        add("agent.gold_success", ca["gold_success_rate"] == 1 and ca["gold_success_rate"] >= ba["gold_success_rate"], ca["gold_success_rate"], "100% and >= main")
        add("agent.controlled_termination", ca["controlled_termination_rate"] == 1, ca["controlled_termination_rate"], "100%", True)
        add("agent.uncaught_recursion", ca["uncaught_recursion_errors"] == 0, ca["uncaught_recursion_errors"], "0", True)
        add("agent.latency", ca["gold_p95_ms"] <= ba["gold_p95_ms"] * 1.15, ca["gold_p95_ms"], "<= main * 1.15")
        add("agent.judge", ca["judge_average"] >= 4 and ca["judge_minimum"] >= 3, {"avg": ca["judge_average"], "min": ca["judge_minimum"]}, "avg >= 4, min >= 3")

    br, cr = baseline["rag"], candidate["rag"]
    if "rag" in selected:
        add("rag.recall", cr["recall_at_5"] >= br["recall_at_5"] and cr["recall_at_5"] >= 0.90, cr["recall_at_5"], ">= main and >= 0.90")
        add("rag.mrr", cr["mrr_at_5"] >= br["mrr_at_5"] and cr["mrr_at_5"] >= 0.80, cr["mrr_at_5"], ">= main and >= 0.80")
        add("rag.ndcg", cr["ndcg_at_5"] >= br["ndcg_at_5"] and cr["ndcg_at_5"] >= 0.82, cr["ndcg_at_5"], ">= main and >= 0.82")
        add("rag.duplicates", cr["duplicate_count"] == 0, cr["duplicate_count"], "0")
        add("rag.fallback", cr["failure_cases_passed"] == cr["failure_case_count"] == 9, cr["failure_cases_passed"], "9/9", True)
        add("rag.isolation", cr["isolation_errors"] == 0, cr["isolation_errors"], "0", True)
        latency_factor = 1.2 if mode == "real" else 0.7
        add("rag.latency", cr["p95_ms"] <= br["p95_ms"] * latency_factor, cr["p95_ms"], f"<= main * {latency_factor}")

    bm, cm = baseline["memory"], candidate["memory"]
    if "memory" in selected:
        add("memory.fact_recall", cm["fact_recall"] >= bm["fact_recall"] and cm["fact_recall"] >= 0.90, cm["fact_recall"], ">= main and >= 0.90")
        add("memory.cross_dialog", cm["cross_dialog_recall"] >= 0.90 and cm["cross_dialog_recall"] > bm["cross_dialog_recall"], cm["cross_dialog_recall"], ">= 0.90 and > main")
        add("memory.stale", cm["stale_memory_count"] == 0, cm["stale_memory_count"], "0", True)
        add("memory.false", cm["false_memory_count"] == 0, cm["false_memory_count"], "0", True)
        add("memory.isolation", cm["isolation_errors"] == 0, cm["isolation_errors"], "0", True)
        add("memory.history", cm["recent_history_pass_rate"] == 1, cm["recent_history_pass_rate"], "100%")
        add("memory.context_budget", cm["max_context_tokens"] <= 2000, cm["max_context_tokens"], "<= 2000")
        if mode == "real":
            add("memory.latency", cm["p95_ms"] <= bm["p95_ms"] * 1.2, cm["p95_ms"], "<= main * 1.20")
    return all(check["passed"] for check in checks), checks


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    baseline = payload["summary"]["main"]
    candidate = payload["summary"]["improve"]
    lines = [
        "# OmniAgent main vs improve Evaluation",
        "",
        f"- Run: `{payload['run_id']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Baseline: `{payload['commits']['main']}`",
        f"- Candidate: `{payload['commits']['improve']}`",
        f"- Gate: **{'PASS' if payload['gate']['passed'] else 'BLOCK'}**",
        "",
        "## Core metrics",
        "",
        "| Metric | main | improve | absolute delta |",
        "| --- | ---: | ---: | ---: |",
    "",
        "## Gate checks",
        "",
    ]
    metric_rows = [
        ("Agent controlled termination", baseline["agent"]["controlled_termination_rate"], candidate["agent"]["controlled_termination_rate"], ".3f"),
        ("RAG Recall@5", baseline["rag"]["recall_at_5"], candidate["rag"]["recall_at_5"], ".3f"),
        ("RAG MRR@5", baseline["rag"]["mrr_at_5"], candidate["rag"]["mrr_at_5"], ".3f"),
        ("RAG NDCG@5", baseline["rag"]["ndcg_at_5"], candidate["rag"]["ndcg_at_5"], ".3f"),
        ("RAG P95 ms", baseline["rag"]["p95_ms"], candidate["rag"]["p95_ms"], ".2f"),
        ("Memory Fact Recall", baseline["memory"]["fact_recall"], candidate["memory"]["fact_recall"], ".3f"),
        ("Memory Cross-dialog Recall", baseline["memory"]["cross_dialog_recall"], candidate["memory"]["cross_dialog_recall"], ".3f"),
        ("Memory false facts", baseline["memory"]["false_memory_count"], candidate["memory"]["false_memory_count"], ".0f"),
    ]
    insert_at = lines.index("") + 1
    # The first blank belongs to the report header; append metrics before Gate checks.
    gate_heading = lines.index("## Gate checks")
    rows = [
        f"| {name} | {base:{fmt}} | {cand:{fmt}} | {cand - base:+{fmt}} |"
        for name, base, cand, fmt in metric_rows
    ]
    lines[gate_heading - 1:gate_heading] = rows + [""]
    for check in payload["gate"]["checks"]:
        lines.append(f"- {'✅' if check['passed'] else '❌'} `{check['name']}`: {check['actual']} (expected {check['expected']})")
    lines.extend([
        "",
        "## Regressions",
        "",
        *(
            [f"- `{item['case_id']}`: {item['baseline_status']} -> {item['candidate_status']} ({item.get('failure_code') or 'no code'})" for item in payload["regressions"]]
            or ["- None." ]
        ),
        "",
        "## Artifacts",
        "",
        "- Raw branch results: `main/raw.jsonl`, `improve/raw.jsonl`",
        "- Candidate failures: `failures.jsonl`",
        "- Environment and immutable dataset hashes: `environment.json`",
        "",
        "## Interpretation",
        "",
        "This is a fixed V1 benchmark, not a production effectiveness claim. Mock and real results must not be averaged together.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> int:
    if args.mode == "real":
        raise InfrastructureError(
            "Real adapters are not implemented; use --mode mock until model, retrieval and memory adapters are available"
        )
    baseline_commit = _resolve_ref(args.baseline_ref)
    candidate_commit = _resolve_ref(args.candidate_ref)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    result_dir = (RESULT_ROOT / run_id).resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    baseline_tree = _safe_worktree_path(run_id, "main")
    candidate_tree = _safe_worktree_path(run_id, "improve")
    cleanup_errors = []

    try:
        _add_worktree(baseline_tree, baseline_commit)
        _add_worktree(candidate_tree, candidate_commit)
        for label, commit, tree in (
            ("main", baseline_commit, baseline_tree),
            ("improve", candidate_commit, candidate_tree),
        ):
            output = result_dir / label / "raw.jsonl"
            _run_worker(
                worktree=tree,
                label=label,
                commit=commit,
                run_id=run_id,
                suite=args.suite,
                mode=args.mode,
                output=output,
            )
    finally:
        for tree in (candidate_tree, baseline_tree):
            try:
                _remove_worktree(tree)
            except Exception as exc:
                cleanup_errors.append(str(exc))

    main_results = _load_jsonl(result_dir / "main" / "raw.jsonl")
    improve_results = _load_jsonl(result_dir / "improve" / "raw.jsonl")
    main_summary = _summarize(main_results)
    improve_summary = _summarize(improve_results)
    gate_passed, gate_checks = _gate(main_summary, improve_summary, args.mode, args.suite)

    dataset_dir = ROOT / "evals" / "datasets" / "v1"
    dataset_checksums = _verify_dataset_snapshot(dataset_dir)
    environment = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "mode": args.mode,
        "suite": args.suite,
        "configuration": {
            "candidate_model": "deepseek-4-flash",
            "judge_model": "deepseek-4-pro",
            "embedding_model": "Qwen/Qwen3-VL-Embedding-8B",
            "rerank_model": "Qwen/Qwen3-VL-Reranker-8B",
            "temperature": 0,
            "executor": "deterministic_fake" if args.mode == "mock" else "real",
        },
        "commits": {"main": baseline_commit, "improve": candidate_commit},
        "dataset_checksums": dataset_checksums,
        "cleanup_errors": cleanup_errors,
        "secrets_recorded": False,
    }
    (result_dir / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")

    failures = [item for item in improve_results if item["status"] != "PASS"]
    with (result_dir / "failures.jsonl").open("w", encoding="utf-8") as handle:
        seen = set()
        for item in failures:
            fingerprint = hashlib.sha256(
                f"{item['suite']}|{item['case_id']}|{item.get('failure_code')}".encode()
            ).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            handle.write(json.dumps({"fingerprint": fingerprint, **item}, ensure_ascii=False) + "\n")
    shutil.copyfile(result_dir / "failures.jsonl", result_dir / "failure_candidates.jsonl")

    status_rank = {"FAIL": 0, "PARTIAL": 1, "PASS": 2}
    main_by_case = {item["case_id"]: item for item in main_results}
    regressions = []
    for item in improve_results:
        baseline_item = main_by_case.get(item["case_id"])
        if baseline_item and status_rank[item["status"]] < status_rank[baseline_item["status"]]:
            regressions.append({
                "case_id": item["case_id"],
                "baseline_status": baseline_item["status"],
                "candidate_status": item["status"],
                "failure_code": item.get("failure_code"),
            })

    payload = {
        "run_id": run_id,
        "mode": args.mode,
        "commits": environment["commits"],
        "summary": {"main": main_summary, "improve": improve_summary},
        "gate": {"passed": gate_passed and not cleanup_errors, "checks": gate_checks},
        "regressions": regressions,
        "cleanup_errors": cleanup_errors,
    }
    (result_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(result_dir / "comparison.md", payload)
    print(result_dir / "comparison.md")
    return 0 if payload["gate"]["passed"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OmniAgent refs using isolated worktrees")
    parser.add_argument("--baseline-ref", default="284305d")
    parser.add_argument("--candidate-ref", default="e3fc22b")
    parser.add_argument("--suite", choices=["all", "agent", "rag", "memory"], default="all")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        raise SystemExit(run(args))
    except (InfrastructureError, subprocess.TimeoutExpired) as exc:
        print(f"Infrastructure error: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
