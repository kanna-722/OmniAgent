from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evals.compare import InfrastructureError, _gate, _safe_worktree_path, _summarize
from evals.datasets.v1 import agent_cases, memory_cases, rag_cases
from evals.metrics import mrr_at_k, ndcg_at_k, percentile, recall_at_k


class DatasetTests(unittest.TestCase):
    def test_counts_and_ids_are_stable_and_unique(self):
        cases = [*agent_cases(), *rag_cases(), *memory_cases()]
        self.assertEqual((20, 60, 50), (len(agent_cases()), len(rag_cases()), len(memory_cases())))
        self.assertEqual(len(cases), len({case.case_id for case in cases}))

    def test_rag_qrels_reference_existing_policy_chunks(self):
        valid = {f"policy-{policy:02d}-chunk-{chunk:02d}" for policy in range(1, 13) for chunk in range(1, 6)}
        for case in rag_cases():
            self.assertTrue(set(case.expected["relevance"]).issubset(valid), case.case_id)


class MetricTests(unittest.TestCase):
    def test_retrieval_metrics(self):
        ranked = ["noise", "a", "b"]
        self.assertEqual(recall_at_k(ranked, {"a", "b"}, 2), 0.5)
        self.assertEqual(mrr_at_k(ranked, {"a"}, 5), 0.5)
        self.assertGreater(ndcg_at_k(ranked, {"a": 3, "b": 1}, 5), 0.0)
        self.assertEqual(recall_at_k([], {}, 5), 1.0)

    def test_nearest_rank_percentile(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.95), 4)
        self.assertEqual(percentile([], 0.95), 0.0)


class GateTests(unittest.TestCase):
    def _result(self, suite, case_id, set_type, metrics, latency=1.0, artifacts=None):
        return {
            "suite": suite,
            "case_id": case_id,
            "set_type": set_type,
            "status": "PASS",
            "metrics": metrics,
            "latency_ms": latency,
            "artifacts": artifacts or {},
        }

    def test_suite_specific_gate_does_not_require_other_suites(self):
        baseline = _summarize([
            self._result("agent", "agent-gold-01", "gold", {"normal_termination": True}),
            self._result("agent", "agent-failure-01", "failure", {"controlled_termination": False, "judge_score": 1}),
        ])
        candidate = _summarize([
            self._result("agent", "agent-gold-01", "gold", {"normal_termination": True}),
            self._result("agent", "agent-failure-01", "failure", {"controlled_termination": True, "judge_score": 5}),
        ])
        passed, checks = _gate(baseline, candidate, "mock", "agent")
        self.assertTrue(passed)
        self.assertTrue(all(item["name"].startswith("agent.") for item in checks))

    def test_history_metric_only_uses_dedicated_pattern(self):
        summary = _summarize([
            self._result(
                "memory", "memory-boundary-01", "boundary",
                {"recent_history_count": 6}, artifacts={"pattern": "history_over_six"},
            ),
            self._result(
                "memory", "memory-boundary-02", "boundary",
                {"recent_history_count": 0}, artifacts={"pattern": "duplicate"},
            ),
        ])
        self.assertEqual(summary["memory"]["recent_history_pass_rate"], 1.0)


class WorktreeSafetyTests(unittest.TestCase):
    def test_generated_path_is_below_worktree_root(self):
        path = _safe_worktree_path("safe-run", "main")
        self.assertIn("safe-run-main", path.name)

    def test_path_traversal_is_rejected(self):
        with self.assertRaises(InfrastructureError):
            _safe_worktree_path("../escape", "outside")


if __name__ == "__main__":
    unittest.main()
