import asyncio
import math
import time
import unittest
from dataclasses import dataclass

from agentchat.services.rag.retrieval import (
    MixRetrival,
    RetrievalCandidate,
    RetrievalRanking,
    normalize_rewritten_queries,
    reciprocal_rank_fusion,
    rerank_candidates_with_fallback,
)


@dataclass
class FakeDocument:
    chunk_id: str
    content: str
    score: float
    file_id: str = "file-1"
    file_name: str = "file.md"
    update_time: str = ""
    knowledge_id: str = "kb-1"
    summary: str = ""


class DelayedRetriever:
    def __init__(self, delay=0.05, failing_queries=None):
        self.delay = delay
        self.failing_queries = set(failing_queries or [])
        self.active = 0
        self.max_active = 0

    async def search(self, query, knowledge_id, top_k=10):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            if query in self.failing_queries:
                raise RuntimeError("fake retrieval failure")
            return [
                FakeDocument(
                    chunk_id=f"{query}-{knowledge_id}",
                    content=f"{query}:{knowledge_id}",
                    score=0.5,
                    knowledge_id=knowledge_id,
                )
            ]
        finally:
            self.active -= 1

    async def search_summary(self, query, knowledge_id, top_k=10):
        return await self.search(query, knowledge_id, top_k)


@dataclass
class FakeRerankResult:
    index: int
    score: float


def recall_at_k(ranked_ids, relevant_ids, k):
    relevant_ids = set(relevant_ids)
    if not relevant_ids:
        return 1.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank_at_k(ranked_ids, relevant_ids, k):
    relevant_ids = set(relevant_ids)
    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids, relevance, k):
    def dcg(scores):
        return sum(score / math.log2(rank + 1) for rank, score in enumerate(scores, start=1))

    actual = [relevance.get(chunk_id, 0) for chunk_id in ranked_ids[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    ideal_score = dcg(ideal)
    return dcg(actual) / ideal_score if ideal_score else 1.0


class QueryNormalizationTest(unittest.TestCase):
    def test_original_query_is_first_and_rewrites_are_cleaned(self):
        result = normalize_rewritten_queries(
            "  OmniAgent 是什么？  ",
            ["", "omniagent 是什么？", "  如何理解 OmniAgent？ ", 123, "功能是什么？"],
            max_rewrites=2,
        )

        self.assertEqual(
            result,
            ["OmniAgent 是什么？", "如何理解 OmniAgent？", "功能是什么？"],
        )

    def test_invalid_rewrite_falls_back_to_original(self):
        self.assertEqual(normalize_rewritten_queries("原始问题", None), ["原始问题"])
        self.assertEqual(
            normalize_rewritten_queries("原始问题", {"variations": ["改写"]}, 0),
            ["原始问题"],
        )


class ParallelRetrievalTest(unittest.IsolatedAsyncioTestCase):
    async def test_queries_and_knowledge_bases_run_concurrently(self):
        retriever = DelayedRetriever(delay=0.05)
        serial_started_at = time.perf_counter()
        for query in ("q1", "q2"):
            for knowledge_id in ("kb1", "kb2"):
                await retriever.search(query, knowledge_id)
        serial_elapsed = time.perf_counter() - serial_started_at

        started_at = time.perf_counter()

        rankings = await MixRetrival.retrieve_ranked_lists(
            queries=["q1", "q2"],
            collection_names=["kb1", "kb2"],
            index_names=None,
            search_field="content",
            enable_elasticsearch=False,
            concurrency=4,
            top_k=5,
            vector_client=retriever,
        )
        elapsed = time.perf_counter() - started_at

        self.assertEqual(len(rankings), 4)
        self.assertEqual(retriever.max_active, 4)
        self.assertLess(elapsed, 0.15)
        self.assertLess(elapsed, serial_elapsed * 0.6)

    async def test_one_failed_task_does_not_discard_other_results(self):
        retriever = DelayedRetriever(delay=0, failing_queries={"bad"})

        rankings = await MixRetrival.retrieve_ranked_lists(
            queries=["bad", "good"],
            collection_names=["kb"],
            index_names=None,
            search_field="content",
            enable_elasticsearch=False,
            concurrency=2,
            top_k=5,
            vector_client=retriever,
        )

        self.assertEqual(len(rankings), 1)
        self.assertEqual(rankings[0].query, "good")

    async def test_slow_task_is_stopped_by_retrieval_timeout(self):
        retriever = DelayedRetriever(delay=0.05)
        started_at = time.perf_counter()

        rankings = await MixRetrival.retrieve_ranked_lists(
            queries=["slow"],
            collection_names=["kb"],
            index_names=None,
            search_field="content",
            enable_elasticsearch=False,
            concurrency=1,
            top_k=5,
            timeout_seconds=0.01,
            vector_client=retriever,
        )
        elapsed = time.perf_counter() - started_at

        self.assertEqual(rankings, [])
        self.assertLess(elapsed, 0.04)


class ReciprocalRankFusionTest(unittest.TestCase):
    def test_rrf_merges_duplicate_chunks_and_preserves_evidence(self):
        rankings = [
            RetrievalRanking(
                source="vector",
                search_field="content",
                query="q1",
                knowledge_id="kb",
                documents=[
                    FakeDocument("shared", "共享内容", 0.01),
                    FakeDocument("vector-only", "向量结果", 0.99),
                ],
            ),
            RetrievalRanking(
                source="elasticsearch",
                search_field="content",
                query="q1",
                knowledge_id="kb",
                documents=[
                    FakeDocument("shared", "共享内容", 9999),
                    FakeDocument("es-only", "关键词结果", 0.01),
                ],
            ),
        ]

        candidates = reciprocal_rank_fusion(rankings, rrf_k=60)

        self.assertEqual([candidate.chunk_id for candidate in candidates].count("shared"), 1)
        self.assertEqual(candidates[0].chunk_id, "shared")
        self.assertEqual(candidates[0].sources, {"vector", "elasticsearch"})
        self.assertEqual(len(candidates[0].evidence), 2)

    def test_raw_score_scale_does_not_change_rrf_order(self):
        def ranking(vector_score, es_score):
            return [
                RetrievalRanking(
                    "vector",
                    "content",
                    "q",
                    "kb",
                    [FakeDocument("a", "A", vector_score), FakeDocument("b", "B", 0)],
                ),
                RetrievalRanking(
                    "elasticsearch",
                    "content",
                    "q",
                    "kb",
                    [FakeDocument("b", "B", es_score), FakeDocument("a", "A", 0)],
                ),
            ]

        first = reciprocal_rank_fusion(ranking(0.001, 100000))
        second = reciprocal_rank_fusion(ranking(99999, 0.00001))

        self.assertEqual(
            [candidate.chunk_id for candidate in first],
            [candidate.chunk_id for candidate in second],
        )

    def test_rrf_quality_metrics_outperform_raw_cross_source_score_sort(self):
        vector_documents = [
            FakeDocument("target", "正确答案", 0.7),
            *[
                FakeDocument(f"vector-noise-{index}", "噪声", 0.6 - index * 0.01)
                for index in range(1, 6)
            ],
        ]
        es_documents = [
            *[
                FakeDocument(f"es-noise-{index}", "噪声", 101 - index)
                for index in range(1, 6)
            ],
            FakeDocument("target", "正确答案", 95),
        ]
        rankings = [
            RetrievalRanking("vector", "content", "q", "kb", vector_documents),
            RetrievalRanking("elasticsearch", "content", "q", "kb", es_documents),
        ]

        before = [
            document.chunk_id
            for document in sorted(
                [*vector_documents, *es_documents],
                key=lambda document: document.score,
                reverse=True,
            )
        ]
        after = [candidate.chunk_id for candidate in reciprocal_rank_fusion(rankings)]
        relevance = {"target": 3}

        self.assertGreater(recall_at_k(after, relevance, 5), recall_at_k(before, relevance, 5))
        self.assertGreater(
            reciprocal_rank_at_k(after, relevance, 5),
            reciprocal_rank_at_k(before, relevance, 5),
        )
        self.assertGreater(ndcg_at_k(after, relevance, 5), ndcg_at_k(before, relevance, 5))


class RerankFallbackTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def candidates():
        return [
            RetrievalCandidate("a", "A", "", "f", "f.md", "kb", "", fusion_score=0.3),
            RetrievalCandidate("b", "B", "", "f", "f.md", "kb", "", fusion_score=0.2),
            RetrievalCandidate("c", "C", "", "f", "f.md", "kb", "", fusion_score=0.1),
        ]

    async def test_successful_rerank_maps_back_to_full_candidates(self):
        async def fake_rerank(query, documents, **kwargs):
            return [FakeRerankResult(1, 0.9), FakeRerankResult(0, 0.1)]

        ranked, succeeded = await rerank_candidates_with_fallback(
            "query",
            self.candidates(),
            fake_rerank,
            top_k=2,
            min_score=0.2,
            timeout_seconds=1,
        )

        self.assertTrue(succeeded)
        self.assertEqual([candidate.chunk_id for candidate in ranked], ["b"])
        self.assertEqual(ranked[0].file_name, "f.md")
        self.assertEqual(ranked[0].rerank_score, 0.9)

    async def test_rerank_failure_falls_back_to_rrf_without_score_filter(self):
        async def failing_rerank(query, documents, **kwargs):
            raise RuntimeError("rerank unavailable")

        ranked, succeeded = await rerank_candidates_with_fallback(
            "query",
            self.candidates(),
            failing_rerank,
            top_k=2,
            min_score=0.99,
            timeout_seconds=1,
        )

        self.assertFalse(succeeded)
        self.assertEqual([candidate.chunk_id for candidate in ranked], ["a", "b"])

    async def test_rerank_timeout_falls_back_to_rrf(self):
        async def slow_rerank(query, documents, **kwargs):
            await asyncio.sleep(0.05)
            return []

        ranked, succeeded = await rerank_candidates_with_fallback(
            "query",
            self.candidates(),
            slow_rerank,
            top_k=1,
            min_score=0.2,
            timeout_seconds=0.01,
        )

        self.assertFalse(succeeded)
        self.assertEqual([candidate.chunk_id for candidate in ranked], ["a"])


if __name__ == "__main__":
    unittest.main()
