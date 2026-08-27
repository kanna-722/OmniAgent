import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalEvidence:
    source: str
    search_field: str
    query: str
    knowledge_id: str
    original_rank: int
    original_score: float | None


@dataclass
class RetrievalCandidate:
    chunk_id: str
    content: str
    summary: str
    file_id: str
    file_name: str
    knowledge_id: str
    update_time: str
    fusion_score: float = 0.0
    rerank_score: float | None = None
    best_original_rank: int = 2**31 - 1
    evidence: list[RetrievalEvidence] = field(default_factory=list)

    @property
    def sources(self) -> set[str]:
        return {item.source for item in self.evidence}

    @property
    def queries(self) -> set[str]:
        return {item.query for item in self.evidence}


@dataclass(frozen=True)
class RetrievalRanking:
    source: str
    search_field: str
    query: str
    knowledge_id: str
    documents: Sequence[Any]


class MixRetrival:
    """并行执行各检索列表；跨来源排序由 RRF 完成。"""

    @staticmethod
    def _normalize_values(values: str | Iterable[str] | None) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            values = [values]
        normalized = []
        seen = set()
        for value in values:
            text = str(value).strip()
            if text and text not in seen:
                seen.add(text)
                normalized.append(text)
        return normalized

    @classmethod
    async def retrieve_ranked_lists(
        cls,
        queries: Sequence[str],
        collection_names: Sequence[str],
        index_names: Sequence[str] | None,
        search_field: str,
        *,
        enable_elasticsearch: bool,
        concurrency: int,
        top_k: int,
        timeout_seconds: float = 10,
        vector_client: Any | None = None,
        keyword_client: Any | None = None,
    ) -> list[RetrievalRanking]:
        queries = cls._normalize_values(queries)
        collection_names = cls._normalize_values(collection_names)
        index_names = cls._normalize_values(index_names or collection_names)

        if collection_names and vector_client is None:
            from agentchat.services.rag.vector_db import milvus_client

            vector_client = milvus_client
        if enable_elasticsearch and keyword_client is None:
            from agentchat.services.rag.es_client import client as es_client

            keyword_client = es_client

        semaphore = asyncio.Semaphore(max(1, concurrency))
        task_specs = [
            ("vector", vector_client, query, knowledge_id)
            for query in queries
            for knowledge_id in collection_names
        ]
        if enable_elasticsearch:
            task_specs.extend(
                ("elasticsearch", keyword_client, query, knowledge_id)
                for query in queries
                for knowledge_id in index_names
            )

        async def run_task(source, client, query, knowledge_id):
            async with semaphore:
                try:
                    if source == "elasticsearch":
                        method_name = (
                            "search_documents_summary"
                            if search_field == "summary"
                            else "search_documents"
                        )
                    else:
                        method_name = "search_summary" if search_field == "summary" else "search"

                    documents = await asyncio.wait_for(
                        getattr(client, method_name)(
                            query,
                            knowledge_id,
                            top_k=top_k,
                        ),
                        timeout=timeout_seconds,
                    )
                    return RetrievalRanking(
                        source=source,
                        search_field=search_field,
                        query=query,
                        knowledge_id=knowledge_id,
                        documents=documents or [],
                    )
                except Exception as err:
                    logger.warning(
                        "RAG retrieval task failed: source=%s field=%s knowledge=%s query=%r error=%s: %s",
                        source,
                        search_field,
                        knowledge_id,
                        query,
                        type(err).__name__,
                        err,
                    )
                    return None

        results = await asyncio.gather(
            *(run_task(*task_spec) for task_spec in task_specs),
            return_exceptions=True,
        )
        rankings = []
        for result in results:
            if isinstance(result, RetrievalRanking):
                rankings.append(result)
            elif isinstance(result, Exception):
                logger.warning("Unexpected RAG retrieval error: %s", result)
        return rankings

    @classmethod
    async def retrival_milvus_documents(cls, query, knowledges_id, search_field):
        rankings = await cls.retrieve_ranked_lists(
            cls._normalize_values(query),
            knowledges_id,
            None,
            search_field,
            enable_elasticsearch=False,
            concurrency=8,
            top_k=10,
        )
        return [document for ranking in rankings for document in ranking.documents]

    @classmethod
    async def retrival_es_documents(cls, query, knowledges_id, search_field):
        rankings = await cls.retrieve_ranked_lists(
            cls._normalize_values(query),
            [],
            knowledges_id,
            search_field,
            enable_elasticsearch=True,
            concurrency=8,
            top_k=10,
        )
        return [document for ranking in rankings for document in ranking.documents]

    @classmethod
    async def mix_retrival_documents(cls, query_list, knowledges_id, search_field):
        rankings = await cls.retrieve_ranked_lists(
            cls._normalize_values(query_list),
            knowledges_id,
            knowledges_id,
            search_field,
            enable_elasticsearch=True,
            concurrency=8,
            top_k=10,
        )
        es_documents = [
            document
            for ranking in rankings
            if ranking.source == "elasticsearch"
            for document in ranking.documents
        ]
        vector_documents = [
            document
            for ranking in rankings
            if ranking.source == "vector"
            for document in ranking.documents
        ]
        return es_documents, vector_documents


def normalize_rewritten_queries(
    original_query: str,
    rewritten: Any,
    max_rewrites: int = 3,
) -> list[str]:
    original_query = str(original_query).strip()
    if not original_query:
        return []
    if isinstance(rewritten, dict):
        rewritten = rewritten.get("variations", [])
    if not isinstance(rewritten, (list, tuple)):
        rewritten = []

    queries = [original_query]
    max_rewrites = max(0, max_rewrites)
    if max_rewrites == 0:
        return queries
    seen = {original_query.casefold()}
    for item in rewritten:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        dedup_key = normalized.casefold()
        if not normalized or dedup_key in seen:
            continue
        seen.add(dedup_key)
        queries.append(normalized)
        if len(queries) >= max_rewrites + 1:
            break
    return queries


def reciprocal_rank_fusion(
    rankings: Sequence[RetrievalRanking],
    rrf_k: int = 60,
) -> list[RetrievalCandidate]:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer")

    candidates: dict[str, RetrievalCandidate] = {}
    for ranking in rankings:
        seen_in_ranking = set()
        for rank, document in enumerate(ranking.documents, start=1):
            chunk_id = str(getattr(document, "chunk_id", "") or "").strip()
            fallback_key = "|".join([
                str(getattr(document, "knowledge_id", "") or ranking.knowledge_id),
                str(getattr(document, "file_id", "") or ""),
                str(getattr(document, "content", "") or ""),
            ])
            candidate_key = chunk_id or fallback_key
            if not candidate_key or candidate_key in seen_in_ranking:
                continue
            seen_in_ranking.add(candidate_key)

            raw_score = getattr(document, "score", None)
            try:
                original_score = float(raw_score) if raw_score is not None else None
            except (TypeError, ValueError):
                original_score = None

            evidence = RetrievalEvidence(
                source=ranking.source,
                search_field=ranking.search_field,
                query=ranking.query,
                knowledge_id=ranking.knowledge_id,
                original_rank=rank,
                original_score=original_score,
            )
            candidate = candidates.get(candidate_key)
            if candidate is None:
                candidate = RetrievalCandidate(
                    chunk_id=chunk_id,
                    content=str(getattr(document, "content", "") or ""),
                    summary=str(getattr(document, "summary", "") or ""),
                    file_id=str(getattr(document, "file_id", "") or ""),
                    file_name=str(getattr(document, "file_name", "") or ""),
                    knowledge_id=str(
                        getattr(document, "knowledge_id", "") or ranking.knowledge_id
                    ),
                    update_time=str(getattr(document, "update_time", "") or ""),
                )
                candidates[candidate_key] = candidate
            else:
                for attribute in (
                    "content",
                    "summary",
                    "file_id",
                    "file_name",
                    "knowledge_id",
                    "update_time",
                ):
                    if not getattr(candidate, attribute):
                        setattr(candidate, attribute, str(getattr(document, attribute, "") or ""))

            candidate.evidence.append(evidence)
            candidate.best_original_rank = min(candidate.best_original_rank, rank)
            candidate.fusion_score += 1.0 / (rrf_k + rank)

    return sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate.fusion_score,
            candidate.best_original_rank,
            candidate.chunk_id,
        ),
    )


async def rerank_candidates_with_fallback(
    query: str,
    candidates: Sequence[RetrievalCandidate],
    rerank_callable,
    *,
    top_k: int,
    min_score: float | None,
    timeout_seconds: float,
) -> tuple[list[RetrievalCandidate], bool]:
    if not candidates:
        return [], False

    documents = [candidate.content for candidate in candidates]
    try:
        reranked_docs = await asyncio.wait_for(
            rerank_callable(
                query,
                documents,
                top_n=len(documents),
                timeout_seconds=timeout_seconds,
            ),
            timeout=timeout_seconds,
        )
        if not reranked_docs:
            raise ValueError("Rerank returned no results")

        ranked_candidates = []
        seen_indices = set()
        for reranked in reranked_docs:
            index = int(reranked.index)
            if index < 0 or index >= len(candidates) or index in seen_indices:
                continue
            seen_indices.add(index)
            candidate = candidates[index]
            candidate.rerank_score = float(reranked.score)
            ranked_candidates.append(candidate)

        if not ranked_candidates:
            raise ValueError("Rerank returned no valid candidate index")

        filtered = [
            candidate
            for candidate in ranked_candidates
            if min_score is None or candidate.rerank_score >= min_score
        ]
        return filtered[:top_k], True
    except Exception as err:
        logger.warning(
            "Rerank failed, fallback to RRF order: %s: %s",
            type(err).__name__,
            err,
        )
        return list(candidates[:top_k]), False
