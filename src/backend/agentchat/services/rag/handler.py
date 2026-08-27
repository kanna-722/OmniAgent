import asyncio
from typing import Optional, Sequence

from loguru import logger

from agentchat.services.rag.es_client import client as es_client
from agentchat.services.rag.retrieval import (
    MixRetrival,
    RetrievalCandidate,
    normalize_rewritten_queries,
    reciprocal_rank_fusion,
    rerank_candidates_with_fallback,
)
from agentchat.services.rag.rerank import Reranker
from agentchat.services.rag.vector_db import milvus_client
from agentchat.services.rewrite.query_write import query_rewriter
from agentchat.settings import app_settings


NO_RELEVANT_DOCUMENTS = "No relevant documents found."


class RagHandler:
    @staticmethod
    def _retrieval_setting(name: str, default):
        value = app_settings.rag.retrival.get(name, default)
        return default if value is None else value

    @classmethod
    def _int_setting(cls, name: str, default: int, minimum: int = 1) -> int:
        try:
            value = int(cls._retrieval_setting(name, default))
            if value < minimum:
                raise ValueError
            return value
        except (TypeError, ValueError):
            logger.warning(f"Invalid RAG integer setting {name}, fallback to {default}")
            return default

    @classmethod
    def _float_setting(cls, name: str, default: float, minimum: float = 0) -> float:
        try:
            value = float(cls._retrieval_setting(name, default))
            if value < minimum:
                raise ValueError
            return value
        except (TypeError, ValueError):
            logger.warning(f"Invalid RAG numeric setting {name}, fallback to {default}")
            return default

    @classmethod
    async def query_rewrite(cls, query: str) -> list[str]:
        original_query = str(query).strip()
        if not original_query:
            return []

        max_rewrites = cls._int_setting("rewrite_max_queries", 3, minimum=0)
        try:
            rewritten = await asyncio.wait_for(
                query_rewriter.rewrite(original_query),
                timeout=10,
            )
        except Exception as err:
            logger.warning(f"Query rewrite failed, fallback to original query: {err}")
            rewritten = []

        return normalize_rewritten_queries(original_query, rewritten, max_rewrites)

    @classmethod
    async def index_milvus_documents(cls, collection_name, chunks):
        await milvus_client.insert(collection_name, chunks)

    @classmethod
    async def index_es_documents(cls, index_name, chunks):
        await es_client.index_documents(index_name, chunks)

    @classmethod
    async def mix_retrival_documents(
        cls,
        query_list,
        knowledges_id,
        search_field="summary",
        *,
        index_names=None,
        candidate_k: int | None = None,
    ) -> list[RetrievalCandidate]:
        if candidate_k is None:
            candidate_k = cls._int_setting("candidate_k", 20)
        rankings = await MixRetrival.retrieve_ranked_lists(
            queries=query_list,
            collection_names=knowledges_id,
            index_names=index_names or knowledges_id,
            search_field=search_field,
            enable_elasticsearch=bool(app_settings.rag.enable_elasticsearch),
            concurrency=cls._int_setting("retrieval_concurrency", 8),
            top_k=max(1, candidate_k),
            timeout_seconds=cls._float_setting(
                "retrieval_timeout_seconds",
                10,
                minimum=0.001,
            ),
            vector_client=milvus_client,
            keyword_client=es_client,
        )
        return reciprocal_rank_fusion(
            rankings,
            rrf_k=cls._int_setting("rrf_k", 60),
        )[:candidate_k]

    @classmethod
    async def _rank_candidates(
        cls,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        top_k: int,
        min_score: float | None,
    ) -> tuple[list[RetrievalCandidate], bool]:
        if not candidates:
            return [], False
        timeout = cls._float_setting("rerank_timeout_seconds", 10, minimum=0.001)
        return await rerank_candidates_with_fallback(
            query,
            candidates,
            Reranker.rerank_documents,
            top_k=top_k,
            min_score=min_score,
            timeout_seconds=timeout,
        )

    @classmethod
    async def _retrieve_candidates(
        cls,
        query: str,
        collection_names,
        index_names,
        *,
        search_field: str,
        top_k: int,
        needs_query_rewrite: bool,
        prepared_queries: Sequence[str] | None = None,
    ) -> list[RetrievalCandidate]:
        if prepared_queries is not None:
            queries = list(prepared_queries)
        elif needs_query_rewrite:
            queries = await cls.query_rewrite(query)
        else:
            queries = [str(query).strip()] if str(query).strip() else []

        configured_candidate_k = cls._int_setting("candidate_k", 20)
        candidate_k = max(top_k * 4, configured_candidate_k)
        return await cls.mix_retrival_documents(
            queries,
            collection_names,
            search_field,
            index_names=index_names,
            candidate_k=candidate_k,
        )

    @staticmethod
    def _merge_unique_candidates(
        primary: Sequence[RetrievalCandidate],
        fallback: Sequence[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        merged = []
        seen = set()
        for candidate in [*primary, *fallback]:
            key = candidate.chunk_id or (
                candidate.knowledge_id,
                candidate.file_id,
                candidate.content,
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(candidate)
            if len(merged) >= top_k:
                break
        return merged

    @classmethod
    async def rag_query_summary(
        cls,
        query,
        knowledges_id,
        min_score: Optional[float] = None,
        top_k: Optional[int] = None,
        needs_query_rewrite: bool = True,
    ):
        if min_score is None:
            min_score = cls._float_setting("min_score", 0.2)
        if top_k is None:
            top_k = cls._int_setting("top_k", 5)
        top_k = max(1, int(top_k))
        min_score = float(min_score) if min_score is not None else None

        prepared_queries = (
            await cls.query_rewrite(query)
            if needs_query_rewrite
            else [str(query).strip()]
        )

        summary_candidates = await cls._retrieve_candidates(
            query,
            knowledges_id,
            knowledges_id,
            search_field="summary",
            top_k=top_k,
            needs_query_rewrite=needs_query_rewrite,
            prepared_queries=prepared_queries,
        )
        ranked_summaries, _ = await cls._rank_candidates(
            query,
            summary_candidates,
            top_k=top_k,
            min_score=min_score,
        )

        if len(ranked_summaries) < top_k:
            logger.info("Summary recall below top_k, fallback to content retrieval")
            content_candidates = await cls._retrieve_candidates(
                query,
                knowledges_id,
                knowledges_id,
                search_field="content",
                top_k=top_k,
                needs_query_rewrite=needs_query_rewrite,
                prepared_queries=prepared_queries,
            )
            ranked_content, _ = await cls._rank_candidates(
                query,
                content_candidates,
                top_k=top_k,
                min_score=min_score,
            )
            ranked_summaries = cls._merge_unique_candidates(
                ranked_summaries,
                ranked_content,
                top_k,
            )

        if not ranked_summaries:
            return NO_RELEVANT_DOCUMENTS
        return "\n".join(candidate.content for candidate in ranked_summaries)

    @classmethod
    async def retrieve_ranked_documents(
        cls,
        query,
        collection_names,
        index_names=None,
        min_score: Optional[float] = None,
        top_k: Optional[int] = None,
        needs_query_rewrite: bool = True,
    ):
        """查询改写、并行召回、RRF 融合、Rerank 与过滤的兼容入口。"""
        if min_score is None:
            min_score = cls._float_setting("min_score", 0.2)
        if top_k is None:
            top_k = cls._int_setting("top_k", 5)
        top_k = max(1, int(top_k))
        min_score = float(min_score) if min_score is not None else None
        index_names = index_names or collection_names

        candidates = await cls._retrieve_candidates(
            query,
            collection_names,
            index_names,
            search_field="content",
            top_k=top_k,
            needs_query_rewrite=needs_query_rewrite,
        )
        ranked_candidates, _ = await cls._rank_candidates(
            query,
            candidates,
            top_k=top_k,
            min_score=min_score,
        )
        if not ranked_candidates:
            return NO_RELEVANT_DOCUMENTS
        return "\n".join(candidate.content for candidate in ranked_candidates)

    @classmethod
    async def delete_documents_es_milvus(cls, file_id, knowledge_id):
        if app_settings.rag.enable_elasticsearch:
            await es_client.delete_documents(file_id, knowledge_id)
        await milvus_client.delete_by_file_id(file_id, knowledge_id)
