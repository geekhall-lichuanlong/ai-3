import math
import re
import time
from collections import Counter, defaultdict

from app.config import RRF_K
from app.embedding import HashingEmbedding, cosine_similarity, tokenize
from app.models import Chunk, SearchHit, SearchResponse, Source
from app.storage import TenantStorage


ANSWER_CUE_RE = re.compile(r"\d|为|是|目标|默认|必须|需要")
QUESTION_CUE_RE = re.compile(r"多少|多久|谁|为什么|怎么|是否|配额|sla", re.IGNORECASE)
QUESTION_LIST_RE = re.compile(r"推荐测试问题|问题\s*\d|？|\?")


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk.text) for chunk in chunks]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))

    def score(self, query: str) -> list[tuple[Chunk, float]]:
        query_terms = tokenize(query)
        total_docs = len(self.chunks)
        scores: list[tuple[Chunk, float]] = []
        if not query_terms or not total_docs:
            return []
        for chunk, tokens, doc_len in zip(self.chunks, self.doc_tokens, self.doc_lengths, strict=True):
            frequencies = Counter(tokens)
            score = 0.0
            for term in query_terms:
                if term not in frequencies:
                    continue
                idf = math.log(1 + (total_docs - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
                tf = frequencies[term]
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
                score += idf * (tf * (self.k1 + 1)) / denominator
            if score > 0:
                scores.append((chunk, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)


class HybridRetriever:
    def __init__(self, embedding: HashingEmbedding | None = None) -> None:
        self.embedding = embedding or HashingEmbedding()

    def search(self, tenant_id: str, query: str, top_k: int) -> SearchResponse:
        started = time.perf_counter()
        chunks = TenantStorage(tenant_id).load_chunks()
        query_embedding = self.embedding.embed(query)
        vector_ranked = sorted(
            ((chunk, cosine_similarity(query_embedding, chunk.embedding)) for chunk in chunks),
            key=lambda item: item[1],
            reverse=True,
        )
        vector_ranked = [(chunk, score) for chunk, score in vector_ranked if score > 0][: max(top_k * 4, 10)]
        keyword_ranked = BM25(chunks).score(query)[: max(top_k * 4, 10)]

        fused: dict[str, float] = defaultdict(float)
        vector_ranks: dict[str, int] = {}
        keyword_ranks: dict[str, int] = {}
        chunk_by_id = {chunk.id: chunk for chunk in chunks}

        for rank, (chunk, _) in enumerate(vector_ranked, start=1):
            vector_ranks[chunk.id] = rank
            fused[chunk.id] += 1 / (RRF_K + rank)
        for rank, (chunk, _) in enumerate(keyword_ranked, start=1):
            keyword_ranks[chunk.id] = rank
            fused[chunk.id] += 1 / (RRF_K + rank)

        self._boost_answer_like_chunks(query, chunk_by_id, fused)

        hits = []
        for chunk_id, score in sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]:
            chunk = chunk_by_id[chunk_id]
            hits.append(
                SearchHit(
                    chunk_id=chunk.id,
                    score=score,
                    vector_rank=vector_ranks.get(chunk.id),
                    keyword_rank=keyword_ranks.get(chunk.id),
                    source=Source(
                        file_id=chunk.file_id,
                        filename=chunk.filename,
                        page=chunk.page,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        text=chunk.text,
                    ),
                )
            )
        retrieval_ms = (time.perf_counter() - started) * 1000
        return SearchResponse(tenant_id=tenant_id, hits=hits, retrieval_ms=retrieval_ms)

    def _boost_answer_like_chunks(
        self,
        query: str,
        chunks: dict[str, Chunk],
        fused: dict[str, float],
    ) -> None:
        if not QUESTION_CUE_RE.search(query):
            return
        query_tokens = set(tokenize(query))
        for chunk_id in list(fused):
            chunk = chunks[chunk_id]
            chunk_tokens = set(tokenize(chunk.text))
            overlap = len(query_tokens & chunk_tokens)
            if "推荐测试问题" in chunk.text:
                fused[chunk_id] *= 0.35
                continue
            if QUESTION_LIST_RE.search(chunk.text) and not re.search(r"\d{2,}|为\s*\d|默认|必须", chunk.text):
                fused[chunk_id] *= 0.55
                continue
            if overlap < 2 or not ANSWER_CUE_RE.search(chunk.text):
                continue
            fused[chunk_id] += min(0.01, overlap * 0.0015)
