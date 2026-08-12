"""Retrieval baselines for FAQ evaluation.

Moved 2026-04-28 from ``src/baselines/retrieval.py`` (alongside corpus
loading and the embedding client). Legacy import path
``src.baselines.retrieval`` is preserved as a re-export shim.

B1. BM25                – rank_bm25 over corpus texts (Korean tokeniser)
B3. Hybrid              – linear combination of BM25 + Dense scores
B5_legacy.              – Hybrid top-N → Qwen3.6-35B yes/no reranking
B5 (neo). Qwen3Reranker — Hybrid top-N → Qwen3-Reranker-4B softmax score

The OKG-aware family (former B4 / B6) was retired: B4 (decay-only
expansion) never scored neighbours against the query, and B6
(Reranker(B4)) was a strict subset of WalkerRetrievalPipeline with a
narrower candidate pool. The canonical OKG-aware retriever is now
``src.retrieval.walker_retrieval.WalkerRetrievalPipeline`` (hybrid
top-30 + 1-hop typed expansion + Qwen3-Reranker over the union).
"""

from __future__ import annotations

import re

import numpy as np
import requests
from rank_bm25 import BM25Okapi

from src.retrieval.corpus import Corpus
from src.retrieval.embedding import DenseRetriever


# Legacy 35B yes/no reranker target. The dedicated
# ``src.retrieval.reranker.Qwen3Reranker`` is the modern path.
LLM_URL = "http://localhost:8035/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"


# ─── Tokenizer (whitespace + jamo-level for Korean) ──────────────────

_RE_TOKEN = re.compile(r"[가-힣]+|[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _RE_TOKEN.findall(text.lower())


# ─── BM25 baseline ──────────────────────────────────────────────────


class BM25Retriever:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        tokenized = [_tokenize(t) for t in corpus.texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.corpus.node_ids[i], float(scores[i])) for i in top_idx]

    def scores_all(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(_tokenize(query))


# ─── Hybrid baseline ────────────────────────────────────────────────


class HybridRetriever:
    def __init__(
        self, bm25: BM25Retriever, dense: DenseRetriever, alpha: float = 0.5
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.alpha = alpha
        self.corpus = bm25.corpus

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        bm25_scores = self.bm25.scores_all(query)
        dense_scores = self.dense.scores_all(query)

        bm25_norm = _min_max_norm(bm25_scores)
        dense_norm = _min_max_norm(dense_scores)

        combined = self.alpha * bm25_norm + (1 - self.alpha) * dense_norm
        top_idx = np.argsort(combined)[::-1][:top_k]
        return [(self.corpus.node_ids[i], float(combined[i])) for i in top_idx]


def _min_max_norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ─── LLM Reranker (legacy 35B yes/no) ───────────────────────────────


_RERANK_PROMPT = (
    "아래 법령 조항이 질문에 답하는 데 직접적으로 관련이 있습니까?\n\n"
    "질문: {question}\n\n"
    "조항 ({node_id}):\n{text}\n\n"
    "yes 또는 no로만 답하세요."
)


class LLMReranker:
    """Legacy 35B prompt-based reranker. Preserved as B5_Reranker_legacy."""

    def __init__(
        self,
        base_retriever,
        corpus: Corpus,
        candidate_k: int = 20,
    ) -> None:
        self.base = base_retriever
        self.corpus = corpus
        self.candidate_k = candidate_k

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        candidates = self.base.search(query, top_k=self.candidate_k)

        yes_list: list[tuple[str, float]] = []
        no_list: list[tuple[str, float]] = []

        for nid, sc in candidates:
            idx = self.corpus.id_to_idx.get(nid)
            if idx is None:
                no_list.append((nid, sc))
                continue
            text = self.corpus.texts[idx]
            if len(text) > 1500:
                text = text[:1500]
            relevant = self._judge(query, nid, text)
            if relevant:
                yes_list.append((nid, sc))
            else:
                no_list.append((nid, sc))

        reranked = yes_list + no_list
        return reranked[:top_k]

    @staticmethod
    def _judge(question: str, node_id: str, text: str) -> bool:
        prompt = _RERANK_PROMPT.format(
            question=question, node_id=node_id, text=text
        )
        try:
            resp = requests.post(
                LLM_URL,
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 16,
                    "temperature": 0.0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"].get("content") or ""
            answer = content.strip().lower()
            return answer.startswith("yes") or answer.startswith("예")
        except Exception:
            return False


# ─── Qwen3-Reranker wrapper (drop-in retriever) ────────────────────


class Qwen3VLRerankerRetriever:
    """Drop-in ``search(query, top_k)`` retriever that pulls N
    candidates from a base retriever and reranks them with the
    Qwen3-Reranker (``src.retrieval.reranker.Qwen3Reranker``).

    Used for B5_Reranker(neo). The reranker client is lazy-constructed
    so importing this module does not require the endpoint to be up.
    The class name keeps the legacy ``Qwen3VL`` prefix for backward
    compat with older imports — under the hood it now talks to the
    text-only Qwen3-Reranker-4B endpoint.
    """

    def __init__(
        self,
        base_retriever,
        corpus: Corpus,
        endpoint: str | None = None,
        instruction: str | None = None,
    ) -> None:
        self.base = base_retriever
        self.corpus = corpus
        self._endpoint = endpoint
        self._instruction = instruction
        self._client = None  # lazy

    def _get_client(self):
        if self._client is None:
            from src.retrieval.reranker import (
                DEFAULT_ENDPOINT,
                DEFAULT_INSTRUCTION,
                Qwen3Reranker,
            )
            self._client = Qwen3Reranker(
                endpoint=self._endpoint or DEFAULT_ENDPOINT,
                instruction=(
                    self._instruction
                    if self._instruction is not None
                    else DEFAULT_INSTRUCTION
                ),
            )
        return self._client

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        pool = self.base.search(query, top_k=top_k * 2)
        if not pool:
            return []

        from src.retrieval.reranker import candidates_from_pairs

        id_to_text = {
            nid: self.corpus.texts[self.corpus.id_to_idx[nid]]
            for nid, _ in pool
            if nid in self.corpus.id_to_idx
        }
        candidates = candidates_from_pairs(
            pool, id_to_text, source="hybrid", score_key="hybrid_score"
        )
        reranked = self._get_client().rerank(
            query=query, candidates=candidates, top_k=top_k
        )
        return [(r["node_id"], r["rerank_score"]) for r in reranked]
