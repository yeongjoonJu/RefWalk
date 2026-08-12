"""Embedding API client + Dense retriever (Qwen3-Embedding-4B).

Updated 2026-04-28: switched from ``Qwen3-VL-Embedding-2B`` (multimodal,
1536-d) to ``Qwen/Qwen3-Embedding-4B`` (text-only, 2560-d, 8192-token
context).

Query side uses the official ``get_detailed_instruct(task, query)``
formatting (prepends ``Instruct: …\\nQuery:…`` as a single text input);
documents are embedded with no instruction prefix.

The HTTP wire format is the standard OpenAI-compatible
``POST /v1/embeddings`` with ``input`` as a list of strings — the
multimodal ``messages`` payload that the prior 2B model required is
gone.

Public surface
--------------
- ``EMBED_URL`` / ``EMBED_MODEL`` / ``EMBED_BATCH`` / ``EMBED_MAX_CHARS``
  — configuration constants.
- ``DEFAULT_TASK_INSTRUCTION`` — Korean R&D regulations task prompt.
- ``get_detailed_instruct(task, query)`` — query-side formatter.
- ``_embed_documents(texts)`` / ``_embed_query(question)`` — HTTP helpers.
- ``DenseRetriever`` — FAISS cosine-similarity retriever over corpus
  embeddings (cache-aware).
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import requests

from src.retrieval.corpus import Corpus


EMBED_URL = "http://localhost:8090/v1/embeddings"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBED_BATCH = 64
# Qwen3-Embedding-4B is served with --max-model-len 8192 tokens.
# Empirically Korean hangul tokenises at ~0.68 tokens/char (12000
# chars → 8193 tokens, which the server rejects). 8000 chars stays
# safely under the limit (~5500 tokens) with headroom for the
# instruction prefix and BOS/EOS, and is robust to mixed-script docs
# (Latin/digits tokenise denser than hangul).
EMBED_MAX_CHARS = 8000

# English task prompt — Qwen3-Embedding's training format expects an
# English ``Instruct: …`` line followed by the (possibly Korean)
# query body.
DEFAULT_TASK_INSTRUCTION = (
    "Given a question about Korean national R&D regulations, "
    "retrieve relevant legal clauses that answer the question."
)


# ─── Query-side formatter ──────────────────────────────────────────


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Build the ``Instruct: <task>\\nQuery:<query>`` line that
    Qwen3-Embedding expects on the query side. Document side uses the
    raw text without a prefix.
    """
    return f"Instruct: {task_description}\nQuery:{query}"


# ─── HTTP helpers ──────────────────────────────────────────────────


def _post_embeddings(payload: dict, timeout: float) -> dict:
    """POST to the embeddings endpoint. On HTTP error, surface the
    server's response body — vLLM 4xx responses carry the actual
    reason (token-length, batch-token cap, model-name mismatch, etc.)
    which is otherwise hidden inside ``raise_for_status``.
    """
    resp = requests.post(EMBED_URL, json=payload, timeout=timeout)
    if not resp.ok:
        body = (resp.text or "").strip()
        raise requests.HTTPError(
            f"{resp.status_code} {resp.reason} from {EMBED_URL} "
            f"(model={payload.get('model')!r}, "
            f"n_inputs={len(payload.get('input', []))}, "
            f"max_input_chars="
            f"{max((len(s) for s in payload.get('input', [])), default=0)}). "
            f"Server said: {body[:500]}"
        )
    return resp.json()


def _embed_documents(texts: list[str]) -> np.ndarray:
    """Embed a list of document texts in batches. No instruction prefix
    on the doc side (Qwen3-Embedding's training convention)."""
    all_vecs: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = [t[:EMBED_MAX_CHARS] for t in texts[i : i + EMBED_BATCH]]
        body = _post_embeddings(
            {"model": EMBED_MODEL, "input": batch}, timeout=120,
        )
        data = body["data"]
        # OpenAI spec doesn't promise order; sort by ``index``.
        data.sort(key=lambda x: x["index"])
        all_vecs.extend(e["embedding"] for e in data)
    return np.array(all_vecs, dtype=np.float32)


def _embed_query(
    question: str,
    task_description: str = DEFAULT_TASK_INSTRUCTION,
) -> np.ndarray:
    """Embed a single query with the official Qwen3-Embedding
    instruction format (``Instruct: …\\nQuery:…`` as a single text input).
    """
    prompt = get_detailed_instruct(task_description, question)
    body = _post_embeddings(
        {"model": EMBED_MODEL, "input": [prompt[:EMBED_MAX_CHARS]]}, timeout=60,
    )
    vec = body["data"][0]["embedding"]
    return np.array(vec, dtype=np.float32)


# ─── DenseRetriever (FAISS over corpus embeddings) ─────────────────


class DenseRetriever:
    def __init__(self, corpus: Corpus, embeddings: np.ndarray) -> None:
        self.corpus = corpus
        self.embeddings = embeddings
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    @classmethod
    def build(cls, corpus: Corpus) -> "DenseRetriever":
        print(f"  Embedding {len(corpus.texts)} documents …")
        embs = _embed_documents(corpus.texts)
        return cls(corpus, embs)

    @classmethod
    def from_cache(cls, corpus: Corpus, cache_path: Path) -> "DenseRetriever":
        embs = np.load(cache_path)
        if embs.shape[0] != len(corpus.texts):
            raise ValueError(
                f"Embedding cache size mismatch: cache {cache_path} has "
                f"{embs.shape[0]} vectors but corpus has {len(corpus.texts)} "
                f"entries. The cache was built for a different corpus "
                f"(likely a different --corpus-source / --use-indexed-text / "
                f"--articles combo). Delete the cache or point --embed-cache "
                f"at a fresh path so it gets rebuilt."
            )
        return cls(corpus, embs)

    def save_cache(self, cache_path: Path) -> None:
        np.save(cache_path, self.embeddings)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        qvec = _embed_query(query).reshape(1, -1)
        faiss.normalize_L2(qvec)
        scores, indices = self.index.search(qvec, top_k)
        return [
            (self.corpus.node_ids[int(idx)], float(sc))
            for sc, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    def scores_all(self, query: str) -> np.ndarray:
        qvec = _embed_query(query).reshape(1, -1)
        faiss.normalize_L2(qvec)
        scores, _ = self.index.search(qvec, len(self.corpus.node_ids))
        out = np.zeros(len(self.corpus.node_ids), dtype=np.float32)
        _, indices = self.index.search(qvec, len(self.corpus.node_ids))
        for sc, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                out[int(idx)] = float(sc)
        return out
