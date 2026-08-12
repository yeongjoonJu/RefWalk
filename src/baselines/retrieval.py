"""Backward-compat shim for ``src.baselines.retrieval``.

The actual retrieval primitives live under ``src.retrieval`` as of
2026-04-28:

  - ``src.retrieval.corpus``    — ``Corpus``, ``load_corpus``,
                                   ``load_faq``, ``_canon_ref``,
                                   ``_article_ancestor``.
  - ``src.retrieval.embedding`` — ``EMBED_*`` constants,
                                   ``get_detailed_instruct``,
                                   ``_embed_documents``,
                                   ``_embed_query``,
                                   ``DenseRetriever``.
  - ``src.retrieval.baselines`` — ``BM25Retriever``,
                                   ``HybridRetriever``,
                                   ``LLMReranker``,
                                   ``Qwen3VLRerankerRetriever``,
                                   ``_min_max_norm``, ``_tokenize``.

This module re-exports those symbols at the legacy import path so
existing CLI / scripts / tests keep working without churn. New code
should import from the canonical ``src.retrieval.*`` modules.
"""

from __future__ import annotations

from src.retrieval.baselines import (  # noqa: F401
    LLM_MODEL,
    LLM_URL,
    BM25Retriever,
    HybridRetriever,
    LLMReranker,
    Qwen3VLRerankerRetriever,
    _min_max_norm,
    _tokenize,
)
from src.retrieval.corpus import (  # noqa: F401
    Corpus,
    _article_ancestor,
    _canon_ref,
    _prepend_header,
    _readable_doc_name,
    load_corpus,
    load_faq,
)
from src.retrieval.embedding import (  # noqa: F401
    DEFAULT_TASK_INSTRUCTION,
    EMBED_BATCH,
    EMBED_MAX_CHARS,
    EMBED_MODEL,
    EMBED_URL,
    DenseRetriever,
    _embed_documents,
    _embed_query,
    get_detailed_instruct,
)


__all__ = [
    # corpus
    "Corpus",
    "load_corpus",
    "load_faq",
    "_canon_ref",
    "_article_ancestor",
    "_prepend_header",
    "_readable_doc_name",
    # embedding
    "EMBED_URL",
    "EMBED_MODEL",
    "EMBED_BATCH",
    "EMBED_MAX_CHARS",
    "DEFAULT_TASK_INSTRUCTION",
    "get_detailed_instruct",
    "_embed_documents",
    "_embed_query",
    "DenseRetriever",
    # baselines
    "BM25Retriever",
    "HybridRetriever",
    "LLMReranker",
    "Qwen3VLRerankerRetriever",
    "LLM_URL",
    "LLM_MODEL",
    "_tokenize",
    "_min_max_norm",
]
