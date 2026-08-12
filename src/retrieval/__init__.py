"""Retrieval infrastructure for RefWalk.

  - corpus           : Corpus loading, node_id canonicalisation / rollup
  - embedding        : dense retriever (Qwen3-Embedding via vLLM) + FAISS
  - baselines        : BM25 / Hybrid / reranker-backed retrievers
  - reranker         : Qwen3-Reranker client (vLLM)
  - walker_retrieval : anchoring + 1-hop OKG expansion + 3-view RRM fusion
  - dedup            : article-level candidate de-duplication
"""
