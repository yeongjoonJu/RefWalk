#!/usr/bin/env bash
# Retrieval evaluation over the shipped baselines on RegOps-Bench.
#
# Baselines (see cli/run_retrieval_eval.py):
#   B1_BM25, B2_Dense, B3_Hybrid, B5_Hybrid_Reranker, Walker_Retrieval
#
# Prerequisites (see README.md):
#   - data/RegOps-Bench/regops_bench.jsonl  (download from HuggingFace)
#   - data/okg/okg.gpickle + data/okg/okg_nodes.jsonl
#       (build once:  python cli/run_build_okg.py
#                       --articles data/RegOps-Bench/articles.jsonl --out-dir data/okg)
#   - Embedding server  @ :8090   (scripts/serve_vllm.sh embed)
#   - Reranker server   @ :8095   (scripts/serve_vllm.sh rerank)
#   - LLM server        @ :8035   (scripts/serve_vllm.sh 35b; used by Walker's
#                                   online TopicExtractor for anchoring)
#
# The dense embedding cache (data/okg/doc_embeddings_okg_nodes.npy) is built
# automatically on the first run and reused afterwards.
set -euo pipefail

BENCH="${BENCH:-data/RegOps-Bench/regops_bench.jsonl}"
ARTICLES="${ARTICLES:-data/okg/okg_nodes.jsonl}"
OKG="${OKG:-data/okg/okg.gpickle}"
OUT="${OUT:-experiments/retrieval_baselines}"
ONLY="${ONLY:-}"   # e.g. ONLY=B1_BM25,Walker_Retrieval

python cli/run_retrieval_eval.py \
    --faq        "$BENCH" \
    --articles   "$ARTICLES" \
    --okg        "$OKG" \
    --out        "$OUT" \
    --top-k-probe 10 \
    --rerank-pool 50 \
    --okg-expand-seed 10 \
    --hit-rule   hierarchical \
    ${ONLY:+--only "$ONLY"}

echo "[done] retrieval metrics + traces written to $OUT/"
