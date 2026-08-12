#!/usr/bin/env bash
# NativeRAG inference + scoring.
#
# Standardized fair-comparison knobs:
#   - LLM:        Qwen/Qwen3.5-4B @ :8040 (4B mode, default)
#   - Judge:      Qwen/Qwen3.6-35B-A3B-FP8 @ :8035 (default)
#   - max_tokens=2048, profile=exact, context-top-k=10, reranker OFF
#     (NativeRAG is the plain dense-retrieval floor; ENABLE_RERANK=1 to add it)
#
# Switch model variant by overriding LLM_BASE_URL/LLM_MODEL/OUT_TAG.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

OUT_TAG="${OUT_TAG:-4b}"

# ── Standardized fair-comparison knobs ─────────────────────────────────
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8040/v1}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3.5-4B}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
PROFILE="${PROFILE:-exact}"
CONTEXT_TOP_K="${CONTEXT_TOP_K:-10}"
RETRIEVE_TOP_K="${RETRIEVE_TOP_K:-20}"
ENABLE_RERANK="${ENABLE_RERANK:-0}"   # 0 = off (NativeRAG default), 1 = on
RERANK_BASE_URL="${RERANK_BASE_URL:-http://localhost:8095}"

# Eval (35B judge) — separate from inference LLM
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8035/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"

# ── Paths ──────────────────────────────────────────────────────────────
ARTICLES="${ARTICLES:-data/RegOps-Bench/articles.jsonl}"
BENCH="${BENCH:-data/RegOps-Bench/regops_bench.jsonl}"
EMBED_CACHE="${EMBED_CACHE:-data/okg/native_rag_embeddings.npy}"
CORPUS="${CORPUS:-data/okg/okg_nodes.jsonl}"
OUT="${OUT:-experiments/native_rag/preds_${OUT_TAG}.jsonl}"
SCORES_DIR="${SCORES_DIR:-experiments/native_rag/scores_${OUT_TAG}}"
LANGUAGE="${LANGUAGE:-ko}"

SKIP_GEN="${SKIP_GEN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

# ── Inference ──────────────────────────────────────────────────────────
if [[ "$SKIP_GEN" != "1" ]]; then
    rerank_arg=()
    if [[ "$ENABLE_RERANK" == "1" ]]; then
        rerank_arg=(--enable-reranker --rerank-base-url "$RERANK_BASE_URL")
    fi

    mkdir -p "$(dirname "$OUT")"
    echo "[nativerag] OUT_TAG=$OUT_TAG llm=$LLM_BASE_URL → $OUT"
    python cli/run_native_rag_eval.py \
        --articles      "$ARTICLES" \
        --bench         "$BENCH" \
        --out           "$OUT" \
        --embed-cache   "$EMBED_CACHE" \
        --retrieve-top-k "$RETRIEVE_TOP_K" \
        --context-top-k "$CONTEXT_TOP_K" \
        --llm-base-url  "$LLM_BASE_URL" \
        --llm-model     "$LLM_MODEL" \
        --profile       "$PROFILE" \
        --max-tokens    "$MAX_TOKENS" \
        --language      "$LANGUAGE" \
        "${rerank_arg[@]}"
else
    echo "[nativerag] SKIP_GEN=1, using existing $OUT"
fi

# ── Scoring ────────────────────────────────────────────────────────────
if [[ "$SKIP_EVAL" != "1" ]]; then
    mkdir -p "$SCORES_DIR"
    echo "[nativerag] eval $OUT → $SCORES_DIR (judge=$JUDGE_BASE_URL)"
    python cli/run_generation_eval.py \
        --runs "$OUT" \
        --faq  "$BENCH" \
        --out  "$SCORES_DIR" \
        --base-url "$JUDGE_BASE_URL" \
        --model-id "$JUDGE_MODEL" \
        --language "$LANGUAGE"
else
    echo "[nativerag] SKIP_EVAL=1"
fi

# ── Unified retrieval + citation report ────────────────────────────────
# Framework-agnostic evaluator; pure Python, so it always runs. This is
# the same command an external RAG system uses on its own predictions.
# See docs/evaluation.md.
python cli/evaluate.py \
    --predictions "$OUT" \
    --bench       "$BENCH" \
    --corpus      "$CORPUS" \
    --label       "native_rag_${OUT_TAG}" \
    --dedup-questions \
    --out         "${SCORES_DIR}/unified"
