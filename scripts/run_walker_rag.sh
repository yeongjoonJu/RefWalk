#!/usr/bin/env bash
# WalkerRetrieval (3-view RRM) + NativeRAG generation.
#
# Same generation stack as NativeRAG (same prompt, same LLM, same citation
# footer parser); only the retrieval step is swapped to WalkerRetrievalPipeline.
# Round-5 default pipeline: anchored seed query (mid view) → top-50 dense seeds
# → 1-hop OKG expansion (REF / DEL / SPEC, decay 0.7) → 3-view rerank
# (narrow / mid / wide) → Reciprocal Rank MAX fusion → top-K.
#
# Standardized fair-comparison knobs:
#   - LLM:        Qwen/Qwen3.6-35B-A3B-FP8 @ :8035 (35B; same as judge) — MODEL_SIZE=35b
#                 Qwen/Qwen3.5-4B            @ :8040 (4B; cross-model eval) — MODEL_SIZE=4b
#   - max_tokens=2048, profile=exact, context-top-k=10, reranker ON
#
# MODEL_SIZE env var (default 35b) selects the inference LLM preset:
#   - MODEL_SIZE=35b  Qwen3.6-35B @ :8035, OUT_TAG defaults to "35b"
#   - MODEL_SIZE=4b   Qwen3.5-4B  @ :8040, OUT_TAG defaults to "4b"
#   Judge stays at 35B in both presets for cross-model fairness on 4b.
#   Manual LLM_BASE_URL / LLM_MODEL / OUT_TAG env vars still override.
#
#
# OUT_TAG env var suffixes outputs:
#   experiments/walker_rag/preds_${OUT_TAG}.jsonl
#
# Skip flags:
#   SKIP_GEN=1 to reuse existing predictions
#   SKIP_EVAL=1 to skip the judge eval

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODEL_SIZE="${MODEL_SIZE:-35b}"

# ── Inference LLM preset by MODEL_SIZE ─────────────────────────────────
case "$MODEL_SIZE" in
    35b)
        _LLM_BASE_URL_DEFAULT="http://localhost:8035/v1"
        _LLM_MODEL_DEFAULT="Qwen/Qwen3.6-35B-A3B-FP8"
        _OUT_TAG_DEFAULT="35b"
        ;;
    4b)
        _LLM_BASE_URL_DEFAULT="http://localhost:8040/v1"
        _LLM_MODEL_DEFAULT="Qwen/Qwen3.5-4B"
        _OUT_TAG_DEFAULT="4b"
        ;;
    *)
        echo "[error] unknown MODEL_SIZE=$MODEL_SIZE (use 4b|35b)" >&2
        exit 1
        ;;
esac

OUT_TAG="${OUT_TAG:-$_OUT_TAG_DEFAULT}"

# ── Standardized fair-comparison knobs ─────────────────────────────────
LLM_BASE_URL="${LLM_BASE_URL:-$_LLM_BASE_URL_DEFAULT}"
LLM_MODEL="${LLM_MODEL:-$_LLM_MODEL_DEFAULT}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
PROFILE="${PROFILE:-exact}"
CONTEXT_TOP_K="${CONTEXT_TOP_K:-10}"

# Walker retrieval tunables (Round-5 default: pool=50, 3-view RRM).
RERANK_POOL="${RERANK_POOL:-50}"
OKG_EXPAND_SEED="${OKG_EXPAND_SEED:-10}"

# Reranker server (used inside Walker, not optional — Walker requires it).
RERANK_BASE_URL="${RERANK_BASE_URL:-http://localhost:8095}"
RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-Reranker-0.6B}"

# Eval (35B judge) — same model as inference here.
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8035/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"

SKIP_GEN="${SKIP_GEN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

# ── Paths ──────────────────────────────────────────────────────────────
ARTICLES="${ARTICLES:-data/RegOps-Bench/articles.jsonl}"
OKG="${OKG:-data/okg/okg.gpickle}"
BENCH="${BENCH:-data/RegOps-Bench/regops_bench.jsonl}"
EMBED_CACHE="${EMBED_CACHE:-data/okg/doc_embeddings_parse_articles.npy}"
OUT="${OUT:-experiments/walker_rag/preds_${OUT_TAG}.jsonl}"
SCORES_DIR="${SCORES_DIR:-experiments/walker_rag/scores_${OUT_TAG}}"
LANGUAGE="${LANGUAGE:-ko}"

echo
echo "============================================================"
echo "[walker-rag] OUT_TAG=$OUT_TAG  pool=$RERANK_POOL"
echo "  bench  = $BENCH"
echo "  preds  → $OUT"
echo "  scores → $SCORES_DIR"
echo "============================================================"

# ── Inference ──────────────────────────────────────────────────────
if [[ "$SKIP_GEN" != "1" ]]; then
    mkdir -p "$(dirname "$OUT")"
    python cli/run_walker_rag_eval.py \
        --articles        "$ARTICLES" \
        --okg             "$OKG" \
        --bench           "$BENCH" \
        --out             "$OUT" \
        --embed-cache     "$EMBED_CACHE" \
        --rerank-pool     "$RERANK_POOL" \
        --okg-expand-seed "$OKG_EXPAND_SEED" \
        --context-top-k   "$CONTEXT_TOP_K" \
        --rerank-base-url "$RERANK_BASE_URL" \
        --rerank-model    "$RERANK_MODEL" \
        --llm-base-url    "$LLM_BASE_URL" \
        --llm-model       "$LLM_MODEL" \
        --profile         "$PROFILE" \
        --max-tokens      "$MAX_TOKENS" \
        --language        "$LANGUAGE"
else
    echo "[walker-rag] SKIP_GEN=1, reusing $OUT"
fi

# ── Scoring ────────────────────────────────────────────────────────
if [[ "$SKIP_EVAL" != "1" ]]; then
    mkdir -p "$SCORES_DIR"
    echo "[walker-rag] eval $OUT → $SCORES_DIR (judge=$JUDGE_BASE_URL)"
    python cli/run_generation_eval.py \
        --runs "$OUT" \
        --faq  "$BENCH" \
        --out  "$SCORES_DIR" \
        --base-url "$JUDGE_BASE_URL" \
        --model-id "$JUDGE_MODEL" \
        --language "$LANGUAGE"
else
    echo "[walker-rag] SKIP_EVAL=1"
fi

CORPUS="${CORPUS:-data/okg/okg_nodes.jsonl}"
python cli/evaluate.py \
    --predictions "$OUT" \
    --bench       "$BENCH" \
    --corpus      "$CORPUS" \
    --label       "walker_rag_${OUT_TAG}" \
    --dedup-questions \
    --out         "${SCORES_DIR}/unified"

echo
echo "[walker-rag] done."
