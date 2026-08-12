#!/usr/bin/env bash
# RefWalk = WalkerRetrieval (3-view RRM, pool 50) × REFWALK_SYSTEM JSON
# answer / per-clause citations. Evaluates the 4B and 35B answer LLMs on
# the same retrieval stack so any quality delta attributes to the
# generator size.
#
# Same retrieval defaults as run_walker_rag.sh (anchored seed → top-50
# dense seeds → 1-hop OKG expansion → 3-view RRM rerank → top-10), but
# the generation prompt is REFWALK_SYSTEM + get_user_prompt, which forces
# the LLM to emit a strict JSON object whose keys are the cited node_ids
# and whose ``answer`` field carries the synthesised answer.
#
# MODEL env var:
#   - MODEL=4b           run only the 4B answer LLM
#   - MODEL=35b          run only the 35B answer LLM
#   - MODEL=all (default) run both, sequentially
#
# Skip flags:
#   SKIP_GEN=1   reuse existing predictions
#   SKIP_EVAL=1  skip the judge eval

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODEL="${MODEL:-all}"

# ── Walker tunables (Round-5 default: pool=50, 3-view RRM) ─────────────
RERANK_POOL="${RERANK_POOL:-50}"
OKG_EXPAND_SEED="${OKG_EXPAND_SEED:-10}"
# RegOps articles are long and chained, so 10 context passages help.
CONTEXT_TOP_K="${CONTEXT_TOP_K:-10}"
EVAL_TOP_K="${EVAL_TOP_K:-10}"
PASSAGE_LIMIT="${PASSAGE_LIMIT:-3000}"

# ── Generation profile ────────────────────────────────────────────────
# REFWALK_SYSTEM is tuned for ``exact`` (low-temperature, no thinking) —
# it must emit a strict JSON object.
PROFILE="${PROFILE:-exact}" # exact
MAX_TOKENS="${MAX_TOKENS:-2048}" # 2048

# ── Reranker (shared) ──────────────────────────────────────────────────
RERANK_BASE_URL="${RERANK_BASE_URL:-http://localhost:8095}"
RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-Reranker-0.6B}"

# ── Judge (35B; matches the other RAG baselines) ──────────────────────
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8035/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"

# ── Per-model endpoints ────────────────────────────────────────────────
LLM_4B_BASE_URL="${LLM_4B_BASE_URL:-http://localhost:8040/v1}"
LLM_4B_MODEL="${LLM_4B_MODEL:-Qwen/Qwen3.5-4B}"
LLM_35B_BASE_URL="${LLM_35B_BASE_URL:-http://localhost:8035/v1}"
LLM_35B_MODEL="${LLM_35B_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"

LANGUAGE="${LANGUAGE:-ko}"
SKIP_GEN="${SKIP_GEN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

# ── Per-model runner ───────────────────────────────────────────────────
run_one() {
    local tag="$1"

    local bench articles okg embed_cache out_root corpus
    bench="${BENCH:-data/RegOps-Bench/regops_bench.jsonl}"
    articles="${ARTICLES:-data/RegOps-Bench/articles.jsonl}"
    corpus="${CORPUS:-data/okg/okg_nodes.jsonl}"
    okg="${OKG:-data/okg/okg.gpickle}"
    embed_cache="${EMBED_CACHE:-data/okg/doc_embeddings_parse_articles.npy}"
    out_root="${OUT_ROOT:-experiments/refwalk}"

    local llm_base_url llm_model
    if [[ "$tag" == "4b" ]]; then
        llm_base_url="$LLM_4B_BASE_URL"
        llm_model="$LLM_4B_MODEL"
    elif [[ "$tag" == "35b" ]]; then
        llm_base_url="$LLM_35B_BASE_URL"
        llm_model="$LLM_35B_MODEL"
    else
        echo "[error] unknown model tag '$tag' (use 4b|35b)" >&2
        return 1
    fi

    local out="${out_root}/preds_${tag}.jsonl"
    local scores="${out_root}/scores_${tag}"

    echo
    echo "============================================================"
    echo "[refwalk :: ${tag}] llm=${llm_base_url} model=${llm_model}"
    echo "  bench  = $bench"
    echo "  preds  → $out"
    echo "  scores → $scores"
    echo "============================================================"

    if [[ "$SKIP_GEN" != "1" ]]; then
        mkdir -p "$(dirname "$out")"
        python cli/run_refwalk_eval.py \
            --bench           "$bench" \
            --articles        "$articles" \
            --okg             "$okg" \
            --embed-cache     "$embed_cache" \
            --out             "$out" \
            --rerank-pool     "$RERANK_POOL" \
            --okg-expand-seed "$OKG_EXPAND_SEED" \
            --context-top-k   "$CONTEXT_TOP_K" \
            --eval-top-k      "$EVAL_TOP_K" \
            --passage-limit   "$PASSAGE_LIMIT" \
            --rerank-base-url "$RERANK_BASE_URL" \
            --rerank-model    "$RERANK_MODEL" \
            --llm-base-url    "$llm_base_url" \
            --llm-model       "$llm_model" \
            --profile         "$PROFILE" \
            --max-tokens      "$MAX_TOKENS" \
            --language        "$LANGUAGE"
    else
        echo "[refwalk :: ${tag}] SKIP_GEN=1, reusing $out"
    fi

    if [[ "$SKIP_EVAL" != "1" ]]; then
        mkdir -p "$scores"
        echo "[refwalk :: ${tag}] eval $out → $scores (judge=$JUDGE_BASE_URL)"
        python cli/run_generation_eval.py \
            --runs "$out" \
            --faq  "$bench" \
            --out  "$scores" \
            --base-url "$JUDGE_BASE_URL" \
            --model-id "$JUDGE_MODEL" \
            --language "$LANGUAGE"
    else
        echo "[refwalk :: ${tag}] SKIP_EVAL=1"
    fi

    # Unified retrieval + citation report via the framework-agnostic
    # evaluator. Pure Python (no LLM), so it always runs — this is the
    # same command an external RAG system would use on its own
    # predictions. See docs/evaluation.md.
    python cli/evaluate.py \
        --predictions "$out" \
        --bench       "$bench" \
        --corpus      "$corpus" \
        --label       "refwalk_${tag}" \
        --dedup-questions \
        --out         "${scores}/unified"
}

# ── Dispatch ──────────────────────────────────────────────────────────
models=()
case "$MODEL" in
    4b|35b) models=("$MODEL") ;;
    all)    models=(4b 35b) ;;
    *) echo "[error] unknown MODEL=$MODEL (use 4b|35b|all)" >&2; exit 1 ;;
esac

for m in "${models[@]}"; do
    run_one "$m"
done

echo
echo "[refwalk] done."
