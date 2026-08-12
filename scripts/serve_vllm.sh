#!/usr/bin/env bash
# One vLLM launcher for every model RefWalk talks to.
#
#   bash scripts/serve_vllm.sh <role> [--gpu N] [--port N] [--model ID] [-- <extra vllm args>]
#
# Roles:
#   35b      answer / judge LLM   Qwen/Qwen3.6-35B-A3B-FP8   :8035  (GPU 0)
#   4b       answer LLM           Qwen/Qwen3.5-4B            :8040  (GPU 0)
#   embed    embedding server     Qwen/Qwen3-Embedding-0.6B  :8090  (GPU 0)
#   rerank   reranker server      Qwen/Qwen3-Reranker-0.6B   :8095  (GPU 0)
#
# Every default is overridable by flag or by environment variable
# (GPU, PORT, MODEL, MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION, MAX_NUM_SEQS, HOST):
#
#   bash scripts/serve_vllm.sh 35b                      # GPU 0, port 8035
#   bash scripts/serve_vllm.sh 4b --gpu 5               # GPU 5, port 8040
#   bash scripts/serve_vllm.sh embed --gpu 3
#   bash scripts/serve_vllm.sh rerank --gpu 7 --port 9095
#   GPU=2 bash scripts/serve_vllm.sh 35b                # env form
#   bash scripts/serve_vllm.sh 35b -- --tensor-parallel-size 2
#
# A full RefWalk run needs three servers up at once — an answer LLM, the
# embedder, and the reranker — so give them different GPUs:
#
#   bash scripts/serve_vllm.sh 4b     --gpu 0 &
#   bash scripts/serve_vllm.sh embed  --gpu 1 &
#   bash scripts/serve_vllm.sh rerank --gpu 2 &

set -euo pipefail

usage() {
    # Print the header comment block: every line after the shebang, up to
    # the first line that isn't a comment.
    awk 'NR > 1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
    exit "${1:-0}"
}

[[ $# -ge 1 ]] || usage 1
case "$1" in -h|--help|help) usage 0 ;; esac

ROLE="$1"; shift

# ── Per-role defaults (env wins over these, flags win over env) ────────
case "$ROLE" in
    35b)
        DEF_MODEL="Qwen/Qwen3.6-35B-A3B-FP8"; DEF_PORT=8035 ;;
    4b)
        DEF_MODEL="Qwen/Qwen3.5-4B";          DEF_PORT=8040 ;;
    embed)
        DEF_MODEL="Qwen/Qwen3-Embedding-0.6B"; DEF_PORT=8090 ;;
    rerank)
        DEF_MODEL="Qwen/Qwen3-Reranker-0.6B";  DEF_PORT=8095 ;;
    *)
        echo "[error] unknown role '$ROLE' (use 35b|4b|embed|rerank)" >&2
        usage 1 ;;
esac

MODEL="${MODEL:-$DEF_MODEL}"
PORT="${PORT:-$DEF_PORT}"
GPU="${GPU:-0}"
HOST="${HOST:-0.0.0.0}"

# ── Flag overrides ─────────────────────────────────────────────────────
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)   GPU="$2";   shift 2 ;;
        --port)  PORT="$2";  shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --host)  HOST="$2";  shift 2 ;;
        --)      shift; EXTRA=("$@"); break ;;
        -h|--help) usage 0 ;;
        *) echo "[error] unknown option '$1' (pass extra vllm args after --)" >&2
           usage 1 ;;
    esac
done

# ── Per-role vLLM arguments ────────────────────────────────────────────
ARGS=(--host "$HOST" --port "$PORT")
case "$ROLE" in
    35b|4b)
        ARGS+=(
            --max-model-len "${MAX_MODEL_LEN:-32768}"
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
            --max-num-seqs "${MAX_NUM_SEQS:-8}"
            --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
            --reasoning-parser qwen3
        )
        # The 4B checkpoint is bf16; the 35B one is already FP8-quantised.
        [[ "$ROLE" == "4b" ]] && ARGS+=(--dtype bfloat16)
        ;;
    embed)
        ARGS+=(
            --runner pooling --trust-remote-code --dtype bfloat16
            --max-model-len "${MAX_MODEL_LEN:-8192}"
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.60}"
        )
        ;;
    rerank)
        # Qwen3-Reranker ships as a causal LM; these overrides turn it into
        # the yes/no sequence classifier the scorer expects.
        ARGS+=(
            --dtype bfloat16
            --max-model-len "${MAX_MODEL_LEN:-8192}"
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.60}"
            --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}'
        )
        ;;
esac

echo "[serve] role=$ROLE model=$MODEL gpu=$GPU port=$PORT"
exec env CUDA_VISIBLE_DEVICES="$GPU" \
    vllm serve "$MODEL" "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
