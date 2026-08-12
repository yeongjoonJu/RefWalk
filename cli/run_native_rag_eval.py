"""Run NativeRAG inference over the RegOps QA benchmark.

NativeRAG = the simplest classical RAG floor:
    dense retrieval (Qwen3-Embedding) → optional cross-rerank
    (Qwen3-Reranker) → LLM answer with a ``[참조] node_id, ...`` footer.

Output schema is identical to ``cli/run_pike_rag_eval.py`` /
``cli/run_hipporag2_eval.py`` so ``cli/run_generation_eval.py`` can score
it without changes.

Usage::

    # default: no reranker
    python cli/run_native_rag_eval.py \\
        --articles data/RegOps-Bench/articles.jsonl \\
        --bench    data/RegOps-Bench/regops_bench.jsonl \\
        --out      experiments/native_rag/preds.jsonl

    # with cross-encoder rerank (Qwen3-Reranker on :8095)
    python cli/run_native_rag_eval.py --enable-reranker

    # score
    python cli/run_generation_eval.py \\
        --runs experiments/native_rag/preds.jsonl \\
        --faq  data/RegOps-Bench/regops_bench.jsonl \\
        --out  experiments/native_rag/scores
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load .env (GEMINI_API_KEY etc.) so hosted backends work without
# exporting the key manually. No-op when python-dotenv is absent or the
# file does not exist; the local-vLLM Qwen path needs no key.
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except Exception:
    pass

from src.baselines.native_rag import NativeRagResult, NativeRagWorkflow
from src.retrieval.reranker import Qwen3Reranker


# ─── I/O ───────────────────────────────────────────────────────────


def _read_bench(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _read_done_qa_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            qa_id = row.get("qa_id")
            if qa_id:
                done.add(qa_id)
    return done


# ─── Result -> runs.jsonl row ──────────────────────────────────────


def _result_to_row(qa: dict, result: NativeRagResult, variant: str) -> dict:
    return {
        "qa_id": qa.get("qa_id"),
        "variant": variant,
        "difficulty": qa.get("difficulty") or qa.get("target_difficulty"),
        "language": qa.get("language"),
        "status": result.status,
        "error": result.error,
        "question": qa.get("question"),
        "gt_answer": qa.get("answer"),
        "gt_references": qa.get("gt_references") or [],
        "answer": result.answer,
        "cited_references": result.cited_references,
        "retrieved_node_ids": result.retrieved_node_ids,
        "rerank_scores": result.rerank_scores,
    }


# ─── Args ──────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run NativeRAG inference.")
    p.add_argument("--articles", default="data/RegOps-Bench/articles.jsonl")
    p.add_argument("--bench", default="data/RegOps-Bench/regops_bench.jsonl")
    p.add_argument("--out", default="experiments/native_rag/preds.jsonl")
    p.add_argument(
        "--embed-cache",
        default="data/okg/native_rag_embeddings.npy",
        help="Path to the .npy doc-embedding cache (rebuilt on size mismatch).",
    )

    p.add_argument("--retrieve-top-k", type=int, default=20,
                   help="Initial dense candidates fetched.")
    p.add_argument("--context-top-k", type=int, default=10,
                   help="Passages fed to the LLM after rerank/truncation.")

    p.add_argument(
        "--enable-reranker",
        action="store_true",
        help="Apply Qwen3-Reranker between dense retrieval and the LLM. "
             "Default off — reranker is optional.",
    )
    p.add_argument("--rerank-base-url", default="http://localhost:8095")
    p.add_argument("--rerank-model", default="Qwen/Qwen3-Reranker-0.6B")

    # Defaults target the local Qwen 4B vLLM server. To run on Gemini,
    # point --llm-base-url at the Gemini OpenAI-compatible endpoint and
    # set --llm-model. Use --profile gemini_min_thinking for thinking-only
    # models (gemini-3.1-pro-preview), or --profile gemini_exact for true
    # non-thinking on flash-class models. See scripts/run_nativerag_gemini.sh.
    p.add_argument("--llm-base-url", default="http://localhost:8040/v1",
                   help="Standardized 4B endpoint. Override per shell var if needed.")
    p.add_argument("--llm-model", default="Qwen/Qwen3.5-4B")
    p.add_argument(
        "--llm-api-key", default=None,
        help="API key for hosted LLM backends (e.g. Gemini). Defaults to "
             "the GEMINI_API_KEY env var (loaded from .env). Ignored by "
             "the local Qwen vLLM server.",
    )
    p.add_argument("--profile", default="exact",
                   help="Key into src.utils.hparams.Qwen3_5_HParams. "
                        "'exact' (temp=0.1, Qwen-recommended presence_penalty=1.5) "
                        "is the standard fair-comparison profile across baselines; "
                        "use 'gemini_exact' (flash non-thinking) or "
                        "'gemini_min_thinking' (Pro, thinking-only) for Gemini.")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--passage-max-chars", type=int, default=2400)
    p.add_argument(
        "--language",
        default="ko",
        choices=["ko", "en"],
        help="Prompt/answer language. 'ko' = RegOps (Korean R&D regs), "
             "'en' renders the same prompts in English.",
    )

    p.add_argument(
        "--oracle", action="store_true",
        help="Oracle generation mode: context = GT articles + random "
             "1-hop OKG distractors (totalling --oracle-total-k) instead "
             "of dense retrieval. Isolates generation quality from "
             "retrieval error while keeping citation non-trivial.",
    )
    p.add_argument(
        "--okg", default="data/okg/okg.gpickle",
        help="OKG gpickle — only loaded in --oracle mode for 1-hop "
             "distractor sampling.",
    )
    p.add_argument(
        "--oracle-total-k", type=int, default=10,
        help="Oracle mode: total context passages = GT + 1-hop "
             "distractors (default 10). GT is never truncated.",
    )
    p.add_argument(
        "--oracle-distractor-seed", type=int, default=42,
        help="Oracle mode: base RNG seed for distractor sampling / "
             "shuffle (seeded per qa_id for resume-stable contexts).",
    )

    p.add_argument(
        "--variant",
        default=None,
        help="Override variant name in runs.jsonl. Default: native_rag (or "
             "native_rag_rerank when --enable-reranker is set; "
             "native_rag_oracle when --oracle is set).",
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="Run on at most N benchmark rows (0 = all).",
    )
    p.add_argument(
        "--difficulty", default=None,
        help="Optional comma list filter, e.g. 'L1' or 'L1,L2'.",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Overwrite output file instead of appending / skipping done.",
    )
    return p.parse_args(argv)


# ─── Main ──────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.no_resume and out_path.exists():
        out_path.unlink()

    if args.variant:
        variant = args.variant
    elif args.oracle:
        variant = "native_rag_oracle"
    elif args.enable_reranker:
        variant = "native_rag_rerank"
    else:
        variant = "native_rag"

    # 1. Reranker (optional).
    reranker = None
    if args.enable_reranker:
        reranker = Qwen3Reranker(
            endpoint=args.rerank_base_url,
            model=args.rerank_model,
        )

    # OKG graph: only needed for oracle 1-hop distractor sampling.
    okg_graph = None
    if args.oracle:
        import pickle

        print(f"[native-rag] loading OKG from {args.okg}")
        with open(args.okg, "rb") as f:
            okg_graph = pickle.load(f)

    # 2. Build workflow once; re-use for every question.
    print("[native-rag] building workflow …")
    workflow = NativeRagWorkflow(
        articles_path=Path(args.articles),
        embed_cache=Path(args.embed_cache) if args.embed_cache else None,
        reranker=reranker,
        retrieve_top_k=args.retrieve_top_k,
        context_top_k=args.context_top_k,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key or os.environ.get("GEMINI_API_KEY"),
        profile_name=args.profile,
        max_tokens=args.max_tokens,
        passage_max_chars=args.passage_max_chars,
        language=args.language,
        okg_graph=okg_graph,
        oracle_total_k=args.oracle_total_k,
        oracle_seed=args.oracle_distractor_seed,
    )
    print(
        f"[native-rag] corpus={len(workflow.corpus.node_ids)} articles "
        f"reranker={'on' if reranker is not None else 'off'} "
        f"mode={'oracle-gt' if args.oracle else 'retrieval'} "
        f"language={args.language} variant={variant}"
    )

    # 3. Load benchmark + filter.
    bench = _read_bench(Path(args.bench))
    if args.difficulty:
        keep = {d.strip() for d in args.difficulty.split(",") if d.strip()}
        bench = [
            q for q in bench
            if (q.get("difficulty") or q.get("target_difficulty")) in keep
        ]

    done = set() if args.no_resume else _read_done_qa_ids(out_path)
    pending = [q for q in bench if q.get("qa_id") not in done]
    if args.limit:
        pending = pending[: args.limit]

    print(
        f"[native-rag] bench total={len(bench)} done={len(done)} "
        f"pending={len(pending)}"
    )
    if not pending:
        print("[native-rag] nothing to do.")
        return

    # 4. Run inference; flush per row so Ctrl-C leaves a usable file.
    t0 = time.time()
    n_ok = n_err = 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for i, qa in enumerate(pending, 1):
            qa_id = qa.get("qa_id")
            question = qa.get("question") or ""
            if not question.strip():
                row = {
                    "qa_id": qa_id,
                    "variant": variant,
                    "status": "error",
                    "error": "empty question",
                    "answer": "",
                    "cited_references": [],
                    "retrieved_node_ids": [],
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                n_err += 1
                continue

            t_start = time.time()
            result = workflow.answer_gt(qa) if args.oracle else workflow.answer(question)
            elapsed = time.time() - t_start

            row = _result_to_row(qa, result, variant=variant)
            row["latency_sec"] = round(elapsed, 3)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            if result.status == "ok":
                n_ok += 1
            else:
                n_err += 1

            print(
                f"  [{i:>3}/{len(pending)}] {qa_id:<20s} "
                f"status={result.status:<5s} "
                f"hits={len(result.retrieved_node_ids)} "
                f"refs={len(result.cited_references)} "
                f"t={elapsed:.1f}s"
            )

    dt = time.time() - t0
    print(
        f"[native-rag] done. ok={n_ok} err={n_err} "
        f"total_time={dt:.1f}s -> {out_path}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
