"""Run RefWalk (WalkerRetrieval + REFWALK_SYSTEM JSON answer) generation.

The prediction file matches ``cli/run_walker_rag_eval.py`` schema so
``cli/run_generation_eval.py`` scores it without changes.

Default Qwen3_5_HParams profile is ``exact`` — REFWALK_SYSTEM enforces
strict JSON output and the low-temperature profile is what the prompt
was tuned with.

Usage::

    # 4B
    python cli/run_refwalk_eval.py \\
        --bench data/RegOps-Bench/regops_bench.jsonl \\
        --out experiments/refwalk/preds_4b.jsonl \\
        --llm-base-url http://localhost:8040/v1 --llm-model Qwen/Qwen3.5-4B

    # 35B
    python cli/run_refwalk_eval.py \\
        --out experiments/refwalk/preds_35b.jsonl \\
        --llm-base-url http://localhost:8035/v1 \\
        --llm-model Qwen/Qwen3.6-35B-A3B-FP8

    # score
    python cli/run_generation_eval.py \\
        --runs experiments/refwalk/preds_4b.jsonl \\
        --faq  data/RegOps-Bench/regops_bench.jsonl \\
        --out  experiments/refwalk/scores_4b
"""

from __future__ import annotations

import argparse
import json
import pickle
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

from src.refwalk import RefWalkWorkflow
from src.retrieval.corpus import load_corpus
from src.retrieval.embedding import DenseRetriever
from src.retrieval.reranker import Qwen3Reranker
from src.retrieval.walker_retrieval import TopicExtractor, WalkerRetrievalPipeline


# ─── I/O ───────────────────────────────────────────────────────────


def _read_bench(path: Path) -> list[dict]:
    """Accept .jsonl (one row per line) or .json (a single list)."""
    if path.suffix == ".jsonl":
        out: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    return json.loads(path.read_text(encoding="utf-8"))


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


def _result_to_row(qa: dict, result, variant: str) -> dict:
    """Mirror the schema used by run_walker_rag_eval / run_native_rag_eval
    so cli/run_generation_eval.py treats this baseline identically."""
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
        "raw_output": result.raw_output,
    }


# ─── Args ──────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", default="data/RegOps-Bench/regops_bench.jsonl")
    p.add_argument("--articles", default="data/RegOps-Bench/articles.jsonl")
    p.add_argument("--okg", default="data/okg/okg.gpickle")
    p.add_argument(
        "--embed-cache",
        default="data/okg/doc_embeddings_parse_articles.npy",
    )
    p.add_argument("--out", default="experiments/refwalk/preds.jsonl")

    # Walker tunables — Round-5 default.
    p.add_argument("--rerank-pool", type=int, default=50)
    p.add_argument("--okg-expand-seed", type=int, default=10)
    p.add_argument("--context-top-k", type=int, default=10,
                   help="Reranked passages stitched into the LLM prompt.")
    p.add_argument("--eval-top-k", type=int, default=None,
                   help="Retrieved passages stored in preds for R@K "
                        "analytics. Defaults to context_top_k. Set higher "
                        "(e.g. 10 when context_top_k=5) to measure R@10 "
                        "without inflating the LLM context budget.")
    p.add_argument(
        "--no-topic-extractor", action="store_true",
        help="Disable online topic anchoring (only valid when bench is "
             "already pre-anchored).",
    )
    p.add_argument(
        "--no-condition-anchor", action="store_true",
        help="Generation-side ablation: do NOT condition the generation "
             "prompt on the topic anchor (topic + structured conditions). "
             "Retrieval still uses the anchor; only Context + Question are "
             "shown to the generator. Isolates the §4.2 anchor-injection "
             "contribution from the retrieval-side --no-topic-extractor.",
    )

    # Reranker.
    p.add_argument("--rerank-base-url", default="http://localhost:8095")
    p.add_argument("--rerank-model", default="Qwen/Qwen3-Reranker-0.6B")
    p.add_argument("--rerank-batch-size", type=int, default=64)

    # LLM. Defaults target the local Qwen3.6 vLLM server. To run RefWalk
    # on Gemini instead, point --llm-base-url at the Gemini
    # OpenAI-compatible endpoint and set --llm-model. Use --profile
    # gemini_min_thinking for thinking-only models (gemini-3.1-pro-preview),
    # or --profile gemini_exact for true non-thinking on flash-class
    # models (gemini-3-flash-preview). See scripts/run_refwalk_gemini.sh.
    p.add_argument("--llm-base-url", default="http://localhost:8035/v1")
    p.add_argument("--llm-model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    p.add_argument(
        "--llm-api-key", default=None,
        help="API key for hosted LLM backends (e.g. Gemini). Defaults to "
             "the GEMINI_API_KEY env var (loaded from .env). Ignored by "
             "the local Qwen vLLM server.",
    )
    p.add_argument(
        "--profile", default="exact",
        help="Sampling profile from src.utils.hparams.Qwen3_5_HParams. "
             "REFWALK_SYSTEM is tuned for 'exact' (Qwen, low temp, no "
             "thinking); use 'gemini_exact' (flash non-thinking) or "
             "'gemini_min_thinking' (Pro, thinking-only).",
    )
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--passage-limit", type=int, default=3000,
                   help="Per-passage character cap inside the Context block.")
    p.add_argument(
        "--language", default="ko", choices=["ko", "en"],
        help="Prompt/answer language. 'ko' is the RegOps default; 'en' "
             "renders the same prompts in English for an English corpus.",
    )

    p.add_argument(
        "--oracle", action="store_true",
        help="Oracle generation mode: feed the bench ``gt_references`` "
             "corpus bodies as context instead of the Walker retrieval "
             "stack. Isolates pure generation quality (answer + citation) "
             "from retrieval error. Topic anchoring still runs. "
             "Context = GT articles + random 1-hop OKG distractors, "
             "totalling --oracle-total-k passages.",
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

    # Variant + iteration.
    p.add_argument("--variant", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--difficulty", default=None,
                   help="Optional comma list filter, e.g. 'L1,L2'.")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args(argv)


# ─── Main ──────────────────────────────────────────────────────────


def _default_variant(args: argparse.Namespace) -> str:
    suffix = ""
    m = (args.llm_model or "").lower()
    if "35b" in m:
        suffix = "_35b"
    elif "4b" in m:
        suffix = "_4b"
    oracle = "_oracle" if getattr(args, "oracle", False) else ""
    return f"refwalk{oracle}{suffix}"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.no_resume and out_path.exists():
        out_path.unlink()

    variant = args.variant or _default_variant(args)

    # Resolve the LLM API key: explicit flag wins, else GEMINI_API_KEY
    # from the environment / .env. Stays None for the local Qwen path
    # (BaseLLM then falls back to the "dummy" key vLLM ignores).
    llm_api_key = args.llm_api_key or os.environ.get("GEMINI_API_KEY")

    # 1. Build retrieval stack.
    print(f"[refwalk] loading corpus from {args.articles}")
    corpus = load_corpus(Path(args.articles), source="parse")
    embed_cache = Path(args.embed_cache) if args.embed_cache else None
    if embed_cache and embed_cache.exists():
        dense = DenseRetriever.from_cache(corpus, embed_cache)
    else:
        dense = DenseRetriever.build(corpus)
        if embed_cache:
            embed_cache.parent.mkdir(parents=True, exist_ok=True)
            dense.save_cache(embed_cache)

    print(f"[refwalk] loading OKG from {args.okg}")
    with open(args.okg, "rb") as f:
        graph = pickle.load(f)

    reranker = Qwen3Reranker(
        endpoint=args.rerank_base_url,
        model=args.rerank_model,
        batch_size=args.rerank_batch_size,
    )

    topic_extractor: TopicExtractor | None
    if args.no_topic_extractor:
        topic_extractor = None
    else:
        if args.language == "ko":
            tx_domain = "Korean R&D funding regulations"
            tx_language = "Korean"
        else:
            tx_domain = "regulations"
            tx_language = "English"
        topic_extractor = TopicExtractor(
            base_url=args.llm_base_url,
            model_id=args.llm_model,
            profile_name=args.profile,
            domain=tx_domain,
            language=tx_language,
            api_key=llm_api_key,
        )

    pipeline = WalkerRetrievalPipeline(
        corpus=corpus,
        graph=graph,
        seed_retriever=dense,
        reranker=reranker,
        rerank_pool=args.rerank_pool,
        okg_expand_seed=args.okg_expand_seed,
        topic_extractor=topic_extractor,
    )

    # 2. Build generation workflow on top of the retrieval pipeline.
    workflow = RefWalkWorkflow(
        pipeline=pipeline,
        context_top_k=args.context_top_k,
        eval_top_k=args.eval_top_k,
        passage_limit=args.passage_limit,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_api_key=llm_api_key,
        profile_name=args.profile,
        max_tokens=args.max_tokens,
        language=args.language,
        oracle_total_k=args.oracle_total_k,
        oracle_seed=args.oracle_distractor_seed,
        condition_anchor=not args.no_condition_anchor,
    )
    print(
        f"[refwalk] corpus={len(corpus.node_ids)} okg=|V|={graph.number_of_nodes()} "
        f"|E|={graph.number_of_edges()} variant={variant} profile={args.profile} "
        f"topic_extractor={'on' if topic_extractor is not None else 'off'} "
        f"mode={'oracle-gt' if args.oracle else 'retrieval'}"
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
        f"[refwalk] bench total={len(bench)} done={len(done)} "
        f"pending={len(pending)}"
    )
    if not pending:
        print("[refwalk] nothing to do.")
        return

    # 4. Inference loop. Flush per row so Ctrl-C leaves a usable file.
    t0 = time.time()
    n_ok = n_err = 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for i, qa in enumerate(pending, 1):
            qa_id = qa.get("qa_id")
            t_start = time.time()
            result = workflow.answer_gt(qa) if args.oracle else workflow.answer(qa)
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
                f"  [{i:>3}/{len(pending)}] {str(qa_id):<20s} "
                f"status={result.status:<5s} "
                f"hits={len(result.retrieved_node_ids)} "
                f"refs={len(result.cited_references)} "
                f"t={elapsed:.1f}s"
            )

    dt = time.time() - t0
    print(
        f"[refwalk] done. ok={n_ok} err={n_err} "
        f"total_time={dt:.1f}s -> {out_path}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
