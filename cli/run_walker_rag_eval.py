"""Run NativeRAG-style generation on top of Walker retrieval.

This is the "Ours retrieval + NativeRAG generation" baseline. The
prediction file matches ``cli/run_native_rag_eval.py`` /
``cli/run_lightrag_eval.py`` schema, so ``cli/run_generation_eval.py``
scores it without changes — and the resulting per-difficulty cells slot
into the same RAG comparison table.

Default corpus / OKG match the "Ours" retrieval setup
(``data/RegOps-Bench/articles.jsonl`` + ``data/okg/okg.gpickle``),
not the bench's compiled-aug set, so the retrieval side reproduces the
numbers in ``experiments/per_difficulty_canonical.md`` §3.

Usage::

    # default: regops_bench.jsonl with online topic anchoring
    python cli/run_walker_rag_eval.py \\
        --bench data/RegOps-Bench/regops_bench.jsonl \\
        --out experiments/walker_rag/preds_4b.jsonl \\
        --llm-base-url http://localhost:8040/v1 --llm-model Qwen/Qwen3.5-4B

    # use the pre-anchored FAQ to skip the per-query topic-extraction LLM call
    python cli/run_walker_rag_eval.py \\
        --bench data/anchoring_test_faq.json \\
        --out experiments/walker_rag/preds_4b.jsonl

    # 35B
    python cli/run_walker_rag_eval.py \\
        --out experiments/walker_rag/preds_35b.jsonl \\
        --llm-base-url http://localhost:8035/v1 --llm-model Qwen/Qwen3.6-35B-A3B-FP8

    # score
    python cli/run_generation_eval.py \\
        --runs experiments/walker_rag/preds_4b.jsonl \\
        --faq data/RegOps-Bench/regops_bench.jsonl \\
        --out experiments/walker_rag/scores_4b
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines.walker_rag import WalkerRagWorkflow
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
    """Match the schema written by run_native_rag_eval / run_lightrag_eval
    so the downstream generation_eval scorer treats this baseline
    identically."""
    return {
        "qa_id": qa.get("qa_id"),
        "variant": variant,
        # Canonical difficulty first; ``target_difficulty`` is an
        # LLM-time intent tag and disagrees on 116/250 rows.
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", default="data/RegOps-Bench/regops_bench.jsonl",
                   help="FAQ source. Pre-anchored ``data/anchoring_test_faq.json`` "
                        "skips the per-query topic-extraction LLM call.")
    p.add_argument("--articles", default="data/RegOps-Bench/articles.jsonl",
                   help="Parse-stage articles for corpus + dense.")
    p.add_argument("--okg", default="data/okg/okg.gpickle",
                   help="OKG gpickle for 1-hop expansion.")
    p.add_argument(
        "--embed-cache",
        default="data/okg/doc_embeddings_parse_articles.npy",
        help="Dense embedding cache for the article corpus.",
    )
    p.add_argument("--out", default="experiments/walker_rag/preds.jsonl")

    # Walker tunables — defaults match WalkerRetrievalPipeline (Round-5 default:
    # pool=50, 3-view RRM rerank). Earlier code used pool=30; the Round-5
    # multi-view RRM analysis showed pool=50 widens the rerank pool just enough
    # for the wide-view to recover L3 specialists without diluting L1/L2.
    p.add_argument("--rerank-pool", type=int, default=50,
                   help="Seed candidates fetched before OKG expansion + rerank.")
    p.add_argument("--okg-expand-seed", type=int, default=10,
                   help="Top seeds that drive 1-hop OKG expansion.")
    p.add_argument("--context-top-k", type=int, default=10,
                   help="Reranked passages fed to the LLM. Standardized to 10 "
                        "across baselines for fair comparison.")
    p.add_argument(
        "--no-topic-extractor",
        action="store_true",
        help="Disable online topic anchoring. Use only when --bench is "
             "already pre-anchored (e.g. data/anchoring_test_faq.json).",
    )

    # Reranker.
    p.add_argument("--rerank-base-url", default="http://localhost:8095")
    p.add_argument("--rerank-model", default="Qwen/Qwen3-Reranker-0.6B")
    p.add_argument("--rerank-batch-size", type=int, default=64)

    # LLM.
    p.add_argument("--llm-base-url", default="http://localhost:8040/v1",
                   help="OpenAI-compatible vLLM endpoint for the answer LLM.")
    p.add_argument("--llm-model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--profile", default="exact",
                   help="Sampling profile from src.utils.hparams.Qwen3_5_HParams.")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--passage-max-chars", type=int, default=2400)
    p.add_argument(
        "--language", default="ko", choices=["ko", "en"],
        help="Prompt/answer language. 'ko' is the RegOps default.",
    )

    # Variant + iteration.
    p.add_argument(
        "--variant", default=None,
        help="Override variant name in preds.jsonl. Default: walker_rag_<llm-suffix>.",
    )
    p.add_argument("--limit", type=int, default=0,
                   help="Run on at most N benchmark rows (0 = all).")
    p.add_argument("--difficulty", default=None,
                   help="Optional comma list filter, e.g. 'L1' or 'L1,L2'.")
    p.add_argument("--no-resume", action="store_true",
                   help="Overwrite output instead of appending / skipping done.")
    return p.parse_args(argv)


# ─── Main ──────────────────────────────────────────────────────────


def _default_variant(args: argparse.Namespace) -> str:
    """Match the naming pattern of other baselines (e.g. native_rag_4b)."""
    suffix = ""
    m = (args.llm_model or "").lower()
    if "35b" in m:
        suffix = "_35b"
    elif "4b" in m:
        suffix = "_4b"
    return f"walker_rag{suffix}"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.no_resume and out_path.exists():
        out_path.unlink()

    variant = args.variant or _default_variant(args)

    # 1. Build retrieval stack (corpus → dense → graph → reranker → walker pipeline).
    print(f"[walker-rag] loading corpus from {args.articles}")
    corpus = load_corpus(Path(args.articles), source="parse")
    embed_cache = Path(args.embed_cache) if args.embed_cache else None
    if embed_cache and embed_cache.exists():
        dense = DenseRetriever.from_cache(corpus, embed_cache)
    else:
        dense = DenseRetriever.build(corpus)
        if embed_cache:
            embed_cache.parent.mkdir(parents=True, exist_ok=True)
            dense.save_cache(embed_cache)

    print(f"[walker-rag] loading OKG from {args.okg}")
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
        # TopicExtractor is lazy — no LLM client until first raw-string
        # query lands. Reuse the answer LLM endpoint for the extractor.
        # ``language`` here is the prompt-template language (full word),
        # not the workflow short code.
        tx_lang = "Korean" if args.language == "ko" else "English"
        tx_domain = (
            "Korean R&D funding regulations" if args.language == "ko"
            else "regulations"
        )
        topic_extractor = TopicExtractor(
            base_url=args.llm_base_url,
            model_id=args.llm_model,
            profile_name=args.profile,
            domain=tx_domain,
            language=tx_lang,
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
    workflow = WalkerRagWorkflow(
        pipeline=pipeline,
        context_top_k=args.context_top_k,
        passage_max_chars=args.passage_max_chars,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        profile_name=args.profile,
        max_tokens=args.max_tokens,
        language=args.language,
    )
    print(
        f"[walker-rag] corpus={len(corpus.node_ids)} okg=|V|={graph.number_of_nodes()} "
        f"|E|={graph.number_of_edges()} variant={variant} language={args.language} "
        f"topic_extractor={'on' if topic_extractor is not None else 'off'}"
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
        f"[walker-rag] bench total={len(bench)} done={len(done)} "
        f"pending={len(pending)}"
    )
    if not pending:
        print("[walker-rag] nothing to do.")
        return

    # 4. Inference loop. Flush per row so Ctrl-C leaves a usable file.
    t0 = time.time()
    n_ok = n_err = 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for i, qa in enumerate(pending, 1):
            qa_id = qa.get("qa_id")
            t_start = time.time()
            result = workflow.answer(qa)
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
        f"[walker-rag] done. ok={n_ok} err={n_err} "
        f"total_time={dt:.1f}s -> {out_path}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
