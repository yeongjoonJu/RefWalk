"""Retrieval evaluation — unified.

Evaluates all retrieval baselines on ``data/FAQ_examples.json`` at both
node and article granularity, and reports two metric families side-by-side:

  Retrieval:  R@5, R@10, nDCG@10, FullCov@10
  Citation:   CP, CR, CFP (top-k retrieved treated as the "cited" set)

Baselines:
  B1_BM25, B2_Dense, B3_Hybrid,
  B5_Reranker_legacy (35B-prompted, opt-in via --include-legacy-reranker),
  B5_Reranker_neo    (hybrid → Qwen3-VL-Reranker-2B; no graph expansion),
  Walker_Retrieval   (hybrid + 1-hop OKG expand + Qwen3-VL-Reranker-2B).

Dropped:
  - B4_OKG_Aware  (decay-only expansion w/o reranker; neighbours never
    scored against the query — empirically near-equivalent to B3 with
    1-3 noise promotions).
  - B6_Combined_neo (Reranker(OKG_Aware)) — structurally identical to
    Walker_Retrieval but with a narrower candidate pool that the
    decay-cap arbitrarily truncates. Walker_Retrieval supersedes it.

Outputs to ``experiments/B5_1_retrieval/``:
  - node_id_results.tsv        / article_id_results.tsv
                               (renamed from node_level / article_level —
                                names now refer to the ID-matching unit,
                                not the corpus structure)
  - per_difficulty_node_id.tsv / per_difficulty_article_id.tsv
  - seed_query_traces.jsonl    — per-query retrieved ids + metric dicts

The ``--top-k-probe`` value is now also surfaced as ``R@<top_k_probe>``
(plus ``nDCG@<top_k_probe>`` / ``FullCov@<top_k_probe>``) in the
output tables, in addition to the always-on R@5 and R@10 baselines.
Previously ``--top-k-probe`` only sized the retriever fetch and was
silently capped at 10 by the metric layer.
  - summary.md                 — human-readable report (this script does
                                 NOT regenerate summary.md; edit it manually
                                 when you want to freeze prose).

Difficulty labels are read from ``data/okg/faq_validation.json``.

Usage:
    python scripts/run_retrieval_eval.py \\
        [--include-legacy-reranker] [--only B1_BM25,Walker_Retrieval] \\
        [--hit-rule strict|hierarchical]   # hierarchical fixes the
                                           # leaf-only corpus artifact
                                           # in --use-indexed-text mode
        [--corpus-source okg_nodes|parse]  # parse = one entry per full
                                           # article (조), with sub-paragraphs
                                           # / items already concatenated
                                           # into text_ko (use parse-stage
                                           # data/parsed/articles_*.jsonl)

Corpus-source matrix:

   okg_nodes (default)   one entry per node (조 + 항 + 호); finest split.
                         pair with okg_nodes-style jsonl such as
                         data/okg/okg_nodes.jsonl.

   parse                 one entry per full article (조). Each row's
                         text_ko already contains all sub-paragraphs/items
                         concatenated. Pair with parse-stage jsonl such
                         as data/RegOps-Bench/articles.jsonl.

   parse + gt at 항/호    use --hit-rule hierarchical so the 조-level
                         retrieval credits sub-clause gt via ancestor.
"""

from __future__ import annotations

import argparse
import json, os
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.utils.io import load_jsonl
from src.baselines.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    LLMReranker,
    Qwen3VLRerankerRetriever,
    load_corpus,
)
from src.eval.metrics import (
    aggregate_all_metrics,
    compute_all_metrics,
)
from src.retrieval.dedup import dedup_by_article
from src.retrieval.reranker import Qwen3Reranker
from src.retrieval.walker_retrieval import TopicExtractor, WalkerRetrievalPipeline


SEED = 42
DEFAULT_EXPERIMENT_DIR = REPO / "experiments" / "retrieval_indexed_dedup_node_hier"

CITATION_COLUMNS = ("CP", "CR", "CFP")


def _ks_for(top_k_probe: int) -> tuple[int, ...]:
    """k values to report. Always include 5 and 10 for cross-paper
    comparability; also include ``top_k_probe`` so the user's chosen
    eval cap is actually measured (previously it was silently capped
    at 10, dropping any retriever output beyond rank 10)."""
    return tuple(sorted({5, 10, max(1, int(top_k_probe))}))


def _retrieval_columns(ks: tuple[int, ...]) -> tuple[str, ...]:
    k_top = ks[-1]
    return tuple(f"R@{k}" for k in ks) + (f"nDCG@{k_top}", f"FullCov@{k_top}")


def _load_faq(faq_path: Path, faq_val_path: Path | None) -> list[dict]:
    """Load FAQ and resolve a difficulty per row.

    Source-of-truth precedence (highest first):
      1. ``difficulty`` already on the FAQ row (the official benchmark
         file owns the label — never overwrite it).
      2. ``per_qa[*].difficulty`` from ``--faq-val`` (compiled artifact;
         used only as a *fallback* for rows missing a label).
      3. ``"?"`` if neither source has it.

    Historically this function blindly overwrote (1) with (2), which
    silently drifts whenever the auto-relabeled validation file lags
    the canonical FAQ. Per-difficulty numbers across baselines then
    bucket into different L1-L4 populations and become non-comparable
    even though the per-query retrieval results are identical.
    """
    if faq_path.suffix == '.jsonl':
        faq = load_jsonl(faq_path)
    else:
        faq = json.loads(faq_path.read_text(encoding="utf-8"))

    diff_map: dict[str, str] = {}
    if faq_val_path and faq_val_path.exists():
        val = json.loads(faq_val_path.read_text(encoding="utf-8"))
        diff_map = {q["qa_id"]: q.get("difficulty", "?") for q in val.get("per_qa", [])}

    n_from_faq = n_from_val = n_missing = 0
    for q in faq:
        own = q.get("difficulty")
        if own and own != "?":
            n_from_faq += 1
            continue
        fallback = diff_map.get(q.get("qa_id"))
        if fallback and fallback != "?":
            q["difficulty"] = fallback
            n_from_val += 1
        else:
            q["difficulty"] = "?"
            n_missing += 1
    print(
        f"[load_faq] difficulty: {n_from_faq} from FAQ, "
        f"{n_from_val} from --faq-val fallback, {n_missing} unresolved"
    )
    return faq


def _run_one(
    name: str,
    search_fn,
    faq: list[dict],
    corpus_ids: set[str],
    top_k_probe: int,
    dedup: bool = False,
    raw_k: int = 20,
    hit_rule: str = "strict",
) -> tuple[dict, dict, list[dict]]:
    """Return (node_agg, article_agg, per_query_trace).

    When ``dedup`` is True, each retriever is called with ``top_k=raw_k``
    and the ranked output is collapsed to at most ``top_k_probe`` unique
    조-ancestors before metrics are computed.

    ``hit_rule`` is forwarded to ``compute_all_metrics`` (see
    src/eval/metrics.py): "strict" (default) requires exact id equality;
    "hierarchical" credits a hit when the retrieved id is an ancestor
    or descendant of a gold id along the PART_OF chain. The latter
    fixes the leaf-only corpus measurement gap when ``--use-indexed-text``
    is on but ``gt_references`` contain non-leaf 조 ids.
    """
    node_per_q: list[dict] = []
    art_per_q: list[dict] = []
    traces: list[dict] = []
    start = time.time()

    fetch_k = raw_k if dedup else top_k_probe

    for qa in tqdm(faq):
        gold = qa.get("gt_references", [])
        if not gold:
            continue
        # search_fn signature: ``(qa: dict, top_k: int) -> list[(id, score)]``.
        # Legacy retrievers (BM25/Dense/Hybrid/Reranker wrappers) only
        # need the question string and are wrapped via ``_legacy_search``;
        # WalkerRetrievalPipeline consumes the full dict to access
        # topic/actor/temporal/… for the validated tagged seed query.
        t_q0 = time.monotonic()
        try:
            hits = search_fn(qa, top_k=fetch_k)
        except Exception as e:
            print(f"  [{name}] {qa.get('qa_id')} failed: {e}")
            continue
        latency_sec = round(time.monotonic() - t_q0, 3)
        # Normalise to (node_id, score) tuples.
        normed: list[tuple[str, float]] = []
        for h in hits:
            if isinstance(h, tuple):
                normed.append((h[0], float(h[1]) if len(h) > 1 else 0.0))
            else:
                normed.append((h, 0.0))
        if dedup:
            normed = dedup_by_article(normed, k=top_k_probe)
        retrieved = [nid for nid, _ in normed]

        node_m = compute_all_metrics(
            retrieved, gold,
            corpus_node_ids=corpus_ids,
            match_unit="node_id",
            ks=_ks_for(top_k_probe),
            hit_rule=hit_rule,
        )
        art_m = compute_all_metrics(
            retrieved, gold,
            corpus_node_ids=corpus_ids,
            match_unit="article_id",
            ks=_ks_for(top_k_probe),
            hit_rule=hit_rule,
        )
        node_per_q.append(node_m)
        art_per_q.append(art_m)
        traces.append(
            {
                "qa_id": qa.get("qa_id"),
                "difficulty": qa.get("difficulty"),
                "category": qa.get("category"),
                "baseline": name,
                "gold": gold,
                "retrieved": retrieved,
                "node_metrics": node_m,
                "article_metrics": art_m,
                "latency_sec": latency_sec,
            }
        )

    elapsed = time.time() - start
    node_agg = aggregate_all_metrics(node_per_q)
    art_agg = aggregate_all_metrics(art_per_q)
    node_agg["seconds"] = round(elapsed, 1)
    art_agg["seconds"] = round(elapsed, 1)
    return node_agg, art_agg, traces


def _format_row(baseline: str, agg: dict, retrieval_cols: tuple[str, ...]) -> dict:
    row = {"Baseline": baseline}
    for m in retrieval_cols:
        v = agg.get(f"retrieval_{m}")
        row[m] = f"{v:.4f}" if isinstance(v, (int, float)) else "-"
    for m in CITATION_COLUMNS:
        v = agg.get(f"citation_{m}")
        row[m] = f"{v:.4f}" if isinstance(v, (int, float)) else "-"
    row["n_queries"] = agg.get("n_queries", 0)
    row["seconds"] = agg.get("seconds", 0)
    return row


def _write_tsv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, "")) for c in columns) + "\n")


def _per_difficulty(
    traces: list[dict], key: str
) -> dict[str, dict[str, dict]]:
    buckets: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for t in traces:
        if t[key].get("retrieval_n_gold", 0) == 0:
            continue
        buckets[t["baseline"]][t.get("difficulty") or "?"].append(t[key])
    out: dict[str, dict[str, dict]] = {}
    for b, by_lv in buckets.items():
        out[b] = {lv: aggregate_all_metrics(items) for lv, items in by_lv.items()}
    return out


def _write_difficulty_tsv(
    path: Path,
    by_baseline: dict[str, dict[str, dict]],
    retrieval_cols: tuple[str, ...],
) -> None:
    levels = ["L1", "L2", "L3", "L4"]
    metric_keys = [*retrieval_cols, "CP", "CR", "CFP"]
    columns = ["Baseline"]
    for lv in levels:
        for m in metric_keys:
            columns.append(f"{lv}_{m}")
        columns.append(f"{lv}_n")
    rows = []
    for b, by_lv in by_baseline.items():
        r = {"Baseline": b}
        for lv in levels:
            agg = by_lv.get(lv, {})
            for m in metric_keys:
                key = (f"retrieval_{m}" if m in retrieval_cols
                       else f"citation_{m}")
                v = agg.get(key)
                r[f"{lv}_{m}"] = f"{v:.4f}" if isinstance(v, (int, float)) else "-"
            r[f"{lv}_n"] = agg.get("n_queries", 0)
        rows.append(r)
    _write_tsv(path, rows, columns)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faq", type=Path, default=Path("data/RegOps-Bench/regops_bench.jsonl"))
    ap.add_argument(
        "--faq-val", type=Path,
        default=Path("data/okg/faq_validation.json"),
    )
    ap.add_argument(
        "--articles", type=Path,
        default=Path("data/okg/okg_nodes.jsonl"),
    )
    ap.add_argument(
        "--embed-cache", type=Path,
        default=Path("data/okg/doc_embeddings_okg_nodes.npy"),
    )
    ap.add_argument("--okg", type=Path, default=Path("data/okg/okg.gpickle"))
    ap.add_argument("--top-k-probe", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--rerank-pool", type=int, default=0,
                    help="WalkerRetrievalPipeline rerank_pool. 0 means top_k_probe*2 "
                         "(legacy default); set 50 to match the paper config.")
    ap.add_argument("--okg-expand-seed", type=int, default=0,
                    help="WalkerRetrievalPipeline okg_expand_seed. 0 means top_k_probe.")
    ap.add_argument("--topic-llm-base-url", default=None,
                    help="Override the LLM endpoint used by Walker's online "
                         "TopicExtractor (defaults to localhost:8035).")
    ap.add_argument("--topic-llm-model", default=None,
                    help="Override the model id passed to TopicExtractor.")
    ap.add_argument("--only", type=str, default="", help="comma list of baselines to run")
    ap.add_argument(
        "--out", type=Path, default=DEFAULT_EXPERIMENT_DIR,
        help="Output directory for TSVs + summary + traces.",
    )
    ap.add_argument(
        "--corpus-source", choices=["okg_nodes", "parse"], default="okg_nodes",
        help=(
            "How to build the retrieval corpus. "
            "okg_nodes (default): one entry per OKG node (조/항/호), pulled "
            "from --articles which must be an okg_nodes-style jsonl. "
            "parse: one entry per full article (조), with sub-paragraphs/"
            "items already concatenated into text_ko. Use with parse-stage "
            "files like data/parsed/articles_*.jsonl when you want "
            "article-unit retrieval. NOTE: with gt at 항/호 level, use "
            "--hit-rule hierarchical so 조 retrieval credits sub-clause gt."
        ),
    )
    ap.add_argument(
        "--use-indexed-text", action="store_true",
        help=(
            "Load OKG nodes' ancestor-prefixed indexed_text. "
            "Only valid with --corpus-source=okg_nodes."
        ),
    )
    ap.add_argument(
        "--dedup-by-article", action="store_true",
        help="Collapse retrieval hits to at most one candidate per 조-ancestor.",
    )
    ap.add_argument(
        "--raw-k", type=int, default=20,
        help="Raw candidates to request before dedup (only used when --dedup-by-article).",
    )
    ap.add_argument(
        "--hit-rule", choices=["strict", "hierarchical"], default="strict",
        help=(
            "How a retrieved id is judged 'hit' against gt_references. "
            "strict (default): exact id match. hierarchical: a hit when "
            "retrieved id is an ancestor/descendant of gt id along the "
            "PART_OF chain. Use hierarchical with --use-indexed-text so "
            "leaf-only corpus can still credit hits when gt is a 조 id."
        ),
    )
    args = ap.parse_args()

    if args.use_indexed_text and args.corpus_source != "okg_nodes":
        ap.error(
            "--use-indexed-text requires --corpus-source=okg_nodes "
            "(parse-stage files don't carry indexed_text metadata)."
        )

    random.seed(SEED)
    try:
        import numpy as np
        np.random.seed(SEED)
    except ImportError:
        pass

    experiment_dir = args.out if args.out.is_absolute() else (REPO / args.out).resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading FAQ: {args.faq}")
    faq = _load_faq(args.faq, args.faq_val)
    print(f"  {len(faq)} queries  (difficulty: "
          f"{dict((lv, sum(1 for q in faq if q['difficulty']==lv)) for lv in ['L1','L2','L3','L4'])})")

    print(f"Loading corpus (source={args.corpus_source}) + embeddings …")
    corpus = load_corpus(
        args.articles, source=args.corpus_source,
        use_indexed_text=args.use_indexed_text,
    )

    bm25 = BM25Retriever(corpus)

    if args.embed_cache is None or not os.path.exists(args.embed_cache):
        dense = DenseRetriever.build(corpus)
        dense.save_cache(args.embed_cache)
    else:
        dense = DenseRetriever.from_cache(corpus, args.embed_cache)
    hybrid = HybridRetriever(bm25, dense, alpha=args.alpha)
    graph = pickle.load(open(args.okg, "rb"))
    print(f"  corpus={len(corpus.node_ids)} nodes, graph={graph.number_of_nodes()} nodes")

    corpus_ids = set(corpus.node_ids)

    b4 = Qwen3VLRerankerRetriever(dense, corpus)
    b5 = Qwen3VLRerankerRetriever(hybrid, corpus)

    # WalkerRetrievalPipeline auto-constructs a default TopicExtractor
    # for raw-string queries. The eval CLI feeds pre-anchored FAQ rows
    # (dicts with topic + actor/temporal/magnitude/situational), so the
    # extractor never fires here — but keeping it on by default lets
    # the same pipeline serve online raw-string traffic without changes.
    topic_extractor_kwargs = {}
    if args.topic_llm_base_url:
        topic_extractor_kwargs["base_url"] = args.topic_llm_base_url
    if args.topic_llm_model:
        topic_extractor_kwargs["model_id"] = args.topic_llm_model
    topic_extractor = (
        TopicExtractor(**topic_extractor_kwargs)
        if topic_extractor_kwargs else TopicExtractor()
    )
    walker_pipeline = WalkerRetrievalPipeline(
        corpus=corpus, seed_retriever=dense, graph=graph,
        reranker=Qwen3Reranker(batch_size=64),
        rerank_pool=args.rerank_pool or args.top_k_probe * 2,
        okg_expand_seed=args.okg_expand_seed or args.top_k_probe,
        topic_extractor=topic_extractor,
    )

    def _legacy_search(retriever_search):
        """Wrap a ``(query: str, top_k)`` retriever to accept a qa dict."""
        def fn(qa, top_k):
            q = qa["question"] if isinstance(qa, dict) else qa
            return retriever_search(q, top_k=top_k)
        return fn

    def _walker_search(qa, top_k):
        # Walker consumes the full qa dict to build the validated tagged
        # seed query (`[TOPIC] … [Q] … [ACTOR] … [TEMPORAL] …`).
        hits = walker_pipeline.retrieve(qa, top_k=top_k)
        return [(h.node_id, h.rerank_score) for h in hits]

    plan: list[tuple[str, callable]] = [
        ("B1_BM25", _legacy_search(bm25.search)),
        ("B2_Dense", _legacy_search(dense.search)),
        ("B3_Hybrid", _legacy_search(hybrid.search)),
        ("B4_Dense_Reranker", _legacy_search(b4.search)),
        ("B5_Hybrid_Reranker", _legacy_search(b5.search)),
        ("Walker_Retrieval", _walker_search),
    ]

    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        plan = [p for p in plan if p[0] in wanted]
        print(f"Filtered to: {[p[0] for p in plan]}")

    ks = _ks_for(args.top_k_probe)
    retrieval_cols = _retrieval_columns(ks)

    node_rows, article_rows = [], []
    all_traces: list[dict] = []

    for name, fn in plan:
        print(f"\n=== {name} ===")
        node_agg, art_agg, traces = _run_one(
            name, fn, faq, corpus_ids, args.top_k_probe,
            dedup=args.dedup_by_article,
            raw_k=args.raw_k,
            hit_rule=args.hit_rule,
        )
        for label, agg in (("node_id", node_agg), ("article_id", art_agg)):
            brief = {k: round(v, 4) for k, v in agg.items()
                     if isinstance(v, (int, float))}
            print(f"  {label:<10}: {brief}")
        node_rows.append(_format_row(name, node_agg, retrieval_cols))
        article_rows.append(_format_row(name, art_agg, retrieval_cols))
        all_traces.extend(traces)

    # ── Write tables ──
    cols = ["Baseline", *retrieval_cols, *CITATION_COLUMNS,
            "n_queries", "seconds"]
    _write_tsv(experiment_dir / "node_id_results.tsv", node_rows, cols)
    _write_tsv(experiment_dir / "article_id_results.tsv", article_rows, cols)

    _write_difficulty_tsv(
        experiment_dir / "per_difficulty_node_id.tsv",
        _per_difficulty(all_traces, "node_metrics"),
        retrieval_cols,
    )
    _write_difficulty_tsv(
        experiment_dir / "per_difficulty_article_id.tsv",
        _per_difficulty(all_traces, "article_metrics"),
        retrieval_cols,
    )

    with open(
        experiment_dir / "seed_query_traces.jsonl", "w", encoding="utf-8"
    ) as f:
        for t in all_traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print()
    print(f"Results written to {experiment_dir}")
    for name in ("node_id_results.tsv", "article_id_results.tsv",
                 "per_difficulty_node_id.tsv", "per_difficulty_article_id.tsv",
                 "seed_query_traces.jsonl"):
        print(f" - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
