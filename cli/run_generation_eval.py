"""Generation-metric pipeline runner (Phase C).

Reads a ``runs.jsonl`` produced by the smoke / grounding pipeline and
an FAQ JSON with GT answers & citations, then for every run computes:

  1. Claim P/R/F1 (pred claims decomposed → matched to GT claims)
  2. Citation P/R/F1 (article / exact / article_or_desc rollups)
  3. Fallback detector flags (citation_class, caveat, degraded_stop)

Per-run records are written as ``<out>/generation_eval.jsonl`` and a
summary (per-variant × per-difficulty) as ``<out>/summary.json``.

Usage::

    python scripts/run_generation_eval.py \
        --runs data/reports/smoke_oks_walk_v7/runs.jsonl \
        --faq data/RegOps-Bench/regops_bench.jsonl \
        --out data/reports/smoke_oks_walk_v7_eval \
        --base-url http://localhost:8035/v1 \
        --matcher-base-url http://localhost:8038/v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import load_jsonl
from src.eval.citation_metrics import score_citations
from src.eval.claim_decomposer import ClaimDecomposer
from src.eval.claim_matcher import ClaimMatcher
from src.eval.fallback_detector import aggregate_fallback, detect_fallback


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _index_faq(faq_path: Path) -> dict[str, dict]:
    if faq_path.suffix=='.json':
        with open(faq_path, encoding="utf-8") as f:
            faq = json.load(f)
    else:
        faq = load_jsonl(faq_path)
    return {q["qa_id"]: q for q in faq}


def _run_answer(row: dict) -> str:
    """Prefer full `answer`, fall back to preview."""
    return row.get("answer") or row.get("answer_preview") or ""


def _mean(xs: list[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0


def _claim_eligible(r: dict) -> bool:
    """Rows where the claim metric is structurally defined (n_gt > 0).

    When the GT decomposer returns 0 claims (often on very short
    one-sentence answers) the matcher returns precision=recall=0, which
    poisons the mean. Those rows are excluded from claim averages and
    counted separately as ``n_skipped_no_gt``.
    """
    return int(r.get("claim", {}).get("n_gt", 0) or 0) > 0


def _per_bucket(
    per_run: list[dict], key: str
) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = {}
    for r in per_run:
        buckets.setdefault(r[key], []).append(r)

    out: dict[str, dict] = {}
    for k, rows in buckets.items():
        eligible = [r for r in rows if _claim_eligible(r)]
        out[k] = {
            "n": len(rows),
            "n_claim_eligible": len(eligible),
            "n_skipped_no_gt": len(rows) - len(eligible),
            "claim_f1_mu": round(_mean([r["claim"]["f1"] for r in eligible]), 4),
            "claim_p_mu": round(_mean([r["claim"]["precision"] for r in eligible]), 4),
            "claim_r_mu": round(_mean([r["claim"]["recall"] for r in eligible]), 4),
            "citation_f1_mu": round(
                _mean([r["citation"]["article"]["f1"] for r in rows]), 4
            ),
            "citation_p_mu": round(
                _mean([r["citation"]["article"]["precision"] for r in rows]), 4
            ),
            "citation_r_mu": round(
                _mean([r["citation"]["article"]["recall"] for r in rows]), 4
            ),
            "citation_exact_f1_mu": round(
                _mean([r["citation"]["exact"]["f1"] for r in rows]), 4
            ),
            "citation_exact_p_mu": round(
                _mean([r["citation"]["exact"]["precision"] for r in rows]), 4
            ),
            "citation_exact_r_mu": round(
                _mean([r["citation"]["exact"]["recall"] for r in rows]), 4
            ),
            "citation_cfp_mu": round(
                _mean([r["citation"]["article"]["cfp"] for r in rows]), 4
            ),
            "citation_context_cfp_mu": round(
                _mean([r["citation"]["article"]["context_cfp"] for r in rows]), 4
            ),
            "citation_exact_cfp_mu": round(
                _mean([r["citation"]["exact"]["cfp"] for r in rows]), 4
            ),
            "citation_exact_context_cfp_mu": round(
                _mean([r["citation"]["exact"]["context_cfp"] for r in rows]), 4
            ),
            "fallback": aggregate_fallback([r["fallback"] for r in rows]),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--faq", type=Path, default=REPO_ROOT / "data/RegOps-Bench/regops_bench.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", default="http://localhost:8035/v1",
                    help="vLLM endpoint for decomposer.")
    ap.add_argument("--matcher-base-url", default=None,
                    help="vLLM endpoint for matcher (can differ from decomposer "
                         "for cross-model bias reduction). Defaults to --base-url.")
    ap.add_argument("--model-id", default="Qwen/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--limit", type=int, default=0,
                    help="Debug: process only the first N runs.")
    ap.add_argument("--only-ok", action="store_true",
                    help="Skip runs with status=error.")
    ap.add_argument(
        "--language",
        default="ko",
        choices=["ko", "en"],
        help="Eval language. 'ko' = RegOps (Korean R&D regs), "
             "'en' scores an English corpus. Drives decomposer/matcher "
             "system prompts and authority classification.",
    )
    ap.add_argument(
        "--articles", type=Path, default=None,
        help="Optional articles jsonl for corpus universe (enables citation "
             "CFP). When supplied, score_citations adds two hallucination "
             "metrics: 'cfp' (cited − corpus) and 'context_cfp' (cited − "
             "retrieved). Each row in --runs should also carry "
             "'retrieved_node_ids' for context_cfp to be meaningful.",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(args.runs)
    if args.only_ok:
        rows = [r for r in rows if r.get("status") == "ok"]
    if args.limit > 0:
        rows = rows[: args.limit]

    faq = _index_faq(args.faq)

    decomposer = ClaimDecomposer(
        base_url=args.base_url, model_id=args.model_id, language=args.language,
    )
    matcher = ClaimMatcher(
        base_url=args.matcher_base_url or args.base_url,
        model_id=args.model_id,
        language=args.language,
    )

    print(f"runs: {len(rows)} | out: {args.out} | language: {args.language}")
    print(
        f"decomposer: {args.base_url} | "
        f"matcher: {args.matcher_base_url or args.base_url}"
    )

    # Optional corpus universe for CFP (citation hallucination) metric.
    corpus_node_ids: list[str] | None = None
    if args.articles is not None:
        from src.retrieval.corpus import load_corpus
        corpus_node_ids = list(load_corpus(args.articles, source="parse").node_ids)
        print(f"corpus: {len(corpus_node_ids)} nodes loaded (CFP enabled)")

    eval_path = args.out / "generation_eval.jsonl"
    eval_path.write_text("", encoding="utf-8")

    # Pre-decompose GT once per qa_id (shared across variants).
    print("\n[1/3] decomposing GT answers …")
    gt_claims_by_qa: dict[str, list[dict]] = {}
    gt_refs_by_qa: dict[str, list[str]] = {}
    for qa_id, qa in tqdm(faq.items()):
        if "answer" not in qa:
            continue
        gt_claims_by_qa[qa_id] = decomposer.decompose(
            qa["answer"], answer_id=f"gt:{qa_id}"
        )
        gt_refs_by_qa[qa_id] = list(qa.get("gt_references", []) or [])
    print(f"  decomposed {len(gt_claims_by_qa)} GT answers; "
          f"stats={decomposer.stats}")

    print("\n[2/3] decomposing + matching predictions …")
    per_run: list[dict] = []
    t0 = time.time()
    for i, row in enumerate(rows):
        qa_id = row.get("qa_id")
        variant = row.get("variant", "?")
        difficulty = row.get("difficulty", "?")
        status = row.get("status", "?")
        if qa_id not in gt_claims_by_qa:
            print(f"  [{i+1}/{len(rows)}] {qa_id} — skip (not in FAQ)")
            continue

        pred_text = _run_answer(row)
        gt_claims = gt_claims_by_qa[qa_id]
        if status == "error" or not pred_text:
            pred_claims: list[dict] = []
            match = matcher.match([], gt_claims)
        else:
            pred_claims = decomposer.decompose(
                pred_text, answer_id=f"pred:{qa_id}:{variant}"
            )
            match = matcher.match(pred_claims, gt_claims)

        citation = score_citations(
            predicted=row.get("cited_references") or row.get("cited_node_ids") or [],
            gt=gt_refs_by_qa.get(qa_id, []),
            corpus_node_ids=corpus_node_ids,
            retrieved_node_ids=row.get("retrieved_node_ids") or [],
        )
        fallback = detect_fallback(row, language=args.language)

        rec = {
            "qa_id": qa_id,
            "variant": variant,
            "difficulty": difficulty,
            "status": status,
            "pred_claims": pred_claims,
            "gt_claims_ids": [c["id"] for c in gt_claims],
            "claim": {
                "precision": match["precision"],
                "recall": match["recall"],
                "f1": match["f1"],
                "n_pred": match["n_pred"],
                "n_gt": match["n_gt"],
                "n_match": match["n_match"],
                "n_partial": match["n_partial"],
                "pairs": match["pairs"],
            },
            "citation": citation,
            "fallback": fallback,
        }
        per_run.append(rec)
        with open(eval_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        elapsed = time.time() - t0
        print(
            f"  [{i+1:>3}/{len(rows)}] [{difficulty}] {qa_id:<28s} {variant:<32s} "
            f"claim_f1={match['f1']:.2f} cite_f1={citation['article']['f1']:.2f} "
            f"exact_f1={citation['exact']['f1']:.2f} "
            f"cls={fallback['citation_class']:<13s} ({elapsed:.0f}s total)"
        )

    print(f"\n  decomposer stats: {decomposer.stats}")
    print(f"  matcher stats: {matcher.stats}")

    print("\n[3/3] computing summary …")
    eligible = [r for r in per_run if _claim_eligible(r)]
    summary = {
        "per_run_count": len(per_run),
        "overall": {
            "n_claim_eligible": len(eligible),
            "n_skipped_no_gt": len(per_run) - len(eligible),
            "claim_f1_mu": round(_mean([r["claim"]["f1"] for r in eligible]), 4),
            "claim_p_mu": round(_mean([r["claim"]["precision"] for r in eligible]), 4),
            "claim_r_mu": round(_mean([r["claim"]["recall"] for r in eligible]), 4),
            "citation_f1_mu": round(
                _mean([r["citation"]["article"]["f1"] for r in per_run]), 4
            ),
            "citation_p_mu": round(
                _mean([r["citation"]["article"]["precision"] for r in per_run]), 4
            ),
            "citation_r_mu": round(
                _mean([r["citation"]["article"]["recall"] for r in per_run]), 4
            ),
            "citation_exact_f1_mu": round(
                _mean([r["citation"]["exact"]["f1"] for r in per_run]), 4
            ),
            "citation_exact_p_mu": round(
                _mean([r["citation"]["exact"]["precision"] for r in per_run]), 4
            ),
            "citation_exact_r_mu": round(
                _mean([r["citation"]["exact"]["recall"] for r in per_run]), 4
            ),
            "citation_cfp_mu": round(
                _mean([r["citation"]["article"]["cfp"] for r in per_run]), 4
            ),
            "citation_context_cfp_mu": round(
                _mean([r["citation"]["article"]["context_cfp"] for r in per_run]), 4
            ),
            "citation_exact_cfp_mu": round(
                _mean([r["citation"]["exact"]["cfp"] for r in per_run]), 4
            ),
            "citation_exact_context_cfp_mu": round(
                _mean([r["citation"]["exact"]["context_cfp"] for r in per_run]), 4
            ),
            "fallback": aggregate_fallback([r["fallback"] for r in per_run]),
        },
        "by_variant": _per_bucket(per_run, "variant"),
        "by_difficulty": _per_bucket(per_run, "difficulty"),
        "decomposer_stats": {
            "calls": decomposer.stats.calls,
            "cache_hits": decomposer.stats.cache_hits,
            "parse_failures": decomposer.stats.parse_failures,
            "empty_answers": decomposer.stats.empty_answers,
            "truncated": decomposer.stats.truncated,
        },
        "matcher_stats": {
            "calls": matcher.stats.calls,
            "cache_hits": matcher.stats.cache_hits,
            "parse_failures": matcher.stats.parse_failures,
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  → {args.out/'summary.json'}")
    print(f"  → {eval_path}")

    # # v7-specific diagnostics
    # print("\n[4/3] v7 diagnostics …")
    # diagnostics = compute_v7_diagnostics(rows, per_run)
    # (args.out / "v7_diagnostics.json").write_text(
    #     json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    # )
    # print(f"  → {args.out/'v7_diagnostics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
