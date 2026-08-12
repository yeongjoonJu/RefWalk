"""RegOps-Bench evaluator — framework-agnostic.

Scores any RAG system on RegOps-Bench from
a prediction JSONL file. The evaluator never runs the system under test:
it joins predictions to gold labels by ``qa_id`` and reports

  Retrieval   R@5, R@10, nDCG@10, FullCov@10
  Citation    precision / recall / F1 at three granularities — 조-level
              (CP/CR/CF1), granularity-preserving (CF1_strict), and
              항/호-level (CP_para/CR_para/CF1_para) — plus CFP (cited ids
              absent from the corpus) and ctxCFP (cited ids absent from
              the system's own context)
  Claim       precision / recall / F1 over atomic claims  [--judge only]

Retrieval and citation metrics are pure Python — no LLM, no GPU, no
network. Claim metrics need an OpenAI-compatible endpoint and are opt-in.

The prediction format is documented in ``docs/evaluation.md``; the short
version is one JSON object per question with ``qa_id``,
``retrieved_node_ids``, ``answer``, and ``cited_node_ids``.

Examples
--------
Score one system::

    python cli/evaluate.py \\
        --predictions experiments/my_rag/preds.jsonl \\
        --bench data/RegOps-Bench/regops_bench.jsonl \\
        --corpus data/okg/okg_nodes.jsonl \\
        --out experiments/my_rag/scores

Compare several systems in one report (each file becomes a row)::

    python cli/evaluate.py \\
        --predictions experiments/*/preds.jsonl \\
        --bench data/RegOps-Bench/regops_bench.jsonl \\
        --corpus data/okg/okg_nodes.jsonl \\
        --out experiments/comparison

Check a prediction file without scoring it::

    python cli/evaluate.py --validate-only \\
        --predictions preds.jsonl --bench data/RegOps-Bench/regops_bench.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.citation_metrics import score_citations
from src.eval.clause_parser import expand_clause_citations, is_clause_id
from src.eval.metrics import compute_retrieval_metrics
from src.eval.predictions import (
    Prediction,
    PredictionFormatError,
    extract_citations,
    load_predictions,
    validate,
)


DIFFICULTIES = ("L1", "L2", "L3", "L4")


# ─── Loading ────────────────────────────────────────────────────────


def load_bench(path: Path) -> dict[str, dict]:
    """Index the benchmark by ``qa_id``. Accepts JSONL or a JSON list."""
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    bench = {}
    for row in rows:
        qa_id = row.get("qa_id") or row.get("id")
        if qa_id:
            bench[qa_id] = row
    if not bench:
        raise SystemExit(f"no qa_id-bearing rows found in {path}")
    return bench


def dedup_bench(bench: dict[str, dict]) -> tuple[dict[str, dict], list[str]]:
    """Drop questions whose text duplicates an earlier one.

    RegOps-Bench shipped with 19 duplicate question strings among its 250
    entries (all augmented, all L3/L4). The corrected evaluation set keeps
    the lowest ``qa_id`` of each duplicate group, leaving 231 questions.
    Idempotent: running this on an already-deduplicated file drops nothing.
    """
    by_question: dict[str, list[str]] = {}
    for qa_id, row in bench.items():
        text = (row.get("question") or "").strip()
        by_question.setdefault(text, []).append(qa_id)

    dropped = sorted(
        qa_id
        for ids in by_question.values() if len(ids) > 1
        for qa_id in sorted(ids)[1:]
    )
    kept = {k: v for k, v in bench.items() if k not in set(dropped)}
    return kept, dropped


def load_corpus_ids(path: Path | None) -> list[str] | None:
    """Read the universe of valid node_ids from a corpus JSONL.

    Works with both ``okg_nodes.jsonl`` and ``articles.jsonl`` — the only
    field consumed is ``node_id``.
    """
    if path is None:
        return None
    ids: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            nid = json.loads(line).get("node_id")
            if nid:
                ids.append(nid)
    if not ids:
        raise SystemExit(f"no node_id found in corpus file {path}")
    return ids


# ─── Scoring ────────────────────────────────────────────────────────


def _prf(pred: set[str], gold: set[str]) -> dict:
    """Plain set P/R/F1. Empty-vs-empty is vacuously perfect."""
    if not pred and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def score_one(
    pred: Prediction,
    gold: dict,
    corpus_ids: list[str] | None,
    *,
    ks: tuple[int, ...],
    hit_rule: str,
    match_unit: str,
    cite_from_answer: bool,
    clause_aware: bool,
) -> dict:
    """Score a single prediction against its gold benchmark row."""
    gt_refs = list(gold.get("gt_references") or [])

    cited = pred.cited_node_ids
    if not cited and cite_from_answer:
        cited = extract_citations(pred.answer, corpus_ids)

    retrieval = compute_retrieval_metrics(
        pred.retrieved_node_ids,
        gt_refs,
        match_unit=match_unit,
        ks=ks,
        hit_rule=hit_rule,
    )

    # Article-level metrics stay on the declared citation list. The
    # clause-expanded set may mine additional 조 numbers out of prose, and
    # those are lower-confidence than a system's own citations — the
    # headline metric should not be inflated by them.
    citation = score_citations(
        predicted=cited,
        gt=gt_refs,
        corpus_node_ids=corpus_ids,
        retrieved_node_ids=pred.retrieved_node_ids,
    )

    # Clause-level metrics need the expanded set, because a system that
    # names 항 in a tag or in prose (rather than in its citation list) would
    # otherwise score 0 at full granularity for a formatting reason.
    if clause_aware:
        cited_clause, clause_stats = expand_clause_citations(
            pred.raw, cited, pred.retrieved_node_ids,
        )
    else:
        cited_clause, clause_stats = list(cited), {"source": "disabled", "n_added": 0}

    strict = score_citations(predicted=cited_clause, gt=gt_refs)["exact"]
    gold_clause = {g for g in gt_refs if is_clause_id(g)}
    pred_clause = {c for c in cited_clause if is_clause_id(c)}
    paragraph = _prf(pred_clause, gold_clause)

    return {
        "qa_id": pred.qa_id,
        "variant": pred.variant,
        "difficulty": gold.get("difficulty") or gold.get("target_difficulty") or "?",
        "status": pred.status,
        "n_retrieved": len(pred.retrieved_node_ids),
        "n_cited": len(cited),
        "n_gold": len(gt_refs),
        "cited_node_ids": cited,
        "retrieval": retrieval,
        "citation": citation,
        # Clause (항/호) granularity.
        "strict": strict,                       # whole benchmark, no rollup
        "paragraph": paragraph,                 # clause ids only, both sides
        "has_clause_gold": bool(gold_clause),   # gates the paragraph average
        "clause_expansion": clause_stats,
    }


# ─── Aggregation ────────────────────────────────────────────────────


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def _get(row: dict, path: tuple[str, ...]) -> float | None:
    cur: object = row
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur if isinstance(cur, (int, float)) else None


# A metric may only be defined on a subset of queries. Each spec entry
# carries the gate that decides which rows enter its average.
#   None      — every row
#   "claim"   — rows whose gold answer decomposed to >=1 claim
#   "clause"  — rows whose gold carries >=1 항/호 reference
def _eligible(row: dict, gate: str | None) -> bool:
    if gate is None:
        return True
    if gate == "claim":
        return int(_get(row, ("claim", "n_gt")) or 0) > 0
    if gate == "clause":
        return bool(row.get("has_clause_gold"))
    raise ValueError(f"unknown eligibility gate {gate!r}")


_GATE_COUNT_KEY = {"claim": "n_claim_eligible", "clause": "n_clause_eligible"}


def metric_spec(
    ks: tuple[int, ...], with_claim: bool,
) -> list[tuple[str, tuple, str | None]]:
    """Ordered (column name → path into a per-query record → gate) mapping."""
    k_top = max(ks)
    spec: list[tuple[str, tuple, str | None]] = []
    for k in ks:
        spec.append((f"R@{k}", ("retrieval", f"R@{k}"), None))
    spec += [
        (f"nDCG@{k_top}", ("retrieval", f"nDCG@{k_top}"), None),
        (f"FullCov@{k_top}", ("retrieval", f"FullCov@{k_top}"), None),
        # 조-level: both sides rolled up to the article ancestor.
        ("CP", ("citation", "article", "precision"), None),
        ("CR", ("citation", "article", "recall"), None),
        ("CF1", ("citation", "article", "f1"), None),
        # Granularity-preserving: no rollup, whole benchmark.
        ("CF1_strict", ("strict", "f1"), None),
        # 항/호-level: both sides restricted to clause ids, scored only on
        # the queries whose gold actually names a clause.
        ("CP_para", ("paragraph", "precision"), "clause"),
        ("CR_para", ("paragraph", "recall"), "clause"),
        ("CF1_para", ("paragraph", "f1"), "clause"),
        ("CFP", ("citation", "article", "cfp"), None),
        ("ctxCFP", ("citation", "article", "context_cfp"), None),
    ]
    if with_claim:
        spec += [
            ("claim_P", ("claim", "precision"), "claim"),
            ("claim_R", ("claim", "recall"), "claim"),
            ("claim_F1", ("claim", "f1"), "claim"),
        ]
    return spec


def aggregate(
    rows: list[dict], spec: list[tuple[str, tuple, str | None]],
) -> dict:
    """Mean-reduce per-query records over one bucket."""
    if not rows:
        return {"n": 0}
    agg: dict = {"n": len(rows)}
    for name, path, gate in spec:
        vals = [
            v for r in rows
            if _eligible(r, gate) and (v := _get(r, path)) is not None
        ]
        if gate is not None:
            agg[_GATE_COUNT_KEY[gate]] = len(vals)
        agg[name] = _mean(vals)
    agg["n_error"] = sum(1 for r in rows if r.get("status") != "ok")
    return agg


def bucket_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(str(r.get(key, "?")), []).append(r)
    return out


# ─── Reporting ──────────────────────────────────────────────────────


def format_table(
    header: str,
    by_row: dict[str, dict],
    spec: list[tuple[str, tuple, str | None]],
) -> str:
    names = [n for n, _, _ in spec]
    label_w = max([len(header)] + [len(k) for k in by_row]) + 2
    cols = [max(len(n), 7) for n in names]

    lines = [
        header.ljust(label_w) + "".join(
            n.rjust(w + 2) for n, w in zip(names, cols)
        ) + "     n"
    ]
    lines.append("-" * len(lines[0]))
    for label, agg in by_row.items():
        cells = "".join(
            f"{agg.get(n, 0.0):.4f}".rjust(w + 2) for n, w in zip(names, cols)
        )
        lines.append(label.ljust(label_w) + cells + f"{agg.get('n', 0):>6}")
    return "\n".join(lines)


def write_tsv(path: Path, by_row: dict[str, dict],
              spec: list[tuple[str, tuple, str | None]],
              label: str) -> None:
    names = [n for n, _, _ in spec]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join([label, *names, "n"]) + "\n")
        for key, agg in by_row.items():
            f.write("\t".join(
                [key]
                + [f"{agg.get(n, 0.0):.4f}" for n in names]
                + [str(agg.get("n", 0))]
            ) + "\n")


# ─── Claim judging (optional) ───────────────────────────────────────


def run_claim_judge(rows: list[dict], bench: dict[str, dict], args) -> None:
    """Attach LLM-judged claim metrics to each per-query record in place.

    Gold answers are decomposed once per ``qa_id`` and reused across
    every variant, so comparing N systems costs one gold pass plus N
    prediction passes.
    """
    from tqdm import tqdm

    from src.eval.claim_decomposer import ClaimDecomposer
    from src.eval.claim_matcher import ClaimMatcher

    decomposer = ClaimDecomposer(
        base_url=args.judge_base_url,
        model_id=args.judge_model,
        language=args.language,
    )
    matcher = ClaimMatcher(
        base_url=args.matcher_base_url or args.judge_base_url,
        model_id=args.judge_model,
        language=args.language,
    )

    needed = sorted({r["qa_id"] for r in rows})
    print(f"[judge] decomposing {len(needed)} gold answers "
          f"({args.judge_base_url}, {args.judge_model})")
    gold_claims: dict[str, list[dict]] = {}
    for qa_id in tqdm(needed, desc="gold"):
        answer = bench[qa_id].get("answer") or ""
        gold_claims[qa_id] = decomposer.decompose(answer, answer_id=f"gt:{qa_id}")

    print(f"[judge] scoring {len(rows)} predictions")
    for row in tqdm(rows, desc="pred"):
        gt = gold_claims.get(row["qa_id"], [])
        answer = row.pop("_answer", "")
        if row.get("status") != "ok" or not answer.strip():
            match = matcher.match([], gt)
        else:
            pred_claims = decomposer.decompose(
                answer, answer_id=f"pred:{row['variant']}:{row['qa_id']}"
            )
            match = matcher.match(pred_claims, gt)
        row["claim"] = {
            "precision": match["precision"],
            "recall": match["recall"],
            "f1": match["f1"],
            "n_pred": match["n_pred"],
            "n_gt": match["n_gt"],
            "n_match": match["n_match"],
        }
    print(f"[judge] decomposer={decomposer.stats} matcher={matcher.stats}")


# ─── CLI ────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate any RAG system on RegOps-Bench.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--predictions", type=Path, nargs="+", required=True,
        help="Prediction JSONL file(s). Each file is one system unless its "
             "records carry a 'variant' field; the file stem is the default "
             "label.",
    )
    p.add_argument(
        "--bench", type=Path,
        default=REPO_ROOT / "data/RegOps-Bench/regops_bench.jsonl",
        help="Benchmark file holding the gold labels (source of truth for "
             "gt_references and difficulty).",
    )
    p.add_argument(
        "--corpus", type=Path, default=None,
        help="Corpus JSONL (okg_nodes.jsonl or articles.jsonl). Supplies the "
             "node_id universe that makes CFP meaningful and lets "
             "--cite-from-answer recognise ids in prose. Strongly recommended.",
    )
    p.add_argument(
        "--label", nargs="+", default=None,
        help="Report label(s) for the systems. Give one label to apply to "
             "every file, or one per --predictions file. Overrides any "
             "'variant' field inside the records, which is useful when that "
             "field is stale or absent.",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Directory for summary.json / per_query.jsonl / *.tsv. "
             "Prints to stdout only when omitted.",
    )
    p.add_argument(
        "--ks", type=int, nargs="+", default=[5, 10],
        help="k values for R@k. nDCG and FullCov use max(ks). Default: 5 10.",
    )
    p.add_argument(
        "--hit-rule", choices=["hierarchical", "strict"], default="hierarchical",
        help="hierarchical (default): a retrieved id counts when it equals a "
             "gold id or is its ancestor/descendant along the id hierarchy. "
             "RegOps-Bench gold spans 조/항/호 levels, so a system that "
             "indexes at 조 granularity cannot emit a 항-level id at all — "
             "strict matching would score it 0 for structural reasons rather "
             "than retrieval quality. strict: exact id equality only; use it "
             "when your index granularity matches the gold labels.",
    )
    p.add_argument(
        "--match-unit", choices=["node_id", "article_id"], default="node_id",
        help="node_id (default): compare ids verbatim. article_id: roll both "
             "sides up to their 조-level ancestor before scoring.",
    )
    p.add_argument(
        "--cite-from-answer", action="store_true",
        help="For records with no citation field, recover cited ids from the "
             "answer text (citation footer, else node_ids named in prose — "
             "the latter needs --corpus).",
    )
    p.add_argument(
        "--no-clause-aware", dest="clause_aware", action="store_false",
        help="Disable clause recovery for the 항/호-level metrics. By default "
             "the evaluator widens the cited set with the clause ids the "
             "answer actually names — bracketed tags in a structured "
             "'raw_output', otherwise clause references in the answer prose "
             "grounded through the system's own cited/retrieved ids. Without "
             "it, a system that reports 조-level citations and names the 항 in "
             "its text scores 0 at clause granularity for a formatting "
             "reason rather than a grounding one. Article-level CP/CR/CF1 are "
             "never affected either way.",
    )
    p.add_argument(
        "--dedup-questions", action="store_true",
        help="Score on the deduplicated benchmark: drop questions whose text "
             "repeats an earlier entry, keeping the lowest qa_id of each "
             "group. RegOps-Bench's original 250 entries contain 19 such "
             "duplicates (all augmented, all L3/L4), so this yields the "
             "corrected 231-question set the reported numbers use. "
             "Idempotent on an already-deduplicated file.",
    )
    p.add_argument(
        "--skip-errors", action="store_true",
        help="Drop records with status=error instead of scoring them as "
             "failures. Reported systems should normally keep them.",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero on validation warnings, not just errors.",
    )
    p.add_argument(
        "--validate-only", action="store_true",
        help="Check the prediction file(s) against the benchmark and exit "
             "without scoring.",
    )

    j = p.add_argument_group("claim metrics (LLM judge, opt-in)")
    j.add_argument(
        "--judge", action="store_true",
        help="Also compute claim precision/recall/F1 by decomposing gold and "
             "predicted answers into atomic claims and matching them with an "
             "LLM. Requires an OpenAI-compatible endpoint.",
    )
    j.add_argument("--judge-base-url", default="http://localhost:8035/v1",
                   help="Endpoint for the claim decomposer.")
    j.add_argument("--matcher-base-url", default=None,
                   help="Endpoint for the claim matcher. Point it at a "
                        "different model than --judge-base-url to reduce "
                        "self-preference bias. Defaults to --judge-base-url.")
    j.add_argument("--judge-model", default="Qwen/Qwen3.6-35B-A3B-FP8",
                   help="Model id passed to the judge endpoints.")
    j.add_argument("--language", choices=["ko", "en"], default="ko",
                   help="Judge prompt language. ko is the RegOps default.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    bench = load_bench(args.bench)
    print(f"[load] bench={len(bench)} questions from {args.bench}")
    if args.dedup_questions:
        bench, dropped = dedup_bench(bench)
        print(f"[dedup] dropped {len(dropped)} duplicate-question entries → "
              f"{len(bench)} questions")

    corpus_ids = load_corpus_ids(args.corpus)
    if corpus_ids:
        print(f"[load] corpus={len(corpus_ids)} node_ids from {args.corpus}")
    else:
        print("[load] no --corpus given: CFP is reported as 0.0 "
              "(metric undefined without the node_id universe)")

    labels = args.label
    if labels and len(labels) not in (1, len(args.predictions)):
        print(f"[error] got {len(labels)} --label values for "
              f"{len(args.predictions)} prediction file(s); pass one label or "
              f"one per file", file=sys.stderr)
        return 2

    preds: list[Prediction] = []
    try:
        for i, path in enumerate(args.predictions):
            batch = load_predictions(path)
            if labels:
                forced = labels[0] if len(labels) == 1 else labels[i]
                for p in batch:
                    p.variant = forced
            preds.extend(batch)
    except PredictionFormatError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    print(f"[load] predictions={len(preds)} records from "
          f"{len(args.predictions)} file(s)")

    if args.dedup_questions:
        # Predictions for dropped duplicates are simply out of scope here,
        # not a format error — silently narrow to the deduplicated set.
        before = len(preds)
        preds = [p for p in preds if p.qa_id in bench]
        if before != len(preds):
            print(f"[dedup] ignored {before - len(preds)} prediction records "
                  f"for dropped questions")

    report = validate(preds, bench.keys(), corpus_ids)
    for w in report["warnings"]:
        print(f"[warn] {w}")
    for e in report["errors"]:
        print(f"[error] {e}", file=sys.stderr)
    if not report["ok"]:
        return 2
    if args.validate_only:
        print(f"[ok] {report['n_predictions']} records, "
              f"{report['n_variants']} variant(s): "
              f"{', '.join(report['variants'])}")
        return 1 if (args.strict and report["warnings"]) else 0

    if args.skip_errors:
        before = len(preds)
        preds = [p for p in preds if p.ok]
        if before != len(preds):
            print(f"[filter] dropped {before - len(preds)} error records")

    rows = [
        score_one(
            p, bench[p.qa_id], corpus_ids,
            ks=tuple(args.ks),
            hit_rule=args.hit_rule,
            match_unit=args.match_unit,
            cite_from_answer=args.cite_from_answer,
            clause_aware=args.clause_aware,
        )
        for p in preds
    ]

    if args.judge:
        for row, p in zip(rows, preds):
            row["_answer"] = p.answer
        run_claim_judge(rows, bench, args)

    spec = metric_spec(tuple(args.ks), with_claim=args.judge)
    by_variant = {
        v: aggregate(rs, spec) for v, rs in sorted(bucket_by(rows, "variant").items())
    }

    by_variant_difficulty: dict[str, dict[str, dict]] = {}
    for variant, vrows in sorted(bucket_by(rows, "variant").items()):
        buckets = bucket_by(vrows, "difficulty")
        by_variant_difficulty[variant] = {
            lv: aggregate(buckets[lv], spec)
            for lv in DIFFICULTIES if lv in buckets
        }

    print()
    print(format_table("system", by_variant, spec))
    if len(by_variant) == 1:
        (only,) = by_variant
        per_diff = by_variant_difficulty[only]
        if per_diff:
            print()
            print(format_table("difficulty", per_diff, spec))

    summary = {
        "config": {
            "bench": str(args.bench),
            "corpus": str(args.corpus) if args.corpus else None,
            "predictions": [str(p) for p in args.predictions],
            "ks": list(args.ks),
            "hit_rule": args.hit_rule,
            "match_unit": args.match_unit,
            "cite_from_answer": args.cite_from_answer,
            "clause_aware": args.clause_aware,
            "skip_errors": args.skip_errors,
            "judge": args.judge,
            "judge_model": args.judge_model if args.judge else None,
        },
        "validation": report,
        "overall": by_variant,
        "by_difficulty": by_variant_difficulty,
    }

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with open(args.out / "per_query.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                row.pop("_answer", None)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_tsv(args.out / "summary.tsv", by_variant, spec, "system")
        flat = {
            f"{v}/{lv}": agg
            for v, per_lv in by_variant_difficulty.items()
            for lv, agg in per_lv.items()
        }
        write_tsv(args.out / "per_difficulty.tsv", flat, spec, "system/difficulty")
        print(f"\n[write] {args.out}/summary.json, per_query.jsonl, "
              f"summary.tsv, per_difficulty.tsv")

    return 1 if (args.strict and report["warnings"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
