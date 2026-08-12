"""Build the Operational Knowledge Graph from parsed articles.

Pipeline (Steps 1–4 of the OKG build task):
  1. Load data/parsed/articles.jsonl
  2. Extract inline references and resolve them (src.compile.reference_extractor)
  3. Build NetworkX DiGraph (src.compile.okg.build_okg)
  4. Persist to data/okg/ and validate FAQ gt_references

Usage:
    python scripts/run_build_okg.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.compile.ancestor_prefix import apply_ancestor_prefix
from src.compile.okg import build_okg, save_okg, validate_faq
from src.utils.io import load_jsonl

def _load_articles(jsonl_path: Path) -> list[dict]:
    with open(jsonl_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_text_comparison(
    graph, out_path: Path, sample_size: int = 10, seed: int = 42
) -> None:
    import random
    rng = random.Random(seed)
    leaf_nids = [
        n for n, d in graph.nodes(data=True)
        if (d.get("data") or {}).get("is_leaf")
        and (d.get("data") or {}).get("indexed_text")
    ]
    rng.shuffle(leaf_nids)
    sample = leaf_nids[:sample_size]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("node_id\told_text_100\tindexed_text_200\n")
        for nid in sample:
            data = graph.nodes[nid].get("data") or {}
            old = (data.get("text_ko") or "").replace("\t", " ").replace("\n", " ")[:100]
            new = (data.get("indexed_text") or "").replace("\t", " ").replace("\n", " ")[:200]
            f.write(f"{nid}\t{old}\t{new}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", default="data/RegOps-Bench/articles.jsonl", type=Path)
    ap.add_argument("--faq", default="data/FAQ_examples.json", type=Path)
    ap.add_argument("--out-dir", default="data/okg", type=Path)
    ap.add_argument(
        "--out-suffix", default="",
        help="Suffix appended to output filenames (e.g. '_v2' → okg_v2.gpickle).",
    )
    ap.add_argument(
        "--no-ancestor-prefix", action="store_true",
        help="Skip ancestor-prefix text composition (legacy behaviour).",
    )
    args = ap.parse_args()

    def _abs(p: Path) -> Path:
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    articles_path = _abs(args.articles)
    faq_path = _abs(args.faq)
    out_dir = _abs(args.out_dir)

    if not articles_path.exists():
        print(f"articles file missing: {articles_path}", file=sys.stderr)
        return 2

    articles = _load_articles(articles_path)
    print(f"Loaded {len(articles)} articles from {articles_path}")

    faq_refs: list[str] = []
    if faq_path.exists():
        if faq_path.suffix=='.json':
            with open(faq_path, encoding="utf-8") as f:
                faq = json.load(f)
        else:
            faq = load_jsonl(faq_path)
        for qa in faq:
            faq_refs.extend(qa.get("gt_references", []))

    result = build_okg(articles, faq_refs=faq_refs)

    if not args.no_ancestor_prefix:
        prefix_stats = apply_ancestor_prefix(result.graph, articles)
        print("\n── Ancestor prefix ──")
        for k, v in prefix_stats.items():
            print(f"  {k}: {v}")
        _write_text_comparison(
            result.graph, out_dir / f"okg_text_comparison{args.out_suffix}.tsv"
        )

    paths = save_okg(result.graph, out_dir, suffix=args.out_suffix)

    # ── Report ──
    print("\n── OKG summary ──")
    for k, v in result.stats.items():
        print(f"  {k}: {v}")
    print("\n── Reference resolution ──")
    for k, v in result.resolution.as_dict().items():
        print(f"  {k}: {v}")
    print("\nWrote:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    # ── FAQ validation ──
    if faq_path.exists():
        validation = validate_faq(result.graph, faq)
        summary = validation["summary"]
        print("\n── FAQ validation ──")
        print(f"  qa_count: {summary['qa_count']}")
        print(
            f"  coverage: {summary['matched_refs']}/{summary['total_refs']} "
            f"({summary['coverage']:.1%})"
        )
        print(f"  difficulty_distribution: {summary['difficulty_distribution']}")

        report_path = out_dir / "faq_validation.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(validation, f, ensure_ascii=False, indent=2)
        print(f"  details → {report_path}")

        # Show hop_count histogram
        from collections import Counter
        hops = Counter(x["hop_count"] for x in validation["per_qa"])
        print(f"  hop_count histogram: {dict(sorted(hops.items()))}")

        # Show a few examples
        print("\n  Sample per-QA:")
        for row in validation["per_qa"][:5]:
            print(
                f"    {row['qa_id']}: hop={row['hop_count']} "
                f"difficulty={row['difficulty']} "
                f"cross={row['cross_document']} "
                f"missing={row['missing']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
