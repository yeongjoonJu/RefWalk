from __future__ import annotations

import math
from typing import Iterable, Literal, Optional

from src.retrieval.corpus import _article_ancestor, _canon_ref


# New (preferred) naming.
MatchUnit = Literal["node_id", "article_id"]
# Legacy alias kept for older callers / scripts.
Granularity = Literal["node", "article"]
HitRule = Literal["strict", "hierarchical"]


def _normalize_match_unit(
    match_unit: Optional[str],
    granularity: Optional[str] = None,
) -> str:
    """Resolve the (possibly aliased) match-unit selector to a canonical
    new-form string ("node_id" or "article_id").

    Accepts:
      - new values: "node_id", "article_id"
      - legacy values: "node" → "node_id", "article" → "article_id"
      - either keyword may carry the value; when both are provided,
        ``granularity`` (legacy) wins iff ``match_unit`` is left at its
        default — this lets old callers keep working without surprising
        new callers who explicitly set match_unit.
    """
    chosen = granularity if granularity is not None else match_unit
    if chosen is None:
        return "node_id"
    if chosen in ("node", "node_id"):
        return "node_id"
    if chosen in ("article", "article_id"):
        return "article_id"
    raise ValueError(
        f"match_unit must be one of node_id|article_id (or legacy node|"
        f"article); got {chosen!r}"
    )


# Separators that delimit one level of a node_id hierarchy. RegOps ids
# chain 조/항/호 with "_" (``doc_제10조_제1항_제1호``); path-style corpora
# chain with "/". Ancestry is a string-prefix test that
# must land on one of these boundaries, so `제10조` never matches
# `제100조`.
_HIER_SEPARATORS: tuple[str, ...] = ("_", "/")


def _hier_match(a: str, b: str) -> bool:
    """True iff ``a`` and ``b`` refer to the same node, or one is a
    hierarchical ancestor/descendant of the other."""
    if a == b:
        return True
    for sep in _HIER_SEPARATORS:
        if a.startswith(b + sep) or b.startswith(a + sep):
            return True
    return False


def _covered(r: str, gold: set[str], hit_rule: HitRule) -> bool:
    if hit_rule == "strict":
        return r in gold
    return any(_hier_match(r, g) for g in gold)


def _gold_covered(g: str, retrieved: Iterable[str], hit_rule: HitRule) -> bool:
    if hit_rule == "strict":
        return g in set(retrieved)
    return any(_hier_match(r, g) for r in retrieved)


# ─── Rollup helpers ─────────────────────────────────────────────────

def rollup_to_article(node_ids: Iterable[str]) -> list[str]:
    """Map a sequence of node_ids to their 조-level ancestors, preserving
    order and de-duplicating (first occurrence wins)."""
    out: list[str] = []
    seen: set[str] = set()
    for nid in node_ids:
        art = _article_ancestor(_canon_ref(nid))
        if art and art not in seen:
            seen.add(art)
            out.append(art)
    return out


def _prep(ids: Iterable[str], match_unit: str) -> list[str]:
    """Internal: caller is expected to have already canonicalised
    ``match_unit`` via :func:`_normalize_match_unit`. Returns a list
    in original order with duplicates removed (article mode rolls up
    each id to its 조 ancestor first)."""
    ids_canon = [_canon_ref(i) for i in ids if i]
    if match_unit == "article_id":
        return rollup_to_article(ids_canon)
    # node_id mode: dedup-in-order, keep original ids.
    out: list[str] = []
    seen: set[str] = set()
    for nid in ids_canon:
        if nid not in seen:
            seen.add(nid)
            out.append(nid)
    return out


def _prep_gold(ids: Iterable[str], match_unit: str) -> set[str]:
    return set(_prep(ids, match_unit))


# ─── Individual metric primitives ──────────────────────────────────

def recall_at_k(
    retrieved: list[str],
    gold: set[str],
    k: int,
    hit_rule: HitRule = "strict",
) -> float:
    """Share of distinct golds covered by retrieved[:k].

    Under ``strict``, equivalent to ``|retrieved[:k] ∩ gold| / |gold|``.
    Under ``hierarchical``, a gold is "covered" if any retrieved id is
    equal to it, its descendant, or its ancestor. Golds are counted
    once each (gold-side distinct counting).
    """
    if not gold:
        return 0.0
    topk = retrieved[:k]
    covered = sum(1 for g in gold if _gold_covered(g, topk, hit_rule))
    return covered / len(gold)


def ndcg_at_k(
    retrieved: list[str],
    gold: set[str],
    k: int,
    hit_rule: HitRule = "strict",
) -> float:
    if not gold:
        return 0.0
    dcg = 0.0
    for i, r in enumerate(retrieved[:k]):
        if _covered(r, gold, hit_rule):
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def full_coverage_at_k(
    retrieved: list[str],
    gold: set[str],
    k: int,
    hit_rule: HitRule = "strict",
) -> bool:
    if not gold:
        return False
    if hit_rule == "strict":
        return gold.issubset(set(retrieved[:k]))
    topk = retrieved[:k]
    return all(_gold_covered(g, topk, hit_rule) for g in gold)


# ─── Public API: retrieval metrics ─────────────────────────────────

def compute_retrieval_metrics(
    predicted_nodes: list[str],
    gold_nodes: list[str],
    match_unit: MatchUnit = "node_id",
    ks: tuple[int, ...] = (5, 10),
    hit_rule: HitRule = "strict",
    *,
    granularity: Optional[str] = None,
) -> dict:
    """Return retrieval metrics for a SINGLE query.

    Parameters
    ----------
    predicted_nodes : ranked list of retrieved node_ids.
    gold_nodes      : ground-truth node_ids (unordered).
    match_unit      : "node_id" (default) — match ids verbatim;
                      "article_id" — roll both sides up to 조 ancestor
                      before scoring.
                      Legacy aliases ``"node"`` / ``"article"`` are
                      accepted (and the legacy keyword ``granularity=``
                      is too).
    ks              : k values to report R@k for. nDCG / FullCov use
                      max(ks) (conventionally 10).
    hit_rule        : "strict" (default) → id equality; "hierarchical" →
                      equality or ancestor/descendant match via
                      :func:`_hier_match`.
    granularity     : (deprecated) old keyword for ``match_unit``.

    Returns
    -------
    dict like {"R@5", "R@10", "nDCG@10", "FullCov@10", "n_gold",
               "n_retrieved"}.
    """
    unit = _normalize_match_unit(match_unit, granularity)
    retrieved = _prep(predicted_nodes, unit)
    gold = _prep_gold(gold_nodes, unit)

    k_top = max(ks)
    out: dict = {}
    for k in ks:
        out[f"R@{k}"] = recall_at_k(retrieved, gold, k, hit_rule=hit_rule)
    out[f"nDCG@{k_top}"] = ndcg_at_k(retrieved, gold, k_top, hit_rule=hit_rule)
    out[f"FullCov@{k_top}"] = 1.0 if full_coverage_at_k(
        retrieved, gold, k_top, hit_rule=hit_rule,
    ) else 0.0
    out["n_gold"] = len(gold)
    out["n_retrieved"] = len(retrieved)
    return out


def aggregate_retrieval_metrics(per_query: list[dict]) -> dict:
    """Mean-reduce a list of per-query retrieval metric dicts.

    Silently drops queries with n_gold == 0.
    """
    usable = [q for q in per_query if q.get("n_gold", 0) > 0]
    if not usable:
        return {}
    agg: dict = {}
    keys = [k for k in usable[0] if isinstance(usable[0][k], (int, float))]
    for k in keys:
        if k in ("n_gold", "n_retrieved"):
            continue
        vals = [q[k] for q in usable if k in q]
        agg[k] = round(sum(vals) / len(vals), 4)
    agg["n_queries"] = len(usable)
    return agg


# ─── Public API: generation (citation) metrics ─────────────────────

def compute_generation_metrics(
    predicted_citations: list[str],
    gold_citations: list[str],
    corpus_node_ids: set[str],
    match_unit: MatchUnit = "node_id",
    *,
    granularity: Optional[str] = None,
) -> dict:
    """Return generation-stage citation metrics for a SINGLE answer.

    Parameters
    ----------
    predicted_citations : references the model cited in its answer.
    gold_citations      : ground-truth references for the question.
    corpus_node_ids     : universe of ids that exist in the OKG
                          (for CFP = hallucination detection). Provide
                          at node level; article rollup is applied
                          automatically when match_unit="article_id".
    match_unit          : "node_id" (default) | "article_id".
                          Legacy aliases ``"node"`` / ``"article"`` are
                          accepted; the legacy keyword ``granularity=``
                          is too.
    granularity         : (deprecated) old keyword for ``match_unit``.

    Returns
    -------
    dict with keys {"CP", "CR", "CFP", "n_pred", "n_gold",
                    "tp", "fp", "fn", "hallucinated"}.
    """
    unit = _normalize_match_unit(match_unit, granularity)
    pred = _prep(predicted_citations, unit)
    gold = set(_prep(gold_citations, unit))
    corpus = set(_prep(corpus_node_ids, unit))

    pred_set = set(pred)
    tp = len(pred_set & gold)
    fp = len(pred_set - gold)
    fn = len(gold - pred_set)
    hallucinated = {p for p in pred_set if p not in corpus}

    cp = tp / len(pred_set) if pred_set else 0.0
    cr = tp / len(gold) if gold else 0.0
    cfp = len(hallucinated) / len(pred_set) if pred_set else 0.0

    return {
        "CP": round(cp, 4),
        "CR": round(cr, 4),
        "CFP": round(cfp, 4),
        "n_pred": len(pred_set),
        "n_gold": len(gold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "hallucinated": sorted(hallucinated),
    }


def aggregate_generation_metrics(per_query: list[dict]) -> dict:
    if not per_query:
        return {}
    agg: dict = {}
    for key in ("CP", "CR", "CFP"):
        vals = [q[key] for q in per_query if key in q]
        if vals:
            agg[key] = round(sum(vals) / len(vals), 4)
    agg["n_queries"] = len(per_query)
    return agg


# ─── Combined reporting (retrieval + citation) ─────────────────────

def compute_all_metrics(
    predicted: list[str],
    gold: list[str],
    corpus_node_ids: set[str],
    match_unit: MatchUnit = "node_id",
    ks: tuple[int, ...] = (5, 10),
    hit_rule: HitRule = "strict",
    *,
    granularity: Optional[str] = None,
) -> dict:
    """Return retrieval **and** citation metrics for one query in a single
    dict. The "citation" side treats ``predicted`` as the cited set.

    Retrieval: R@k, nDCG@10, FullCov@10. Honours ``hit_rule``.
    Citation : CP, CR, CFP — always strict equality (hit_rule does not
               apply to citation metrics).

    Keys are prefixed so they do not collide: retrieval_* / citation_*.

    ``match_unit`` defaults to ``"node_id"`` (verbatim id matching);
    set to ``"article_id"`` to roll both retrieved and gold up to their
    조 ancestor before scoring. Legacy ``granularity=`` keyword and
    legacy values ``"node"``/``"article"`` are still accepted.
    """
    unit = _normalize_match_unit(match_unit, granularity)
    retrieval = compute_retrieval_metrics(
        predicted, gold, match_unit=unit, ks=ks, hit_rule=hit_rule,
    )
    k_top = max(ks)
    # Cap the "cited" set at top-k so citation precision is well-defined
    # on rankings longer than k.
    pred_topk = predicted[:k_top]
    citation = compute_generation_metrics(
        predicted_citations=pred_topk,
        gold_citations=gold,
        corpus_node_ids=corpus_node_ids,
        match_unit=unit,
    )
    out: dict = {}
    for k, v in retrieval.items():
        out[f"retrieval_{k}"] = v
    for k, v in citation.items():
        if k == "hallucinated":
            continue
        out[f"citation_{k}"] = v
    return out


def aggregate_all_metrics(per_query: list[dict]) -> dict:
    """Mean-reduce a list of per-query `compute_all_metrics` dicts.

    Silently drops queries where `retrieval_n_gold == 0`.
    """
    usable = [q for q in per_query if q.get("retrieval_n_gold", 0) > 0]
    if not usable:
        return {}
    agg: dict = {}
    for key in usable[0]:
        if key.endswith("_n_gold") or key.endswith("_n_retrieved") \
                or key.endswith("_n_pred") or key.endswith("_tp") \
                or key.endswith("_fp") or key.endswith("_fn"):
            continue
        vals = [q[key] for q in usable if isinstance(q.get(key), (int, float))]
        if vals:
            agg[key] = round(sum(vals) / len(vals), 4)
    agg["n_queries"] = len(usable)
    return agg
