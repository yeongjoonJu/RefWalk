"""Citation Precision / Recall / F1 with article-level rollup.

A predicted citation (항/호-level node_id or 조-level literal) counts
as correct iff its 조-level ancestor is present in the GT article
set. Rollup uses ``_article_ancestor`` from ``src.retrieval.corpus``
which already handles the 제N조 / 제N조의M pattern, and
``_canon_ref`` which normalises one legacy typo in the corpus.

Three levels of strictness are computed so we can see cost/benefit:

  - **article**  (default) : ancestor-to-ancestor, dedup on 조-level.
                             Matches ``evaluate()`` in retrieval eval.
  - **exact**              : string equality (after canonicalisation),
                             no rollup. Strictest.
  - **article_or_desc**    : a gt 조 counts as hit if any pred starts
                             with gt 조-id (pred may be 항/호-level).

The rollup is lossless for our smoke QAs — gt_references are always
조-level literals, and predicted `cited_references` come back from the
planner as 항/호/조 ids. article rollup is the faithful default.

Returned dict per run::

    {
      "article": {"precision": .., "recall": .., "f1": .., "tp": int,
                  "fp": int, "fn": int, "pred_articles": [...],
                  "gt_articles": [...]},
      "exact":   {... same keys ...},
      "article_or_desc": {... same ...},
    }
"""

from __future__ import annotations

from src.retrieval.corpus import _article_ancestor, _canon_ref


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def _score_set(pred: set[str], gt: set[str]) -> dict:
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    p, r, f = _prf(tp, fp, fn)
    return {
        "precision": p, "recall": r, "f1": f,
        "tp": tp, "fp": fp, "fn": fn,
        "n_pred": len(pred), "n_gt": len(gt),
    }


def _compute_cfp(
    pred_set: set[str],
    universe: set[str] | None,
) -> tuple[float, list[str]]:
    """Citation False-Positive rate vs a reference universe.

    CFP = |pred − universe| / |pred|. Returns ``(cfp, hallucinated_list)``.
    A vacuous pred (empty) returns 0.0 — no citations means no hallucination.
    A None universe (caller didn't supply one) returns 0.0 too — the metric
    is undefined.
    """
    if not pred_set or universe is None:
        return 0.0, []
    halluc = sorted(p for p in pred_set if p not in universe)
    return round(len(halluc) / len(pred_set), 4), halluc


def score_citations(
    predicted: list[str],
    gt: list[str],
    corpus_node_ids: list[str] | None = None,
    retrieved_node_ids: list[str] | None = None,
) -> dict:
    """Compute citation P/R/F1 at three rollup levels.

    Both ``predicted`` and ``gt`` are canonicalised with ``_canon_ref``
    before scoring. ``corpus_node_ids`` and ``retrieved_node_ids`` are
    optional and drive two hallucination metrics when supplied:

      * ``cfp`` (corpus-CFP) — |pred − corpus| / |pred|. Captures
        citations the LLM emitted that don't exist anywhere in the OKG
        ("pure" hallucination). Mirrors the retrieval-side CFP defined
        in ``src/eval/metrics.py``.
      * ``context_cfp`` (context-CFP) — |pred − retrieved| / |pred|.
        Stricter: citations the LLM emitted that weren't even in its
        own retrieved context window. A generation-specific signal.

    Both are reported at both ``article`` and ``exact`` levels. ``cfp``
    is 0.0 when ``corpus_node_ids`` is missing; ``context_cfp`` is 0.0
    when ``retrieved_node_ids`` is missing.

    Empty pred with empty gt → all metrics = 1.0 (vacuously correct).
    Empty pred with non-empty gt → 0.0. Non-empty pred with empty gt
    → precision 0.0, recall undefined (reported as 0.0 to stay comparable).
    """
    pred_canon = [_canon_ref(p) for p in (predicted or []) if p]
    gt_canon = [_canon_ref(g) for g in (gt or []) if g]
    corpus_canon = (
        {_canon_ref(c) for c in corpus_node_ids}
        if corpus_node_ids is not None else None
    )
    retrieved_canon = (
        {_canon_ref(r) for r in retrieved_node_ids}
        if retrieved_node_ids is not None else None
    )
    corpus_articles = (
        {_article_ancestor(c) for c in corpus_canon}
        if corpus_canon is not None else None
    )
    retrieved_articles = (
        {_article_ancestor(r) for r in retrieved_canon}
        if retrieved_canon is not None else None
    )

    if not pred_canon and not gt_canon:
        base = {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                "tp": 0, "fp": 0, "fn": 0, "n_pred": 0, "n_gt": 0,
                "cfp": 0.0, "context_cfp": 0.0,
                "hallucinated": [], "context_hallucinated": []}
        return {
            "article": {**base, "pred_articles": [], "gt_articles": []},
            "exact": {**base, "pred": [], "gt": []},
            "article_or_desc": {**base, "pred_articles": [], "gt_articles": []},
        }

    pred_articles = {_article_ancestor(p) for p in pred_canon}
    gt_articles = {_article_ancestor(g) for g in gt_canon}

    article = _score_set(pred_articles, gt_articles)
    article["pred_articles"] = sorted(pred_articles)
    article["gt_articles"] = sorted(gt_articles)
    art_cfp, art_halluc = _compute_cfp(pred_articles, corpus_articles)
    art_ctx_cfp, art_ctx_halluc = _compute_cfp(pred_articles, retrieved_articles)
    article["cfp"] = art_cfp
    article["context_cfp"] = art_ctx_cfp
    article["hallucinated"] = art_halluc
    article["context_hallucinated"] = art_ctx_halluc

    exact_pred = set(pred_canon)
    exact_gt = set(gt_canon)
    exact = _score_set(exact_pred, exact_gt)
    exact["pred"] = sorted(exact_pred)
    exact["gt"] = sorted(exact_gt)
    ex_cfp, ex_halluc = _compute_cfp(exact_pred, corpus_canon)
    ex_ctx_cfp, ex_ctx_halluc = _compute_cfp(exact_pred, retrieved_canon)
    exact["cfp"] = ex_cfp
    exact["context_cfp"] = ex_ctx_cfp
    exact["hallucinated"] = ex_halluc
    exact["context_hallucinated"] = ex_ctx_halluc

    # article_or_desc: gt 조 is hit if any pred (at any granularity)
    # has that 조 as prefix OR equals it exactly (after canon).
    hits: set[str] = set()
    for g_art in gt_articles:
        for p in pred_canon:
            if p == g_art or p.startswith(g_art + "_") or p.startswith(g_art):
                if p == g_art or p.startswith(g_art + "_"):
                    hits.add(g_art)
                    break
    tp = len(hits)
    fn = len(gt_articles) - tp
    # fp: pred_articles that map to no gt_article prefix match.
    gt_prefixes = tuple(gt_articles)
    fp_preds = {
        p for p in pred_canon
        if not any(p == g or p.startswith(g + "_") for g in gt_prefixes)
    }
    fp = len({_article_ancestor(p) for p in fp_preds})
    p_val, r_val, f_val = _prf(tp, fp, fn)
    article_or_desc = {
        "precision": p_val, "recall": r_val, "f1": f_val,
        "tp": tp, "fp": fp, "fn": fn,
        "n_pred": len(pred_articles), "n_gt": len(gt_articles),
        "pred_articles": sorted(pred_articles),
        "gt_articles": sorted(gt_articles),
    }

    return {
        "article": article,
        "exact": exact,
        "article_or_desc": article_or_desc,
    }
