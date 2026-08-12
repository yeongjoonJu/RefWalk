"""Article-level dedup for retrieval output.

After the reranker has scored a candidate pool, multiple candidates from
the same 조 may survive with similar scores (especially when leaf texts
carry an ancestor prefix — they share article context, so cosine +
cross-encoder both give them near-identical scores). Under a top-k budget
we prefer *diversity* across articles to redundancy within one.

Rule: walk the input ranking in order; for each 조-ancestor, keep the
**first** occurrence (i.e., the highest-ranked). Drop subsequent hits
from the same 조. Truncate to ``k``.

The input is any sequence of ``(node_id, score)`` — the common output
shape of every baseline in ``src.baselines.retrieval``. The Walker's
``RetrievalHit`` objects are normalised to tuples upstream in the eval
harness, so a single utility suffices.
"""

from __future__ import annotations

from typing import Sequence

from src.baselines.retrieval import _article_ancestor


def dedup_by_article(
    ranked: Sequence[tuple[str, float]], k: int
) -> list[tuple[str, float]]:
    """Return up to ``k`` (node_id, score) tuples, one per 조-ancestor,
    preserving input order. Ids whose ancestor falls back to themselves
    (non-matching regex) are treated as distinct groups.

    The sort order of the input is respected — higher-scoring tuples must
    already appear first. No internal re-sorting.
    """
    if k <= 0:
        return []
    seen: set[str] = set()
    out: list[tuple[str, float]] = []
    for item in ranked:
        # Defensive: allow bare node_id lists too.
        if isinstance(item, tuple):
            nid = item[0]
        else:
            nid = item
        art = _article_ancestor(nid)
        if art in seen:
            continue
        seen.add(art)
        out.append(item if isinstance(item, tuple) else (nid, 0.0))
        if len(out) >= k:
            break
    return out
