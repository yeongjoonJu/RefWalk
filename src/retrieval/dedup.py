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
