from __future__ import annotations

import random
from typing import Optional


def _rollup(node_id: str, valid: set[str]) -> Optional[str]:
    """Longest underscore-ancestor of ``node_id`` present in ``valid``."""
    if node_id in valid:
        return node_id
    segs = node_id.split("_")
    while len(segs) > 1:
        segs.pop()
        candidate = "_".join(segs)
        if candidate in valid:
            return candidate
    return None


def resolve_gt_ids(gt_references, valid: set[str]) -> list[str]:
    """Resolve + de-duplicate (order-preserving) gt refs to corpus ids."""
    out: list[str] = []
    seen: set[str] = set()
    for ref in gt_references or []:
        if not isinstance(ref, str):
            continue
        r = _rollup(ref.strip(), valid)
        if r is None or r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


def one_hop_distractor_pool(
    gt_ids: list[str], graph, valid: set[str]
) -> list[str]:
    """Corpus articles 1 OKG hop from any GT article, minus the GT set.

    Both edge directions count as a hop. Neighbours are rolled up to
    valid corpus ids and de-duplicated; GT articles are excluded so a
    distractor is always a genuinely different article.
    """
    gt_set = set(gt_ids)
    pool: list[str] = []
    seen: set[str] = set()
    for g in gt_ids:
        if g not in graph:
            continue
        neighbours = set(graph.successors(g)) | set(graph.predecessors(g))
        for nb in neighbours:
            r = _rollup(str(nb), valid)
            if r is None or r in gt_set or r in seen:
                continue
            seen.add(r)
            pool.append(r)
    return pool


def build_oracle_context(
    gt_references,
    graph,
    valid: set[str],
    *,
    total_k: int = 10,
    seed: int = 42,
    qa_id: str = "",
    fill_from_corpus: bool = True,
) -> tuple[list[str], set[str], set[str]]:
    """Return ``(ordered_ids, gt_set, distractor_set)``.

    ``ordered_ids`` is the GT articles plus ``total_k - len(gt)``
    distractors, shuffled. Distractors are drawn **1-hop OKG neighbours
    of the GT first**; when that pool is exhausted before reaching the
    budget the remainder is filled with random corpus articles (still
    excluding GT and already-chosen distractors) so the context always
    contains exactly ``total_k`` passages. Set ``fill_from_corpus=False``
    to keep the strict 1-hop-only behaviour (context then shorter when
    the 1-hop pool is small).

    When GT alone already meets/exceeds ``total_k`` no distractors are
    added and **all** GT articles are kept (GT is never truncated — it
    must stay in context for a faithful generation measurement).

    Randomness is seeded by ``f"{seed}:{qa_id}"`` so the same row yields
    the same context across resume and re-runs, independent of row order.
    """
    gt_ids = resolve_gt_ids(gt_references, valid)
    rng = random.Random(f"{seed}:{qa_id}")

    n_distract = max(0, int(total_k) - len(gt_ids))
    distractors: list[str] = []
    if n_distract > 0:
        pool = one_hop_distractor_pool(gt_ids, graph, valid)
        rng.shuffle(pool)
        distractors = pool[:n_distract]

        if fill_from_corpus and len(distractors) < n_distract:
            # 1-hop pool exhausted — top up with random corpus articles.
            excluded = set(gt_ids) | set(distractors)
            remaining = sorted(valid - excluded)
            rng.shuffle(remaining)
            distractors.extend(
                remaining[: n_distract - len(distractors)]
            )

    combined = gt_ids + distractors
    rng.shuffle(combined)
    return combined, set(gt_ids), set(distractors)
