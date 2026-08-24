from __future__ import annotations

import re

# Heuristic: authority-level classification from node_id prefix.
# For the Korean R&D corpus we only have 7 document names; each maps
# deterministically to an authority level (checked against
# data/okg/okg_nodes.jsonl catalog). Using a prefix table avoids
# needing the graph at eval time.
_DOC_AUTHORITY: list[tuple[str, str]] = [
    ("국가연구개발혁신법_매뉴얼_2025", "MANUAL"),
    ("국가연구개발_시설장비의_관리_등에_관한_표준지침", "NOTICE"),
    ("국가연구개발사업_연구개발비_사용기준", "NOTICE"),
    ("정보통신방송_연구개발_관리규정", "NOTICE"),
    ("국가연구개발혁신법_시행령", "DECREE"),
    ("국가연구개발혁신법_시행규칙", "RULE"),
    ("국가연구개발혁신법", "STATUTE"),
    ("과학기술기본법_시행령", "DECREE"),
    ("과학기술기본법_시행규칙", "RULE"),
    ("과학기술기본법", "STATUTE"),
]

_PRIMARY_LEVELS = {"STATUTE", "DECREE", "RULE", "NOTICE"}

_CAVEAT_TOKEN_RE = re.compile(
    r"(CAVEAT|caveat|주의|매뉴얼만|매뉴얼\s*근거|disclaimer|caveats?)",
    re.I,
)
_FALLBACK_SENTINEL_RE = re.compile(r"\[planner-fallback\]")


def classify_authority(node_id: str, language: str = "ko") -> str:
    """Map a node_id to an authority level.

    For ``language="ko"`` (RegOps), uses the Korean R&D document prefix
    table. For ``language="en"``, every non-empty node_id maps to ``RULE``:
    an English rule corpus carries no MANUAL/STATUTE/DECREE distinction to
    key off, so the authority dimension collapses to a single level.
    """
    if not node_id:
        return "UNKNOWN"
    if language == "en":
        return "RULE"
    for prefix, level in _DOC_AUTHORITY:
        if node_id.startswith(prefix):
            return level
    return "UNKNOWN"


def detect_fallback(run_row: dict, language: str = "ko") -> dict:
    """Classify one run. `run_row` is a parsed row from runs.jsonl.

    Fields consumed:
      - cited_references: list[str]
      - caveats_count: int
      - answer_preview / answer: str
      - trajectory_actions: list[dict]  (for planner-fallback sentinel)
      - planner_error: str | None

    ``language`` controls authority classification — see
    :func:`classify_authority`.
    """
    # Mirror ``src.eval.citation_metrics.score_citations``: prefer
    # ``cited_references`` but fall back to ``cited_node_ids`` for
    # baselines (LightRAG, HippoRAG2, …) that emit the latter. Without
    # this fallback the detector saw an empty list for those baselines
    # and classified every row as ``citation_class="none"``.
    cited = list(
        run_row.get("cited_references")
        or run_row.get("cited_node_ids")
        or []
    )
    levels = [classify_authority(c, language=language) for c in cited]
    level_set = set(levels) - {"UNKNOWN"}

    n_primary = sum(1 for lv in levels if lv in _PRIMARY_LEVELS)
    n_manual = sum(1 for lv in levels if lv == "MANUAL")

    if not cited:
        citation_class = "none"
    elif n_primary > 0 and n_manual > 0:
        citation_class = "mixed_manual"
    elif n_primary > 0:
        citation_class = "primary"
    elif n_manual > 0:
        citation_class = "manual_only"
    else:
        citation_class = "other"

    answer_text = run_row.get("answer") or run_row.get("answer_preview") or ""
    has_caveat_field = int(run_row.get("caveats_count", 0) or 0) > 0
    has_caveat_token = bool(_CAVEAT_TOKEN_RE.search(answer_text))
    has_caveat = has_caveat_field or has_caveat_token

    # Degraded stop: planner-fallback sentinel in trajectory reflections
    # or any planner_error; or answer non-empty but empty citations.
    planner_error = (run_row.get("planner_error") or "") or ""
    degraded_stop = bool(_FALLBACK_SENTINEL_RE.search(planner_error))
    # Scan answer too (v7 prompt may leak sentinel through caveats).
    if _FALLBACK_SENTINEL_RE.search(answer_text):
        degraded_stop = True
    if not cited and (run_row.get("answer_len", 0) or 0) > 0:
        degraded_stop = True

    manual_fallback_ok = citation_class == "manual_only" and has_caveat
    manual_fallback_violation = citation_class == "manual_only" and not has_caveat

    return {
        "citation_class": citation_class,
        "n_primary": n_primary,
        "n_manual": n_manual,
        "authority_levels": sorted(level_set),
        "has_caveat": has_caveat,
        "degraded_stop": degraded_stop,
        "manual_fallback_ok": manual_fallback_ok,
        "manual_fallback_violation": manual_fallback_violation,
    }


def aggregate_fallback(results: list[dict]) -> dict:
    """Aggregate a list of per-run detect_fallback() results."""
    n = len(results)
    if n == 0:
        return {"n": 0}
    c = lambda key: sum(1 for r in results if r.get(key))
    cls_counts: dict[str, int] = {}
    for r in results:
        k = r.get("citation_class", "unknown")
        cls_counts[k] = cls_counts.get(k, 0) + 1
    return {
        "n": n,
        "citation_class_counts": cls_counts,
        "rate_primary": cls_counts.get("primary", 0) / n,
        "rate_manual_only": cls_counts.get("manual_only", 0) / n,
        "rate_mixed_manual": cls_counts.get("mixed_manual", 0) / n,
        "rate_none": cls_counts.get("none", 0) / n,
        "rate_caveat": c("has_caveat") / n,
        "rate_degraded_stop": c("degraded_stop") / n,
        "rate_manual_fallback_ok": c("manual_fallback_ok") / n,
        "rate_manual_fallback_violation": c("manual_fallback_violation") / n,
    }
