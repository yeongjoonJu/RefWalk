from __future__ import annotations

import re
from typing import Optional

import networkx as nx


_RE_ARTICLE_TITLE = re.compile(r"^제(\d+)조(?:의(\d+))?\s*[(（]([^)）]+)[)）]")
_RE_ARTICLE_TAIL = re.compile(r"(제\d+조(?:의\d+)?)$")
_RE_PARA_TAIL = re.compile(r"_제(\d+)항$")


# ─── helpers ────────────────────────────────────────────────────────


def _article_title_from(node_data: dict, node_id: str) -> str:
    """Return '제N조 (제목)' when the title parenthetical is present in the
    first line of text_ko; else fall back to the '제N조' tail from node_id."""
    text_ko = (node_data.get("text_ko") or "").strip()
    first = text_ko.splitlines()[0] if text_ko else ""
    m = _RE_ARTICLE_TITLE.match(first)
    if m:
        num = m.group(1)
        sub = m.group(2)
        title = (m.group(3) or "").strip()
        head = f"제{num}조의{sub}" if sub else f"제{num}조"
        return f"{head} ({title})" if title else head
    tail = _RE_ARTICLE_TAIL.search(node_id)
    return tail.group(1) if tail else node_id


def is_leaf(graph: nx.DiGraph, node_id: str) -> bool:
    """Leaf iff no PART_OF edge is incoming (no child points to it).

    Uses ``graph.predecessors`` because PART_OF edges run child → parent.
    """
    if node_id not in graph.nodes:
        return True
    for pred in graph.predecessors(node_id):
        if graph.edges[pred, node_id].get("type") == "PART_OF":
            return False
    return True


def _part_of_parent(graph: nx.DiGraph, node_id: str) -> Optional[str]:
    """Return the immediate PART_OF parent (via graph successors) or None."""
    if node_id not in graph.nodes:
        return None
    for succ in graph.successors(node_id):
        if graph.edges[node_id, succ].get("type") == "PART_OF":
            return succ
    return None


def _find_article_ancestor(graph: nx.DiGraph, node_id: str) -> Optional[str]:
    """Walk up PART_OF edges to the top of the chain (the 조)."""
    cur = node_id
    for _ in range(10):
        nxt = _part_of_parent(graph, cur)
        if nxt is None:
            return cur
        cur = nxt
    return None


def _paragraph_body_map(articles: list[dict]) -> dict[tuple[str, int], str]:
    """Map ``(article_id, paragraph_num) → 항 body`` — the sub_item.text
    only, *without* the rendered 호 lines. Pulled from the parser's
    structured form on the parent 조 article."""
    out: dict[tuple[str, int], str] = {}
    for a in articles:
        aid = a.get("node_id")
        if not aid:
            continue
        for si in a.get("sub_items", []):
            num = si.get("number")
            if num is None:
                continue
            try:
                key = (aid, int(num))
            except (TypeError, ValueError):
                continue
            out[key] = (si.get("text") or "").strip()
    return out


def _strip_article_header(text_ko: str) -> str:
    """Drop the leading '제N조(제목)' echo so we can re-attach a clean
    article_title without duplication."""
    if not text_ko:
        return ""
    first, sep, rest = text_ko.partition("\n")
    m = _RE_ARTICLE_TITLE.match(first.strip())
    if not m:
        return text_ko.strip()
    tail = first.strip()[m.end():].strip()
    if tail and rest:
        return (tail + "\n" + rest).strip()
    if tail:
        return tail
    return rest.strip()


# ─── Manual (장/절) ─────────────────────────────────────────────────


def _manual_indexed_text(data: dict) -> str:
    hp = data.get("hierarchy_path") or []
    chap_title = (data.get("chapter_title") or "").strip()
    sec_title = (data.get("section_title") or "").strip()
    body = (data.get("text_ko") or "").strip()
    parts: list[str] = []
    if hp:
        head0 = str(hp[0])
        parts.append(f"{head0} ({chap_title})" if chap_title else head0)
        if len(hp) > 1:
            head1 = str(hp[1])
            parts.append(f"{head1} ({sec_title})" if sec_title else head1)
    prefix = " / ".join(parts)
    if prefix and body:
        return f"{prefix}: {body}"
    return prefix or body


# ─── Main composer ─────────────────────────────────────────────────


def build_indexed_text(
    graph: nx.DiGraph,
    node_id: str,
    paragraph_body_map: dict[tuple[str, int], str],
) -> str:
    """Return the ancestor-prefixed text for a leaf node, or '' if non-leaf
    or text unavailable."""
    if not is_leaf(graph, node_id):
        return ""
    nd = graph.nodes[node_id]
    data = nd.get("data") or {}

    if nd.get("authority_level") == "MANUAL":
        return _manual_indexed_text(data)

    self_text = (data.get("text_ko") or "").strip()
    article_id = _find_article_ancestor(graph, node_id)

    # Case 1: self is the article (no 항/호 child; whole article indexed).
    if article_id is None or article_id == node_id:
        art_data = data
        title = _article_title_from(art_data, node_id)
        body = _strip_article_header(self_text)
        if body:
            return f"{title}: {body}"
        return title

    # Case 2: descendant of a 조 node. Look up article title from the 조.
    art_data = graph.nodes[article_id].get("data") or {}
    article_title = _article_title_from(art_data, article_id)

    # Determine parent to decide 호-under-항 vs direct-child layout.
    parent_id = _part_of_parent(graph, node_id)
    para_num: Optional[int] = None
    if parent_id:
        pm = _RE_PARA_TAIL.search(parent_id)
        if pm:
            try:
                para_num = int(pm.group(1))
            except ValueError:
                para_num = None

    if para_num is not None:
        # 호 leaf beneath a 항. Fetch 항 body (text-only, no 호 echoes).
        para_body = paragraph_body_map.get((article_id, para_num), "")
        parts = [article_title]
        if para_body:
            parts.append(para_body)
        if self_text:
            parts.append(self_text)
        return " / ".join(parts)

    # 항 leaf (or direct-호 leaf under article): parent is the 조 itself.
    if self_text:
        return f"{article_title} / {self_text}"
    return article_title


# ─── Graph-wide application ─────────────────────────────────────────


def apply_ancestor_prefix(
    graph: nx.DiGraph, articles: list[dict]
) -> dict:
    """Mutate every node's ``data`` dict in-place, adding:

      - ``data["article_title"]`` — string (derived from the 조 ancestor)
      - ``data["is_leaf"]`` — bool (false iff a child has PART_OF to it)
      - ``data["indexed_text"]`` — ancestor-prefixed text for leaves; '' otherwise

    Non-operational nodes (definition / form / external / synthesized-only
    stubs without text) are still annotated with ``is_leaf`` + empty
    ``indexed_text`` so downstream code can filter uniformly.

    Returns a stats dict for the build report.
    """
    paragraph_body_map = _paragraph_body_map(articles)

    stats = {
        "total": 0,
        "leaves": 0,
        "non_leaves": 0,
        "indexed_set": 0,
        "manual_leaves": 0,
        "empty_leaves": 0,
    }

    for nid in list(graph.nodes):
        nd = graph.nodes[nid]
        stats["total"] += 1
        data = nd.setdefault("data", {}) or {}
        if not isinstance(data, dict):
            # Definition / form nodes sometimes have no data dict.
            data = {}
        nd["data"] = data

        leaf = is_leaf(graph, nid)
        data["is_leaf"] = leaf

        if leaf:
            stats["leaves"] += 1
            article_id = _find_article_ancestor(graph, nid) or nid
            art_data = (
                graph.nodes[article_id].get("data") or {}
                if article_id in graph.nodes
                else {}
            )
            data["article_title"] = _article_title_from(art_data, article_id)
            indexed = build_indexed_text(graph, nid, paragraph_body_map)
            data["indexed_text"] = indexed
            if indexed:
                stats["indexed_set"] += 1
            else:
                stats["empty_leaves"] += 1
            if nd.get("authority_level") == "MANUAL":
                stats["manual_leaves"] += 1
        else:
            stats["non_leaves"] += 1
            data["indexed_text"] = ""
            # Still populate article_title for non-leaf nodes (useful
            # for UIs / lookup_by_node_id).
            article_id = _find_article_ancestor(graph, nid) or nid
            art_data = (
                graph.nodes[article_id].get("data") or {}
                if article_id in graph.nodes
                else {}
            )
            data.setdefault(
                "article_title", _article_title_from(art_data, article_id)
            )

    return stats
