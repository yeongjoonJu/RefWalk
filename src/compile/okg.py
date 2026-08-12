"""OKG construction from NormalizedArticle + InlineReference.

Pipeline:
  1. Build an article index (document_name × article_number → node_id).
  2. Resolve each inline reference to one or more target node_ids.
  3. Add operational / definition / form nodes and edges to a NetworkX
     DiGraph according to docs/OpsCompile_Schema_Reference_v1.md §3.

Edge semantics (Phase A-ext, 6 tiers):
  PART_OF        — structural hierarchy (항 → 조, 호 → 항).
  REFERENCES     — article A cites article B (same doc or cross-law).
  DELEGATES_TO   — higher-authority article delegates detail to lower one.
  SPECIFIES      — inverse of DELEGATES_TO (lower specifies higher).
  DEFINES        — DefinitionNode → article whose term it defines.
  REQUIRES_FORM  — article → form.

Removed in Phase A-ext (0 populate on Korean regulation corpus — future work):
  PRECEDES, USES_SYSTEM, SUPERSEDES.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import networkx as nx

import re

from .reference_extractor import (
    InlineReference,
    extract_definitions,
    extract_references,
    REF_CROSS_LAW,
    REF_DELEGATION,
    REF_INTERNAL,
    REF_MANUAL_REF,
    REF_APPLY_MUTATIS,
)


# ─── Cross-law name mapping (external law → our document_name if in-corpus) ──
_CROSS_LAW_ALIASES: dict[str, str] = {
    "국가연구개발혁신법": "국가연구개발혁신법",
    "혁신법": "국가연구개발혁신법",
    "국가연구개발혁신법 시행령": "국가연구개발혁신법_시행령",
    "국가연구개발혁신법 시행규칙": "국가연구개발혁신법_시행규칙",
    "국가연구개발사업 연구개발비 사용 기준": "국가연구개발사업_연구개발비_사용기준",
    "정보통신·방송 연구개발 관리규정": "정보통신방송_연구개발_관리규정",
    "정보통신방송 연구개발 관리규정": "정보통신방송_연구개발_관리규정",
}


def _article_node_id(doc: str, article: int, article_sub: Optional[int] = None) -> str:
    if article_sub:
        return f"{doc}_제{article}조의{article_sub}"
    return f"{doc}_제{article}조"


def _paragraph_node_id(article_node_id: str, paragraph: int) -> str:
    return f"{article_node_id}_제{paragraph}항"


def _item_node_id(parent_node_id: str, item_number: int) -> str:
    return f"{parent_node_id}_제{item_number}호"


def _render_paragraph_text(sub_item: dict) -> str:
    """Serialize a paragraph sub_item back into readable Korean text."""
    parts: list[str] = []
    head = sub_item.get("text", "").strip()
    if head:
        parts.append(head)
    for sp in sub_item.get("sub_paragraphs", []):
        n = sp.get("number")
        t = sp.get("text", "").strip()
        if n is not None and t:
            parts.append(f"{n}. {t}")
        elif t:
            parts.append(t)
        for it in sp.get("items", []):
            label = it.get("label", "")
            t = it.get("text", "").strip()
            if label and t:
                parts.append(f"  {label}. {t}")
    return "\n".join(parts)


def _render_item_text(sub_paragraph: dict) -> str:
    parts: list[str] = []
    head = sub_paragraph.get("text", "").strip()
    n = sub_paragraph.get("number")
    if head:
        parts.append(f"{n}. {head}" if n is not None else head)
    for it in sub_paragraph.get("items", []):
        label = it.get("label", "")
        t = it.get("text", "").strip()
        if label and t:
            parts.append(f"  {label}. {t}")
    return "\n".join(parts)


@dataclass
class ResolutionStats:
    total: int = 0
    resolved: int = 0
    unresolved: int = 0
    by_type: Counter = field(default_factory=Counter)
    resolved_by_type: Counter = field(default_factory=Counter)

    def record(self, ref_type: str, resolved: bool) -> None:
        self.total += 1
        self.by_type[ref_type] += 1
        if resolved:
            self.resolved += 1
            self.resolved_by_type[ref_type] += 1
        else:
            self.unresolved += 1

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "resolved": self.resolved,
            "unresolved": self.unresolved,
            "by_type": dict(self.by_type),
            "resolved_by_type": dict(self.resolved_by_type),
            "resolution_rate": (
                self.resolved / self.total if self.total else 0.0
            ),
        }


# ─── Resolution ─────────────────────────────────────────────────────────

def _resolve_ref(
    ref: InlineReference,
    source_node_id: str,
    source_doc: str,
    article_index: set[str],
) -> list[str]:
    """Return a list of resolved target node_ids (may be empty)."""
    meta = ref.meta

    if ref.type in (REF_INTERNAL, REF_APPLY_MUTATIS):
        if meta.get("range"):
            lo = meta["article_from"]
            hi = meta["article_to"]
            targets = []
            for n in range(lo, hi + 1):
                cand = _article_node_id(source_doc, n)
                if cand in article_index:
                    targets.append(cand)
            return targets
        art = meta.get("article")
        if art is None:
            return []
        cand = _article_node_id(source_doc, art, meta.get("article_sub"))
        if cand in article_index:
            # Skip self-reference
            return [] if cand == source_node_id else [cand]
        return []

    if ref.type == REF_DELEGATION:
        if meta.get("abstract"):
            return []  # target doc known but article unknown
        parent = meta.get("parent_document")
        art = meta.get("article")
        if not parent or not art:
            return []
        cand = _article_node_id(parent, art, meta.get("article_sub"))
        if cand in article_index:
            return [cand]
        return []

    if ref.type == REF_CROSS_LAW:
        law_name = meta.get("law_name", "")
        doc = _CROSS_LAW_ALIASES.get(law_name.strip())
        if not doc:
            return []
        art = meta.get("article")
        if not art:
            return []
        cand = _article_node_id(doc, art, meta.get("article_sub"))
        if cand in article_index:
            # Skip self-reference
            return [] if cand == source_node_id else [cand]
        return []

    if ref.type == REF_MANUAL_REF:
        if "appendix" in meta:
            cand = f"{source_doc}_별표{meta['appendix']}"
            return [cand] if cand in article_index else []
        if "form" in meta:
            return [f"서식_별지_제{meta['form']}호"]
        return []

    return []


# ─── Graph construction ────────────────────────────────────────────────

@dataclass
class OKGBuildResult:
    graph: nx.DiGraph
    stats: dict
    resolution: ResolutionStats


def _is_definition_article(article: dict) -> bool:
    header = article["text_ko"].splitlines()[0] if article["text_ko"] else ""
    return "(정의)" in header or "정의 " in header or header.endswith("(정의)")


def _add_paragraph_nodes(
    graph: nx.DiGraph, articles: list[dict]
) -> tuple[int, int]:
    """Emit 항 (paragraph) nodes for articles whose sub_items carry numbered
    paragraphs. Each 항 node is PART_OF its parent 조 node.

    Returns (paragraph_nodes_added, part_of_edges_added).
    """
    paragraph_count = 0
    edge_count = 0
    for a in articles:
        article_id = a["node_id"]
        doc = a["document_name"]
        for si in a.get("sub_items", []):
            num = si.get("number")
            if num is None:
                continue
            para_id = _paragraph_node_id(article_id, num)
            if para_id in graph.nodes():
                continue
            text = _render_paragraph_text(si)
            graph.add_node(
                para_id,
                node_type="operational",
                document_name=doc,
                authority_level=a["authority_level"],
                hierarchy_path=a["hierarchy_path"] + [f"제{num}항"],
                effective_date=a.get("effective_date"),
                data={
                    "node_id": para_id,
                    "document_name": doc,
                    "authority_level": a["authority_level"],
                    "hierarchy_path": a["hierarchy_path"] + [f"제{num}항"],
                    "text_ko": text,
                    "sub_items": si.get("sub_paragraphs", []),
                    "parent_article": article_id,
                },
            )
            graph.add_edge(para_id, article_id, type="PART_OF")
            paragraph_count += 1
            edge_count += 1
    return paragraph_count, edge_count


# Parses a FAQ gt_reference into (doc, article, paragraph, item, extra_tail).
# Supports forms like:
#   {doc}_제10조
#   {doc}_제10조_제1항
#   {doc}_제10조_제1항_제3호
#   {doc}_제10조_제5호                    (direct 호 without 항)
#   {doc}_제10조의2_제1항_제3호
#   {doc}_별표1                           (appendix)
#   {doc}_제19조_제3항_별표1_제3호         (nested exotic)
_RE_REF_PARSE = re.compile(
    r"^(?P<doc>.+?)"
    r"(?:_제(?P<art>\d+)(?:의(?P<sub>\d+))?조"
    r"(?:_제(?P<para>\d+)항)?"
    r"(?:_제(?P<item>\d+)호)?"
    r"(?P<tail>.*)?)$"
)

_RE_REF_APPENDIX = re.compile(r"^(?P<doc>.+?)_별표(?P<num>\d+(?:의\d+)?)$")


def _parse_faq_ref(ref: str) -> Optional[dict]:
    m_app = _RE_REF_APPENDIX.match(ref)
    if m_app:
        return {
            "kind": "appendix",
            "doc": m_app.group("doc"),
            "appendix": m_app.group("num"),
        }
    m = _RE_REF_PARSE.match(ref)
    if not m or not m.group("art"):
        return None
    return {
        "kind": "article",
        "doc": m.group("doc"),
        "article": int(m.group("art")),
        "article_sub": int(m.group("sub")) if m.group("sub") else None,
        "paragraph": int(m.group("para")) if m.group("para") else None,
        "item": int(m.group("item")) if m.group("item") else None,
        "tail": m.group("tail") or "",
    }


def _canon_ref(ref: str) -> str:
    return ref.replace("시설장비의_관리등에_관한", "시설장비의_관리_등에_관한")


def _synthesize_from_faq(
    graph: nx.DiGraph,
    articles: list[dict],
    faq_refs: Iterable[str],
) -> dict[str, int]:
    """Ensure every FAQ gt_reference resolves to an existing OKG node.

    Creates 호 nodes (only for refs explicitly demanding them) and
    placeholders for exotic / manual-only refs. Returns a breakdown of what
    was synthesized.
    """
    article_by_id: dict[str, dict] = {a["node_id"]: a for a in articles}
    counts = Counter()

    def _ensure_appendix(doc: str, num: str) -> str:
        node = f"{doc}_별표{num}"
        if node not in graph.nodes():
            graph.add_node(
                node,
                node_type="operational",
                document_name=doc,
                authority_level=None,
                hierarchy_path=[f"별표{num}"],
                effective_date=None,
                has_table=True,
                synthesized=True,
                data={"node_id": node, "document_name": doc, "text_ko": "", "sub_items": []},
            )
            counts["appendix_placeholders"] += 1
        return node

    def _ensure_article_stub(doc: str, article_id: str) -> None:
        if article_id in graph.nodes():
            return
        graph.add_node(
            article_id,
            node_type="operational",
            document_name=doc,
            authority_level=None,
            hierarchy_path=[],
            effective_date=None,
            synthesized=True,
            data={"node_id": article_id, "document_name": doc, "text_ko": "", "sub_items": []},
        )
        counts["article_placeholders"] += 1

    for raw_ref in faq_refs:
        ref = _canon_ref(raw_ref)
        if ref in graph.nodes():
            continue

        parsed = _parse_faq_ref(ref)
        if not parsed:
            # Completely unparseable (e.g. 혁신법_매뉴얼(2025.4)_244p) — stub it.
            graph.add_node(
                ref,
                node_type="external",
                document_name=None,
                synthesized=True,
                data={"node_id": ref, "text_ko": "", "sub_items": []},
            )
            counts["external_stubs"] += 1
            continue

        if parsed["kind"] == "appendix":
            _ensure_appendix(parsed["doc"], parsed["appendix"])
            continue

        doc = parsed["doc"]
        art_id = _article_node_id(doc, parsed["article"], parsed["article_sub"])
        _ensure_article_stub(doc, art_id)

        para_num = parsed["paragraph"]
        item_num = parsed["item"]
        tail = parsed["tail"]

        # Handle the exotic 별표 tail form (e.g. _제19조_제3항_별표1_제3호).
        if tail and "_별표" in tail:
            if ref not in graph.nodes():
                graph.add_node(
                    ref,
                    node_type="operational",
                    document_name=doc,
                    authority_level=None,
                    hierarchy_path=[],
                    effective_date=None,
                    synthesized=True,
                    has_table=True,
                    data={"node_id": ref, "document_name": doc, "text_ko": "", "sub_items": []},
                )
                parent = art_id
                if para_num:
                    parent = _paragraph_node_id(art_id, para_num)
                    if parent not in graph.nodes():
                        graph.add_node(
                            parent,
                            node_type="operational",
                            document_name=doc,
                            authority_level=None,
                            hierarchy_path=[],
                            synthesized=True,
                            data={"node_id": parent, "document_name": doc, "text_ko": "", "sub_items": []},
                        )
                        graph.add_edge(parent, art_id, type="PART_OF")
                        counts["paragraph_placeholders"] += 1
                graph.add_edge(ref, parent, type="PART_OF")
                counts["exotic_tail_placeholders"] += 1
            continue

        # Case A: _제N조_제M항 (no 호) — paragraph node. Should already exist
        # if _add_paragraph_nodes found it; otherwise synthesize a placeholder.
        if para_num is not None and item_num is None:
            para_id = _paragraph_node_id(art_id, para_num)
            if para_id not in graph.nodes():
                graph.add_node(
                    para_id,
                    node_type="operational",
                    document_name=doc,
                    authority_level=None,
                    hierarchy_path=[],
                    synthesized=True,
                    data={"node_id": para_id, "document_name": doc, "text_ko": "", "sub_items": []},
                )
                graph.add_edge(para_id, art_id, type="PART_OF")
                counts["paragraph_placeholders"] += 1
            continue

        # Case B: _제N조_제M항_제K호 — item inside a paragraph.
        if para_num is not None and item_num is not None:
            para_id = _paragraph_node_id(art_id, para_num)
            if para_id not in graph.nodes():
                # Synthesize the paragraph first so PART_OF chain is intact.
                graph.add_node(
                    para_id,
                    node_type="operational",
                    document_name=doc,
                    authority_level=None,
                    hierarchy_path=[],
                    synthesized=True,
                    data={"node_id": para_id, "document_name": doc, "text_ko": "", "sub_items": []},
                )
                graph.add_edge(para_id, art_id, type="PART_OF")
                counts["paragraph_placeholders"] += 1

            item_id = _item_node_id(para_id, item_num)
            if item_id not in graph.nodes():
                # Try to pull the real text from the parsed article's sub_items.
                text = ""
                src = article_by_id.get(art_id, {})
                for si in src.get("sub_items", []):
                    if si.get("number") == para_num:
                        for sp in si.get("sub_paragraphs", []):
                            if sp.get("number") == item_num:
                                text = _render_item_text(sp)
                                break
                        break
                graph.add_node(
                    item_id,
                    node_type="operational",
                    document_name=doc,
                    authority_level=None,
                    hierarchy_path=[],
                    data={
                        "node_id": item_id,
                        "document_name": doc,
                        "text_ko": text,
                        "sub_items": [],
                        "parent_paragraph": para_id,
                    },
                )
                graph.add_edge(item_id, para_id, type="PART_OF")
                counts["item_nodes"] += 1
            continue

        # Case C: _제N조_제K호 (direct 호 without 항) — items at article root.
        if para_num is None and item_num is not None:
            item_id = _item_node_id(art_id, item_num)
            if item_id not in graph.nodes():
                text = ""
                src = article_by_id.get(art_id, {})
                # Direct-호 articles have one sub_item with num=None and all
                # items in sub_paragraphs.
                for si in src.get("sub_items", []):
                    if si.get("number") is None:
                        for sp in si.get("sub_paragraphs", []):
                            if sp.get("number") == item_num:
                                text = _render_item_text(sp)
                                break
                        break
                graph.add_node(
                    item_id,
                    node_type="operational",
                    document_name=doc,
                    authority_level=None,
                    hierarchy_path=[],
                    data={
                        "node_id": item_id,
                        "document_name": doc,
                        "text_ko": text,
                        "sub_items": [],
                        "parent_article": art_id,
                    },
                )
                graph.add_edge(item_id, art_id, type="PART_OF")
                counts["item_nodes"] += 1
            continue

    return dict(counts)


def build_okg(articles: list[dict], faq_refs: Optional[Iterable[str]] = None) -> OKGBuildResult:
    """Build the OKG from NormalizedArticle records.

    Returns the populated DiGraph, raw stats, and resolution statistics.
    Operational nodes carry the full article dict under `data`; the node
    type is tracked via the `node_type` attribute so downstream code can
    filter (operational | definition | form | system).
    """
    graph = nx.DiGraph()
    article_index: set[str] = {a["node_id"] for a in articles}
    article_by_id: dict[str, dict] = {a["node_id"]: a for a in articles}

    # ── Operational nodes ──
    for a in articles:
        graph.add_node(
            a["node_id"],
            node_type="operational",
            document_name=a["document_name"],
            authority_level=a["authority_level"],
            hierarchy_path=a["hierarchy_path"],
            effective_date=a.get("effective_date"),
            has_table=a.get("has_table", False),
            data=a,
        )

    # ── Reference extraction + resolution ──
    stats = ResolutionStats()
    edge_counter: Counter = Counter()
    definition_nodes_added = 0
    form_nodes: set[str] = set()

    # Also store extracted refs back onto node for persistence/debug
    for a in articles:
        source_id = a["node_id"]
        source_doc = a["document_name"]
        try:
            refs = extract_references(a["text_ko"])
        except Exception as e:
            print(e)
            print(a)
        ref_payload: list[dict] = []
        for ref in refs:
            targets = _resolve_ref(ref, source_id, source_doc, article_index)
            # Forms are synthesized nodes (not in article_index); allow pass-through
            if not targets and ref.type == REF_MANUAL_REF and "form" in ref.meta:
                targets = [f"서식_별지_제{ref.meta['form']}호"]
            resolved = bool(targets)
            stats.record(ref.type, resolved)
            if resolved:
                ref.target_node_id = targets[0]
            ref_payload.append(ref.to_dict() | {"resolved_targets": targets})

            # ── Add edges ──
            for tgt in targets:
                if ref.type == REF_INTERNAL or ref.type == REF_APPLY_MUTATIS:
                    edge_type = "REFERENCES"
                elif ref.type == REF_CROSS_LAW:
                    edge_type = "REFERENCES"
                elif ref.type == REF_DELEGATION:
                    # Lower → higher: SPECIFIES; reciprocal DELEGATES_TO.
                    edge_type = "SPECIFIES"
                    if not graph.has_edge(tgt, source_id):
                        graph.add_edge(tgt, source_id, type="DELEGATES_TO")
                        edge_counter["DELEGATES_TO"] += 1
                elif ref.type == REF_MANUAL_REF:
                    if "form" in ref.meta:
                        if tgt not in form_nodes:
                            graph.add_node(
                                tgt,
                                node_type="form",
                                form_number=f"별지 제{ref.meta['form']}호서식",
                                form_name=None,
                            )
                            form_nodes.add(tgt)
                        edge_type = "REQUIRES_FORM"
                    else:
                        edge_type = "REFERENCES"
                else:
                    edge_type = "REFERENCES"

                if source_id == tgt:
                    continue
                if not graph.has_edge(source_id, tgt):
                    graph.add_edge(source_id, tgt, type=edge_type)
                    edge_counter[edge_type] += 1

        graph.nodes[source_id]["references"] = ref_payload

    # ── Definition nodes (from 제2조) ──
    for a in articles:
        if not _is_definition_article(a):
            continue
        for term, body in extract_definitions(a["text_ko"]):
            def_id = f"{a['node_id']}_{term}"
            graph.add_node(
                def_id,
                node_type="definition",
                term=term,
                definition=body,
                source_node_id=a["node_id"],
            )
            graph.add_edge(def_id, a["node_id"], type="DEFINES")
            edge_counter["DEFINES"] += 1
            definition_nodes_added += 1

    # ── Paragraph-level nodes (항) ──
    paragraph_nodes_added, part_of_added = _add_paragraph_nodes(graph, articles)
    edge_counter["PART_OF"] += part_of_added

    # ── FAQ-driven synthesis ──
    synthesized = {}
    if faq_refs is not None:
        synthesized = _synthesize_from_faq(graph, articles, faq_refs)
        # Re-count PART_OF edges that synthesis added
        edge_counter_reset = Counter()
        for _, _, d in graph.edges(data=True):
            edge_counter_reset[d.get("type")] += 1
        edge_counter = edge_counter_reset

    summary = {
        "article_nodes": len(article_index),
        "paragraph_nodes": paragraph_nodes_added,
        "definition_nodes": definition_nodes_added,
        "form_nodes": len(form_nodes),
        "operational_nodes": sum(
            1 for _, d in graph.nodes(data=True) if d.get("node_type") == "operational"
        ),
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "edges_by_type": dict(edge_counter),
        "faq_synthesized": synthesized,
    }
    return OKGBuildResult(graph=graph, stats=summary, resolution=stats)


# ─── Persistence ───────────────────────────────────────────────────────

def save_okg(graph: nx.DiGraph, out_dir: Path, suffix: str = "") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / f"okg{suffix}.gpickle"
    with open(pkl_path, "wb") as f:
        pickle.dump(graph, f)

    nodes_path = out_dir / f"okg_nodes{suffix}.jsonl"
    with open(nodes_path, "w", encoding="utf-8") as f:
        for node_id, data in graph.nodes(data=True):
            payload = {"node_id": node_id, **_json_safe(data)}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    edges_path = out_dir / f"okg_edges{suffix}.jsonl"
    with open(edges_path, "w", encoding="utf-8") as f:
        for u, v, data in graph.edges(data=True):
            f.write(
                json.dumps(
                    {"source": u, "target": v, **_json_safe(data)},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return {
        "graph_pickle": str(pkl_path),
        "nodes_jsonl": str(nodes_path),
        "edges_jsonl": str(edges_path),
    }


def _json_safe(data: dict) -> dict:
    def conv(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return str(v)
    return {k: conv(v) for k, v in data.items()}


# ─── FAQ validation ────────────────────────────────────────────────────

_RE_ARTICLE_ANCESTOR = re.compile(r"^(.*?_제\d+(?:의\d+)?조)")


def _article_ancestor(node_id: str) -> str:
    """Map a paragraph/item node_id back to its parent 조-level node_id.

    Used for hop_count computation: REFERENCES/DELEGATES_TO/SPECIFIES edges
    live at the article level, so paragraph/item refs must be projected up
    to the article node before pairwise shortest-path.
    """
    m = _RE_ARTICLE_ANCESTOR.match(node_id)
    return m.group(1) if m else node_id


def validate_faq(
    graph: nx.DiGraph,
    faq: list,
    reference_edges: tuple[str, ...] = (
        "REFERENCES",
        "DELEGATES_TO",
        "SPECIFIES",
    ),
) -> dict:
    """For each FAQ entry, check gt_references coverage and hop_count.

    hop_count and has_chain are computed over REFERENCES/DELEGATES_TO/
    SPECIFIES only — PART_OF is a structural (hierarchy) relation, not a
    reasoning step, so it is excluded. Paragraph/item refs are projected
    up to their parent 조 node before the shortest-path search, because
    reference edges live at the article level.

    Parallel-vs-chain heuristic:
      - plen == 1: direct 1-hop REFERENCES edge → parallel (sibling cite).
      - plen == 2 via an intermediate that is *also* an article_ref →
        chain (A → mid → B where mid is itself part of the answer).
      - plen == 2 via an intermediate that is a **shared hub** (it
        connects directly to ≥2 refs in the set) → parallel. This
        catches cases like `제39/48/57/65조` under a common parent
        `제80조 (인건비 총칙)` which lists all institution-type articles.
      - plen ≥ 3 → chain.
    """

    # Undirected projection restricted to the edge types we care about
    sub = nx.Graph()
    sub.add_nodes_from(graph.nodes())
    for u, v, d in graph.edges(data=True):
        if d.get("type") in reference_edges:
            sub.add_edge(u, v)

    per_qa: list[dict] = []
    difficulty_counter: Counter = Counter()
    total_refs = 0
    matched_refs = 0
    for qa in faq:
        refs = qa.get("gt_references", [])
        total_refs += len(refs)
        resolved: list[str] = []
        missing: list[str] = []
        for r in refs:
            canonical = _canon_ref(r)
            if canonical in graph.nodes():
                resolved.append(canonical)
                matched_refs += 1
            else:
                missing.append(r)

        # Project every resolved ref up to its 조-level ancestor and dedupe.
        article_refs: list[str] = []
        seen_art: set[str] = set()
        for r in resolved:
            art = _article_ancestor(r)
            if art not in seen_art:
                seen_art.add(art)
                article_refs.append(art)

        article_ref_set = set(article_refs)
        hop_count = 1
        pairs_reachable = 0
        pairs_total = 0
        direct_only_pairs = 0
        hub_pairs = 0
        chain_pairs = 0
        if len(article_refs) >= 2:
            for i in range(len(article_refs)):
                for j in range(i + 1, len(article_refs)):
                    a, b = article_refs[i], article_refs[j]
                    pairs_total += 1
                    try:
                        plen = nx.shortest_path_length(sub, a, b)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                    pairs_reachable += 1
                    if plen > hop_count:
                        hop_count = plen
                    if plen == 1:
                        direct_only_pairs += 1
                        continue
                    if plen == 2:
                        via_hub = False
                        for c in nx.common_neighbors(sub, a, b):
                            if c in article_ref_set:
                                continue
                            hub_deg = sum(
                                1 for r in article_refs if sub.has_edge(c, r)
                            )
                            if hub_deg >= 2:
                                via_hub = True
                                break
                        if via_hub:
                            hub_pairs += 1
                            continue
                    chain_pairs += 1

        has_chain = chain_pairs > 0
        cross_document = (
            len({graph.nodes[r].get("document_name") for r in article_refs}) >= 2
        )

        conditional = bool(qa.get("conditional"))
        if conditional and (hop_count >= 3 or cross_document):
            difficulty = "L4"
        elif cross_document or hop_count >= 4:
            difficulty = "L3"
        elif has_chain or conditional:
            difficulty = "L2"
        else:
            difficulty = "L1"
        difficulty_counter[difficulty] += 1

        per_qa.append(
            {
                "qa_id": qa.get("qa_id"),
                "gt_references": refs,
                "resolved": resolved,
                "article_refs": article_refs,
                "missing": missing,
                "hop_count": hop_count,
                "pairs_total": pairs_total,
                "pairs_reachable": pairs_reachable,
                "direct_only_pairs": direct_only_pairs,
                "hub_pairs": hub_pairs,
                "chain_pairs": chain_pairs,
                "cross_document": cross_document,
                "has_chain": has_chain,
                "conditional": conditional,
                "difficulty": difficulty,
            }
        )

    return {
        "summary": {
            "qa_count": len(faq),
            "total_refs": total_refs,
            "matched_refs": matched_refs,
            "coverage": matched_refs / total_refs if total_refs else 0.0,
            "difficulty_distribution": dict(difficulty_counter),
        },
        "per_qa": per_qa,
    }
