from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


# ─── Data classes ────────────────────────────────────────────────────


@dataclass
class Corpus:
    node_ids: list[str]
    texts: list[str]
    id_to_idx: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}


# ─── Header helpers (used by okg_nodes path_prepend mode) ──────────


def _readable_doc_name(name: str) -> str:
    return (name or "").replace("_", " ").strip()


def _prepend_header(document_name: str, hierarchy_path: list | None) -> str:
    parts = [_readable_doc_name(document_name)]
    if hierarchy_path:
        parts.append(" ".join(str(p) for p in hierarchy_path))
    header = " ".join(p for p in parts if p).strip()
    return f"[{header}]" if header else ""


# ─── load_corpus ────────────────────────────────────────────────────


def load_corpus(
    articles_path: Path | None = None,
    source: str = "parse",
    path_prepend: bool = True,
    use_indexed_text: bool = False,
) -> Corpus:
    """Load retrieval corpus.

    source="parse" (legacy): read parse-stage articles.jsonl
        (조-level + 별표). Each row's ``text_ko`` already contains all
        sub-paragraphs/items concatenated.
    source="okg_nodes": read data/okg/okg_nodes.jsonl (조/항/호
        decomposed). Nodes without text_ko are filtered out.
        If path_prepend=True, prefix each text with a single header
        line "[{document_name_readable} {hierarchy_joined}]\\n".
        Header is retrieval-only; ``lookup_by_node_id`` returns its
        own header independently.
    use_indexed_text=True (Phase B.5.1b): when the OKG was built with
        ``apply_ancestor_prefix``, each leaf carries
        ``data.indexed_text`` (ancestor-prefixed: article title / 항
        body / self) and ``data.is_leaf``. In this mode non-leaf
        nodes are filtered out and ``indexed_text`` replaces
        ``text_ko``; ``path_prepend`` is implicitly disabled (the
        prefix is already inside the text).
    """
    if source == "parse":
        if articles_path is None:
            raise ValueError("source='parse' requires articles_path")
        node_ids, texts = [], []
        seen_total = 0
        with open(articles_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                art = json.loads(line)
                seen_total += 1
                text = (art.get("text_ko") or "").strip()
                if not text:
                    continue
                node_ids.append(art["node_id"])
                texts.append(text)
        if not node_ids:
            raise ValueError(
                f"load_corpus produced an empty corpus from {articles_path} "
                f"(mode=source='parse'). seen_total={seen_total}, "
                f"non-empty text_ko=0. "
                "Hint: verify the file is a parse-stage articles jsonl with "
                "non-empty text_ko per row (e.g. data/parsed/articles_*.jsonl)."
            )
        return Corpus(node_ids=node_ids, texts=texts)

    if source == "okg_nodes":
        default_path = Path("data/okg/okg_nodes.jsonl")
        nodes_path = articles_path or default_path
        node_ids, texts = [], []
        # Counters drive a friendly diagnostic when the filters drain
        # the corpus to zero — otherwise BM25Okapi raises a cryptic
        # ZeroDivisionError several frames downstream.
        seen_total = 0
        leaf_seen = 0
        indexed_seen = 0
        textko_seen = 0
        with open(nodes_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                n = json.loads(line)
                data = n.get("data") or {}
                seen_total += 1
                if data.get("is_leaf") is True:
                    leaf_seen += 1
                if (data.get("indexed_text") or "").strip():
                    indexed_seen += 1
                if (data.get("text_ko") or "").strip():
                    textko_seen += 1

                if use_indexed_text:
                    if data.get("is_leaf") is False:
                        continue
                    text = (data.get("indexed_text") or "").strip()
                    if not text:
                        continue
                    node_ids.append(n["node_id"])
                    texts.append(text)
                    continue

                text_ko = (data.get("text_ko") or "").strip()
                if not text_ko:
                    continue
                if path_prepend:
                    header = _prepend_header(
                        data.get("document_name") or n.get("document_name") or "",
                        data.get("hierarchy_path") or n.get("hierarchy_path") or [],
                    )
                    text_ko = f"{header}\n{text_ko}" if header else text_ko
                node_ids.append(n["node_id"])
                texts.append(text_ko)

        if not node_ids:
            mode = "use_indexed_text=True" if use_indexed_text else "text_ko"
            raise ValueError(
                f"load_corpus produced an empty corpus from {nodes_path} "
                f"(mode={mode}). seen_total={seen_total}, "
                f"is_leaf=True rows={leaf_seen}, "
                f"non-empty indexed_text={indexed_seen}, "
                f"non-empty text_ko={textko_seen}. "
                + (
                    "Hint: this articles file lacks ancestor-prefix metadata "
                    "(re-run cli/run_build_okg.py without --no-ancestor-prefix, "
                    "or point --articles at the file matching your --okg, "
                    "e.g. data/benchmark/okg_nodes_aug.jsonl)."
                    if use_indexed_text and indexed_seen == 0
                    else "Hint: every row had empty text_ko — check the source jsonl."
                )
            )
        return Corpus(node_ids=node_ids, texts=texts)

    raise ValueError(f"unknown source={source!r}; must be 'parse' or 'okg_nodes'")


def load_faq(faq_path: Path) -> list[dict]:
    with open(faq_path, encoding="utf-8") as f:
        return json.load(f)


# ─── node_id helpers (used by metrics + retrieval code) ────────────


_TYPO_FIXES: tuple[tuple[str, str], ...] = (
    # 시설장비 표준지침: missing underscore between 관리/등에.
    ("시설장비의_관리등에_관한", "시설장비의_관리_등에_관한"),
    # 정보통신방송 관리규정: middle-dot variant in some refs.
    ("정보통신·방송_연구개발_관리규정", "정보통신방송_연구개발_관리규정"),
)


def _canon_ref(ref: str) -> str:
    """Normalise legacy typos in node_id-style refs.

    Applied wherever a ``gt_references`` / ``anchor_node`` / OKG node_id
    is consumed downstream; the OKG itself uses the canonical (right-
    hand) forms. Adding a new typo? Append to ``_TYPO_FIXES``.
    """
    out = ref
    for old, new in _TYPO_FIXES:
        out = out.replace(old, new)
    return out


_RE_ARTICLE_ANCESTOR = re.compile(r"^(.*?_제\d+(?:의\d+)?조)")


def _article_ancestor(node_id: str) -> str:
    """Map any 항/호/별표-level node_id to its 조-level ancestor."""
    m = _RE_ARTICLE_ANCESTOR.match(node_id)
    return m.group(1) if m else node_id


_RE_PARAGRAPH_ANCESTOR = re.compile(r"^(.*?_제\d+(?:의\d+)?조(?:_제\d+(?:의\d+)?항)?)")


def _paragraph_ancestor(node_id: str) -> str:
    """Map a 호-level node_id to its 항-level ancestor.

    The middle rung between :func:`_article_ancestor` (조) and verbatim
    ids. Truncates below 항 and leaves everything at or above it alone,
    so a 조-level id stays 조-level — an id is never pushed *down* to a
    granularity it doesn't have::

        A_제48조             → A_제48조
        A_제48조_제8항        → A_제48조_제8항
        A_제48조_제8항_제1호   → A_제48조_제8항
    """
    m = _RE_PARAGRAPH_ANCESTOR.match(node_id)
    return m.group(1) if m else node_id
