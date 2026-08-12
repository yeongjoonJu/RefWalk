"""Inline reference extraction for OKS/OKG.

Scans NormalizedArticle text_ko and emits InlineReference records
(schema: docs/OpsCompile_Schema_Reference_v1.md §2, §4.1). Each reference
carries its type, the exact raw substring, and character span. Resolution
to target node_ids happens later in `okg.py` once the full article index
is available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional


# ─── Reference types (mirror the Enum in the schema reference) ──────────
REF_INTERNAL = "INTERNAL"
REF_DELEGATION = "DELEGATION"
REF_CROSS_LAW = "CROSS_LAW"
REF_MANUAL_REF = "MANUAL_REF"
REF_APPLY_MUTATIS = "APPLY_MUTATIS"


@dataclass
class InlineReference:
    type: str
    raw_text: str
    span: tuple[int, int]
    # Populated during resolution:
    target_node_id: Optional[str] = None
    # Parsed numeric hints (article/paragraph/sub-paragraph, law name, form id, etc.)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["span"] = list(self.span)
        return d


# ─── Patterns ────────────────────────────────────────────────────────────
# "제N조" / "제N조의M" (optional), optional "제Y항", optional "제Z호"
_ART_CORE = r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?(?:\s*제\s*(\d+)\s*항)?(?:\s*제\s*(\d+)\s*호)?"

# Standalone internal reference: e.g. "제32조에 따라", "제10조 제1항"
# Only match when NOT preceded by "「...」" (that's handled by CROSS_LAW below).
_RE_INTERNAL = re.compile(_ART_CORE)

# Range: "제9조부터 제19조까지"
_RE_RANGE = re.compile(
    r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?\s*부터\s*제\s*(\d+)\s*조(?:\s*의\s*(\d+))?\s*까지"
)

# Cross-law: 「법명」 제N조 (optionally 제Y항 제Z호)
_RE_CROSS_LAW = re.compile(
    r"\u300C([^\u300D]+?)\u300D"  # 「law name」
    r"(?:\s*" + _ART_CORE + r")?"
)

# Reverse-delegation markers inside subordinate docs:
#   "법 제12조제4항" (ref to parent 법)
#   "영 제18조" (ref to parent 시행령)
#   "혁신법 제10조"
_RE_PARENT_LAW = re.compile(r"(?<![\uAC00-\uD7A3])법\s*" + _ART_CORE)
_RE_PARENT_DECREE = re.compile(r"(?<![\uAC00-\uD7A3])영\s*" + _ART_CORE)
_RE_PARENT_RULE = re.compile(r"(?<![\uAC00-\uD7A3])규칙\s*" + _ART_CORE)

# Delegation-out (abstract, target document known but specific article not):
_DELEG_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"대통령령으로\s*정하는"), "국가연구개발혁신법_시행령"),
    (re.compile(r"대통령령으로\s*정한다"), "국가연구개발혁신법_시행령"),
    (re.compile(r"시행령으로\s*정하는"), "국가연구개발혁신법_시행령"),
    (re.compile(r"과학기술정보통신부령으로\s*정하는"), "국가연구개발혁신법_시행규칙"),
    (re.compile(r"과학기술정보통신부령으로\s*정한다"), "국가연구개발혁신법_시행규칙"),
    (
        re.compile(r"과학기술정보통신부장관이\s*정하여\s*고시"),
        None,  # generic NOTICE — target unresolved
    ),
]

# Manual refs / forms
_RE_APPENDIX = re.compile(r"별표\s*(\d+(?:\s*의\s*\d+)?)")
_RE_FORM = re.compile(r"별지\s*(?:서식\s*)?제\s*(\d+(?:\s*의\s*\d+)?)\s*호\s*서식")
_RE_FORM_ALT = re.compile(r"별지\s*제\s*(\d+(?:\s*의\s*\d+)?)\s*호")

# Apply-mutatis: "제X조를 준용한다"
_RE_MUTATIS = re.compile(_ART_CORE + r"(?:\s*를)?\s*준용")


# ─── Context detection ──────────────────────────────────────────────────

def _in_cross_law_span(pos: int, cross_spans: list[tuple[int, int]]) -> bool:
    for s, e in cross_spans:
        if s <= pos < e:
            return True
    return False


def extract_references(text: str) -> list[InlineReference]:
    """Extract all inline references from a single article's text.

    Order of extraction is important: cross-law matches are done first so
    that article-number patterns inside `「...」 제X조` are not double-counted
    as internal references.
    """
    refs: list[InlineReference] = []
    cross_spans: list[tuple[int, int]] = []

    # 1) CROSS_LAW — capture law name + optional article
    for m in _RE_CROSS_LAW.finditer(text):
        law_name = m.group(1).strip()
        art = m.group(2)
        meta = {"law_name": law_name}
        if art:
            meta["article"] = int(art)
            if m.group(3):
                meta["article_sub"] = int(m.group(3))
            if m.group(4):
                meta["paragraph"] = int(m.group(4))
            if m.group(5):
                meta["sub_paragraph"] = int(m.group(5))
        refs.append(
            InlineReference(
                type=REF_CROSS_LAW,
                raw_text=m.group(0),
                span=(m.start(), m.end()),
                meta=meta,
            )
        )
        cross_spans.append((m.start(), m.end()))

    # 2) APPLY_MUTATIS — "제X조를 준용"
    mutatis_spans: list[tuple[int, int]] = []
    for m in _RE_MUTATIS.finditer(text):
        if _in_cross_law_span(m.start(), cross_spans):
            continue
        meta = {
            "article": int(m.group(1)),
        }
        if m.group(2):
            meta["article_sub"] = int(m.group(2))
        if m.group(3):
            meta["paragraph"] = int(m.group(3))
        if m.group(4):
            meta["sub_paragraph"] = int(m.group(4))
        refs.append(
            InlineReference(
                type=REF_APPLY_MUTATIS,
                raw_text=m.group(0),
                span=(m.start(), m.end()),
                meta=meta,
            )
        )
        mutatis_spans.append((m.start(), m.end()))

    # 3) Ranges (expand on resolution): "제9조부터 제19조까지"
    range_spans: list[tuple[int, int]] = []
    for m in _RE_RANGE.finditer(text):
        if _in_cross_law_span(m.start(), cross_spans):
            continue
        meta = {
            "article_from": int(m.group(1)),
            "article_to": int(m.group(3)),
            "range": True,
        }
        if m.group(2):
            meta["article_from_sub"] = int(m.group(2))
        if m.group(4):
            meta["article_to_sub"] = int(m.group(4))
        refs.append(
            InlineReference(
                type=REF_INTERNAL,
                raw_text=m.group(0),
                span=(m.start(), m.end()),
                meta=meta,
            )
        )
        range_spans.append((m.start(), m.end()))

    # 4) Internal single references
    for m in _RE_INTERNAL.finditer(text):
        if _in_cross_law_span(m.start(), cross_spans):
            continue
        if _in_cross_law_span(m.start(), mutatis_spans):
            continue
        if _in_cross_law_span(m.start(), range_spans):
            continue
        # Guard: avoid matching "법 제X조" / "영 제X조" (captured separately)
        prev = text[max(0, m.start() - 2) : m.start()].strip()
        if prev.endswith("법") or prev.endswith("영") or prev.endswith("규칙"):
            continue
        meta = {"article": int(m.group(1))}
        if m.group(2):
            meta["article_sub"] = int(m.group(2))
        if m.group(3):
            meta["paragraph"] = int(m.group(3))
        if m.group(4):
            meta["sub_paragraph"] = int(m.group(4))
        refs.append(
            InlineReference(
                type=REF_INTERNAL,
                raw_text=m.group(0),
                span=(m.start(), m.end()),
                meta=meta,
            )
        )

    # 5) Parent-law / decree / rule reverse references (inside subordinate docs)
    for regex, parent_doc in (
        (_RE_PARENT_LAW, "국가연구개발혁신법"),
        (_RE_PARENT_DECREE, "국가연구개발혁신법_시행령"),
        (_RE_PARENT_RULE, "국가연구개발혁신법_시행규칙"),
    ):
        for m in regex.finditer(text):
            if _in_cross_law_span(m.start(), cross_spans):
                continue
            meta = {
                "parent_document": parent_doc,
                "article": int(m.group(1)),
            }
            if m.group(2):
                meta["article_sub"] = int(m.group(2))
            if m.group(3):
                meta["paragraph"] = int(m.group(3))
            if m.group(4):
                meta["sub_paragraph"] = int(m.group(4))
            refs.append(
                InlineReference(
                    type=REF_DELEGATION,
                    raw_text=m.group(0),
                    span=(m.start(), m.end()),
                    meta=meta,
                )
            )

    # 6) Delegation-out markers (abstract)
    for pat, target_doc in _DELEG_PATTERNS:
        for m in pat.finditer(text):
            if _in_cross_law_span(m.start(), cross_spans):
                continue
            refs.append(
                InlineReference(
                    type=REF_DELEGATION,
                    raw_text=m.group(0),
                    span=(m.start(), m.end()),
                    meta={"abstract": True, "target_document": target_doc},
                )
            )

    # 7) Manual refs: 별표
    for m in _RE_APPENDIX.finditer(text):
        num = m.group(1).replace(" ", "").replace("의", "의")
        refs.append(
            InlineReference(
                type=REF_MANUAL_REF,
                raw_text=m.group(0),
                span=(m.start(), m.end()),
                meta={"appendix": num},
            )
        )

    # 8) Forms: 별지 제X호서식
    form_spans: list[tuple[int, int]] = []
    for m in _RE_FORM.finditer(text):
        refs.append(
            InlineReference(
                type=REF_MANUAL_REF,
                raw_text=m.group(0),
                span=(m.start(), m.end()),
                meta={"form": m.group(1).replace(" ", "")},
            )
        )
        form_spans.append((m.start(), m.end()))
    for m in _RE_FORM_ALT.finditer(text):
        if _in_cross_law_span(m.start(), form_spans):
            continue
        refs.append(
            InlineReference(
                type=REF_MANUAL_REF,
                raw_text=m.group(0),
                span=(m.start(), m.end()),
                meta={"form": m.group(1).replace(" ", "")},
            )
        )

    # Dedupe exact span+type collisions (a pattern may match in multiple
    # groups above, e.g. the reverse-law regex overlaps INTERNAL guards).
    seen: set[tuple[str, int, int]] = set()
    unique: list[InlineReference] = []
    for r in sorted(refs, key=lambda x: (x.span[0], x.span[1])):
        key = (r.type, r.span[0], r.span[1])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


# ─── Definition extraction (used by OKG building) ──────────────────────

# "용어"이란 ... 을 말한다  /  "용어"란 ... 을 말한다  /  "용어"이라 한다
_RE_DEFINITION = re.compile(
    r"[\u201C\u201D\"]([^\u201C\u201D\"]{1,40})[\u201C\u201D\"]"
    r"(?:이란|란|이라\s*한다|이라\s*한다\.)\s*(.+?)(?=(?:\n|$))"
)


def extract_definitions(text: str) -> list[tuple[str, str]]:
    """Return a list of (term, definition_text) found in a 제2조(정의)-style article."""
    results: list[tuple[str, str]] = []
    for m in _RE_DEFINITION.finditer(text):
        term = m.group(1).strip()
        body = m.group(2).strip()
        # Trim trailing "을 말한다." noise for neatness
        body = re.sub(r"(을|를)\s*말한다\.?$", "", body).strip()
        if term and body:
            results.append((term, body))
    return results


def extract_article_references(articles: Iterable[dict]) -> dict[str, list[InlineReference]]:
    """Run extraction over a list of NormalizedArticle dicts."""
    out: dict[str, list[InlineReference]] = {}
    for art in articles:
        refs = extract_references(art["text_ko"])
        out[art["node_id"]] = refs
    return out
