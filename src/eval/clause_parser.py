"""Clause-level (항/호) citation extraction, fair across output formats.

Article-level citation F1 rolls every reference up to its 조 ancestor, which
hides whether a system found the *right paragraph*. Scoring at full
granularity instead is only meaningful if every system gets credit for the
clauses it actually names — and systems name them in different places:

  * A system emitting structured output puts the clause in a bracketed tag
    inside its per-article payload (``"... [제8항] ..."``), not in its
    citation key list, which stays 조-level.
  * A free-form system names the clause in prose (``"제48조제8항에 따라 …"``)
    and its citation list is likewise 조-level.

Scoring either one against its citation list alone measures output format,
not grounding. This module recovers the clause ids from both shapes so the
strict metric compares grounding.

Dispatch is on the record's shape, never on a system name:

  * ``raw_output`` present (a dict keyed by 조-level node_id) → read the
    bracketed tags in its values.
  * otherwise → parse the answer prose, grounding each 조 number to a law
    prefix through the ids the system itself cited or retrieved.

Ported from the E13 / E1 rebuttal analyses.
"""

from __future__ import annotations

import re

from src.retrieval.corpus import _canon_ref


# A node_id is clause-level (항 or 호) iff it carries one of these tokens.
CLAUSE_RE = re.compile(r"_제\d+(?:의\d+)?항|_제\d+(?:의\d+)?호")


def is_clause_id(node_id: str) -> bool:
    """True for 항/호-level ids; False for 조-level and 별표 ids."""
    return bool(CLAUSE_RE.search(node_id))


# ─── Bracketed-tag grammar (structured output) ─────────────────────

# Anchored so only tags that *start* with a citation token are accepted.
# Appendix prose such as "[가.가중기준 1) 법 제32조...]" starts with "가." and
# is rejected, which keeps stray article numbers out of the cited set.
_TAG_JO = re.compile(
    r"^제(\d+)조(?:의(\d+))?(?:제(\d+)(?:의\d+)?항)?(?:제(\d+)(?:의\d+)?호)?"
)
_TAG_HANG = re.compile(r"^제(\d+)(?:의\d+)?항(?:\s*제(\d+)(?:의\d+)?호)?")
_TAG_HO = re.compile(r"^제(\d+)(?:의\d+)?호")
_BRACKET = re.compile(r"\[([^\]]{1,25})\]")


def law_prefix(node_id: str) -> str:
    """Everything before the first ``_제N조`` / ``_별표`` token."""
    m = re.match(r"^(.*?)_제\d+", node_id)
    if m:
        return m.group(1)
    m = re.match(r"^(.*?)_별표", node_id)
    return m.group(1) if m else node_id


def tag_clause_ids(raw_output: dict) -> set[str]:
    """Clause ids named by ``[제N항]`` / ``[제N호]`` / ``[제N조…]`` tags in a
    structured ``raw_output`` payload.

    ``raw_output`` maps a 조-level node_id to a list of strings; a tag inside
    those strings is read relative to that key, except for a ``제N조`` tag,
    which names a different article under the same law.

    Deepest granularity only, mirroring the gold convention — gold never
    lists a 항 parent alongside its 호 child.
    """
    out: set[str] = set()
    if not isinstance(raw_output, dict):
        return out

    for key, val in raw_output.items():
        if key == "answer" or not isinstance(val, list):
            continue
        base = key                      # the 조-level node_id this text sits under
        law = law_prefix(key)
        for s in val:
            if not isinstance(s, str):
                continue
            for content in _BRACKET.findall(s):
                c = content.replace(" ", "")

                m = _TAG_JO.match(c)          # cross-article: 제N조[의M][항][호]
                if m:
                    jo = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
                    art = f"{law}_{jo}"
                    out.add(art)
                    if m.group(3) and m.group(4):
                        out.add(f"{art}_제{m.group(3)}항_제{m.group(4)}호")
                    elif m.group(3):
                        out.add(f"{art}_제{m.group(3)}항")
                    elif m.group(4):
                        out.add(f"{art}_제{m.group(4)}호")
                    continue

                m = _TAG_HANG.match(c)        # 제N항[제M호] under the current article
                if m:
                    if m.group(2):
                        out.add(f"{base}_제{m.group(1)}항_제{m.group(2)}호")
                    else:
                        out.add(f"{base}_제{m.group(1)}항")
                    continue

                m = _TAG_HO.match(c)          # 제N호 directly under the article
                if m:
                    out.add(f"{base}_제{m.group(1)}호")

    return {_canon_ref(x) for x in out}


# ─── Prose grammar (free-form output) ──────────────────────────────

# "제48조제8항", "제48조 제8항", "제48조(대학 인건비) 제8항 제3호", "제91조의2 제3항"
_PROSE = re.compile(
    r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?"      # 조 [의 M]
    r"(?:\s*\([^)]{0,40}\))?"                  # optional (title)
    r"(?:\s*제\s*(\d+)\s*항)?"                 # optional 항
    r"(?:\s*제\s*(\d+)\s*호)?"                 # optional 호
)


def _ground_law(jo: str, cited: list[str], retrieved: list[str]) -> str | None:
    """Resolve which law ``제{jo}조`` belongs to.

    Prose names an article number but not its statute, so the number is
    grounded against ids the system itself produced — preferring what it
    cited, falling back to what it retrieved. Returns None when neither
    mentions that article, in which case the reference is dropped rather
    than guessed.
    """
    target = re.compile(r"^(.*)_제" + re.escape(jo) + r"조(?:의\d+)?(?:_|$)")
    for pool in (cited, retrieved):
        for nid in pool:
            m = target.match(nid)
            if m:
                return m.group(1)
    return None


def prose_clause_ids(
    answer: str,
    cited: list[str],
    retrieved: list[str],
) -> tuple[set[str], int, int]:
    """Node ids named inline in an answer's prose.

    Returns ``(ids, n_clause_mentions, n_ungrounded)``. An article-only
    mention is credited only when the system also cited or retrieved that
    article, so a stray number in unrelated text cannot invent a citation.
    """
    ids: set[str] = set()
    n_clause = n_ungrounded = 0
    if not isinstance(answer, str):
        return ids, 0, 0

    for m in _PROSE.finditer(answer):
        jo, ui, hang, ho = m.groups()
        if not (hang or ho):
            law = _ground_law(jo, cited, retrieved)
            if law is not None:
                jo_tok = f"제{jo}조" + (f"의{ui}" if ui else "")
                ids.add(_canon_ref(f"{law}_{jo_tok}"))
            continue

        n_clause += 1
        law = _ground_law(jo, cited, retrieved)
        if law is None:
            n_ungrounded += 1
            continue
        jo_tok = f"제{jo}조" + (f"의{ui}" if ui else "")
        art = f"{law}_{jo_tok}"
        ids.add(_canon_ref(art))
        if hang and ho:
            ids.add(_canon_ref(f"{art}_제{hang}항_제{ho}호"))
        elif hang:
            ids.add(_canon_ref(f"{art}_제{hang}항"))
        elif ho:
            ids.add(_canon_ref(f"{art}_제{ho}호"))

    return ids, n_clause, n_ungrounded


# ─── Public entry point ────────────────────────────────────────────


def expand_clause_citations(
    record: dict,
    cited: list[str],
    retrieved: list[str],
) -> tuple[list[str], dict]:
    """Return ``cited`` widened with the clause ids the answer actually names.

    Shape-dispatched, never system-name-dispatched: a record carrying a
    ``raw_output`` dict is read through its bracketed tags; anything else is
    read through its answer prose. The 조-level citation list is always kept,
    so this can only add granularity, never remove a citation.

    The second return value is a stats dict for diagnostics.
    """
    base = [_canon_ref(c) for c in (cited or []) if c]
    retrieved = [_canon_ref(r) for r in (retrieved or []) if r]
    stats = {"source": "none", "n_added": 0, "prose_clause_refs": 0,
             "ungrounded": 0}

    raw_output = record.get("raw_output")
    if isinstance(raw_output, dict) and raw_output:
        found = tag_clause_ids(raw_output)
        stats["source"] = "tags"
    else:
        found, n_clause, n_ungrounded = prose_clause_ids(
            record.get("answer") or "", base, retrieved,
        )
        stats["source"] = "prose"
        stats["prose_clause_refs"] = n_clause
        stats["ungrounded"] = n_ungrounded

    merged = list(base)
    seen = set(base)
    for nid in sorted(found):
        if nid not in seen:
            seen.add(nid)
            merged.append(nid)
    stats["n_added"] = len(merged) - len(base)
    return merged, stats
