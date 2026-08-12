"""Framework-agnostic prediction records for RegOps-Bench evaluation.

Any RAG system can be scored on RegOps-Bench by writing one JSON object
per benchmark question to a JSONL file. Only three things are required:

    {"qa_id": "...", "retrieved_node_ids": [...], "answer": "..."}

Everything else is optional or derived. Gold labels are **never** read
from the prediction file — they are joined from the benchmark file by
``qa_id``, so a system cannot accidentally (or deliberately) score
itself against its own copy of the labels.

Canonical field names
---------------------
========================  ==========================================
``qa_id``                 benchmark question id (join key, required)
``retrieved_node_ids``    ranked list of retrieved corpus ``node_id``
                          (rank 1 first). Required for retrieval
                          metrics; ``[]`` if the system has no
                          retrieval stage.
``answer``                the generated answer text. Required for
                          generation metrics.
``cited_node_ids``        ``node_id`` the answer actually cites. If
                          absent, citations can be recovered from the
                          answer text (see :func:`extract_citations`).
``status``                ``"ok"`` (default) or ``"error"``.
``variant``               free-form system/run label, used to bucket
                          the report when one file holds several runs.
========================  ==========================================

Accepted aliases
----------------
Real frameworks name these fields differently, so the loader accepts
the common spellings (see ``_ALIASES``) — e.g. ``id`` / ``question_id``
for ``qa_id``, ``cited_references`` / ``citations`` for
``cited_node_ids``, ``prediction`` / ``response`` / ``output`` for
``answer``, and ``contexts`` / ``source_nodes`` / ``retrieved`` for
``retrieved_node_ids``.

``retrieved_node_ids`` may also be a list of objects rather than a list
of strings — the LangChain ``Document`` / LlamaIndex ``NodeWithScore``
shape is understood, and the id is pulled from the object itself or
from its ``metadata``. See :func:`coerce_id_list`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from src.retrieval.corpus import _canon_ref


# ─── Field aliases ──────────────────────────────────────────────────

# Canonical name → alternative spellings, in priority order. The first
# key present on the record wins; the canonical name always wins over
# any alias.
_ALIASES: dict[str, tuple[str, ...]] = {
    "qa_id": ("id", "question_id", "qid", "query_id", "sample_id"),
    "retrieved_node_ids": (
        "retrieved", "retrieved_ids", "retrieved_docs", "contexts",
        "context_node_ids", "source_nodes", "sources", "docs",
    ),
    "cited_node_ids": (
        "cited_references", "citations", "cited", "cited_ids",
        "citation_node_ids",
    ),
    "answer": (
        "prediction", "output", "response", "generated_answer",
        "pred_answer", "text",
    ),
    "variant": ("system", "run", "run_id", "method", "model"),
    "question": ("query", "input"),
}

# Keys an id-bearing object may carry, in priority order.
_ID_KEYS = (
    "node_id", "id", "doc_id", "document_id", "source", "file_path",
    "ref", "reference", "chunk_id",
)


def _pick(record: dict, canonical: str) -> Any:
    """Return ``record[canonical]`` if present and non-null, else the
    first non-null alias value."""
    if record.get(canonical) is not None:
        return record[canonical]
    for alias in _ALIASES.get(canonical, ()):
        if record.get(alias) is not None:
            return record[alias]
    return None


# ─── Value coercion ─────────────────────────────────────────────────


def _id_from_object(obj: Any) -> str | None:
    """Pull a node_id out of one retrieved-context object.

    Understands the shapes emitted by the common frameworks:

      * plain string                       → itself
      * ``{"node_id": ...}`` and friends   → that value
      * LangChain ``Document``             → ``metadata.node_id`` /
        ``metadata.source`` / ``metadata.file_path``
      * LlamaIndex ``NodeWithScore``       → ``node.node_id`` /
        ``node.metadata.*``
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj.strip() or None
    if not isinstance(obj, dict):
        return None

    for key in _ID_KEYS:
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # LlamaIndex nests the payload one level down.
    nested = obj.get("node")
    if isinstance(nested, dict):
        found = _id_from_object(nested)
        if found:
            return found

    meta = obj.get("metadata") or obj.get("extra_info")
    if isinstance(meta, dict):
        for key in _ID_KEYS:
            val = meta.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def coerce_id_list(value: Any) -> list[str]:
    """Normalise any of the accepted retrieved/cited shapes to an
    ordered, de-duplicated, canonicalised list of node_ids.

    Order is meaningful (rank 1 first) and is preserved; the first
    occurrence of a repeated id wins, which is the right behaviour for
    ranking metrics.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []

    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        raw = _id_from_object(item)
        if not raw:
            continue
        nid = _canon_ref(raw)
        if nid not in seen:
            seen.add(nid)
            out.append(nid)
    return out


# ─── Citation extraction from free answer text ─────────────────────

# Footer conventions the shipped baselines emit, e.g.
#   [참조] doc_제48조, doc_제48조_제8항
#   [Citations] spans-.../sent_0012-0ec05553
_FOOTER_RE = re.compile(
    r"^\s*\[(?:참조|참고|근거|Citations?|References?)\]\s*(.+)$",
    re.MULTILINE | re.IGNORECASE,
)


def extract_citations(
    answer: str,
    corpus_node_ids: Iterable[str] | None = None,
) -> list[str]:
    """Recover cited node_ids from an answer that has no citation field.

    Two passes, both order-preserving:

    1. **Footer** — any ``[참조] a, b`` / ``[Citations] a, b`` line.
    2. **Inline scan** — if a corpus vocabulary is supplied, every
       node_id from that vocabulary that literally occurs in the answer,
       ordered by first occurrence. Longest ids are matched first so
       that a 항-level id wins over its 조-level prefix at the same
       position.

    Pass 2 only runs when pass 1 found nothing, so a system that emits a
    proper footer is never penalised for also naming articles in prose.
    """
    if not answer:
        return []

    footer_ids: list[str] = []
    for m in _FOOTER_RE.finditer(answer):
        for part in re.split(r"[,;、]|\s{2,}", m.group(1)):
            part = part.strip().strip("`'\"[]()")
            if part:
                footer_ids.append(part)
    if footer_ids:
        return coerce_id_list(footer_ids)

    if not corpus_node_ids:
        return []

    hits: list[tuple[int, str]] = []
    claimed: list[tuple[int, int]] = []
    # Longest-first so `..._제48조_제8항` beats `..._제48조`.
    for nid in sorted(set(corpus_node_ids), key=len, reverse=True):
        start = answer.find(nid)
        if start < 0:
            continue
        end = start + len(nid)
        if any(s <= start and end <= e for s, e in claimed):
            continue
        claimed.append((start, end))
        hits.append((start, nid))
    return coerce_id_list([nid for _, nid in sorted(hits)])


# ─── Record ─────────────────────────────────────────────────────────


@dataclass
class Prediction:
    """One system's response to one benchmark question."""

    qa_id: str
    retrieved_node_ids: list[str] = field(default_factory=list)
    cited_node_ids: list[str] = field(default_factory=list)
    answer: str = ""
    status: str = "ok"
    variant: str = "system"
    question: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class PredictionFormatError(ValueError):
    """Raised when a prediction file cannot be interpreted."""


def parse_prediction(
    record: dict,
    *,
    default_variant: str = "system",
    line_no: int | None = None,
) -> Prediction:
    """Normalise one raw JSON object into a :class:`Prediction`."""
    where = f" (line {line_no})" if line_no is not None else ""

    qa_id = _pick(record, "qa_id")
    if not isinstance(qa_id, str) or not qa_id.strip():
        raise PredictionFormatError(
            f"missing or non-string 'qa_id'{where}. Accepted aliases: "
            f"{', '.join(_ALIASES['qa_id'])}."
        )

    status = record.get("status")
    if not isinstance(status, str) or not status:
        # An `error` payload implies failure even without a status field.
        status = "error" if record.get("error") else "ok"

    variant = _pick(record, "variant")
    answer = _pick(record, "answer")

    return Prediction(
        qa_id=qa_id.strip(),
        retrieved_node_ids=coerce_id_list(_pick(record, "retrieved_node_ids")),
        cited_node_ids=coerce_id_list(_pick(record, "cited_node_ids")),
        answer=answer if isinstance(answer, str) else "",
        status=status,
        variant=str(variant) if variant else default_variant,
        question=str(_pick(record, "question") or ""),
        raw=record,
    )


def iter_predictions(
    path: Path,
    *,
    default_variant: str | None = None,
) -> Iterator[Prediction]:
    """Stream :class:`Prediction` records from a JSONL (or JSON list) file.

    ``default_variant`` labels records that carry no variant of their
    own; it defaults to the file stem, which keeps multi-file reports
    readable without every framework having to agree on a label.
    """
    path = Path(path)
    if not path.exists():
        raise PredictionFormatError(f"prediction file not found: {path}")
    fallback = default_variant or path.stem

    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            records = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PredictionFormatError(f"{path}: invalid JSON — {exc}") from exc
        for i, rec in enumerate(records, 1):
            yield parse_prediction(rec, default_variant=fallback, line_no=i)
        return

    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PredictionFormatError(
                f"{path}: line {i} is not valid JSON — {exc}"
            ) from exc
        yield parse_prediction(rec, default_variant=fallback, line_no=i)


def load_predictions(
    paths: Sequence[Path] | Path,
    *,
    default_variant: str | None = None,
) -> list[Prediction]:
    """Load one or more prediction files into a flat list."""
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    out: list[Prediction] = []
    for p in paths:
        out.extend(iter_predictions(Path(p), default_variant=default_variant))
    return out


# ─── Validation ─────────────────────────────────────────────────────


def validate(
    predictions: Sequence[Prediction],
    bench_ids: Iterable[str],
    corpus_node_ids: Iterable[str] | None = None,
) -> dict:
    """Sanity-check a prediction set before scoring.

    Returns a report dict with ``errors`` (block scoring) and
    ``warnings`` (worth knowing, but scoring proceeds). Nothing here
    raises — the caller decides how strict to be.
    """
    bench = set(bench_ids)
    corpus = {_canon_ref(c) for c in corpus_node_ids} if corpus_node_ids else None

    errors: list[str] = []
    warnings: list[str] = []

    seen: dict[tuple[str, str], int] = {}
    for p in predictions:
        key = (p.variant, p.qa_id)
        seen[key] = seen.get(key, 0) + 1

    dupes = [f"{v}/{q}" for (v, q), n in seen.items() if n > 1]
    if dupes:
        errors.append(
            f"{len(dupes)} duplicate (variant, qa_id) pairs, e.g. "
            f"{', '.join(sorted(dupes)[:3])}"
        )

    unknown = sorted({p.qa_id for p in predictions if p.qa_id not in bench})
    if unknown:
        errors.append(
            f"{len(unknown)} qa_id not present in the benchmark, e.g. "
            f"{', '.join(unknown[:3])}"
        )

    by_variant: dict[str, set[str]] = {}
    for p in predictions:
        by_variant.setdefault(p.variant, set()).add(p.qa_id)
    for variant, ids in sorted(by_variant.items()):
        missing = bench - ids
        if missing:
            warnings.append(
                f"variant '{variant}' covers {len(ids & bench)}/{len(bench)} "
                f"benchmark questions — the {len(missing)} missing ones are "
                f"not scored, so its numbers are not comparable to a full run"
            )

    n_no_retrieval = sum(1 for p in predictions if p.ok and not p.retrieved_node_ids)
    if n_no_retrieval:
        warnings.append(
            f"{n_no_retrieval}/{len(predictions)} ok-status records have an "
            f"empty 'retrieved_node_ids' — retrieval metrics will be 0 for them"
        )

    n_no_answer = sum(1 for p in predictions if p.ok and not p.answer.strip())
    if n_no_answer:
        warnings.append(
            f"{n_no_answer}/{len(predictions)} ok-status records have an empty "
            f"'answer'"
        )

    n_no_citation = sum(1 for p in predictions if p.ok and not p.cited_node_ids)
    if n_no_citation:
        warnings.append(
            f"{n_no_citation}/{len(predictions)} ok-status records have no "
            f"citations — pass --cite-from-answer to recover them from the "
            f"answer text"
        )

    if corpus is not None:
        off_corpus = {
            nid
            for p in predictions
            for nid in p.retrieved_node_ids
            if nid not in corpus
        }
        if off_corpus:
            share = len(off_corpus)
            warnings.append(
                f"{share} distinct retrieved ids are not in the corpus "
                f"(id-space mismatch?), e.g. {', '.join(sorted(off_corpus)[:3])}"
            )

    return {
        "n_predictions": len(predictions),
        "n_variants": len(by_variant),
        "variants": sorted(by_variant),
        "n_bench": len(bench),
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }
