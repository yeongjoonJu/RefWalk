from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Union

import networkx as nx

from src.retrieval.corpus import Corpus
from src.retrieval.reranker import (
    DEFAULT_ENDPOINT,
    DEFAULT_INSTRUCTION,
    Qwen3Reranker,
    RerankItem,
)


# Module-level defaults — kept as constants so tests / external code
# can introspect them.
_DEFAULT_EXPAND_EDGE_TYPES: tuple[str, ...] = (
    "REFERENCES", "DELEGATES_TO", "SPECIFIES",
)
_DEFAULT_HYBRID_POOL = 50
_DEFAULT_OKG_EXPAND_SEED = 10
_DEFAULT_OKG_EXPAND_DECAY = 0.7

# Multi-view rerank fusion defaults — Round-5 finding (cached anchors,
# regops_bench n=250): 3-view RRM beats single-view (mid) by +3.1pp OV
# R@10, +2.0pp OV FC@10, +13.9pp L3 R@10, with no regression on any
# difficulty. RRM (max-aggregation) preserves each view's specialist
# signal; uniform RRF averages and dilutes them.
_DEFAULT_RERANK_VIEWS: tuple[str, ...] = ("narrow", "mid", "wide")
_DEFAULT_RERANK_FUSION: str = "rrm"
_DEFAULT_RERANK_RRF_K: int = 60

_ASPECT_KEYS: tuple[str, ...] = ("actor", "temporal", "magnitude", "situational")
# QueryAnalysis schema in src.ground.schemas uses the typo "magnitute"
# (Literal). We accept both for robustness.
_ASPECT_ALIASES: dict[str, str] = {"magnitute": "magnitude"}

_SEARCH_TEXT_LEN = 200


# ─── Query anchoring ────────────────────────────────────────────────


def _is_unspecified(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip().lower() in ("", "unspecified", "none", "null")
    return False


def _flatten_conditions(anchored: dict) -> dict:
    """Pull a nested ``conditions`` dict up to top level. Idempotent.

    ``QueryAnalysis`` returns ``{topic, conditions: {actor, temporal,
    magnitute, situational}}`` while ``data/anchoring_test_faq*.json``
    flattens those keys to top level. Accept both forms.

    Every aspect key is finally back-filled to ``"unspecified"`` if the
    base model omitted it. The ``TOPIC_ANCHORING`` prompt asks for
    ``"unspecified"`` on unmentioned dimensions and Qwen always complies,
    but other base models (e.g. Gemini in non-thinking mode) sometimes
    drop the key entirely instead. Without this default the hard
    indexing in ``get_user_prompt`` (``anchor['actor']`` …) raises
    ``KeyError``. ``"unspecified"`` is the value the rest of the pipeline
    already treats as "not anchored" (see :func:`_is_unspecified` and the
    ``unspecified``→``all`` rewrite), so this is behaviour-preserving for
    base models that already emit every key.
    """
    out = dict(anchored)
    cond = out.get("conditions")
    if isinstance(cond, dict):
        for k, v in cond.items():
            canon = _ASPECT_ALIASES.get(k, k)
            out.setdefault(canon, v)
    # Resolve typo at top level too.
    for typo, canon in _ASPECT_ALIASES.items():
        if typo in out:
            out.setdefault(canon, out[typo])
    # Guarantee every aspect key exists so downstream consumers can index
    # the anchor unconditionally regardless of base-model compliance.
    for k in _ASPECT_KEYS:
        out.setdefault(k, "unspecified")
    return out


# ─── Fusion (multi-view rerank aggregation) ────────────────────────


def _rrf_fuse(
    rankings_by_view: dict,
    k: int = 60,
    weights: Optional[dict] = None,
) -> dict:
    """Reciprocal Rank Fusion across views, optionally weighted.

    rankings_by_view: ``{view_name: [node_id ordered by descending score]}``
    weights: optional ``{view_name: float}`` — per-view weight (default
        1.0 each). Higher weight on a view → its ranks count more in
        the sum.
    Returns: ``{node_id: rrf_score}``
    """
    weights = weights or {}
    scores: dict = {}
    for view, ranked_ids in rankings_by_view.items():
        w = float(weights.get(view, 1.0))
        for rank0, nid in enumerate(ranked_ids):
            scores[nid] = scores.get(nid, 0.0) + w / (k + rank0 + 1)
    return scores


def _rrm_fuse(rankings_by_view: dict, k: int = 60) -> dict:
    """Reciprocal Rank MAX — winner-take aggregation.

    ``score(d) = max_v 1 / (k + rank_v(d))``

    Each doc keeps the rank of the view that loves it most; other views
    are ignored. Preserves a specialist view's top picks completely
    (no averaging dilution from views that fail to recognize the doc).
    """
    scores: dict = {}
    for view, ranked_ids in rankings_by_view.items():
        for rank0, nid in enumerate(ranked_ids):
            s = 1.0 / (k + rank0 + 1)
            if s > scores.get(nid, 0.0):
                scores[nid] = s
    return scores


def build_tagged_query(anchored: Union[str, dict]) -> tuple[str, str]:
    """Render an anchored payload to ``(seed_query, query_wo_q)``.

    The two share the same ``[TOPIC] …`` and ``Conditions: …`` lines —
    they differ only in whether the raw question text appears as a
    separate ``[Q] …`` line. Unspecified condition slots render as
    ``all`` (per the extraction system prompt's "value 'all' = no
    filter on this aspect" rule), so all four aspect slots are always
    present in both renderings.
    """
    if isinstance(anchored, str):
        topic = ""
        ori = anchored.strip()
        a: dict = {}
    else:
        a = _flatten_conditions(anchored)
        topic = (a.get("topic") or "").strip()
        ori = (a.get("ori_question") or a.get("question") or "").strip()

    headline = topic or ori

    cond_parts: list[str] = []
    for k in _ASPECT_KEYS:
        v = a.get(k)
        if _is_unspecified(v):
            cond_parts.append(f"[{k.upper()}] all")
        else:
            cond_parts.append(f"[{k.upper()}] {str(v).strip()}")
    cond_line = "Conditions: " + ", ".join(cond_parts)

    topic_line = f"[TOPIC] {headline}"
    query_wo_q = f"{topic_line}\n{cond_line}"

    if ori and ori != headline:
        seed_query = f"{topic_line}\n[Q] {ori}\n{cond_line}"
    else:
        seed_query = query_wo_q
    return seed_query, query_wo_q


@dataclass
class TopicExtractor:
    """Topic + condition anchorer for raw queries.

    Mirrors ``cli/test_topic_anchoring.py``: feeds the question to an
    LLM with ``TOPIC_ANCHORING`` system prompt and the ``QueryAnalysis``
    JSON schema, returning a flat dict suitable for
    :func:`build_tagged_query`.

    There is **no cache** — every ``extract()`` call performs a live
    LLM round-trip. Callers that need reproducibility (the eval CLI,
    benchmark replay) should pre-anchor in batch and feed
    :class:`WalkerRetrievalPipeline` an already-anchored dict.

    The OpenAI client is constructed lazily on the first ``extract``
    call so importing this module does not require the LLM endpoint
    to be reachable.
    """

    base_url: str = "http://localhost:8035/v1"
    model_id: str = "Qwen/Qwen3.6-35B-A3B-FP8"
    profile_name: str = "exact"
    domain: str = "Korean R&D funding regulations"
    language: str = "Korean"
    # Real key for hosted backends (Gemini OpenAI-compat); None → "dummy"
    # for local vLLM, so existing Qwen callers are unaffected.
    api_key: Optional[str] = None

    _llm: Any = field(default=None, init=False, repr=False)

    def _get_llm(self):
        if self._llm is None:
            from src.utils.prompts.refwalker import TOPIC_ANCHORING
            from src.schemas import QueryAnalysis
            from src.base_llm import BaseLLM
            self._llm = BaseLLM(
                base_url=self.base_url,
                model_id=self.model_id,
                profile_name=self.profile_name,
                system_prompt=TOPIC_ANCHORING.format(
                    domain=self.domain, language=self.language,
                ),
                schema=QueryAnalysis.model_json_schema(),
                api_key=self.api_key,
            )
        return self._llm

    def extract(self, question: str) -> dict:
        out = self._get_llm()._call_llm(prompt=f"Question: {question}")
        if not isinstance(out, dict):
            return {"topic": question, "ori_question": question}
        out = _flatten_conditions(out)
        out.setdefault("ori_question", question)
        out.setdefault("topic", question)
        return out


@dataclass
class RetrievalHit:
    node_id: str
    text: str
    hybrid_score: float                 # seed retriever's score, kept-named for back-compat
    rerank_score: float
    source: str  # "hybrid" | "okg_expansion"

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "text": self.text,
            "hybrid_score": round(self.hybrid_score, 4),
            "rerank_score": round(self.rerank_score, 4),
            "source": self.source,
        }


@dataclass
class WalkerRetrievalPipeline:
    """Tagged-query seed retrieval + 1-hop OKG expansion + Qwen3 reranker.

    Parameters
    ----------
    corpus, graph
        Required. ``corpus`` is the retrieval pool; ``graph`` is the
        OKG used for typed-edge expansion.
    seed_retriever
        First-stage retriever. Any object with ``search(query, top_k)``
        that returns ``[(node_id, score), …]``. Defaults to a fresh
        ``DenseRetriever`` built from ``corpus``.
    reranker
        :class:`Qwen3Reranker` instance (default factory).
    rerank_pool
        How many seeds the first stage fetches.
    okg_expand_seed
        Of those seeds, how many drive the 1-hop typed walk.
    okg_expand_decay
        Multiplier applied to a seed's score when propagating to a
        neighbor (``0.7`` by default).
    expand_edge_types
        Which OKG edge types qualify. PART_OF is intentionally NOT in
        the default — that's hierarchical containment, not cross-ref.
    topic_extractor
        :class:`TopicExtractor` for online query anchoring. **On by
        default** (lazy-initialised; no LLM client until a raw string
        is passed to ``retrieve``). The tagged seed query that this
        enables is the single largest validated win, so it stays on
        by default. Pass ``topic_extractor=None`` to disable when
        callers always supply pre-anchored dicts.
    hybrid
        **Legacy alias** for ``seed_retriever``. Old callers passing
        ``hybrid=…`` keep working unchanged.
    """

    corpus: Corpus
    graph: nx.DiGraph
    seed_retriever: Any = None
    reranker: Qwen3Reranker = field(default_factory=Qwen3Reranker)
    rerank_pool: int = _DEFAULT_HYBRID_POOL
    okg_expand_seed: int = _DEFAULT_OKG_EXPAND_SEED
    okg_expand_decay: float = _DEFAULT_OKG_EXPAND_DECAY
    expand_edge_types: tuple[str, ...] = _DEFAULT_EXPAND_EDGE_TYPES
    # Authority-aware rerank: multiplicative penalty applied to manual
    # entries' fused rerank score so that 법령 (법/시행령/시행규칙) is
    # preferred over 매뉴얼 when both score similarly. Set to 1.0 to
    # disable. Applied after multi-view fusion so a strongly-matching
    # manual (e.g. unique 양식/한도 content) can still surface.
    manual_rerank_weight: float = 0.7
    # Online topic anchoring — on by default; lazy-initialised.
    topic_extractor: Optional[TopicExtractor] = field(
        default_factory=TopicExtractor,
    )
    # Multi-view rerank fusion (Round-5 default, see module docstring).
    # Set ``rerank_views=("mid",)`` and ``rerank_fusion="rrf"`` to recover
    # the legacy single-view anchored-seed rerank behaviour.
    rerank_views: tuple[str, ...] = _DEFAULT_RERANK_VIEWS
    rerank_fusion: str = _DEFAULT_RERANK_FUSION
    rerank_view_weights: Optional[dict] = None
    rerank_rrf_k: int = _DEFAULT_RERANK_RRF_K
    # Legacy alias for ``seed_retriever``.
    hybrid: Any = None

    def __post_init__(self) -> None:
        # Legacy → canonical: if caller passed hybrid= but not
        # seed_retriever=, route it through. Caller-provided
        # seed_retriever wins if both are supplied.
        if self.seed_retriever is None and self.hybrid is not None:
            self.seed_retriever = self.hybrid
        # Default seed retriever: dense over the corpus.
        if self.seed_retriever is None:
            from src.retrieval.embedding import DenseRetriever
            self.seed_retriever = DenseRetriever.build(self.corpus)
        # Mirror back so external code reading ``pipeline.hybrid`` (rare
        # but possible) still resolves to the seed retriever.
        if self.hybrid is None:
            self.hybrid = self.seed_retriever

    # ── Query normalisation ────────────────────────────────────────

    def _anchor_query(self, query: Union[str, dict]) -> dict:
        """Return a flat dict with ``topic`` / ``ori_question`` / aspects.

        Topic + condition extraction is the **start** of the WalkerRetrieval
        pipeline; it is mandatory in normal operation. The only way to
        skip the live extraction call is to feed an already-anchored dict
        (i.e. ``topic`` set or at least one aspect specified — the
        ``data/anchoring_test_faq.json`` shape). That path exists for
        reproducibility / ablation against a pre-extracted FAQ.

        Raw inputs (a string question, or a bench dict whose only
        retrieval-relevant field is ``question``) **must** go through
        ``topic_extractor``. If the extractor is None for a raw input,
        this raises — silently degrading would invalidate the
        methodology's contract.
        """
        if isinstance(query, dict):
            flat = _flatten_conditions(query)
            has_topic = bool((flat.get("topic") or "").strip())
            has_aspects = any(
                not _is_unspecified(flat.get(k)) for k in _ASPECT_KEYS
            )
            if has_topic or has_aspects:
                return flat
            # Essentially unanchored — extract topic + aspects from the
            # question text. Back-fill onto ``flat`` so qa_id and other
            # benchmark fields are preserved.
            question = (
                flat.get("ori_question") or flat.get("question") or ""
            ).strip()
            if not question:
                raise RuntimeError(
                    "WalkerRetrieval got a dict with no topic, no "
                    "aspects, and no question text — nothing to anchor."
                )
            if self.topic_extractor is None:
                raise RuntimeError(
                    "WalkerRetrieval requires topic+condition extraction "
                    "at the start of the pipeline, but topic_extractor "
                    "is disabled and the input dict is unanchored "
                    "(no topic, no aspects). This combination is only "
                    "valid for ablation against a pre-anchored bench "
                    "(e.g. data/anchoring_test_faq.json). Re-enable the "
                    "extractor (drop --no-topic-extractor) or pass a "
                    "dict carrying topic/aspects."
                )
            extracted = self.topic_extractor.extract(question)
            for k, v in extracted.items():
                flat.setdefault(k, v)
            return flat
        if self.topic_extractor is None:
            raise RuntimeError(
                "WalkerRetrieval got a raw string question but "
                "topic_extractor is disabled. Topic extraction is the "
                "start of the pipeline; running it without an extractor "
                "is only valid when the caller pre-anchors via dict. "
                "Re-enable the extractor."
            )
        return self.topic_extractor.extract(query)

    # ── Steps ──────────────────────────────────────────────────────

    def _seed_candidates(self, query: str) -> dict[str, float]:
        hits = self.seed_retriever.search(query, top_k=self.rerank_pool)
        return {nid: sc for nid, sc in hits}

    # Legacy alias (some external scripts may grep for this name).
    _hybrid_candidates = _seed_candidates

    def _okg_expand(self, seed_scores: dict[str, float]) -> dict[str, float]:
        """1-hop typed-edge expansion around the top-N seeds.

        Uses ``self.okg_expand_seed`` as the seed-count, ``self.okg_expand_decay``
        as the score multiplier, and ``self.expand_edge_types`` as the
        whitelist of edge types to walk. Only neighbours present in
        ``self.corpus`` (i.e., with retrievable text) are kept.
        """
        seeds = sorted(seed_scores.items(), key=lambda x: -x[1])[
            : self.okg_expand_seed
        ]
        decay = self.okg_expand_decay
        edge_types = set(self.expand_edge_types)
        expanded: dict[str, float] = {}
        for nid, sc in seeds:
            if nid not in self.graph:
                continue
            for neighbor in self.graph.successors(nid):
                etype = self.graph.edges[nid, neighbor].get("type")
                if etype not in edge_types:
                    continue
                if neighbor not in self.corpus.id_to_idx:
                    continue
                bonus = sc * decay
                if expanded.get(neighbor, -1.0) < bonus:
                    expanded[neighbor] = bonus
            for neighbor in self.graph.predecessors(nid):
                etype = self.graph.edges[neighbor, nid].get("type")
                if etype not in edge_types:
                    continue
                if neighbor not in self.corpus.id_to_idx:
                    continue
                bonus = sc * decay
                if expanded.get(neighbor, -1.0) < bonus:
                    expanded[neighbor] = bonus
        return expanded

    def _build_candidates(
        self,
        seed_scores: dict[str, float],
        okg_scores: dict[str, float],
    ) -> list[RerankItem]:
        items: list[RerankItem] = []
        seen: set[str] = set()
        # Seeds first, so that ties attribute to the seed retriever.
        for nid, sc in seed_scores.items():
            if nid in seen:
                continue
            idx = self.corpus.id_to_idx.get(nid)
            if idx is None:
                continue
            items.append(
                RerankItem(
                    node_id=nid,
                    text=self.corpus.texts[idx],
                    metadata={"hybrid_score": float(sc), "source": "hybrid"},
                )
            )
            seen.add(nid)
        for nid, sc in okg_scores.items():
            if nid in seen:
                continue
            idx = self.corpus.id_to_idx.get(nid)
            if idx is None:
                continue
            items.append(
                RerankItem(
                    node_id=nid,
                    text=self.corpus.texts[idx],
                    metadata={
                        "hybrid_score": float(sc),
                        "source": "okg_expansion",
                    },
                )
            )
            seen.add(nid)
        return items

    # ── Multi-view rerank ─────────────────────────────────────────

    def _multi_view_rerank(
        self,
        candidates: list[RerankItem],
        *,
        narrow_q: str,
        mid_q: str,
        wide_q: str,
        top_k: int,
    ) -> list[dict]:
        """Run reranker once per configured view, fuse via RRF/RRM.

        Returns reranker rows in fused order (each row from the view that
        scored that doc highest). For single-view configs (e.g.
        ``rerank_views=("mid",)``) this degenerates to a one-shot rerank
        with the chosen view's query — identical to the legacy path.
        """
        view_query = {"narrow": narrow_q, "mid": mid_q, "wide": wide_q}
        rankings: dict[str, list[str]] = {}
        # Track each candidate's strongest reranker row across views,
        # so the final list carries the highest rerank_score (informative
        # for downstream visited-filter ordering and trace analysis).
        best_row: dict[str, dict] = {}
        pool_size = len(candidates)
        for v in self.rerank_views:
            q = (view_query.get(v) or "").strip()
            if not q:
                continue
            ranked = self.reranker.rerank(
                query=q, candidates=candidates, top_k=pool_size,
            )
            rankings[v] = [r["node_id"] for r in ranked]
            for r in ranked:
                nid = r["node_id"]
                cur = best_row.get(nid)
                if cur is None or r["rerank_score"] > cur["rerank_score"]:
                    best_row[nid] = r

        if not rankings:
            return []

        if self.rerank_fusion == "rrm":
            scores = _rrm_fuse(rankings, k=self.rerank_rrf_k)
        elif self.rerank_fusion == "rrf":
            scores = _rrf_fuse(
                rankings, k=self.rerank_rrf_k,
                weights=self.rerank_view_weights,
            )
        else:
            raise ValueError(
                f"unknown rerank_fusion {self.rerank_fusion!r}; "
                f"choose from {{'rrf', 'rrm'}}"
            )

        # Authority-aware penalty: push down 매뉴얼 entries so 법령 wins
        # ties. Applied to the fused score so it directly affects the
        # final ordering; a strongly-matched manual can still rank above
        # a weakly-matched 법령 entry.
        manual_w = self.manual_rerank_weight
        if manual_w != 1.0:
            for nid in list(scores.keys()):
                if "매뉴얼" in nid:
                    scores[nid] *= manual_w

        # Tie-break: secondary key = mid-view rank (smallest = best).
        # Falls back to narrow then wide if mid is not in ``rerank_views``.
        tiebreak_view = next(
            (v for v in ("mid", "narrow", "wide") if v in rankings), None
        )
        tb_rank = (
            {nid: i for i, nid in enumerate(rankings[tiebreak_view])}
            if tiebreak_view else {}
        )
        order = sorted(
            scores,
            key=lambda nid: (-scores[nid], tb_rank.get(nid, 1_000_000)),
        )
        out: list[dict] = []
        for nid in order[:top_k]:
            row = best_row.get(nid)
            if row is None:
                continue
            row = dict(row)
            row["fusion_score"] = float(scores[nid])
            out.append(row)
        return out

    def retrieve_for_generation(
        self,
        query: Union[str, dict],
        top_k: int = 10,
        visited_node_ids: Optional[Iterable[str]] = None,
        diagnostics: Optional[dict] = None,
    ) -> tuple[list[RetrievalHit], str]:
        """Run anchoring → tagged seed retrieval → OKG expand → rerank,
        then optionally drop hits whose ``node_id`` is in
        ``visited_node_ids``.

        Returns a 2-tuple ``(hits, query_wo_q)`` — the second element is
        the structured ``[TOPIC] / [CONDITIONS]`` query rendering used
        by downstream rule-extraction prompts (see
        :class:`src.chain_of_rules.ChainOfRulesWorkflow`).

        ``query`` accepts:
        - ``str`` — raw question. Anchored on the fly via
          ``topic_extractor`` (a fresh LLM call; no cache). When
          ``topic_extractor`` is None the string is used as-is.
        - ``dict`` — pre-anchored ``{topic, ori_question, actor,
          temporal, magnitude, situational}`` (flat or nested under
          ``conditions``). The eval CLI feeds rows of
          ``data/anchoring_test_faq.json`` through this path.

        When ``diagnostics`` is supplied, the visited-filter count is
        written to ``diagnostics["filtered_visited"]`` for tooling.
        """
        anchored = self._anchor_query(query)
        seed_query, query_wo_q = build_tagged_query(anchored)
        ori_question = (anchored.get("ori_question") or "").strip()

        seed_scores = self._seed_candidates(seed_query)
        okg_scores = self._okg_expand(seed_scores)
        candidates = self._build_candidates(seed_scores, okg_scores)

        if not candidates:
            if diagnostics is not None:
                diagnostics["filtered_visited"] = 0
            return [], query_wo_q

        # Multi-view rerank → RRM/RRF fusion. Rerank the full pool so the
        # visited filter can pull the next-ranked candidate instead of
        # underfilling the returned top-k.
        rerank_pool = max(top_k, len(candidates))
        reranked = self._multi_view_rerank(
            candidates,
            narrow_q=ori_question,
            mid_q=seed_query,
            wide_q=query_wo_q,
            top_k=rerank_pool,
        )

        visited = set(visited_node_ids or ())
        out: list[RetrievalHit] = []
        filtered = 0
        for r in reranked:
            nid = r["node_id"]
            if nid in visited:
                filtered += 1
                continue
            md = r.get("metadata") or {}
            out.append(
                RetrievalHit(
                    node_id=nid,
                    text=r["text"],
                    hybrid_score=float(md.get("hybrid_score", 0.0)),
                    rerank_score=float(r["rerank_score"]),
                    source=str(md.get("source", "hybrid")),
                )
            )
            if len(out) >= top_k:
                break

        return out, anchored

    def retrieve(
        self,
        query: Union[str, dict],
        top_k: int = 5,
        visited_node_ids: Optional[Iterable[str]] = None,
        diagnostics: Optional[dict] = None,
    ) -> list[RetrievalHit]:
        """Run anchoring → tagged seed retrieval → OKG expand → rerank,
        then optionally drop hits whose ``node_id`` is in
        ``visited_node_ids``.

        ``query`` accepts:
        - ``str`` — raw question. Anchored on the fly via
          ``topic_extractor`` (a fresh LLM call; no cache). When
          ``topic_extractor`` is None the string is used as-is.
        - ``dict`` — pre-anchored ``{topic, ori_question, actor,
          temporal, magnitude, situational}`` (flat or nested under
          ``conditions``). The eval CLI feeds rows of
          ``data/anchoring_test_faq.json`` through this path.

        When ``diagnostics`` is supplied, the visited-filter count is
        written to ``diagnostics["filtered_visited"]`` for tooling.
        """
        anchored = self._anchor_query(query)
        seed_query, query_wo_q = build_tagged_query(anchored)
        ori_question = (anchored.get("ori_question") or "").strip()

        seed_scores = self._seed_candidates(seed_query)
        okg_scores = self._okg_expand(seed_scores)
        candidates = self._build_candidates(seed_scores, okg_scores)

        if not candidates:
            if diagnostics is not None:
                diagnostics["filtered_visited"] = 0
            return []

        # Multi-view rerank → RRM/RRF fusion. Rerank the full pool so the
        # visited filter can pull the next-ranked candidate instead of
        # underfilling the returned top-k.
        rerank_pool = max(top_k, len(candidates))
        reranked = self._multi_view_rerank(
            candidates,
            narrow_q=ori_question,
            mid_q=seed_query,
            wide_q=query_wo_q,
            top_k=rerank_pool,
        )

        visited = set(visited_node_ids or ())
        out: list[RetrievalHit] = []
        filtered = 0
        for r in reranked:
            nid = r["node_id"]
            if nid in visited:
                filtered += 1
                continue
            md = r.get("metadata") or {}
            out.append(
                RetrievalHit(
                    node_id=nid,
                    text=r["text"],
                    hybrid_score=float(md.get("hybrid_score", 0.0)),
                    rerank_score=float(r["rerank_score"]),
                    source=str(md.get("source", "hybrid")),
                )
            )
            if len(out) >= top_k:
                break
        if diagnostics is not None:
            diagnostics["filtered_visited"] = filtered
        return out


# ─── Formatting helper ────────────────────────────────────────────

def format_hits_for_walker(
    hits: list[RetrievalHit],
    toolkit,  # src.ground.tools.OKSWalkToolkit; duck-typed to avoid cycle
    *,
    query: str = "",
) -> str:
    """Render retrieval hits into the string layout the Walker prompt
    already expects from `search_by_query` — headed sections separated
    by "\\n---\\n", with the new `source` / `rerank_score` fields
    surfaced at the top of each block.

    The `[already_visited]` label / `already_visited=` field used to
    flag nodes the walker had already touched is gone: the upstream
    pipeline now filters visited candidates before this formatter runs,
    so every rendered hit is guaranteed fresh. This removes the "why
    is the same article reappearing?" noise in the Walker prompt.
    """
    if not hits:
        return f"No results for query: {query!r}"

    sections: list[str] = []
    for h in hits:
        text = (h.text or "").strip().replace("\n", " ")
        snippet_text = text[:_SEARCH_TEXT_LEN] + (
            "…" if len(text) > _SEARCH_TEXT_LEN else ""
        )
        edge_block = toolkit._edge_summary(h.node_id) or "  (no expandable edges)"
        sections.append(
            f"[{h.node_id}] rerank={h.rerank_score:.4f} "
            f"hybrid={h.hybrid_score:.4f} source={h.source}\n"
            f"text: {snippet_text}\n"
            f"edges:\n{edge_block}"
        )
        toolkit._mark_visited(h.node_id)

    return "\n---\n".join(sections)
