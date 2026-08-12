"""RefWalk = WalkerRetrieval + structured-JSON answer + per-clause citations.

Pipeline for one question:

  1. ``WalkerRetrievalPipeline.retrieve_for_generation`` runs the full
     anchored retrieval stack (topic+conditions extraction → tagged seed
     dense → 1-hop OKG expansion → 3-view RRM rerank) and returns
     ``(hits, anchored_dict)``.
  2. The hits + anchor go through ``get_user_prompt`` from
     ``src.utils.prompts.refwalker`` to build the Korean Context/Question
     prompt. ``REFWALK_SYSTEM`` (same module) is used as the system role.
  3. The answer LLM is called in JSON-object mode (``Qwen3_5_HParams['exact']``
     by default — deterministic enough for citation extraction). The
     prompt fixes the schema as
     ``{"<node_id>": [<claim>, ...], ..., "answer": "<str>"}`` so the
     emitted JSON keys *are* the cited node_ids.
  4. ``cited_references`` are pulled with the trivial logic the user
     specified::

         cited_references = [k for k in output.keys() if k != "answer"]

     We additionally roll over-specific keys (e.g. ``..._제22조_제4항_제4호``
     when only ``..._제22조_제4항`` is in the corpus) up to the nearest
     valid ancestor — this matches ``src.baselines.native_rag._parse_cited_ids``
     so the downstream ``cli/run_generation_eval.py`` scorer treats
     RefWalk citations identically to NativeRAG / Walker-RAG.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Union

from src.base_llm import BaseLLM
from src.retrieval.walker_retrieval import RetrievalHit, WalkerRetrievalPipeline

# REFWALK_SYSTEM / get_user_prompt live in ``src.utils.prompts.refwalker``,
# but importing that submodule eagerly drags in the package
# ``src.utils.prompts.__init__`` which currently re-exports legacy
# Planner/Walker symbols that no longer exist in the refactored module.
# We mirror the lazy-import pattern used by
# ``src.retrieval.walker_retrieval.TopicExtractor`` to avoid taking that
# dependency at module import time.


# ─── Result dataclass ─────────────────────────────────────────────────


@dataclass
class RefWalkResult:
    answer: str
    cited_references: list[str]
    retrieved_node_ids: list[str]
    raw_output: Optional[dict] = None
    raw_retrieved: list[tuple[str, float]] = field(default_factory=list)
    rerank_scores: list[float] = field(default_factory=list)
    anchor: Optional[dict] = None
    status: str = "ok"
    error: Optional[str] = None


# ─── Citation parsing ─────────────────────────────────────────────────


_KOR_BOUNDARY_FIXES: tuple[tuple[str, str], ...] = (
    ("법시행령", "법_시행령"),
    ("법시행규칙", "법_시행규칙"),
)


def _rollup(node_id: str, valid: set[str]) -> Optional[str]:
    if node_id in valid:
        return node_id
    segs = node_id.split("_")
    while len(segs) > 1:
        segs.pop()
        candidate = "_".join(segs)
        if candidate in valid:
            return candidate
    return None


def _resolve_to_valid(node_id: str, valid: set[str]) -> Optional[str]:
    """Walk up the underscore hierarchy to find the longest valid ancestor.

    Adds a Korean-legal-document boundary repair pass before the rollup:
    LLMs occasionally emit ``..법시행령..`` as one token where the corpus
    uses ``..법_시행령..`` (i.e., drop the underscore between a parent law
    name and its 시행령/시행규칙 child). When the raw ``node_id`` has no
    valid ancestor by underscore rollup, we re-attempt resolution after
    applying each known boundary fix.
    """
    direct = _rollup(node_id, valid)
    if direct is not None:
        return direct
    for old, new in _KOR_BOUNDARY_FIXES:
        if old in node_id:
            fixed = node_id.replace(old, new)
            r = _rollup(fixed, valid)
            if r is not None:
                return r
    return None


def _extract_citations(
    output: dict, valid: Optional[set[str]] = None
) -> list[str]:
    """Pull cited node_ids from the JSON output.

    Trivial logic per spec:

        cited_references = [k for k in output.keys() if k != "answer"]

    Defensive normalisation:
      * Leading/trailing whitespace is stripped.
      * The user prompt's Context block displays each passage header as
        ``[<node_id>]`` so the LLM (especially smaller models) can mimic
        that format and emit keys like ``"[<node_id>]"``. Strip a single
        layer of surrounding ``[ ]`` before resolution.

    When ``valid`` is supplied, over-specific ids are rolled up to the
    longest ancestor present in the corpus (mirrors NativeRAG /
    WalkerRAG citation parsing so the eval scorer is identical).
    """
    cited: list[str] = []
    seen: set[str] = set()
    for cite in output.keys():
        if cite == "answer":
            continue
        if not isinstance(cite, str):
            continue
        cite = cite.strip()
        # Strip a single layer of surrounding `[ ]` — Context block format
        # leakage from the user prompt header.
        if len(cite) >= 2 and cite.startswith("[") and cite.endswith("]"):
            cite = cite[1:-1].strip()
        if not cite or "_" not in cite:
            continue
        if valid is not None:
            resolved = _resolve_to_valid(cite, valid)
            if resolved is None:
                continue
            cite = resolved
        if cite in seen:
            continue
        seen.add(cite)
        cited.append(cite)
    return cited


# ─── Workflow ─────────────────────────────────────────────────────────


class RefWalkWorkflow:
    """End-to-end RefWalk pipeline for one question.

    Parameters
    ----------
    pipeline
        Pre-built :class:`WalkerRetrievalPipeline`. Caller owns the
        corpus / dense / OKG / reranker so they can be shared across
        CLIs and notebooks.
    context_top_k
        How many reranked passages to feed to the LLM.
    passage_limit
        Per-passage character cap inside the Context block (matches
        the ``passage_limit`` knob of :func:`get_user_prompt`).
    profile_name
        Sampling profile from ``src.utils.hparams.Qwen3_5_HParams``.
        Defaults to ``"exact"`` (low-temperature, no thinking) — the
        REFWALK system prompt expects strict JSON output.
    """

    def __init__(
        self,
        pipeline: WalkerRetrievalPipeline,
        *,
        context_top_k: int = 10,
        eval_top_k: Optional[int] = None,
        passage_limit: int = 3000,
        llm_base_url: str = "http://localhost:8035/v1",
        llm_model: str = "Qwen/Qwen3.6-35B-A3B-FP8",
        llm_api_key: Optional[str] = None,
        profile_name: str = "exact",
        max_tokens: int = 2048,
        language: str = "ko",
        domain: str = "regops",
        oracle_total_k: int = 10,
        oracle_seed: int = 42,
        condition_anchor: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.context_top_k = int(context_top_k)
        # eval_top_k decouples the retrieval pool size (used for downstream
        # retrieval R@K analytics) from the prompt's passage budget. When
        # the caller wants context_top_k=5 in the prompt but R@10 in the
        # logs, set eval_top_k=10 — the workflow retrieves 10, feeds the
        # top context_top_k to the LLM, and stores all 10 in
        # ``retrieved_node_ids``. Defaults to ``context_top_k`` (legacy).
        self.eval_top_k = int(eval_top_k) if eval_top_k is not None else self.context_top_k
        if self.eval_top_k < self.context_top_k:
            self.eval_top_k = self.context_top_k
        self.passage_limit = int(passage_limit)
        self.max_tokens = int(max_tokens)
        self.language = language
        self.domain = domain
        self.oracle_total_k = int(oracle_total_k)
        self.oracle_seed = int(oracle_seed)
        # When False, the topic anchor is NOT conditioned into the
        # generation prompt (retrieval still uses it). Ablation for §4.2.
        self.condition_anchor = bool(condition_anchor)

        self.valid_node_ids = set(pipeline.corpus.node_ids)
        # Cache id → body text, used when composing context passages.
        self.id2text = dict(zip(
            pipeline.corpus.node_ids, pipeline.corpus.texts,
        ))

        from src.utils.prompts.refwalker import get_refwalk_system
        # schema={} (non-None) flips BaseLLM into json_object mode so
        # ``_call_llm`` returns a parsed dict.
        self.llm = BaseLLM(
            name="refwalk",
            base_url=llm_base_url,
            model_id=llm_model,
            max_tokens=max_tokens,
            profile_name=profile_name,
            system_prompt=get_refwalk_system(language),
            schema={},
            api_key=llm_api_key,
        )

    # ── Public entry point ────────────────────────────────────────────

    def answer(self, qa: Union[str, dict]) -> RefWalkResult:
        """Retrieve via Walker, then generate JSON-cited answer.

        ``qa`` accepts the same shapes as
        :meth:`WalkerRetrievalPipeline.retrieve_for_generation`:

          * ``str`` — raw question (online topic anchoring fires).
          * ``dict`` — pre-anchored or full benchmark row.
        """
        if isinstance(qa, str):
            payload: Union[str, dict] = qa
        elif isinstance(qa, dict):
            payload = qa
        else:
            return RefWalkResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=[],
                status="error",
                error=f"unsupported_qa_type: {type(qa).__name__}",
            )

        # 1. Retrieval — Walker handles anchoring → seed → OKG expand → rerank.
        # Retrieve eval_top_k for downstream R@K analytics; the LLM only
        # sees the leading context_top_k.
        try:
            hits, anchor = self.pipeline.retrieve_for_generation(
                payload, top_k=self.eval_top_k,
            )
        except Exception as e:
            return RefWalkResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=[],
                status="error",
                error=f"walker_retrieve_failed: {e}",
            )

        if not hits:
            return RefWalkResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=[],
                anchor=anchor if isinstance(anchor, dict) else None,
                status="error",
                error="no_passages_after_retrieval",
            )

        # Full eval_top_k goes into retrieved_node_ids / rerank_scores so
        # downstream R@K analytics can read them; only the leading
        # context_top_k passages are stitched into the LLM prompt.
        retrieved_ids = [h.node_id for h in hits]
        rerank_scores = [round(h.rerank_score, 4) for h in hits]
        raw_retrieved = [(h.node_id, h.hybrid_score) for h in hits]
        prompt_hits = hits[: self.context_top_k]

        # 2. Build prompt via the shared refwalker prompt helpers.
        # ``get_user_prompt`` mutates ``anchor`` in place (unspecified→all);
        # pass a shallow copy so the returned ``anchor`` field still
        # reflects the original extractor output for downstream tooling.
        from src.utils.prompts.refwalker import get_user_prompt
        prompt = get_user_prompt(
            prompt_hits, dict(anchor) if isinstance(anchor, dict) else {},
            passage_limit=self.passage_limit,
            language=self.language,
            domain=self.domain,
            condition_anchor=self.condition_anchor,
        )

        # 3. Generate — JSON-object mode, ``_call_llm`` returns the parsed dict.
        try:
            output = self.llm._call_llm(prompt=prompt)
        except Exception as e:
            return RefWalkResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=retrieved_ids,
                raw_retrieved=raw_retrieved,
                rerank_scores=rerank_scores,
                anchor=anchor if isinstance(anchor, dict) else None,
                status="error",
                error=f"llm_failed: {e}",
            )

        if not isinstance(output, dict):
            return RefWalkResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=retrieved_ids,
                raw_retrieved=raw_retrieved,
                rerank_scores=rerank_scores,
                anchor=anchor if isinstance(anchor, dict) else None,
                status="error",
                error="llm_output_not_json",
            )

        # 4. Extract citations + answer.
        cited = _extract_citations(output, valid=self.valid_node_ids)
        answer_text = str(output.get("answer") or "").strip()

        return RefWalkResult(
            answer=answer_text,
            cited_references=cited,
            retrieved_node_ids=retrieved_ids,
            raw_output=output,
            raw_retrieved=raw_retrieved,
            rerank_scores=rerank_scores,
            anchor=anchor if isinstance(anchor, dict) else None,
            status="ok",
        )

    # ── Oracle (ground-truth context) entry point ─────────────────────

    def _gt_hits(self, qa: dict) -> list[RetrievalHit]:
        """Build the prompt context from the benchmark ``gt_references``
        plus 1-hop OKG distractors, instead of the Walker retrieval stack.

        The context is the GT articles padded up to
        ``self.oracle_total_k`` with randomly-sampled 1-hop OKG
        neighbours of the GT (see
        :func:`src.eval.oracle_context.build_oracle_context`), shuffled
        deterministically per ``qa_id``. GT hits carry
        ``rerank_score=1.0`` / ``source="oracle_gt"``; distractors
        ``rerank_score=0.0`` / ``source="oracle_distractor"`` — the
        prediction row stays schema-identical to :meth:`answer`.
        """
        from src.eval.oracle_context import build_oracle_context

        ordered, gt_set, _distract = build_oracle_context(
            qa.get("gt_references") or [],
            self.pipeline.graph,
            self.valid_node_ids,
            total_k=self.oracle_total_k,
            seed=self.oracle_seed,
            qa_id=str(qa.get("qa_id") or ""),
        )
        hits: list[RetrievalHit] = []
        for nid in ordered:
            is_gt = nid in gt_set
            hits.append(
                RetrievalHit(
                    node_id=nid,
                    text=self.id2text.get(nid, ""),
                    hybrid_score=1.0 if is_gt else 0.0,
                    rerank_score=1.0 if is_gt else 0.0,
                    source="oracle_gt" if is_gt else "oracle_distractor",
                )
            )
        return hits

    def answer_gt(self, qa: Union[str, dict]) -> RefWalkResult:
        """Generate the JSON-cited answer from the **ground-truth**
        articles (oracle context) instead of retrieved passages.

        This isolates pure generation quality (answer faithfulness +
        per-clause citation precision) from retrieval error: the context
        is exactly the bench ``gt_references`` corpus bodies. Topic /
        condition anchoring still runs (it conditions generation, not
        retrieval), so the prompt is otherwise identical to
        :meth:`answer`.
        """
        if not isinstance(qa, dict):
            return RefWalkResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=[],
                status="error",
                error="oracle_mode_requires_dict_with_gt_references",
            )

        try:
            anchor = self.pipeline._anchor_query(qa)
        except Exception as e:
            return RefWalkResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=[],
                status="error",
                error=f"anchor_failed: {e}",
            )

        hits = self._gt_hits(qa)
        if not hits:
            return RefWalkResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=[],
                anchor=anchor if isinstance(anchor, dict) else None,
                status="error",
                error="no_gt_references_resolvable_to_corpus",
            )

        retrieved_ids = [h.node_id for h in hits]
        rerank_scores = [round(h.rerank_score, 4) for h in hits]
        raw_retrieved = [(h.node_id, h.hybrid_score) for h in hits]
        # Oracle context is already bounded by ``oracle_total_k``; feed the
        # full GT+distractor set so shuffling never truncates a GT article.
        prompt_hits = hits

        from src.utils.prompts.refwalker import get_user_prompt
        prompt = get_user_prompt(
            prompt_hits, dict(anchor) if isinstance(anchor, dict) else {},
            passage_limit=self.passage_limit,
            language=self.language,
            domain=self.domain,
            condition_anchor=self.condition_anchor,
        )

        try:
            output = self.llm._call_llm(prompt=prompt)
        except Exception as e:
            return RefWalkResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=retrieved_ids,
                raw_retrieved=raw_retrieved,
                rerank_scores=rerank_scores,
                anchor=anchor if isinstance(anchor, dict) else None,
                status="error",
                error=f"llm_failed: {e}",
            )

        if not isinstance(output, dict):
            return RefWalkResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=retrieved_ids,
                raw_retrieved=raw_retrieved,
                rerank_scores=rerank_scores,
                anchor=anchor if isinstance(anchor, dict) else None,
                status="error",
                error="llm_output_not_json",
            )

        cited = _extract_citations(output, valid=self.valid_node_ids)
        answer_text = str(output.get("answer") or "").strip()

        return RefWalkResult(
            answer=answer_text,
            cited_references=cited,
            retrieved_node_ids=retrieved_ids,
            raw_output=output,
            raw_retrieved=raw_retrieved,
            rerank_scores=rerank_scores,
            anchor=anchor if isinstance(anchor, dict) else None,
            status="ok",
        )
