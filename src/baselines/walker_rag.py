"""WalkerRAG = NativeRAG generation on top of Walker retrieval.

Plumbs the OKG-aware retrieval stack used by the "Ours" / Walker baseline
into the same prompt + LLM + citation-parser scaffolding that
``src.baselines.native_rag.NativeRagWorkflow`` uses, so the prediction
schema remains identical to NativeRAG / LightRAG / HippoRAG-2 / PIKE-RAG.

What's reused from NativeRAG:
  * ``_PROMPTS`` (system + user prompt for ko / en)
  * ``_build_user_prompt`` style — numbered ``[i] node_id`` reference list
  * ``_parse_cited_ids`` — pulls node_ids from the trailing
    ``[참조] …`` / ``[Citations] …`` footer
  * ``NativeRagResult`` dataclass (no fork — downstream
    ``run_generation_eval.py`` doesn't care about the workflow class).

What's swapped in:
  * Retrieval: ``WalkerRetrievalPipeline`` (topic + tagged seed +
    OKG 1-hop expansion + Qwen3-Reranker) instead of plain dense /
    dense+rerank.

The LLM step is identical to NativeRAG so any quality delta in scores
attributes cleanly to the retrieval substitution.
"""

from __future__ import annotations

import re
from typing import Union

from src.base_llm import BaseLLM
from src.baselines.native_rag import (
    NativeRagResult,
    _PROMPTS,
    _parse_cited_ids,
)
from src.retrieval.walker_retrieval import RetrievalHit, WalkerRetrievalPipeline


class WalkerRagWorkflow:
    """End-to-end Walker-retrieval RAG pipeline for one question.

    Parameters
    ----------
    pipeline
        Pre-built :class:`WalkerRetrievalPipeline`. The caller owns
        construction of corpus / dense / graph / reranker so they can be
        shared across CLIs and notebooks.
    context_top_k
        How many reranked passages to feed to the LLM. Matches NativeRAG's
        ``context_top_k`` so the prompt budget is comparable.
    passage_max_chars
        Per-passage character cap inside the prompt.
    llm_base_url, llm_model, profile_name, max_tokens, language
        Identical knobs to ``NativeRagWorkflow`` — same defaults so a
        fair head-to-head only varies the retriever.
    """

    def __init__(
        self,
        pipeline: WalkerRetrievalPipeline,
        *,
        context_top_k: int = 10,
        passage_max_chars: int = 2400,
        llm_base_url: str = "http://localhost:8035/v1",
        llm_model: str = "Qwen/Qwen3.6-35B-A3B-FP8",
        profile_name: str = "exact",
        max_tokens: int = 2048,
        language: str = "ko",
    ) -> None:
        if language not in _PROMPTS:
            raise ValueError(
                f"unsupported language={language!r}; "
                f"choose one of {sorted(_PROMPTS)}"
            )
        self.pipeline = pipeline
        self.context_top_k = int(context_top_k)
        self.passage_max_chars = int(passage_max_chars)
        self.max_tokens = int(max_tokens)
        self.language = language
        self._prompts = _PROMPTS[language]

        # The pipeline owns the corpus, so the citation validator can
        # reuse the same node_id set without re-loading.
        self.valid_node_ids = set(pipeline.corpus.node_ids)

        self.llm = BaseLLM(
            name="walker_rag",
            base_url=llm_base_url,
            model_id=llm_model,
            max_tokens=max_tokens,
            profile_name=profile_name,
            system_prompt=self._prompts["system"],
            schema=None,
        )

    # ── Prompt builder (mirrors NativeRagWorkflow._build_user_prompt) ──

    def _build_user_prompt(
        self, question: str, passages: list[dict]
    ) -> str:
        q_label = self._prompts["question_label"]
        lines: list[str] = [f"{q_label}: {question}", "", "Reference Document List:"]
        for i, p in enumerate(passages, 1):
            nid = p["node_id"]
            text = (p.get("text") or "").strip()
            if len(text) > self.passage_max_chars:
                text = text[: self.passage_max_chars] + "…"
            role = p.get("role")
            header = f"[{i}] [{role}] {nid}" if role else f"[{i}] {nid}"
            lines.append(f"{header}\n{text}")
        lines.append("")
        lines.append(self._prompts["citation_instruction"])
        return "\n".join(lines)

    # ── Public entry point ────────────────────────────────────────────

    def answer(self, qa: Union[str, dict]) -> NativeRagResult:
        """Retrieve via Walker, then generate + parse citations.

        ``qa`` accepts the same shapes as
        :meth:`WalkerRetrievalPipeline.retrieve`:

          * ``str`` — raw question (online topic anchoring fires).
          * ``dict`` — pre-anchored or full benchmark row. The dict's
            ``question`` / ``ori_question`` field is used as the
            human-facing question text in the LLM prompt.
        """
        if isinstance(qa, str):
            question = qa
            payload: Union[str, dict] = qa
        else:
            payload = qa
            question = (qa.get("question") or qa.get("ori_question") or "").strip()

        if not question:
            return NativeRagResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=[],
                status="error",
                error="empty_question",
            )

        # 1. Retrieval — Walker handles anchoring → seed → OKG expand → rerank.
        try:
            hits: list[RetrievalHit] = self.pipeline.retrieve(
                payload, top_k=self.context_top_k,
            )
        except Exception as e:
            return NativeRagResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=[],
                status="error",
                error=f"walker_retrieve_failed: {e}",
            )

        if not hits:
            return NativeRagResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=[],
                status="error",
                error="no_passages_after_retrieval",
            )

        passages = [{"node_id": h.node_id, "text": h.text} for h in hits]
        retrieved_ids = [h.node_id for h in hits]
        rerank_scores = [round(h.rerank_score, 4) for h in hits]
        raw_retrieved = [(h.node_id, h.hybrid_score) for h in hits]

        # 2. LLM answer — identical prompt scaffolding to NativeRAG.
        prompt = self._build_user_prompt(question, passages)
        try:
            raw = self.llm._call_once(prompt, response_format=None)
        except Exception as e:
            return NativeRagResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=retrieved_ids,
                raw_retrieved=raw_retrieved,
                rerank_scores=rerank_scores,
                status="error",
                error=f"llm_failed: {e}",
            )

        answer_text = (raw or "").strip()
        # Strip <think> blocks so the citation-footer parser sees the
        # final answer (matches NativeRagWorkflow behavior).
        answer_text = re.sub(
            r"<think>.*?</think>", "", answer_text, flags=re.DOTALL
        ).strip()
        cited = _parse_cited_ids(answer_text, valid=self.valid_node_ids)

        return NativeRagResult(
            answer=answer_text,
            cited_references=cited,
            retrieved_node_ids=retrieved_ids,
            raw_retrieved=raw_retrieved,
            rerank_scores=rerank_scores,
            status="ok",
        )
