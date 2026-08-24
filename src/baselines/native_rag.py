from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.base_llm import BaseLLM
from src.retrieval.corpus import load_corpus
from src.retrieval.embedding import DenseRetriever
from src.retrieval.reranker import Qwen3Reranker, candidates_from_pairs


# ─── Prompt templates ──────────────────────────────────────────────────
# Two language profiles: "ko" for the Korean RegOps benchmark, "en" for
# an English regulatory corpus. Both emit the same citation footer format
# so the downstream generation_eval scorer can parse either.

_PROMPTS = {
    "ko": {
        "system": (
            "당신은 한국 국가연구개발 규정에 정통한 법무 어시스턴트입니다. "
            "주어진 조항(법령·시행령·고시·기준)에 근거하여 사용자의 질문에 한국어로 "
            "답변하세요. 조항에 명시되지 않은 사항은 추측하지 말고, 부족한 경우 "
            "그 사실을 명시하세요."
        ),
        "citation_instruction": (
            "답변의 마지막 줄에 다음 형식으로 인용한 조항의 node_id 목록을 출력하라.\n"
            "예: [참조] 국가연구개발사업_연구개발비_사용기준_제48조, "
            "국가연구개발사업_연구개발비_사용기준_제48조_제8항\n"
            "node_id는 위 'Reference Document List'에 표기된 항목을 그대로 사용하라."
        ),
        "question_label": "질문",
    },
    "en": {
        "system": (
            "You are a legal/compliance assistant well-versed in regulatory "
            "compliance. Answer the user's question in English, "
            "grounded strictly in the rules provided below. Do not speculate "
            "beyond what the rules state; if the rules are insufficient to "
            "answer, say so explicitly."
        ),
        "citation_instruction": (
            "On the final line of your answer, list the node_ids of the rules "
            "you cited in the format below.\n"
            "Example: [Citations] spans-passthrough-candidates/sent_0012-0ec05553, "
            "spans-passthrough-candidates/sent_0010-99ea86fe\n"
            "Use node_ids exactly as they appear in the 'Reference Document List' "
            "above; do not invent identifiers."
        ),
        "question_label": "Question",
    },
}


# Match a `[참조] ...` / `[Citations] ...` footer line — used to lift cited
# node_ids back out of the answer.
_CITATION_FOOTER_RE = re.compile(
    r"\[(?:참조|Citations?|References?)\][^\n]*", re.IGNORECASE
)


def _parse_cited_ids(answer: str, valid: set[str] | None = None) -> list[str]:
    """Lift node_ids from the trailing ``[참조] a, b, c`` footer.

    Tokens without ``_`` are dropped (keeps Korean phrases like "참조"
    out). When ``valid`` is supplied, the parser keeps only ids that
    resolve into the corpus — but if the LLM emits an *over-specific*
    id (e.g. ``..._별표1_1``, ``..._별표1_10`` when only ``..._별표1``
    exists in the corpus, or ``..._제48조_제8항_제3호`` when only
    ``..._제48조_제8항`` exists), it is **rolled up to the longest
    ancestor in valid** by trimming trailing ``_<segment>`` until a hit.
    Tokens that have no valid ancestor are dropped (true hallucinations).
    """
    if not answer:
        return []
    matches = _CITATION_FOOTER_RE.findall(answer)
    if not matches:
        return []
    tail = matches[-1].split("]", 1)[1] if "]" in matches[-1] else ""
    out: list[str] = []
    seen: set[str] = set()
    for tok in re.split(r"[,\s;、，]+", tail):
        tok = tok.strip().strip(".,")
        if not tok or "_" not in tok:
            continue
        if valid is not None and tok not in valid:
            # Walk up the underscore hierarchy until we hit a valid
            # ancestor. This recovers the over-specific case the LLM
            # routinely produces from rule-extraction notes whose tags
            # include sub-item depth not represented as graph nodes.
            resolved: str | None = None
            segs = tok.split("_")
            while len(segs) > 1:
                segs.pop()
                candidate = "_".join(segs)
                if candidate in valid:
                    resolved = candidate
                    break
            if resolved is None:
                continue
            tok = resolved
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def _resolve_to_valid(node_id: str, valid: set[str]) -> Optional[str]:
    """Roll a (possibly over-specific) gt reference up to the longest
    ancestor present in the corpus.

    The bench ``gt_references`` are frequently clause-level
    (``..._제48조_제8항``) while the parse-stage corpus is article-level
    (``..._제48조``). Trim trailing ``_<segment>`` until a corpus hit;
    return ``None`` when nothing in the hierarchy is in the corpus.
    Mirrors the rollup in :func:`_parse_cited_ids` so oracle context and
    citation scoring share one granularity.
    """
    if node_id in valid:
        return node_id
    segs = node_id.split("_")
    while len(segs) > 1:
        segs.pop()
        candidate = "_".join(segs)
        if candidate in valid:
            return candidate
    return None


# ─── Result dataclass ─────────────────────────────────────────────────


@dataclass
class NativeRagResult:
    answer: str
    cited_references: list[str]
    retrieved_node_ids: list[str]
    raw_retrieved: list[tuple[str, float]] = field(default_factory=list)
    rerank_scores: list[float] = field(default_factory=list)
    status: str = "ok"
    error: Optional[str] = None


# ─── Workflow ─────────────────────────────────────────────────────────


class NativeRagWorkflow:
    """End-to-end NativeRAG pipeline for one question.

    Parameters
    ----------
    articles_path : path to a parse-stage articles jsonl (one row per 조,
        with sub-paragraphs/items already concatenated into ``text_ko``).
    embed_cache : optional ``.npy`` path for the document embedding
        cache. Reused if it matches the corpus size; otherwise rebuilt.
    reranker : pre-constructed ``Qwen3Reranker`` to rerank dense hits;
        pass ``None`` to skip reranking entirely.
    retrieve_top_k : how many candidates to fetch from dense retrieval.
    context_top_k : how many passages to feed to the LLM after rerank
        (or directly from dense if no reranker).
    passage_max_chars : per-passage character cap inside the prompt to
        keep the total context under the LLM's window.
    """

    def __init__(
        self,
        articles_path: Path,
        embed_cache: Path | None = None,
        reranker: Optional[Qwen3Reranker] = None,
        retrieve_top_k: int = 20,
        context_top_k: int = 5,
        llm_base_url: str = "http://localhost:8035/v1",
        llm_model: str = "Qwen/Qwen3.6-35B-A3B-FP8",
        llm_api_key: Optional[str] = None,
        profile_name: str = "non_thinking",
        max_tokens: int = 2048,
        passage_max_chars: int = 2400,
        language: str = "ko",
        okg_graph=None,
        oracle_total_k: int = 10,
        oracle_seed: int = 42,
    ) -> None:
        articles_path = Path(articles_path)
        embed_cache = Path(embed_cache) if embed_cache is not None else None

        if language not in _PROMPTS:
            raise ValueError(
                f"unsupported language={language!r}; "
                f"choose one of {sorted(_PROMPTS)}"
            )
        self.language = language
        self._prompts = _PROMPTS[language]

        self.corpus = load_corpus(articles_path, source="parse")
        self.id_to_text = dict(zip(self.corpus.node_ids, self.corpus.texts))
        self.valid_node_ids = set(self.corpus.node_ids)

        if embed_cache is not None and embed_cache.exists():
            self.dense = DenseRetriever.from_cache(self.corpus, embed_cache)
        else:
            self.dense = DenseRetriever.build(self.corpus)
            if embed_cache is not None:
                embed_cache.parent.mkdir(parents=True, exist_ok=True)
                self.dense.save_cache(embed_cache)

        # OKG graph is only consulted by the oracle path (1-hop
        # distractor sampling); the retrieval path never touches it.
        self.okg_graph = okg_graph
        self.oracle_total_k = int(oracle_total_k)
        self.oracle_seed = int(oracle_seed)

        self.reranker = reranker
        self.retrieve_top_k = max(int(retrieve_top_k), int(context_top_k))
        self.context_top_k = int(context_top_k)
        self.passage_max_chars = int(passage_max_chars)
        self.max_tokens = int(max_tokens)

        self.llm = BaseLLM(
            name="native_rag",
            base_url=llm_base_url,
            model_id=llm_model,
            max_tokens=max_tokens,
            profile_name=profile_name,
            system_prompt=self._prompts["system"],
            schema=None,
            api_key=llm_api_key,
        )

    # ── Prompt builder ────────────────────────────────────────────────

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
            lines.append(f"[{i}] {nid}\n{text}")
        lines.append("")
        lines.append(self._prompts["citation_instruction"])
        return "\n".join(lines)

    # ── Public entry point ────────────────────────────────────────────

    def answer(self, question: str) -> NativeRagResult:
        try:
            raw_pairs = self.dense.search(question, top_k=self.retrieve_top_k)
        except Exception as e:
            return NativeRagResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=[],
                raw_retrieved=[],
                status="error",
                error=f"dense_search_failed: {e}",
            )

        rerank_scores: list[float] = []
        if self.reranker is not None and raw_pairs:
            try:
                items = candidates_from_pairs(
                    raw_pairs, self.id_to_text,
                    source="dense", score_key="dense_score",
                )
                ranked = self.reranker.rerank(
                    question, items, top_k=self.context_top_k,
                )
                passages = [
                    {
                        "node_id": r["node_id"],
                        "text": r["text"],
                        "rerank_score": r["rerank_score"],
                    }
                    for r in ranked
                ]
                rerank_scores = [p["rerank_score"] for p in passages]
            except Exception as e:
                return NativeRagResult(
                    answer="",
                    cited_references=[],
                    retrieved_node_ids=[nid for nid, _ in raw_pairs],
                    raw_retrieved=raw_pairs,
                    status="error",
                    error=f"rerank_failed: {e}",
                )
        else:
            passages = []
            for nid, _ in raw_pairs[: self.context_top_k]:
                text = self.id_to_text.get(nid)
                if text is None:
                    continue
                passages.append({"node_id": nid, "text": text})

        retrieved_ids = [p["node_id"] for p in passages]

        if not passages:
            return NativeRagResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=retrieved_ids,
                raw_retrieved=raw_pairs,
                rerank_scores=rerank_scores,
                status="error",
                error="no_passages_after_retrieval",
            )

        prompt = self._build_user_prompt(question, passages)

        try:
            raw = self.llm._call_once(prompt, response_format=None)
        except Exception as e:
            return NativeRagResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=retrieved_ids,
                raw_retrieved=raw_pairs,
                rerank_scores=rerank_scores,
                status="error",
                error=f"llm_failed: {e}",
            )

        answer_text = (raw or "").strip()
        answer_text = re.sub(
            r"<think>.*?</think>", "", answer_text, flags=re.DOTALL
        ).strip()
        cited = _parse_cited_ids(answer_text, valid=self.valid_node_ids)

        return NativeRagResult(
            answer=answer_text,
            cited_references=cited,
            retrieved_node_ids=retrieved_ids,
            raw_retrieved=raw_pairs,
            rerank_scores=rerank_scores,
            status="ok",
        )

    # ── Oracle (ground-truth context) entry point ─────────────────────

    def answer_gt(self, qa: dict) -> NativeRagResult:
        """Answer from the **ground-truth** articles plus 1-hop OKG
        distractors instead of dense retrieval — isolates pure generation
        quality from retrieval error while keeping citation non-trivial.

        Context = GT articles padded up to ``self.oracle_total_k`` with
        randomly-sampled 1-hop OKG neighbours of the GT (see
        :func:`src.eval.oracle_context.build_oracle_context`), shuffled
        deterministically per ``qa_id``. Dense retrieval / reranker are
        not invoked. Requires ``okg_graph`` to have been supplied at
        construction time.
        """
        if not isinstance(qa, dict):
            return NativeRagResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=[],
                status="error",
                error="oracle_mode_requires_dict_with_gt_references",
            )

        question = (qa.get("question") or "").strip()
        if not question:
            return NativeRagResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=[],
                status="error",
                error="empty question",
            )

        if self.okg_graph is None:
            return NativeRagResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=[],
                status="error",
                error="oracle_mode_requires_okg_graph",
            )

        from src.eval.oracle_context import build_oracle_context

        ordered, _gt_set, _distract = build_oracle_context(
            qa.get("gt_references") or [],
            self.okg_graph,
            self.valid_node_ids,
            total_k=self.oracle_total_k,
            seed=self.oracle_seed,
            qa_id=str(qa.get("qa_id") or ""),
        )
        passages: list[dict] = [
            {"node_id": nid, "text": self.id_to_text.get(nid, "")}
            for nid in ordered
        ]

        retrieved_ids = [p["node_id"] for p in passages]
        if not passages:
            return NativeRagResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=[],
                status="error",
                error="no_gt_references_resolvable_to_corpus",
            )

        prompt = self._build_user_prompt(question, passages)

        try:
            raw = self.llm._call_once(prompt, response_format=None)
        except Exception as e:
            return NativeRagResult(
                answer="",
                cited_references=[],
                retrieved_node_ids=retrieved_ids,
                status="error",
                error=f"llm_failed: {e}",
            )

        answer_text = (raw or "").strip()
        answer_text = re.sub(
            r"<think>.*?</think>", "", answer_text, flags=re.DOTALL
        ).strip()
        cited = _parse_cited_ids(answer_text, valid=self.valid_node_ids)

        return NativeRagResult(
            answer=answer_text,
            cited_references=cited,
            retrieved_node_ids=retrieved_ids,
            rerank_scores=[],
            status="ok",
        )
