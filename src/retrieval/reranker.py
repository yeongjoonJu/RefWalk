from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from transformers import AutoTokenizer


DEFAULT_ENDPOINT = os.environ.get(
    "QWEN3_RERANKER_ENDPOINT", "http://localhost:8095"
)
DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"

DEFAULT_INSTRUCTION = (
    "Given a user question, "
    "retrieve legal clauses or regulations that answer the question."
)

# Conservative character budget per (query, doc). Keeps the
# chat-templated payload well under the 32k context window even for
# 별표 nodes; Korean BPE tokens are roughly half the char count.
_DOC_CHAR_CAP = 3000
_QUERY_CHAR_CAP = 500


# ─── vLLM-style prompt scaffolding ──────────────────────────────────
# Verbatim from the vLLM Qwen3-Reranker example. The server expects
# the caller to splice these around the (instruction, query, doc)
# triple — the chat template is NOT applied server-side on /v1/rerank.

_PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the '
    'Query and the Instruct provided. Note that the answer can only '
    'be "yes" or "no".<|im_end|>\n'
    '<|im_start|>user\n'
)

_SUFFIX = (
    '<|im_end|>\n'
    '<|im_start|>assistant\n'
    '<think>\n\n</think>\n\n'
)

_QUERY_TEMPLATE = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
_DOCUMENT_TEMPLATE = "<Document>: {doc}{suffix}"


# Tokenizer is only needed for the offline ``process_inputs`` helper.
# Loaded lazily so the module can be imported in environments that
# don't have the transformers cache populated yet (the HTTP rerank
# path doesn't need it at all).
_tokenizer: Any = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    return _tokenizer


# Backward-compat module alias — older callers / external scripts may
# import ``tokenizer`` directly. Lazy resolution keeps the original
# eager-load symptom (any failure surfaces on first access).
class _LazyTokenizerProxy:
    def __getattr__(self, name):
        return getattr(_get_tokenizer(), name)

    def __call__(self, *a, **kw):
        return _get_tokenizer()(*a, **kw)


tokenizer = _LazyTokenizerProxy()  # type: ignore[assignment]


# ─── Prompt builders (parity with official README) ─────────────────


def format_instruction(instruction: str, query: str, doc: str) -> list[dict]:
    """Build the chat-message list that Qwen3-Reranker's ship-with
    chat template expects (used by the offline ``process_inputs``
    path; the HTTP path uses the prefix/suffix template literals).

    Roles ``system`` / ``query`` / ``document`` are the model-specific
    role names whose template fills ``<Instruct>:`` / ``<Query>:`` /
    ``<Document>:`` slots.
    """
    return [
        {"role": "system", "content": instruction},
        {"role": "query", "content": query},
        {"role": "document", "content": doc},
    ]


def process_inputs(pairs, task_instruction, max_length, suffix_tokens=()):
    """Offline-vLLM helper: returns ``vllm.inputs.data.TokensPrompt``
    objects suitable for ``LLM.generate(prompts, sampling_params)``.

    Mirrors the official Qwen3-Reranker README. The HTTP
    ``Qwen3Reranker`` client below uses ``/v1/rerank`` instead and
    bypasses this code path.
    """
    from vllm.inputs.data import TokensPrompt

    tok = _get_tokenizer()
    messages = [
        format_instruction(task_instruction, query, doc) for query, doc in pairs
    ]
    messages = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    messages = [ele[:max_length] + list(suffix_tokens) for ele in messages]
    return [TokensPrompt(prompt_token_ids=ele) for ele in messages]


# ─── Helpers ───────────────────────────────────────────────────────


def _truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[:cap]


@dataclass
class RerankItem:
    """One candidate being reranked.

    ``metadata`` is opaque — round-tripped onto the output dict.
    Typical caller supplies ``{"hybrid_score": float, "source": str}``.
    """

    node_id: str
    text: str
    metadata: dict = field(default_factory=dict)


# ─── HTTP client ───────────────────────────────────────────────────


class Qwen3Reranker:
    """Thin client over a vLLM server's ``/v1/rerank`` endpoint.

    Each ``rerank`` call sends one chat-template-formatted query plus
    a list of pre-suffixed documents; the server returns one
    ``relevance_score`` (yes-probability) per document.

    Parameters
    ----------
    endpoint    : base URL of the vLLM server (default env
                  ``QWEN3_RERANKER_ENDPOINT`` or
                  ``http://localhost:8095``).
    model       : model identifier registered on the server.
                  Optional — most vLLM deployments ignore the field on
                  ``/v1/rerank``, but we forward it for symmetry.
    instruction : English instruction injected into the
                  ``<Instruct>:`` slot of the query string.
                  Pass an empty string to skip — the slot becomes
                  empty, which is generally fine for Qwen3 but may
                  reduce score quality.
    timeout     : HTTP request timeout (seconds).
    batch_size  : max documents per ``/v1/rerank`` call. The server
                  batches internally, but very long batches can
                  exceed proxy timeouts — chunk transparently.

    Usage
    -----
    >>> rr = Qwen3Reranker()
    >>> out = rr.rerank(
    ...     query="인건비 계상률 산정",
    ...     candidates=[RerankItem("A", "..."), RerankItem("B", "...")],
    ...     top_k=5,
    ... )
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        instruction: str = DEFAULT_INSTRUCTION,
        timeout: float = 60.0,
        batch_size: int = 32,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.instruction = instruction
        self.timeout = timeout
        self.batch_size = max(1, int(batch_size))

    # ── Prompt formatting (string templates, not chat template) ────

    def _format_query(self, query: str) -> str:
        """Wrap the user query with the system prefix + Instruct/Query
        chat-template lines that the vLLM server expects on the
        ``query`` field of ``/v1/rerank``."""
        return _QUERY_TEMPLATE.format(
            prefix=_PREFIX,
            instruction=self.instruction or "",
            query=query,
        )

    @staticmethod
    def _format_document(doc: str) -> str:
        """Wrap a document with ``<Document>:`` plus the assistant-
        prime suffix that primes the model for the yes/no judgement."""
        return _DOCUMENT_TEMPLATE.format(doc=doc, suffix=_SUFFIX)

    # ── Low-level call ─────────────────────────────────────────────

    def _call_rerank(
        self, query_text: str, doc_texts: list[str]
    ) -> list[float]:
        """POST one batch to ``/v1/rerank``. Returns one
        ``relevance_score`` per input doc, in input order."""
        if not doc_texts:
            return []
        formatted_query = self._format_query(query_text)
        formatted_docs = [self._format_document(d) for d in doc_texts]
        payload: dict[str, Any] = {
            "model": self.model,
            "query": formatted_query,
            "documents": formatted_docs,
        }
        resp = requests.post(
            f"{self.endpoint}/v1/rerank",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        results = body.get("results") or []
        # Default to 0.0 so we never silently return wrong-length list.
        scores = [0.0] * len(doc_texts)
        for r in results:
            idx = r.get("index")
            sc = r.get("relevance_score")
            if idx is None or sc is None:
                continue
            if 0 <= idx < len(scores):
                scores[idx] = float(sc)
        return scores

    # ── Public API ─────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list[RerankItem],
        top_k: int = 10,
    ) -> list[dict]:
        """Return candidates sorted by reranker score, truncated to
        ``top_k``. Each element is a dict with keys:

            node_id, text, rerank_score, metadata

        Metadata from the input RerankItem is preserved verbatim.
        An empty ``candidates`` list returns ``[]``.
        """
        if not candidates:
            return []

        query_text = _truncate(query.strip(), _QUERY_CHAR_CAP)
        doc_texts = [_truncate(c.text or "", _DOC_CHAR_CAP) for c in candidates]

        all_scores: list[float] = []
        for i in range(0, len(doc_texts), self.batch_size):
            chunk = doc_texts[i : i + self.batch_size]
            all_scores.extend(self._call_rerank(query_text, chunk))

        assert len(all_scores) == len(candidates), (
            f"reranker score count mismatch: got {len(all_scores)} "
            f"for {len(candidates)} candidates"
        )

        scored = [
            {
                "node_id": c.node_id,
                "text": c.text,
                "rerank_score": float(s),
                "metadata": dict(c.metadata),
            }
            for c, s in zip(candidates, all_scores)
        ]
        scored.sort(key=lambda d: d["rerank_score"], reverse=True)
        return scored[:top_k]


# ─── Convenience: adapter to retrieval baseline candidate format ──


def candidates_from_pairs(
    pairs: list[tuple[str, float]],
    id_to_text: dict[str, str],
    source: str = "hybrid",
    score_key: str = "hybrid_score",
) -> list[RerankItem]:
    """Convert ``[(node_id, score), …]`` + id→text map into RerankItems.

    Nodes missing from ``id_to_text`` are skipped.
    """
    out: list[RerankItem] = []
    for nid, score in pairs:
        text = id_to_text.get(nid)
        if text is None:
            continue
        out.append(
            RerankItem(
                node_id=nid,
                text=text,
                metadata={score_key: float(score), "source": source},
            )
        )
    return out


