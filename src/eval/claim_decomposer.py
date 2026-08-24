from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI

from src.utils.hparams import Qwen3_5_HParams


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_DEFAULT_CACHE_DIR = Path("data/cache/claim_decomposer")
_MAX_CLAIMS = 20  # truncation guardrail — an answer of 30+ claims is usually over-split


_DECOMPOSER_SYSTEM_KO = """당신은 한국 국가 R&D 규정 답변을 atomic claim 으로 분해하는 평가 도우미입니다.

atomic claim 정의:
- 하나의 주체(actor) + 하나의 행위(action) + 최대 하나의 조건/수식 으로 구성된 단일 규범 문장.
- 인용 번호, 법령 조문 표기(예: 「…」 제N조), 목록 기호(- , * , 1. ), **굵게** 등 서식은 claim 본문에서 제거.
- 두 개 이상의 조건이 접속사(및/또는/다만/그러나/단)로 결합된 경우 별도 claim 으로 분해.
- 단순 서술(배경 설명, 참고 링크)은 claim 으로 포함하지 않음.

출력은 반드시 다음 JSON schema 를 따르십시오:
{
  "claims": [
    {"id": "c1", "text": "<atomic claim 한국어 문장>"},
    {"id": "c2", "text": "..."}
  ]
}

주의:
- claim 수는 1~20 개 사이.
- 원문에 명시된 내용만 claim 으로 추출. 추론/추가 설명 금지.
- 동일 의미의 claim 중복 금지.
"""


_DECOMPOSER_SYSTEM_EN = """You are an evaluation assistant that decomposes a regulation answer into atomic claims.

Atomic claim definition:
- A single normative sentence with one actor + one action + at most one condition/qualifier.
- Strip citation markers, section numbers (e.g. §164.502, 45 CFR 160.103), list bullets (-, *, 1.), and bold (**...**) from the claim text.
- When two or more conditions are joined by "and / or / except / unless / provided that / subject to", split them into separate claims.
- Do NOT include background framing, hedges, or pointers ("see also …") as claims.

Your output MUST follow this JSON schema exactly:
{
  "claims": [
    {"id": "c1", "text": "<atomic claim in English>"},
    {"id": "c2", "text": "..."}
  ]
}

Rules:
- Emit between 1 and 20 claims.
- Extract only what the source text states; do not infer or add commentary.
- Do not emit two claims with the same meaning.
"""


_DECOMPOSER_USER_PREFIX = {
    "ko": "다음 답변을 atomic claim 리스트로 분해하여 JSON 으로 응답하십시오.\n\n답변:\n",
    "en": "Decompose the following answer into a list of atomic claims and respond as JSON.\n\nAnswer:\n",
}

_DECOMPOSER_SYSTEM = {"ko": _DECOMPOSER_SYSTEM_KO, "en": _DECOMPOSER_SYSTEM_EN}


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def _try_extract_json(text: str) -> Optional[dict]:
    text = _strip_thinking(text)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def _model_kwargs_from_profile(profile: dict) -> dict:
    kw: dict = {}
    for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
        if k in profile:
            kw[k] = profile[k]
    if "extra_body" in profile:
        kw["extra_body"] = dict(profile["extra_body"])
    return kw


def _hash_key(
    text: str, profile_name: str, model_id: str, language: str = "ko"
) -> str:
    h = hashlib.sha1()
    h.update(profile_name.encode("utf-8"))
    h.update(b"|")
    h.update(model_id.encode("utf-8"))
    h.update(b"|")
    # Only include language in the hash for non-default languages so that
    # existing Korean-RegOps caches stay hot.
    if language != "ko":
        h.update(f"lang={language}|".encode("utf-8"))
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()[:16]


@dataclass
class DecomposerStats:
    calls: int = 0
    cache_hits: int = 0
    parse_failures: int = 0
    empty_answers: int = 0
    truncated: int = 0


@dataclass
class ClaimDecomposer:
    base_url: str = "http://localhost:8035/v1"
    model_id: str = "Qwen/Qwen3.6-35B-A3B-FP8"
    profile_name: str = "judge_strict"
    max_tokens: int = 2048
    cache_dir: Optional[Path] = None
    stats: DecomposerStats = field(default_factory=DecomposerStats)
    language: str = "ko"

    def __post_init__(self) -> None:
        if self.language not in _DECOMPOSER_SYSTEM:
            raise ValueError(
                f"unsupported language={self.language!r}; "
                f"choose one of {sorted(_DECOMPOSER_SYSTEM)}"
            )
        self._client = OpenAI(api_key="dummy", base_url=self.base_url)
        self._extra = _model_kwargs_from_profile(Qwen3_5_HParams[self.profile_name])
        if self.cache_dir is None:
            self.cache_dir = _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, text: str) -> Path:
        key = _hash_key(text, self.profile_name, self.model_id, self.language)
        return self.cache_dir / f"{key}.json"

    def _call_once(self, answer_text: str) -> str:
        prompt = (
            f"{_DECOMPOSER_USER_PREFIX[self.language]}{answer_text.strip()}\n"
        )
        kwargs: dict = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": _DECOMPOSER_SYSTEM[self.language]},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        for k, v in self._extra.items():
            kwargs[k] = v
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def decompose(self, answer_text: str, answer_id: str = "") -> list[dict]:
        """Decompose one answer into atomic claims.

        Returns list of {"id": str, "text": str}. The LLM is instructed to
        emit citation/marker-stripped text directly; `text` is whitespace-
        trimmed.
        """
        answer_text = (answer_text or "").strip()
        if not answer_text:
            self.stats.empty_answers += 1
            return []

        cache_path = self._cache_path(answer_text)
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                self.stats.cache_hits += 1
                return cached.get("claims", []) or []
            except Exception:
                pass

        self.stats.calls += 1
        raw = self._call_once(answer_text)
        data = _try_extract_json(raw)

        if data is None or "claims" not in data:
            self.stats.parse_failures += 1
            return []

        items = data.get("claims", []) or []
        claims: list[dict] = []
        for i, c in enumerate(items):
            if not isinstance(c, dict):
                continue
            txt = (c.get("text") or "").strip()
            if not txt:
                continue
            claims.append(
                {
                    "id": c.get("id") or f"c{i + 1}",
                    "text": txt,
                }
            )
        if len(claims) > _MAX_CLAIMS:
            self.stats.truncated += 1
            claims = claims[:_MAX_CLAIMS]

        try:
            cache_path.write_text(
                json.dumps(
                    {"answer_id": answer_id, "claims": claims},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

        return claims
