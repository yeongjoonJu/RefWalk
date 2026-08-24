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
_DEFAULT_CACHE_DIR = Path("data/cache/claim_matcher")

_LABEL_PRIORITY = {"match": 2, "partial": 1, "none": 0}


_MATCHER_SYSTEM_KO = """당신은 두 answer 사이의 claim 일치 여부를 판정하는 엄격한 judge 입니다.

입력:
  - predicted claim: 한 문장
  - reference claims: 여러 문장 (0~20 개)

각 reference claim 에 대해 다음 세 label 중 하나를 부여:
  - "match"   : 의미가 동일하거나 더 구체적으로 포함 (조건/수식 포함해 완전 포괄)
  - "partial" : 핵심은 일치하지만 조건/예외/수치가 누락 또는 반대로 서술
  - "none"    : 무관하거나 상충

판정 원칙:
- 한국 법령/규정 담화의 엄밀성을 지킬 것. 숫자(비율, 금액, 일수)나 예외(다만/제외) 가 다르면 partial 이하.
- 동일한 의미를 다른 말투로 쓴 것은 match.
- 단순 인용(「…」 제N조)만 다른 것은 의미 동등하면 match 유지.

출력 JSON schema:
{
  "labels": [
    {"ref_id": "<reference claim id>", "label": "match|partial|none"},
    ...
  ],
  "best_label": "match|partial|none"
}

labels 길이는 reference claims 수와 정확히 같아야 함.
best_label 은 labels 중 우선순위(match > partial > none) 최고값.
"""


_MATCHER_SYSTEM_EN = """You are a strict judge that decides whether two claims agree, in the context of regulatory text.

Inputs:
  - predicted claim: one sentence
  - reference claims: 0–20 sentences

For every reference claim, assign exactly one of:
  - "match"   : same meaning, or strictly more specific (fully covers all conditions/qualifiers).
  - "partial" : core idea agrees but a condition/exception/number is missing, weaker, or stated in the opposite direction.
  - "none"    : unrelated, or in conflict.

Judging principles:
- Hold the line on regulatory precision. If numbers (percentages, dollars, days), thresholds, or qualifiers ("subject to", "except", "unless", "provided that") differ, the label is at most "partial".
- Wording variation alone (active vs passive, paraphrase) does NOT downgrade — same content = "match".
- Differences only in citation style (e.g. "§164.502" vs "45 CFR 164.502") do not downgrade if the substance matches.

Output JSON schema:
{
  "labels": [
    {"ref_id": "<reference claim id>", "label": "match|partial|none"},
    ...
  ],
  "best_label": "match|partial|none"
}

The length of `labels` MUST equal the number of reference claims.
`best_label` is the highest-priority label across `labels` (match > partial > none).
"""


_MATCHER_SYSTEM = {"ko": _MATCHER_SYSTEM_KO, "en": _MATCHER_SYSTEM_EN}


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
    pred_text: str, ref_texts: list[str], model_id: str, language: str = "ko"
) -> str:
    h = hashlib.sha1()
    h.update(model_id.encode("utf-8"))
    h.update(b"||")
    # Only include language for non-default languages so existing
    # Korean-RegOps caches stay valid.
    if language != "ko":
        h.update(f"lang={language}||".encode("utf-8"))
    h.update((pred_text or "").encode("utf-8"))
    h.update(b"||")
    for r in ref_texts:
        h.update(r.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:16]


@dataclass
class MatcherStats:
    calls: int = 0
    cache_hits: int = 0
    parse_failures: int = 0


@dataclass
class ClaimMatcher:
    base_url: str = "http://localhost:8035/v1"
    model_id: str = "Qwen/Qwen3.6-35B-A3B-FP8"
    profile_name: str = "judge_strict"
    max_tokens: int = 1536
    partial_weight: float = 0.5
    cache_dir: Optional[Path] = None
    stats: MatcherStats = field(default_factory=MatcherStats)
    language: str = "ko"

    def __post_init__(self) -> None:
        if self.language not in _MATCHER_SYSTEM:
            raise ValueError(
                f"unsupported language={self.language!r}; "
                f"choose one of {sorted(_MATCHER_SYSTEM)}"
            )
        self._client = OpenAI(api_key="dummy", base_url=self.base_url)
        self._extra = _model_kwargs_from_profile(Qwen3_5_HParams[self.profile_name])
        if self.cache_dir is None:
            self.cache_dir = _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, pred_text: str, ref_texts: list[str]) -> Path:
        key = _hash_key(pred_text, ref_texts, self.model_id, self.language)
        return self.cache_dir / f"{key}.json"

    def _call_once(self, pred_claim: dict, refs: list[dict]) -> str:
        ref_block = "\n".join(
            f'  - {r["id"]}: {r["text"]}' for r in refs
        )
        prompt = (
            f"predicted claim:\n  {pred_claim['text']}\n\n"
            f"reference claims:\n{ref_block}\n"
        )
        kwargs: dict = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": _MATCHER_SYSTEM[self.language]},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        for k, v in self._extra.items():
            kwargs[k] = v
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def _judge_pred(
        self, pred_claim: dict, refs: list[dict]
    ) -> list[dict]:
        """Returns list of {ref_id, label} parallel to `refs`."""
        if not refs:
            return []

        cache_path = self._cache_path(
            pred_claim["text"], [r["text"] for r in refs]
        )
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                self.stats.cache_hits += 1
                return cached.get("labels", []) or []
            except Exception:
                pass

        self.stats.calls += 1
        raw = self._call_once(pred_claim, refs)
        data = _try_extract_json(raw)

        if data is None or "labels" not in data:
            self.stats.parse_failures += 1
            # Conservative: no match on failure.
            return [{"ref_id": r["id"], "label": "none"} for r in refs]

        labels = data.get("labels", []) or []
        # Sanitize: fill missing with "none", cap length to len(refs).
        got: dict[str, str] = {}
        for item in labels:
            if not isinstance(item, dict):
                continue
            rid = item.get("ref_id")
            lab = item.get("label")
            if rid and lab in _LABEL_PRIORITY:
                got[str(rid)] = lab
        out = [
            {"ref_id": r["id"], "label": got.get(r["id"], "none")}
            for r in refs
        ]

        try:
            cache_path.write_text(
                json.dumps({"labels": out}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        return out

    def match(
        self, pred_claims: list[dict], gt_claims: list[dict]
    ) -> dict:
        """Match pred against gt and compute Claim P/R/F1.

        Resolution: for every pred we call the judge once against all
        GT. Then build a list of (pred_id, gt_id, label) triples and
        greedy-assign in priority order (match → partial) with one-to-
        one exclusivity. Any unmatched pred contributes 0 to precision;
        any unmatched gt contributes 0 to recall.

        Returns:
            {
              "pairs": [...],          # final assignments
              "pred_labels": [...],    # per-pred best label after resolution
              "gt_labels": [...],      # per-gt best label after resolution
              "precision": float,
              "recall": float,
              "f1": float,
              "n_pred": int,
              "n_gt": int,
              "n_match": int,
              "n_partial": int,
              "n_none": int,
            }
        """
        n_pred, n_gt = len(pred_claims), len(gt_claims)
        if n_pred == 0 and n_gt == 0:
            return {
                "pairs": [], "pred_labels": [], "gt_labels": [],
                "precision": 1.0, "recall": 1.0, "f1": 1.0,
                "n_pred": 0, "n_gt": 0,
                "n_match": 0, "n_partial": 0, "n_none": 0,
            }
        if n_pred == 0:
            return {
                "pairs": [], "pred_labels": [],
                "gt_labels": [{"gt_id": g["id"], "label": "none"} for g in gt_claims],
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_pred": 0, "n_gt": n_gt,
                "n_match": 0, "n_partial": 0, "n_none": 0,
            }
        if n_gt == 0:
            return {
                "pairs": [],
                "pred_labels": [{"pred_id": p["id"], "label": "none"} for p in pred_claims],
                "gt_labels": [],
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_pred": n_pred, "n_gt": 0,
                "n_match": 0, "n_partial": 0, "n_none": 0,
            }

        # Collect all pred→gt judgments.
        triples: list[tuple[str, str, str]] = []
        for p in pred_claims:
            labs = self._judge_pred(p, gt_claims)
            for item in labs:
                triples.append((p["id"], item["ref_id"], item["label"]))

        # Greedy bipartite: sort by priority, then consume.
        triples.sort(key=lambda t: -_LABEL_PRIORITY[t[2]])
        used_pred: set[str] = set()
        used_gt: set[str] = set()
        pairs: list[dict] = []
        for pid, gid, lab in triples:
            if lab == "none":
                break
            if pid in used_pred or gid in used_gt:
                continue
            pairs.append({"pred_id": pid, "gt_id": gid, "label": lab})
            used_pred.add(pid)
            used_gt.add(gid)

        n_match = sum(1 for p in pairs if p["label"] == "match")
        n_partial = sum(1 for p in pairs if p["label"] == "partial")
        w = self.partial_weight
        matched_weight = n_match + w * n_partial

        precision = matched_weight / n_pred
        recall = matched_weight / n_gt
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        pred_labels = []
        for p in pred_claims:
            hit = next((pr for pr in pairs if pr["pred_id"] == p["id"]), None)
            pred_labels.append(
                {"pred_id": p["id"], "label": hit["label"] if hit else "none"}
            )
        gt_labels = []
        for g in gt_claims:
            hit = next((pr for pr in pairs if pr["gt_id"] == g["id"]), None)
            gt_labels.append(
                {"gt_id": g["id"], "label": hit["label"] if hit else "none"}
            )

        return {
            "pairs": pairs,
            "pred_labels": pred_labels,
            "gt_labels": gt_labels,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "n_pred": n_pred,
            "n_gt": n_gt,
            "n_match": n_match,
            "n_partial": n_partial,
            "n_none": n_pred - n_match - n_partial,
        }
