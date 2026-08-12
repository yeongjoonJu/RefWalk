"""RefWalk prompts.

Holds the RefWalk-facing prompt text: online query anchoring
(``TOPIC_ANCHORING``), the cited-answer generation prompts
(``REFWALK_SYSTEM`` / ``get_user_prompt``), and the Chain-of-Rules
extraction prompts.

All system instructions are in English. User-facing content
(regulations, questions) remains in the corpus language (Korean for
RegOps-Bench).
"""


TOPIC_ANCHORING = """Extract a procedural topic and structured conditions from a given question for {domain}.

## Topic
- Output a phrase (15 under syllables)
- Express a generalized procedural category that matches chapter/article granularity in the regulation corpus
- Include the cost category, item type, or document type implicit in the question
- Exclude specific numbers, amounts, or actor identities (these belong in conditions)

## Conditions
For each of the four dimensions below, extract the value mentioned in the question. If a dimension is not mentioned, use "unspecified".

- Actor: the top entities whose institutional or organizational type determine which rules apply (e.g., institution type, role, position)
- Magnitude: a quantitative threshold or scale that triggers different rules (e.g., monetary amount, count, percentage; keep specific numbers)
- Temporal: a time point, period, or sequence relative to the action (e.g., before/after action, deadline, period, frequency, duration)
- Situational: any contextual condition not captured by the other dimensions that affects rule applicability (e.g., funding source, collaboration arrangement, employment status, planning status, organizational relation, document type)

## Output format
Use JSON format. Topic and condition values in {language}. The parsed output should allow reconstruction of the original question's meaning.

## Examples

### Example 1
Question: Can a covered entity share a patient's medical records with 
an external researcher who is not affiliated with the entity?

{{
  "topic": "Disclosure of medical records to external researcher",
  "actor": "Covered entity",
  "magnitude": "unspecified",
  "temporal": "unspecified",
  "situational": "External researcher (not affiliated)"
}}

### Example 2
Question: Can I pay the labor cost in cash to a new recruiter at a for-profit organization?

{{
  "topic": "Payment of cash labor costs",
  "actor": "For-profit organization",
  "magnitude": "unspecified",
  "temporal": "unspecified",
  "situational": "new recruit"
}}

### Example 3
Question: 3천만원 이상 GPU를 구매할 때 사전승인이 필요한가요?

{{
  "topic": "장비 구매 사전승인",
  "actor": "unspecified",
  "magnitude": "3천만원 이상",
  "temporal": "구매 전",
  "situational": "unspecified"
}}"""


REFWALK_SYSTEM = """당신은 국가연구개발사업 및 관련 법령/규정 해석을 지원하는 전문가입니다.
사용자가 제공하는 문맥(Context), 질문(Question), 그리고 질문의 주요 조건(주체, 시점, 규모, 상황)을 철저히 분석하여 정확한 답변을 제공해야 합니다.

# Instruction
1. 문맥 의존성: 모든 답변과 근거는 반드시 제공된 'Context' 내에서만 추출해야 합니다. 제공된 문서에 없는 내용을 임의로 지어내거나 외부 지식을 개입하지 마세요.
2. 참조 단위 통일(조 단위): JSON의 Key 값은 Context에 명시된 문서의 `node_id` (예: "국가연구개발사업_연구개발비_사용기준_제22조") 형식과 동일하게 작성해야 합니다. 절대 임의로 Key의 이름을 지어내거나 변형하지 마세요.
3. 근거(Claim) 추출 및 세부 단위 표기: 각 참조 조항에서 질문에 답변하기 위해 필요한 구체적인 사실이나 규정 내용을 추출하여 배열(Array) 형태로 나열하세요. 이때 실제 근거가 되는 세부 단위(항, 호)가 존재한다면 내용 앞에 대괄호(예: [제O항])로 표기하세요.
4. 예외 및 단서 조항 필수 반영: 제공된 조건(주체, 시점, 규모, 상황)의 값이 'all'(특정되지 않음)이거나 포괄적인 경우, 존재하는 예외 조건이나 단서 조항(예: "다만, ~", "~의 경우 예외로 한다", "중앙행정기관의 장이 인정하는 경우" 등)을 반드시 탐색하여 근거(Claim)에 포함시켜야 합니다.
5. 최종 답변(Answer) 작성:
   - 추출한 근거(Claim)와 예외 조항을 종합하여 질문에 부합하는 명확한 답변을 작성하세요.
   - 예외 조건이 있는 경우, 답변 내에 "다만, [예외 조건]의 경우 [예외 결과]가 가능합니다/제한됩니다."의 형태로 명시적으로 고지하세요.
   - 답변 내에 어떤 법령의 몇 조 몇 항에 따른 것인지 명확히 언급하여 신뢰성과 전문성을 높이세요.
6. 출력 형식: 반드시 아래의 JSON format으로만 응답해야 하며, JSON 외의 다른 인사말이나 부연 설명은 절대 포함하지 마세요.

# Output format (JSON)
{
  "OOO_제O조": [
    "[제1항] 해당 조항에서 도출한 원칙적인 근거 원문 및 내용",
    "[제2항] 대상별로 다른 규정이 적용되는 경우의 또 다른 원문 내용",
    "[예외] 해당 조항에 존재하는 예외 및 단서 조항 내용"
  ],
  "OOO_OO_제O조": [
    "해당 조항에서 도출한 원칙적인 근거 내용 (항/호가 없는 경우)"
  ],
  "answer": "원칙적인 규정에 대한 설명. 다만, [예외 조건]에 해당하는 경우 [예외 사항]이 적용됩니다. (도출된 근거를 종합하여 작성한 최종 답변 텍스트)"
}"""


REFWALK_USER = "## Context:\n{ctx}\n\nThe below question is about {topic}. The conditions for the topic are as follows:\n- 주체: {actor}\n- 시점: {temporal}\n- 규모: {magnitude}\n- 상황: {situational}\n\n## Question: {q}"
# REFWALK_USER = "## Context:\n{ctx}\n\n## Question: {q}"


# ── English variant ──────────────────────────────────────────────────
# Tuned for atomic rule corpora, where most questions are answered by a
# single rule: cite minimally and precisely.
REFWALK_SYSTEM_EN = """You are an AI compliance expert in regulatory rules.

## Instruction
1. Context dependency: Every claim and the final answer MUST be drawn from the provided 'Context' only. Do not invoke outside knowledge or fabricate content.
2. Reference-key format: Each JSON key MUST be the `node_id` exactly as it appears between the `[` and `]` of a Context passage header — without the surrounding brackets. Do not invent, modify, or concatenate identifiers. (Example: header `[spans-passthrough-candidates/sent_0012-0ec05553]` → key `"spans-passthrough-candidates/sent_0012-0ec05553"`.)
3. Cite minimally: include only the rule(s) that materially support the answer. The default is one rule per question; cite a second only when the answer genuinely depends on a distinct second rule. Do not echo every passage in the Context.
4. Claim extraction: Under each cited node_id, list the specific facts the rule contributes toward the answer (JSON array of one-sentence English strings). Surface an exception or proviso ("except", "provided that", "unless") only when the cited rule itself explicitly contains one — do NOT search for exceptions when none are present in the cited text.
5. Final 'answer' field:
   - Concise English synthesis grounded in the cited claims.
   - Reference the regulation in-line by its CFR section (e.g., "§164.502(a)(1)") so the reasoning is traceable.
   - State any explicit exception when it applies.
6. Output format: Respond with ONLY the JSON object below. No greetings, no preface, no trailing prose.

# Output format (JSON)
{
  "<node_id>": [
    "Primary normative content derived from this rule",
    "Exception or proviso, only when the cited rule explicitly contains one"
  ],
  "answer": "Final answer text. Reference the regulation in-line (e.g., §164.502(a)) and state any explicit exception when it applies."
}"""

REFWALK_USER_EN = "## Context:\n{ctx}\n\nThe below question is about {topic}. The conditions for the topic are as follows:\n- actor: {actor}\n- temporal: {temporal}\n- magnitude: {magnitude}\n- situational: {situational}\n\n## Question: {q}"

# Ablation template: the topic anchor (topic + structured conditions) is NOT
# conditioned into the generation prompt — only Context + Question are shown.
# Retrieval still uses the anchor; this isolates the generation-side
# contribution of anchor conditioning (§4.2). Language-agnostic.
REFWALK_USER_NOCOND = "## Context:\n{ctx}\n\n## Question: {q}"


# ── Prompt selectors ─────────────────────────────────────────────────

_REFWALK_PROMPTS: dict[str, dict[str, str]] = {
    "ko": {"system": REFWALK_SYSTEM, "user": REFWALK_USER},
    "en": {"system": REFWALK_SYSTEM_EN, "user": REFWALK_USER_EN},
}


def get_refwalk_system(language: str = "ko") -> str:
    """Return the REFWALK system prompt for the given language ('ko'|'en')."""
    if language not in _REFWALK_PROMPTS:
        raise ValueError(
            f"unsupported refwalk language={language!r}; "
            f"choose one of {sorted(_REFWALK_PROMPTS)}"
        )
    return _REFWALK_PROMPTS[language]["system"]


# Domains whose regulations routinely carry exception / proviso clauses
# ("다만, …", "단서") that benefit from the "broad-condition → enumerate
# exceptions" trigger. These are the domains where the prompt rewrites
# ``unspecified`` → ``all``. Other domains — atomic rule corpora, where
# the substitution empirically caused over-citation — keep the
# literal ``unspecified`` so the LLM treats the slot as a knowledge gap
# rather than a license to widen scope.
_REFWALK_BROAD_TRIGGER_DOMAINS: frozenset[str] = frozenset({"regops"})


def get_user_prompt(
    retrieved,
    anchor,
    passage_limit: int = 3000,
    language: str = "ko",
    domain: str = "regops",
    condition_anchor: bool = True,
):
    """Build the REFWALK user prompt.

    ``language`` selects the user-template flavour
      * ``'ko'`` — Korean condition labels (주체/시점/규모/상황) for RegOps.
      * ``'en'`` — English condition labels (actor/temporal/magnitude/situational).
    The Context block (``[node_id]\\ntext``) and the {topic} field are
    domain-agnostic in both renderings.

    ``domain`` selects the condition-value mapping. When the domain is in
    :data:`_REFWALK_BROAD_TRIGGER_DOMAINS` (currently only ``regops``),
    ``unspecified`` is rewritten to ``all`` to fire the "broad-condition →
    enumerate exceptions" rule. Other domains keep ``unspecified`` literal
    — atomic rule corpora over-cited under ``all``.
    """
    if language not in _REFWALK_PROMPTS:
        raise ValueError(
            f"unsupported refwalk language={language!r}; "
            f"choose one of {sorted(_REFWALK_PROMPTS)}"
        )
    user_template = _REFWALK_PROMPTS[language]["user"]

    # construct context
    ctx = []
    for ret in retrieved:
        ctx.append(f"[{ret.node_id}]\n{ret.text[:passage_limit]}")
    ctx = "\n\n".join(ctx)

    # Ablation: drop the topic-anchor conditioning from the generation
    # prompt. Only Context + Question are shown (the unspecified→all
    # rewrite is moot since no condition slots are rendered).
    if not condition_anchor:
        return REFWALK_USER_NOCOND.format(ctx=ctx, q=anchor['ori_question'])

    if domain in _REFWALK_BROAD_TRIGGER_DOMAINS:
        for k, v in anchor.items():
            if v == 'unspecified':
                anchor[k] = 'all'

    # return user_template.format(ctx=ctx, q=anchor['ori_question'])
    return user_template.format(
        ctx=ctx, q=anchor['ori_question'],
        topic=anchor['topic'], actor=anchor['actor'],
        temporal=anchor['temporal'], magnitude=anchor['magnitude'],
        situational=anchor['situational'])


# ── Prompt fragments ─────────────────────────────────────────────────

NOTES_INITIAL_PROMPT = "PlanNotes: (empty — no walks have run yet for this user question)"

TOP_LEVEL_DOCS_HEADER = "Top-level documents indexed in the OKG:"


# ── Judge Prompt (unchanged) ────────────────────────────────────────

JUDGE_GUIDANCE_PROMPT = """\
Compare the system's answer with the gold answer and evaluate.

Gold answer summary: {gold_summary}
System answer: {predicted_answer}

Evaluate on three dimensions:
1. CORE_JUDGMENT: Is the core conclusion correct? (CORRECT / PARTIALLY_CORRECT / INCORRECT)
2. COMPLETENESS: Are all relevant conditions and exceptions mentioned? (score 0-1)
3. CITATION_QUALITY: Are the cited references appropriate? (score 0-1)

Output JSON:
{{"core_judgment": "...", "completeness": 0.X, "citation_quality": 0.X, "reason": "..."}}
"""