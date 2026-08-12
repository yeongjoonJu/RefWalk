Qwen3_5_HParams = {
    "general_thinking": {
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "extra_body": {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": True}}
    },
    "explicit_thinking": {
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "extra_body": {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": True}}
    },
    "coding_thinking": {
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "extra_body": {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": True}}
    },
    "non_thinking": {
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "extra_body": {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": False}}
    },
    "conf": {
        "temperature": 0.3,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "extra_body": {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": False}}
    },
    "exact": {
        "temperature": 0.1,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "extra_body": {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": False}}
    },
    "judge_strict": {
        # Low-temperature deterministic profile for the practitioner-voice
        # judge. Thinking disabled for latency; presence_penalty=0 so the
        # short boolean JSON isn't perturbed by novelty pressure.
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "extra_body": {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": False}}
    },
    # ── Gemini (OpenAI-compatible endpoint) ────────────────────────────
    # The Qwen profiles above carry vLLM-only knobs (top_k, min_p,
    # repetition_penalty, chat_template_kwargs.enable_thinking) that the
    # Gemini OpenAI-compat endpoint rejects, so Gemini gets its own
    # profiles. Thinking is disabled via ``reasoning_effort: "none"`` —
    # the documented OpenAI-compat knob Google maps to thinking-off on
    # flash-class models (replaces Qwen's enable_thinking=False). The key
    # is placed inside ``extra_body`` so the OpenAI SDK forwards it at the
    # top level of the request body (BaseLLM only relays
    # temperature/top_p/penalties/extra_body).
    "gemini_exact": {
        # RefWalk default: deterministic, non-thinking. Mirrors the intent
        # of the Qwen ``exact`` profile (REFWALK_SYSTEM enforces strict
        # JSON, so we want low temperature + no chain-of-thought).
        "temperature": 0.1,
        "top_p": 0.95,
        "extra_body": {"reasoning_effort": "none"}
    },
    "gemini_non_thinking": {
        # Higher-temperature non-thinking variant, parallels Qwen
        # ``non_thinking``.
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_body": {"reasoning_effort": "none"}
    },
    "gemini_min_thinking": {
        # Minimum-thinking deterministic profile for Gemini models that
        # *only* run in thinking mode (e.g. gemini-3.1-pro-preview, which
        # rejects reasoning_effort="none"/"minimal" with HTTP 400
        # "This model only works in thinking mode"). "low" is the floor
        # such models allow — not true non-thinking, but the closest
        # available. Same low temperature as gemini_exact since
        # REFWALK_SYSTEM still needs deterministic strict JSON.
        "temperature": 0.1,
        "top_p": 0.95,
        "extra_body": {"reasoning_effort": "low"}
    }
}