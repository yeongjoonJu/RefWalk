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
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "extra_body": {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": False}}
    },
    # ── Gemini (OpenAI-compatible endpoint) ────────────────────────────
    "gemini_exact": {
        "temperature": 0.1,
        "top_p": 0.95,
        "extra_body": {"reasoning_effort": "none"}
    },
    "gemini_non_thinking": {
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_body": {"reasoning_effort": "none"}
    },
    "gemini_min_thinking": {
        "temperature": 0.1,
        "top_p": 0.95,
        "extra_body": {"reasoning_effort": "low"}
    }
}