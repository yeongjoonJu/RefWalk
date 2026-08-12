from typing import Callable, Optional, Any
import re, json
from pydantic import BaseModel, Field
from openai import OpenAI
from src.utils.hparams import Qwen3_5_HParams


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def _model_kwargs_from_profile(profile: dict) -> dict:
    kw: dict = {}
    for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
        if k in profile:
            kw[k] = profile[k]
    if "extra_body" in profile:
        kw["extra_body"] = dict(profile["extra_body"])
    return kw

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
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


class BaseLLM:
    def __init__(
        self,
        name: str = "base",
        base_url: str = "http://localhost:8035/v1",
        model_id: str = "Qwen/Qwen3.6-35B-A3B-FP8",
        max_tokens: int = 6144,
        profile_name: str = "general_thinking",
        system_prompt: Optional[str] = None,
        schema: Optional[dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> None:

        profile = Qwen3_5_HParams[profile_name]
        self._extra = _model_kwargs_from_profile(profile)
        # Local vLLM servers ignore the key, so the default stays "dummy"
        # and every existing Qwen call site is unaffected. Hosted backends
        # (e.g. the Gemini OpenAI-compatible endpoint) need a real key,
        # passed explicitly by the caller.
        self._client = OpenAI(api_key=api_key or "dummy", base_url=base_url)
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        
        if schema is not None:
            self._response_format_js = {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema}
            }
            self._response_format_jo = {"type": "json_object"}
        else:
            self._response_format_js = None
            self._response_format_jo = None

    def _call_once(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        on_delta: Optional[Callable[[str, str], None]] = None,
        stream = False
    ) -> str:
        
        messages = []
        if self._system_prompt is not None:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        kwargs: dict = {
            "model": self._model_id,
            "messages": messages,
            "max_tokens": self._max_tokens,
        }

        if response_format is not None:
            kwargs["response_format"] = response_format

        for k, v in self._extra.items():
            kwargs[k] = v

        if stream:
            if on_delta is not None:
                try:
                    on_delta(
                        "prompt",
                        f"[system]\n{self._system_prompt}\n\n[user]\n{prompt}",
                    )
                except Exception:
                    pass
            if on_delta is None:
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            kwargs["stream"] = True
            stream = self._client.chat.completions.create(**kwargs)
            parts: list[str] = []
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "reasoning", None
                )
                content = getattr(delta, "content", None)
                if reasoning:
                    try:
                        on_delta("reasoning", reasoning)
                    except Exception:
                        pass
                if content:
                    parts.append(content)
                    try:
                        on_delta("content", content)
                    except Exception:
                        pass
            return "".join(parts)
        else:
            resp = self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip() or ""
        
    def _call_llm(
        self,
        prompt: str,
        on_delta: Optional[Callable[[str, str], None]] = None,
    ):
        for attempt in range(2):
            try:
                raw = self._call_once(
                    prompt, self._response_format_jo, on_delta=on_delta
                )

                if self._response_format_jo is not None:
                    data = _try_extract_json(raw)
                else:
                    data = raw

                if data:
                    return data

            except Exception as e:
                last_err = f"json_schema request error: {e}"
                print(last_err)

        return None