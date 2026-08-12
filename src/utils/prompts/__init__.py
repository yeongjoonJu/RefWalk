"""Prompt package.

Re-exports a few RefWalk prompt symbols at the package root for
convenience. The full set of RefWalk / Chain-of-Rules / anchoring
prompts lives in ``src.utils.prompts.refwalker``.
"""

from src.utils.prompts.refwalker import (  # noqa: F401
    JUDGE_GUIDANCE_PROMPT,
    NOTES_INITIAL_PROMPT,
    TOP_LEVEL_DOCS_HEADER,
)

__all__ = [
    "JUDGE_GUIDANCE_PROMPT",
    "NOTES_INITIAL_PROMPT",
    "TOP_LEVEL_DOCS_HEADER",
]
