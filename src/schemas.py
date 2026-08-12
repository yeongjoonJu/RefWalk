"""Structured-output schemas shared across RefWalk modules.

Currently holds the query-anchoring schema produced by the online
``TopicExtractor`` (see ``src.retrieval.walker_retrieval``). The
``conditions`` keys mirror the four anchoring dimensions used by the
``TOPIC_ANCHORING`` prompt in ``src.utils.prompts.refwalker``.

Note: the ``"magnitute"`` key is an intentional, frozen spelling kept
for backward compatibility with cached anchoring outputs and prompts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class QueryAnalysis(BaseModel):
    topic: str
    conditions: dict[Literal["actor", "magnitute", "temporal", "situational"], str]
