from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class QueryAnalysis(BaseModel):
    topic: str
    conditions: dict[Literal["actor", "magnitute", "temporal", "situational"], str]
