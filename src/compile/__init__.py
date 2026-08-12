"""Offline Compile stage: build the OKG from the released articles.jsonl.

  - okg                 : graph construction (nodes, typed edges, validation)
  - reference_extractor : inline citation extraction + resolution
  - ancestor_prefix     : ancestor-prefixed indexed_text composition

Entry point: ``cli/run_build_okg.py``.
"""
