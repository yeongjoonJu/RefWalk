"""Evaluation utilities.

Modules:
  metrics           — retrieval (R@5/10, nDCG@10, FullCov@10)
                      + generation (CP/CR/CFP) metrics with node/article
                      granularity. This is the canonical module for the
                      Phase B.5.1 retrieval and ground evaluations.
  citation_metrics  — article-level citation P/R/F1 with three rollup
                      variants (legacy, used by run_generation_eval.py).
  claim_decomposer  — LLM-based atomic claim decomposition (Phase C).
  claim_matcher     — cross-model claim entailment judge (Phase C).
  fallback_detector — manual-only / CAVEAT / degraded-stop detection.
"""
