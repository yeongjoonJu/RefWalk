# Evaluating any RAG system on RegOps-Bench

RegOps-Bench scores a system from a **prediction file**, not from its code.
Run your own pipeline however you like, dump one JSON object per benchmark
question, then hand the file to `cli/evaluate.py`. Nothing in the evaluator
imports RefWalk's retriever or generator, so a LangChain chain, a LlamaIndex
query engine, a GraphRAG variant, or a bespoke script are all scored by the
same code path on the same terms.

```bash
python cli/evaluate.py \
    --predictions experiments/my_rag/preds.jsonl \
    --bench       data/RegOps-Bench/regops_bench.jsonl \
    --corpus      data/okg/okg_nodes.jsonl \
    --out         experiments/my_rag/scores
```

RegOps-Bench is **231 questions** (L1/L2/L3/L4 = 48/81/45/57; 56 official +
175 augmented). Predictions whose `qa_id` is not in the benchmark are skipped
with a warning rather than treated as an error, so a file generated against a
pre-release copy still scores on the 231 the release defines.

---

## 1. The prediction format

One JSON object per line (JSONL). A JSON array in a single file also works.

```json
{
  "qa_id": "L1_ko_001",
  "retrieved_node_ids": ["국가연구개발사업_연구개발비_사용기준_제48조",
                         "국가연구개발사업_연구개발비_사용기준_제48조_제8항"],
  "answer": "타 기관 소속 전문가를 참여연구자로 계상하려면 …",
  "cited_node_ids": ["국가연구개발사업_연구개발비_사용기준_제48조_제8항"]
}
```

| Field | Required | Meaning |
|-------|----------|---------|
| `qa_id` | **yes** | Benchmark question id. The join key against `--bench`. |
| `retrieved_node_ids` | for retrieval metrics | Ranked list of corpus `node_id`, best first. Use `[]` if the system has no retrieval stage. |
| `answer` | for generation metrics | The generated answer text. |
| `cited_node_ids` | for citation metrics | The `node_id` the answer actually cites. See [§4](#4-if-your-system-doesnt-emit-a-citation-list). |
| `status` | no | `"ok"` (default) or `"error"`. Error rows are scored as failures unless you pass `--skip-errors`. |
| `variant` | no | System label used to bucket the report. Defaults to the filename stem; `--label` overrides both. |

Extra fields are ignored, so you can keep latency, token counts, or raw model
output in the same file.

**Gold labels are never read from your file.** `gt_references` and `difficulty`
come from `--bench`, so a stray copy of the labels in your predictions cannot
influence the score.

### Field aliases

The loader accepts the spellings common frameworks already produce, so an
adapter is usually unnecessary:

| Canonical | Also accepted |
|-----------|---------------|
| `qa_id` | `id`, `question_id`, `qid`, `query_id`, `sample_id` |
| `retrieved_node_ids` | `retrieved`, `retrieved_ids`, `retrieved_docs`, `contexts`, `context_node_ids`, `source_nodes`, `sources`, `docs` |
| `cited_node_ids` | `cited_references`, `citations`, `cited`, `cited_ids`, `citation_node_ids` |
| `answer` | `prediction`, `output`, `response`, `generated_answer`, `pred_answer`, `text` |
| `variant` | `system`, `run`, `run_id`, `method`, `model` |

### Retrieved contexts as objects

`retrieved_node_ids` may hold objects instead of strings. The evaluator pulls
the id from the object itself or from its `metadata`, which covers the
LangChain `Document` and LlamaIndex `NodeWithScore` shapes directly:

```json
{"id": "L1_ko_001",
 "response": "…",
 "contexts": [
   {"page_content": "…", "metadata": {"node_id": "…_제48조", "score": 0.81}},
   {"node": {"node_id": "…_제48조_제8항"}, "score": 0.77}
 ]}
```

Keys checked, in order: `node_id`, `id`, `doc_id`, `document_id`, `source`,
`file_path`, `ref`, `reference`, `chunk_id` — on the object, then on its
`node`, then on its `metadata` / `extra_info`.

> **The one integration requirement:** whatever you index must carry the corpus
> `node_id` through to retrieval output. When you chunk `articles.jsonl` or
> `okg_nodes.jsonl`, store `node_id` in each chunk's metadata and echo it back.
> Everything else the evaluator can adapt to.

---

## 2. What gets measured

### Retrieval

Computed from `retrieved_node_ids` against `gt_references`.

| Metric | Definition |
|--------|------------|
| `R@5`, `R@10` | Share of distinct gold references covered by the top-k. |
| `nDCG@10` | Rank-aware gain, binary relevance on gold. |
| `FullCov@10` | 1 when the top-10 covers **every** gold reference, else 0. The metric that matters for multi-clause compliance questions. |

### Citation

Computed from `cited_node_ids` against `gt_references`, as sets.

| Metric | Definition |
|--------|------------|
| `CP` / `CR` / `CF1` | Precision / recall / F1 after rolling both sides up to their 조-level ancestor. **The headline citation metric.** |
| `CF1_strict` | No rollup: F1 over ids at whatever granularity each side states, across the whole benchmark. |
| `CP_para` / `CR_para` / `CF1_para` | 항/호 level: both sides restricted to clause ids. Averaged only over the queries whose gold names at least one clause (154 of the 231), reported as `n_clause_eligible`. |
| `CFP` | Share of cited ids absent from the corpus, i.e. references the model invented. Needs `--corpus`. |
| `ctxCFP` | Share of cited ids absent from the system's own `retrieved_node_ids` — it cited something it never retrieved. |

Three granularities, because 조-level F1 alone cannot tell you whether a system
found the right *paragraph*: 조-level gold is 68% of the benchmark, so a system
that never resolves below the article still scores respectably on `CF1`.

### Clause recovery — why `CF1_strict` is not naive string equality

Scoring at full granularity is only fair if every system gets credit for the
clauses it actually names, and systems name them in different places:

* A system emitting structured output puts the clause in a bracketed tag inside
  its per-article payload (`"… [제8항] …"`); its citation list stays 조-level.
* A free-form system names the clause in prose (`"제48조제8항에 따라 …"`); its
  citation list is likewise 조-level.

Comparing citation lists alone would measure output format rather than
grounding. By default the evaluator therefore widens the cited set with the
clause ids the answer actually names — reading bracketed tags when the record
carries a `raw_output` dict, otherwise parsing the answer prose and grounding
each 조 number through the ids the system itself cited or retrieved. An
article-only mention is credited only when the system also cited or retrieved
that article, so a stray number cannot invent a citation.

The dispatch is on record shape, never on a system name. Pass
`--no-clause-aware` to see the naive string-equality version; on the released
predictions it costs every system between 2 and 9 F1 points, unevenly, which is
exactly the format bias it exists to remove.

Article-level `CP`/`CR`/`CF1` are computed from the declared citation list and
are **never** affected by this expansion — prose-mined article numbers are
lower-confidence than a system's own citations, so the headline metric does not
absorb them.

### Claim (opt-in, needs an LLM)

`--judge` decomposes the gold answer and the predicted answer into atomic
claims and matches them with an LLM, yielding `claim_P` / `claim_R` /
`claim_F1`. Questions whose gold answer decomposes to zero claims are excluded
from the average and reported as `n_claim_eligible`.

```bash
python cli/evaluate.py \
    --predictions experiments/my_rag/preds.jsonl \
    --bench   data/RegOps-Bench/regops_bench.jsonl \
    --corpus  data/okg/okg_nodes.jsonl \
    --judge --judge-base-url http://localhost:8035/v1 \
    --judge-model Qwen/Qwen3.6-35B-A3B-FP8 --language ko \
    --out experiments/my_rag/scores
```

Point `--matcher-base-url` at a different model than `--judge-base-url` to
reduce self-preference bias — the paper's numbers use a cross-model matcher.

---

## 3. Matching rules

RegOps-Bench gold references sit at mixed granularity: of 314 distinct
references, ~68% are 조-level and ~32% are 항/호-level. Systems index at
whatever granularity they choose, so the evaluator has to say how a retrieved
id counts as a hit.

`--hit-rule hierarchical` **(default)** — a retrieved id counts when it equals a
gold id **or** is its ancestor/descendant along the id hierarchy
(`_` for RegOps ids, `/` for path-style ids). A system that indexes whole
조 articles gets
credit for retrieving `…_제48조` when the gold label is `…_제48조_제8항`.

`--hit-rule strict` — exact id equality only. Use it when your index
granularity matches the gold labels; otherwise it scores index design rather
than retrieval quality.

`--match-unit article_id` rolls both sides up to their 조 ancestor before
scoring, which is the coarsest reading of retrieval quality. The default
`node_id` compares ids verbatim.

Report the flags alongside your numbers — they are recorded in
`summary.json:config` for exactly this reason.

---

## 4. If your system doesn't emit a citation list

Pass `--cite-from-answer` and the evaluator recovers citations from the answer
text in two passes:

1. **Citation footer** — a line like `[참조] id_a, id_b` or `[Citations] id_a, id_b`.
   Prompt your generator to emit one; it is the unambiguous path.
2. **Inline scan** — otherwise, every corpus `node_id` that literally appears in
   the answer, ordered by first occurrence (needs `--corpus`).

Pass 2 only runs when pass 1 finds nothing, so emitting a proper footer is never
penalised. Note that inline scanning tends to *inflate* recall relative to an
explicit citation list, since it credits every id mentioned in prose. Systems
compared against each other should use the same setting.

---

## 5. Output

```
experiments/my_rag/scores/
├── summary.json        # config + validation report + overall + per-difficulty
├── summary.tsv         # one row per system
├── per_difficulty.tsv  # one row per system × L1–L4
└── per_query.jsonl     # every per-question metric dict, for error analysis
```

and a table on stdout:

```
system                   R@5     R@10  nDCG@10  FullCov@10       CP       CR      CF1   ...     n
-------------------------------------------------------------------------------------------------
refwalk_35b           0.5709   0.6482   0.5772      0.4762   0.6795   0.5360   0.5490   ...   231
```

Pass several `--predictions` files to get one row per system in a single
comparison table:

```bash
python cli/evaluate.py \
    --predictions experiments/refwalk/preds_35b.jsonl \
                  experiments/native_rag/preds_35b.jsonl \
                  experiments/my_rag/preds.jsonl \
    --label RefWalk NativeRAG MyRAG \
    --bench data/RegOps-Bench/regops_bench.jsonl \
    --corpus data/okg/okg_nodes.jsonl \
    --out experiments/comparison
```

---

## 6. Check the file before you run

```bash
python cli/evaluate.py --validate-only \
    --predictions experiments/my_rag/preds.jsonl \
    --bench data/RegOps-Bench/regops_bench.jsonl \
    --corpus data/okg/okg_nodes.jsonl
```

Errors (exit 2) — duplicate `(variant, qa_id)` pairs, or a system that covers
*none* of the benchmark (wrong `--bench` file, or a `qa_id` scheme that doesn't
match).

Warnings — partial benchmark coverage, `qa_id` outside the benchmark (skipped),
empty answers, missing citations, and retrieved ids that aren't in the corpus.
That last one almost always means an id-space mismatch: your index is keyed on
chunk hashes rather than `node_id`, and every retrieval metric will read 0. Add
`--strict` to make warnings exit non-zero in CI.

---

## 7. Worked example: adapting an existing pipeline

```python
import json

bench = [json.loads(l) for l in open("data/RegOps-Bench/regops_bench.jsonl", encoding="utf-8")]

with open("preds.jsonl", "w", encoding="utf-8") as f:
    for q in bench:
        # Your system. `docs` must carry node_id — see §1.
        docs = my_retriever.search(q["question"], k=10)
        answer = my_generator(q["question"], docs)

        f.write(json.dumps({
            "qa_id": q["qa_id"],
            "retrieved_node_ids": [d.metadata["node_id"] for d in docs],
            "answer": answer.text,
            "cited_node_ids": answer.citations,   # or omit + --cite-from-answer
        }, ensure_ascii=False) + "\n")
```

Then score it with the command at the top of this document. To index the corpus
in the first place, read `data/RegOps-Bench/articles.jsonl` (718 조-level
articles, text in `text_ko` / `text_en`) or `data/okg/okg_nodes.jsonl` (2,572
조/항/호-level nodes) and keep `node_id` on every chunk.
