# Data setup

RefWalk evaluates on **RegOps-Bench**, distributed separately on the Hugging
Face Hub. This directory holds the dataset and the locally-built Operational
Knowledge Graph (OKG). None of these files are tracked in git (see
`../.gitignore`).

Expected layout after setup:

```
data/
├── RegOps-Bench/                 # downloaded from Hugging Face (step 1)
│   ├── articles.jsonl            # parsed regulation articles (조-level corpus)
│   └── regops_bench.jsonl        # QA benchmark (question, answer, gt_references, difficulty)
├── okg/                          # built locally from articles.jsonl (step 2)
│   ├── okg.gpickle               # NetworkX DiGraph
│   ├── okg_nodes.jsonl           # node-level corpus (조/항/호)
│   ├── okg_edges.jsonl           # typed edges
│   └── *.npy                     # dense-embedding caches (auto-built on first run)
└── HIPAA/                        # second-domain data, shipped as-is (see below)
    ├── hipaa_articles.jsonl      # corpus
    ├── hipaa_bench.jsonl         # QA benchmark
    ├── hipaa_okg.gpickle         # pre-built OKG (NetworkX DiGraph)
    ├── hipaa_okg_nodes.jsonl     # node-level corpus
    └── hipaa_okg_edges.jsonl     # typed edges
```

`HIPAA/` holds a second-domain evaluation set (English; 45 CFR Parts 160/164) —
corpus, QA, and a pre-built OKG. **The pipeline code in this repository targets
RegOps only**; these files are published for reference and for anyone who wants
to point their own system at a second domain. `cli/evaluate.py` will score
predictions against `hipaa_bench.jsonl` since it is corpus-agnostic, but the
RefWalk / NativeRAG runners are not wired for it.

## 0. Which questions are scored

`regops_bench.jsonl` is **231 QA entries** — L1/L2/L3/L4 = 48/81/45/57,
56 official + 175 augmented. Every reported number uses all 231.
See [`../docs/evaluation.md`](../docs/evaluation.md).

## 1. Download RegOps-Bench

```bash
huggingface-cli download Y-J-Ju/RegOps-Bench --repo-type dataset \
    --local-dir data/RegOps-Bench
```

> The dataset is **private until publication** — request access on the
> [dataset page](https://huggingface.co/datasets/Y-J-Ju/RegOps-Bench) and
> `huggingface-cli login` first.

## 2. Build the OKG

The OKG is built offline (pure Python, no LLM/network) from `articles.jsonl`:

```bash
python cli/run_build_okg.py \
    --articles data/RegOps-Bench/articles.jsonl \
    --out-dir  data/okg
```

This produces `okg.gpickle`, `okg_nodes.jsonl`, and `okg_edges.jsonl` in
`data/okg/`.

## 3. Embedding caches

Dense-embedding caches (`data/okg/*.npy`) are built automatically the first
time a retrieval/RAG run needs them, then reused. This requires the embedding
server to be running (`bash scripts/serve_vllm.sh embed`). To force a rebuild,
delete the relevant `.npy` file.
