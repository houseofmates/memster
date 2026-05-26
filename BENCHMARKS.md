# Memster LongMemEval Benchmarks

This document reports LongMemEval performance for Memster.

## Verified Result

| Backend | Fusion | Reranker | Recall@5 | p95 Latency | Proof |
|---------|--------|----------|----------|-------------|-------|
| NIM (Nemotron) | weighted + RRF k=300 | cross-encoder (MiniLM-L-6) | **95.20%** | ~2.8s | [V6 result file](V6_RESULTS_sem500_bm500_ent500_temp500_k300_w1.5_1.0_5.0_1.0.json) |

Configuration: semantic weight 1.5, BM25 1.0, entity 5.0, temporal 1.0, RRF k=300, cross-encoder ms-marco-MiniLM-L-6-v2.

## How to Reproduce the Verified Result

```bash
# Requires: PostgreSQL with memories table, LongMemEval dataset in benchmarks/LongMemEval_dataset/
# Requires: NVIDIA NIM / OpenRouter API key with access to nvidia/llama-nemotron-embed-vl-1b-v2

EMBEDDING_BACKEND=nim \
  SIGNAL_CANDIDATE_LIMIT=500 \
  WEIGHT_SEM=1.5 WEIGHT_BM25=1.0 WEIGHT_ENT=5.0 WEIGHT_TEMP=1.0 \
  FUSION_METHOD=weighted \
  python3 -c "
from memster.hybrid_retrieval import HybridRetrievalEngine
import psycopg2
conn = psycopg2.connect('postgresql://house:@/memster?host=/run/postgresql&port=5433')
def get_conn(): return conn
engine = HybridRetrievalEngine(get_conn)
# ... (run retrieval for each question, see benchmarks/run_improved_longmemeval.py)
"
```

## Performance Notes

- The 95.20% result uses the cross-encoder reranker (ms-marco-MiniLM-L-6-v2) which takes ~2.7s per query.
- The lightweight reranker (mixedbread-ai/mxbai-rerank-xsmall-v1) is faster (~0.15s per rerank) but achieves lower recall (~60% in testing) because the NIM 2048-dim cosine similarity cannot fully discriminate between similar sessions without cross-encoder attention.
- **To achieve ≥96.2%**, the cross-encoder reranker is essential. The hybrid retrieval engine supports both rerankers — set `USE_LIGHTWEIGHT_RERANKER=false` and `USE_CROSS_ENCODER=true` (or load the cross-encoder manually) for maximum recall.
- The semantic search relies on the `local_embedding` column. When switching between NIM (2048-dim) and local (384-dim) backends, old embeddings are incompatible — re-embed via `engine.embed_and_store()`.
- Query expansion with WordNet is **disabled by default** — testing showed it adds noise rather than signal for this dataset.

## Local Backend

The local backend uses `BAAI/bge-small-en-v1.5` (384-dim, ~33M params, runs on CPU with 8GB RAM). Expected recall is lower than NIM due to reduced embedding resolution. Run the benchmark with `EMBEDDING_BACKEND=local` to generate numbers for your configuration.

## Configuration Options

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `EMBEDDING_BACKEND` | `local`, `nim` | `local` | Which embedding model to use |
| `SIGNAL_CANDIDATE_LIMIT` | int | 500 | Candidates per signal before fusion |
| `USE_LIGHTWEIGHT_RERANKER` | `true`, `false` | `true` | Use mxbai reranker (faster, lower recall) |
| `USE_TWO_STAGE_RERANKER` | `true`, `false` | `false` | Two-stage: hybrid → top-N → reranker → top-K |
| `WEIGHT_SEM` | float | 1.5 | Semantic signal weight |
| `WEIGHT_BM25` | float | 1.0 | BM25 signal weight |
| `WEIGHT_ENT` | float | 5.0 | Entity boosting weight |
| `WEIGHT_TEMP` | float | 1.0 | Temporal decay weight |
| `FUSION_METHOD` | `weighted`, `rrf` | `weighted` | Fusion algorithm |