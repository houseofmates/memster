# Memster LongMemEval Benchmarks

This document reports LongMemEval performance for Memster across different configurations.

## Verified Result

| Backend | Fusion | Reranker | Recall@5 | p95 Latency | Proof |
|---------|--------|----------|----------|-------------|-------|
| NIM (Nemotron) | RRF k=300 | cross-encoder (MiniLM-L-6) | **95.20%** | ~2.8s | [V6 result file](V6_RESULTS_sem500_bm500_ent500_temp500_k300_w1.5_1.0_5.0_1.0.json) |

**Configuration:** semantic top-500 × 1.5 + BM25 top-500 × 1.0 + entity top-500 × 5.0 + temporal top-500 × 1.0 → RRF(k=300) → cross-encoder rerank → top-5.

## Current Measurements

The following results were measured on the current PostgreSQL-based architecture (2048-dim NIM embeddings stored in `local_embedding` column, lightweight mxbai-xsmall reranker):

| Backend | Fusion | Reranker | Recall@5 | Avg Latency | Notes |
|---------|--------|----------|----------|-------------|-------|
| NIM (Nemotron) | weighted | mxbai-xsmall | ~62% (partial) | ~5.8s | Lightweight reranker, 250/500 qns [log](CROSSENCODER_RESULTS_nim_partial_250.txt) |
| NIM (Nemotron) | RRF k=300 | cross-encoder (MiniLM-L-6) | ~58% (partial) | ~8.5s | Cross-encoder rerank on top-200, 150/500 qns |

The 95.20% V6 result used a different embedding storage mechanism (the `nvidia_nim_embeddings` module's own index) and the full cross-encoder on a larger candidate pool. The current unified `local_embedding` column approach produces lower recall.

## How to Reproduce

### Prerequisites

```bash
pip install -e .[all]
```

### Run the Improved Benchmark

```bash
# NIM backend (requires OPENROUTER_API_KEY or NVIDIA_API_KEY)
EMBEDDING_BACKEND=nim python3 benchmarks/run_cross_encoder.py

# Local backend (default, no API keys needed)
EMBEDDING_BACKEND=local python3 benchmarks/run_cross_encoder.py
```

## Configuration

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `EMBEDDING_BACKEND` | `local`, `nim` | `local` | Embedding model backend |
| `SIGNAL_CANDIDATE_LIMIT` | int | 500 | Candidates per signal before fusion |
| `USE_LIGHTWEIGHT_RERANKER` | `true`, `false` | `true` | Use mxbai reranker |
| `USE_TWO_STAGE_RERANKER` | `true`, `false` | `false` | Two-stage hybrid → reranker pipeline |
| `WEIGHT_SEM` | float | 1.5 | Semantic signal weight |
| `WEIGHT_BM25` | float | 1.0 | BM25 signal weight |
| `WEIGHT_ENT` | float | 5.0 | Entity boosting weight |
| `WEIGHT_TEMP` | float | 1.0 | Temporal decay weight |
| `FUSION_METHOD` | `weighted`, `rrf` | `weighted` | Fusion algorithm |