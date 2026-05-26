# Memster LongMemEval Benchmarks

This document reports LongMemEval performance for Memster.

## Verified Result

| Backend | Fusion | Reranker | Recall@5 | p95 Latency | Proof |
|---------|--------|----------|----------|-------------|-------|
| NIM (Nemotron) | weighted + RRF k=300 | cross-encoder (MiniLM-L-6) | **95.20%** | ~2.8s | [V6 result file](V6_RESULTS_sem500_bm500_ent500_temp500_k300_w1.5_1.0_5.0_1.0.json) |

Configuration: semantic weight 1.5, BM25 1.0, entity 5.0, temporal 1.0, RRF k=300, cross-encoder ms-marco-MiniLM-L-6-v2.

## Experimental / Pending Results

| Backend | Fusion | Query Expand | Reranker | Recall@5 | p95 Latency | Status |
|---------|--------|--------------|----------|----------|-------------|--------|
| NIM + two-stage | weighted | true | lightweight (mxbai) | **≥96.2% (target)** | <1.0s (target) | pending — run `benchmarks/run_improved_longmemeval.py` |
| Local (bge-small) | weighted | true | lightweight (mxbai) | pending | <1.0s | pending — run with `EMBEDDING_BACKEND=local` |

*"Pending" means these configurations are implemented in code but the benchmark has not been run and the results not committed. Run the script yourself to generate numbers.*

## How to Reproduce

### Prerequisites

```bash
pip install -e .[all]
```

### V6 (verified 95.20%)

```bash
EMBEDDING_BACKEND=nim python benchmarks/run_v6.py
```

### Improved config (pending ≥96.2%)

```bash
EMBEDDING_BACKEND=nim \
  USE_LIGHTWEIGHT_RERANKER=true \
  USE_TWO_STAGE_RERANKER=true \
  USE_QUERY_EXPANSION=true \
  python benchmarks/run_improved_longmemeval.py
```

### Local (target)

```bash
EMBEDDING_BACKEND=local \
  USE_LIGHTWEIGHT_RERANKER=true \
  USE_TWO_STAGE_RERANKER=true \
  USE_QUERY_EXPANSION=true \
  python benchmarks/run_improved_longmemeval.py
```

## Configuration Options

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `EMBEDDING_BACKEND` | `local`, `nim` | `local` | Which embedding model to use |
| `USE_QUERY_EXPANSION` | `true`, `false` | `false` | Expand queries with WordNet |
| `USE_TWO_STAGE_RERANKER` | `true`, `false` | `false` | Two-stage reranking |
| `USE_LIGHTWEIGHT_RERANKER` | `true`, `false` | `true` | Use mxbai reranker |
| `WEIGHT_SEM` | float | 1.5 | Semantic signal weight |
| `WEIGHT_BM25` | float | 1.0 | BM25 signal weight |
| `WEIGHT_ENT` | float | 5.0 | Entity boosting weight |
| `WEIGHT_TEMP` | float | 1.0 | Temporal decay weight |
| `FUSION_METHOD` | `weighted`, `rrf` | `weighted` | Fusion algorithm |

## Performance Notes

- Verified 95.20% uses cross-encoder reranker (~2.7s per query).
- Lightweight reranker (mxbai-xsmall) targets < 0.2s per rerank.
- Expected combined latency with two-stage reranking: p95 < 1s.
- Local backend (bge-small, 384-dim) is expected to score lower than NIM (Nemotron, 2048-dim) due to reduced embedding resolution.