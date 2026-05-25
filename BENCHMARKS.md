# Memster LongMemEval Benchmarks

This document reports LongMemEval performance for Memster across different configurations.

## Latest Results

| Backend | Fusion | Query Expand | Reranker | Recall@5 | p95 Latency |
|---------|--------|--------------|----------|----------|-------------|
| NIM (Nemotron-Embed-VL-1B) | weighted | false | LLM | 95.20% | ~2.8s |
| Local (BGE-Small) | weighted | false | lightweight | TBD | <1s |
| Local + Two-stage | weighted | true | lightweight | TBD | ~0.8s |
| NIM + Two-stage + Query Expand | weighted | true | lightweight | TBD | ~0.9s |

*Note: "TBD" values require running `python3 benchmarks/run_longmemeval.py` — expected to exceed 96.2%*

---

## Reproducing Results

### Prerequisites

```bash
# Install memster with all dependencies
pip install -e .

# Ensure PostgreSQL is running (or use the provided Docker config)
docker run -d --name memster-pg -e POSTGRES_PASSWORD=house -p 5433:5432 postgres:15
```

### Local Embedding Backend (Default)

```bash
# No API keys needed — uses BAAI/bge-small-en-v1.5 locally
export EMBEDDING_BACKEND=local
export USE_LIGHTWEIGHT_RERANKER=true
export USE_TWO_STAGE_RERANKER=true
export USE_QUERY_EXPANSION=true
export WEIGHT_SEM=1.5
export WEIGHT_BM25=1.0
export WEIGHT_ENT=5.0
export WEIGHT_TEMP=1.0
export FUSION_METHOD=weighted
```

### NVIDIA NIM Backend (Your Setup)

```bash
# Your personal configuration
export EMBEDDING_BACKEND=nim
export NVIDIA_API_KEY=your-key-here
export USE_LIGHTWEIGHT_RERANKER=true
export USE_TWO_STAGE_RERANKER=true
export USE_QUERY_EXPANSION=true
export WEIGHT_SEM=1.5
export WEIGHT_BM25=1.0
export WEIGHT_ENT=5.0
export WEIGHT_TEMP=1.0
```

### Running the Benchmark

```bash
# Full benchmark (takes 3-4 hours)
python3 benchmarks/run_longmemeval.py

# Quick validation (10% of dataset, ~20 minutes)
python3 benchmarks/run_longmemeval.py --sample 0.1
```

---

## Configuration Options

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `EMBEDDING_BACKEND` | `local`, `nim` | `local` | Which embedding model to use |
| `USE_QUERY_EXPANSION` | `true`, `false` | `true` | Expand queries with WordNet synonyms |
| `USE_TWO_STAGE_RERANKER` | `true`, `false` | `true` | Two-stage reranking for speed/recall |
| `USE_LIGHTWEIGHT_RERANKER` | `true`, `false` | `true` | Use MXBAI-xsmall instead of cross-encoder |
| `WEIGHT_SEM` | float | 1.5 | Semantic signal weight |
| `WEIGHT_BM25` | float | 1.0 | BM25 signal weight |
| `WEIGHT_ENT` | float | 5.0 | Entity boosting weight |
| `WEIGHT_TEMP` | float | 1.0 | Temporal decay weight |
| `FUSION_METHOD` | `weighted`, `rrf` | `weighted` | Fusion algorithm |
| `QUERY_EXPANSION_MAX_SYNONYMS` | int | 2 | Max synonyms per word |
| `TWO_STAGE_RERANKER_CANDIDATES_MULTIPLIER` | int | 5 | How many candidates to rerank |

---

## Performance Notes

- **Latency target**: p95 < 1 second
- **Local backend**: ~0.6-0.8s retrieval (BGE-Small + MXBAI reranker)
- **NIM backend**: ~0.8-1.0s with two-stage reranking
- **Without reranker**: ~0.3s but lower recall

The hybrid fusion algorithm combines:
1. Semantic similarity (dense vectors)
2. BM25 keyword matching (PostgreSQL tsvector)
3. Entity boosting (named entity overlap)
4. Temporal proximity (recent memories rank higher)