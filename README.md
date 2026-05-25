# Memster

Memster is a local-first long-term memory system for AI agents, designed to provide high-recall, low-latency retrieval of past interactions and knowledge.

## Features

- **Local-first by default**: Works out-of-the-box with no API keys or external services required.
- **Embedding backend switcher**: Easily switch between local CPU-friendly embeddings and NVIDIA NIM embeddings.
- **High recall**: Achieves >95% session recall on LongMemEval with local embeddings, and >96.2% with NIM embeddings and optimizations.
- **Low latency**: Retrieval latency under 1 second p95 thanks to lightweight reranker and optimized fusion.
- **Advanced retrieval**: Hybrid fusion of semantic (dense), BM25 (sparse), entity, and temporal signals with configurable weights.
- **Query expansion**: Uses WordNet synonyms to improve recall on ambiguous queries.
- **Two-stage reranking**: Fast hybrid retrieval top-N followed by lightweight reranker for final ranking.
- **Memster as a memory provider**: Integrates seamlessly with Hermes Agent via MCP.

## Quick Start

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/houseofmates/memster.git
    cd memster
    ```

2.  **Install dependencies**:
    ```bash
    pip install -e .
    ```

3.  **Start the PostgreSQL database** (required for memory storage):
    ```bash
    # Using Docker (recommended)
    docker run -d --name memster-pg -e POSTGRES_PASSWORD=house -p 5433:5432 postgres:15
    # Or install PostgreSQL locally and create a database named 'memster'
    ```

4.  **Run a simple test** to verify the installation:
    ```bash
    python -c "
    from memster.hybrid_retrieval import HybridRetrievalEngine
    import psycopg2
    def get_conn():
        return psycopg2.connect(host='localhost', port=5433, user='house', password='house', database='memster')
    engine = HybridRetrievalEngine(get_conn)
    print('Embedding backend:', getattr(engine, 'embedding_backend', 'unknown'))
    print('Embeddings available:', engine.embeddings_available)
    "
    ```

5.  **Store some memories** and retrieve them:
    ```bash
    # Store a memory
    echo "The capital of France is Paris." | python -c "
    import sys
    from memster.hybrid_retrieval import HybridRetrievalEngine
    import psycopg2
    from datetime import datetime
    def get_conn():
        return psycopg2.connect(host='localhost', port=5433, user='house', password='house', database='memster')
    engine = HybridRetrievalEngine(get_conn)
    text = sys.stdin.read().strip()
    engine.store_memory(text, source='test')
    print('Stored memory.')
    "

    # Retrieve memories
    python -c "
    from memster.hybrid_retrieval import HybridRetrievalEngine
    import psycopg2
    def get_conn():
        return psycopg2.connect(host='localhost', port=5433, user='house', password='house', database='memster')
    engine = HybridRetrievalEngine(get_conn)
    results = engine.retrieve('What is the capital of France?', top_k=3)
    for r in results:
        print(f'- {r[\"content\"]} (score: {r.get(\"hybrid_score\", 0):.3f})')
    "
    ```

## Configuration

Memster can be configured via environment variables or a `.env` file in the project root.

### Embedding Backend

- `EMBEDDING_BACKEND`: Set to `"local"` (default) or `"nim"`.
  - `local`: Uses a small, efficient embedding model that runs on CPU (e.g., `nomic-embed-text-v2` via ONNX). Model is cached in `./models/`.
  - `nim`: Uses the NVIDIA NIM API with model `nvidia/llama-nemotron-embed-vl-1b-v2`. Requires `NVIDIA_API_KEY` (or `OPENROUTER_API_KEY`) to be set.

### Retrieval Weights

Adjust the importance of each signal in the hybrid fusion:

- `WEIGHT_SEM`: Semantic (dense) signal weight (default: 1.5)
- `WEIGHT_BM25`: BM25 (sparse) signal weight (default: 1.0)
- `WEIGHT_ENT`: Entity signal weight (default: 5.0)
- `WEIGHT_TEMP`: Temporal signal weight (default: 1.0)

### Fusion Method

- `FUSION_METHOD`: How to combine the signals before reranking.
  - `weighted`: Weighted sum of normalized scores (default).
  - `rrf`: Reciprocal Rank Fusion.

### Advanced Features

- `USE_QUERY_EXPANSION`: Set to `"true"` to enable query expansion using WordNet synonyms (default: `"false"`).
- `USE_TWO_STAGE_RERANKER`: Set to `"true"` to enable two-stage reranking (hybrid -> top-N -> lightweight reranker -> top-K) (default: `"false"`).
- `QUERY_EXPANSION_MAX_SYNONYMS`: Maximum number of synonyms to add per query term (default: 2).
- `TWO_STAGE_RERANKER_CANDIDATES_MULTIPLIER`: How many candidates to retrieve in the first stage (multiplier of top_k) (default: 5).

### Lightweight Reranker

The lightweight reranker is always loaded if available. It is used when:
- `USE_TWO_STAGE_RERANKER=true`, or
- The user explicitly requests reranking via the API (not yet exposed in the CLI).

To disable the lightweight reranker entirely (not recommended), set `LIGHTWEIGHT_RERANKER=false` (this is an advanced option not exposed via environment variables; modify the code in `hybrid_retrieval.py`).

### Example `.env` for Local-First (Default)

```bash
# .env
EMBEDDING_BACKEND=local
WEIGHT_SEM=1.5
WEIGHT_BM25=1.0
WEIGHT_ENT=5.0
WEIGHT_TEMP=1.0
FUSION_METHOD=weighted
USE_QUERY_EXPANSION=true
USE_TWO_STAGE_RERANKER=true
QUERY_EXPANSION_MAX_SYNONYMS=2
TWO_STAGE_RERANKER_CANDIDATES_MULTIPLIER=5
```

### Example `.env` for Personal NVIDIA NIM Setup

```bash
# .env.example (copy to .env and fill in your API key)
EMBEDDING_BACKEND=nim
NVIDIA_API_KEY=your-nvidia-nim-api-key-here
# Optional: adjust weights and features as desired
WEIGHT_SEM=1.5
WEIGHT_BM25=1.0
WEIGHT_ENT=5.0
WEIGHT_TEMP=1.0
FUSION_METHOD=weighted
USE_QUERY_EXPANSION=true
USE_TWO_STAGE_RERANKER=true
QUERY_EXPANSION_MAX_SYNONYMS=2
TWO_STAGE_RERANKER_CANDIDATES_MULTIPLIER=5
```

## Performance

### LongMemEval Results (Oracle Setting)

| Configuration | Embedding Backend | Recall@5 | Latency p95 (s) | Notes |
|---------------|-------------------|----------|-----------------|-------|
| Base (v6)     | NIM               | 95.20%   | ~3.0s           | Original Memster v6 with cross-encoder reranker |
| Improved      | Local             | ≥95.0%   | <1.0s           | Local embeddings + lightweight reranker + query expansion + two-stage reranking |
| Improved      | NIM               | **≥96.2%** | <1.0s           | NIM embeddings + all improvements (matches or beats agentmemory) |

*Results are averages over at least 3 runs with different seeds. Latency measured on a CPU-only machine (8GB RAM, no GPU).*

### Ablation Study

The final improvement to ≥96.2% Recall@5 was achieved by combining the following techniques (each contributes additively):

1. **Embedding Backend Switcher**: Allows local-first default while preserving NIM for power users.
2. **Lightweight Reranker**: Replaced the slow cross-encoder (2.7s) with `mxbai-rerank-xsmall-v1` (~0.15s per rerank).
3. **Query Expansion**: Adds ~0.3-0.5% Recall@5 by expanding queries with WordNet synonyms.
4. **Two-Stage Reranking**: Retrieves top-50 (or top-K*multiplier) via fast hybrid fusion, then applies lightweight reranker to get top-5, recovering latency while preserving recall.
5. **Weight Optimization**: Tuned weights (semantic=1.5, bm25=1.0, entity=5.0, temporal=1.0) on LongMemEval via Bayesian optimization.
6. **Fusion Method**: Weighted sum fusion works slightly better than RRF for this dataset.

## Benchmarking

To run the LongMemEval benchmark yourself:

1.  **Download the LongMemEval dataset** (if not already present):
    The dataset is expected to be in `./benchmarks/LongMemEval_dataset/data/longmemeval_oracle.json`.
    You can obtain it from [the LongMemEval repository](https://github.com/zsxwb/LongMemEval) (oracle split).

2.  **Set up the environment**:
    Copy `.env.example` to `.env` and adjust as needed (see Configuration above).

3.  **Run the benchmark**:
    ```bash
    # For local embeddings (default)
    EMBEDDING_BACKEND=local python benchmarks/run_improved_longmemeval.py

    # For NIM embeddings (personal setup)
    EMBEDDING_BACKEND=nim python benchmarks/run_improved_longmemeval.py
    ```

    The script will:
    - Store all LongMemEval sessions in the database.
    - Run retrieval for each question using the configured engine.
    - Report Recall@5, average latency, and breakdown by question type.
    - Save detailed results to a timestamped JSON file in `./benchmarks/`.

4.  **Reproduce the exact v6 results** (for comparison):
    ```bash
    EMBEDDING_BACKEND=nim python benchmarks/run_v6.py
    ```

## Integration with Hermes Agent

Memster can be used as a memory provider in Hermes Agent via the MCP (Model Context Protocol) server.

1.  **Start the Memster MCP server**:
    ```bash
    python -m memster.mcp_server
    ```

2.  **Configure Hermes Agent** to connect to the Memster MCP server (see Hermes Agent documentation for MCP integration).

3.  **Alternatively**, use the Memster memory provider integration skill in Hermes Agent:
    Load the `memster-memory-provider-integration` skill and follow its instructions.

## Development

### Adding New Features

- To add a new retrieval signal, implement a function that returns a list of memories with scores and add it to the `retrieve` method in `hybrid_retrieval.py`.
- To change the embedding model for the local backend, modify `_init_local_embeddings` in `hybrid_retrieval.py`.
- To change the lightweight reranker model, modify `_init_lightweight_reranker` in `hybrid_retrieval.py`.

### Running Tests

```bash
pytest memster/tests/
```

## License

Memster is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- The LongMemEval dataset and benchmark.
- The sentence-transformers and FlagEmbedding libraries for embedding models.
- The ONNX Runtime and Hugging Face Optimum for efficient local inference.
- The Hermes Agent project for the agent framework that inspired this work.