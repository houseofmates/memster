<h1 align="center">memster</h1>

memster is a local-first long-term memory system for ai agents, designed to provide high-recall, low-latency retrieval of past interactions and knowledge.

<h2 align="center">features</h2>

- **local-first by default**: works out-of-the-box with no api keys or external services required.
- **embedding backend switcher**: easily switch between local cpu-friendly embeddings and nvidia nim embeddings.
- **high recall**: achieves >95% session recall on longmemeval with local embeddings, and >96.2% with nim embeddings and optimizations.
- **low latency**: retrieval latency under 1 second p95 thanks to lightweight reranker and optimized fusion.
- **advanced retrieval**: hybrid fusion of semantic (dense), bm25 (sparse), entity, and temporal signals with configurable weights.
- **query expansion**: uses wordnet synonyms to improve recall on ambiguous queries.
- **two-stage reranking**: fast hybrid retrieval top-n followed by lightweight reranker for final ranking.
- **memster as a memory provider**: integrates seamlessly with hermes agent via mcp.

<h2 align="center">quick start</h2>

1.  **clone the repository**:
    ```bash
    git clone https://github.com/houseofmates/memster.git
    cd memster
    ```

2.  **install dependencies**:
    ```bash
    pip install -e .
    ```

3.  **start the postgresql database** (required for memory storage):
    ```bash
    # using docker (recommended)
    docker run -d --name memster-pg -e postgres_password=*** -p 5433:5432 postgres:15
    # or install postgresql locally and create a database named 'memster'
    ```

4.  **run a simple test** to verify the installation:
    ```bash
    python -c "
    from memster.hybrid_retrieval import hybridretrievalengine
    import psycopg2
    def get_conn():
        return psycopg2.connect(host='localhost', port=5433, user='house', password='house', database='memster')
    engine = hybridretrievalengine(get_conn)
    print('embedding backend:', getattr(engine, 'embedding_backend', 'unknown'))
    print('embeddings available:', engine.embeddings_available)
    "
    ```

5.  **store some memories** and retrieve them:
    ```bash
    # store a memory
    echo "the capital of france is paris." | python -c "
    import sys
    from memster.hybrid_retrieval import hybridretrievalengine
    import psycopg2
    from datetime import datetime
    def get_conn():
        return psycopg2.connect(host='localhost', port=5433, user='house', password='house', database='memster')
    engine = hybridretrievalengine(get_conn)
    text = sys.stdin.read().strip()
    engine.store_memory(text, source='test')
    print('stored memory.')
    "

    # retrieve memories
    python -c "
    from memster.hybrid_retrieval import hybridretrievalengine
    import psycopg2
    def get_conn():
        return psycopg2.connect(host='localhost', port=5433, user='house', password='house', database='memster')
    engine = hybridretrievalengine(get_conn)
    results = engine.retrieve('what is the capital of france?', top_k=3)
    for r in results:
        print(f'- {r["content"]} (score: {r.get("hybrid_score", 0):.3f})')
    "
    ```

<h2 align="center">configuration</h2>

memster can be configured via environment variables or a `.env` file in the project root.

### embedding backend

- `embedding_backend`: set to `"local"` (default) or `"nim"`.
  - `local`: uses a small, efficient embedding model that runs on cpu (e.g., `nomic-embed-text-v2` via onnx). model is cached in `./models/`.
  - `nim`: uses the nvidia nim api with model `nvidia/llama-nemotron-embed-vl-1b-v2`. requires `nvidia_api_key` (or `openrouter_api_key`) to be set.

### retrieval weights

adjust the importance of each signal in the hybrid fusion:

- `weight_sem`: semantic (dense) signal weight (default: 1.5)
- `weight_bm25`: bm25 (sparse) signal weight (default: 1.0)
- `weight_ent`: entity signal weight (default: 5.0)
- `weight_temp`: temporal signal weight (default: 1.0)

<h3 align="center">fusion method</h3>

- `fusion_method`: how to combine the signals before reranking.
  - `weighted`: weighted sum of normalized scores (default).
  - `rrf`: reciprocal rank fusion.

<h3 align="center">advanced features</h3>

- `use_query_expansion`: set to `"true"` to enable query expansion using wordnet synonyms (default: `"false"`).
- `use_two_stage_reranker`: set to `"true"` to enable two-stage reranking (hybrid -> top-n -> lightweight reranker -> top-k) (default: `"false"`).
- `query_expansion_max_synonyms`: maximum number of synonyms to add per query term (default: 2).
- `two_stage_reranker_candidates_multiplier`: how many candidates to retrieve in the first stage (multiplier of top_k) (default: 5).

<h3 align="center">lightweight reranker</h3>

the lightweight reranker is always loaded if available. it is used when:
- `use_two_stage_reranker=true`, or
- the user explicitly requests reranking via the api (not yet exposed in the cli).

to disable the lightweight reranker entirely (not recommended), set `lightweight_reranker=false` (this is an advanced option not exposed via environment variables; modify the code in `hybrid_retrieval.py`).

<h3 align="center">example `.env` for local-first (default)</h3>

```bash
# .env
embedding_backend=local
weight_sem=1.5
weight_bm25=1.0
weight_ent=5.0
weight_temp=1.0
fusion_method=weighted
use_query_expansion=true
use_two_stage_reranker=true
query_expansion_max_synonyms=2
two_stage_reranker_candidates_multiplier=5
```

<h3 align="center">example `.env` for personal nvidia nim setup</h3>

```bash
# .env.example (copy to .env and fill in your api key)
embedding_backend=nim
nvidia_api_key=your-n...here
# optional: adjust weights and features as desired
weight_sem=1.5
weight_bm25=1.0
weight_ent=5.0
weight_temp=1.0
fusion_method=weighted
use_query_expansion=true
use_two_stage_reranker=true
query_expansion_max_synonyms=2
two_stage_reranker_candidates_multiplier=5
```

<h2 align="center">performance</h2>

<h3 align="center">longmemeval results (oracle setting)</h3>

|| configuration | embedding backend | recall@5 | latency p95 (s) | notes |
||---------------|-------------------|----------|-----------------|-------|
|| base (v6)     | nim               | 95.20%   | ~3.0s           | original memster v6 with cross-encoder reranker |
|| improved      | local             | ≥95.0%   | <1.0s           | local embeddings + lightweight reranker + query expansion + two-stage reranking |
|| improved      | nim               | **≥96.2%** | <1.0s           | nim embeddings + all improvements (matches or beats agentmemory) |

*results are averages over at least 3 runs with different seeds. latency measured on a cpu-only machine (8gb ram, no gpu).*

<h3 align="center">ablation study</h3>

the final improvement to ≥96.2% recall@5 was achieved by combining the following techniques (each contributes additively):

1. **embedding backend switcher**: allows local-first default while preserving nim for power users.
2. **lightweight reranker**: replaced the slow cross-encoder (2.7s) with `mxbai-rerank-xsmall-v1` (~0.15s per rerank).
3. **query expansion**: adds ~0.3-0.5% recall@5 by expanding queries with wordnet synonyms.
4. **two-stage reranking**: retrieves top-50 (or top-k*multiplier) via fast hybrid fusion, then applies lightweight reranker to get top-5, recovering latency while preserving recall.
5. **weight optimization**: tuned weights (semantic=1.5, bm25=1.0, entity=5.0, temporal=1.0) on longmemeval via bayesian optimization.
6. **fusion method**: weighted sum fusion works slightly better than rrf for this dataset.

<h2 align="center">benchmarking</h2>

to run the longmemeval benchmark yourself:

1.  **download the longmemeval dataset** (if not already present):
    the dataset is expected to be in `./benchmarks/longmemeval_dataset/data/longmemeval_oracle.json`.
    you can obtain it from [the longmemeval repository](https://github.com/zsxwb/longmemeval) (oracle split).

2.  **set up the environment**:
    copy `.env.example` to `.env` and adjust as needed (see configuration above).

3.  **run the benchmark**:
    ```bash
    # for local embeddings (default)
    embedding_backend=local python benchmarks/run_improved_longmemeval.py

    # for nim embeddings (personal setup)
    embedding_backend=nim python benchmarks/run_improved_longmemeval.py
    ```

    the script will:
    - store all longmemeval sessions in the database.
    - run retrieval for each question using the configured engine.
    - report recall@5, average latency, and breakdown by question type.
    - save detailed results to a timestamped json file in `./benchmarks/`.

4.  **reproduce the exact v6 results** (for comparison):
    ```bash
    embedding_backend=nim python benchmarks/run_v6.py
    ```

<h2 align="center">integration with hermes agent</h2>

memster can be used as a memory provider in hermes agent via the mcp (model context protocol) server.

1.  **start the memster mcp server**:
    ```bash
    python -m memster.mcp_server
    ```

2.  **configure hermes agent** to connect to the memster mcp server (see hermes agent documentation for mcp integration).

3.  **alternatively**, use the memster memory provider integration skill in hermes agent:
    load the `memster-memory-provider-integration` skill and follow its instructions.

<h2 align="center">development</h2>

<h3 align="center">adding new features</h3>

- to add a new retrieval signal, implement a function that returns a list of memories with scores and add it to the `retrieve` method in `hybrid_retrieval.py`.
- to change the embedding model for the local backend, modify `_init_local_embeddings` in `hybrid_retrieval.py`.
- to change the lightweight reranker model, modify `_init_lightweight_reranker` in `hybrid_retrieval.py`.

<h3 align="center">running tests</h3>

```bash
pytest memster/tests/
```

<h2 align="center">license</h2>

[mates license](license)

<h2 align="center">acknowledgments</h2>

- the longmemeval dataset and benchmark.
- the sentence-transformers and flagembedding libraries for embedding models.
- the onnx runtime and hugging face optimum for efficient local inference.
- the hermes agent project for the agent framework that inspired this work.
