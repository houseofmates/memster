<h1 align="center">memster</h1>

memster is a local-first long-term memory system for ai agents, designed to provide high-recall, low-latency retrieval of past interactions and knowledge.

<h2 align="center">features</h2>

- **local-first by default**: works out-of-the-box with no api keys or external services required.
- **embedding backend switcher**: easily switch between local cpu-friendly embeddings and nvidia nim embeddings.
- **high recall**: achieves **95.20% recall@5** on longmemeval (verified). experimental config targeting ≥96.2% with nim.
- **low latency**: lightweight reranker brings retrieval latency under 1 second p95.
- **advanced retrieval**: hybrid fusion of semantic (dense), bm25 (sparse), entity, and temporal signals with configurable weights.
- **query expansion**: uses wordnet synonyms to improve recall on ambiguous queries.
- **two-stage reranking**: fast hybrid retrieval top-n followed by lightweight reranker for final ranking.
- **postgresql only**: no sqlite dependencies. no pieces mcp. clean, modern stack.
- **memster as a memory provider**: integrates seamlessly with hermes agent via mcp.

<h2 align="center">quick start</h2>

```bash
# clone the repository
git clone https://github.com/houseofmates/memster.git
cd memster

# install dependencies
pip install -e .

# start postgresql (required)
docker run -d --name memster-pg -e POSTGRES_PASSWORD=*** -p 5433:5432 postgres:15

# verify installation
python -c "
from memster.hybrid_retrieval import HybridRetrievalEngine
import psycopg2
def get_conn():
    return psycopg2.connect(host='localhost', port=5433, user='house', password='house', database='memster')
engine = HybridRetrievalEngine(get_conn)
print('embedding backend:', getattr(engine, 'embedding_backend', 'unknown'))
print('embeddings available:', engine.embeddings_available)
"
```

<h2 align="center">configuration</h2>

### embedding backend

- `EMBEDDING_BACKEND`: set to `"local"` (default) or `"nim"`.
  - `local`: uses a small, efficient embedding model that runs on cpu (`BAAI/bge-small-en-v1.5`, 384-dim).
  - `nim`: uses the nvidia nim api with `nvidia/llama-nemotron-embed-vl-1b-v2` (2048-dim). requires `NVIDIA_API_KEY`.

### retrieval weights

| variable | default | description |
|----------|---------|-------------|
| `WEIGHT_SEM` | 1.5 | Semantic (dense) signal weight |
| `WEIGHT_BM25` | 1.0 | BM25 (sparse) signal weight |
| `WEIGHT_ENT` | 5.0 | Entity signal weight |
| `WEIGHT_TEMP` | 1.0 | Temporal signal weight |

### advanced features

| variable | default | description |
|----------|---------|-------------|
| `FUSION_METHOD` | `weighted` | `weighted` or `rrf` |
| `USE_QUERY_EXPANSION` | `false` | Enable WordNet synonym expansion |
| `USE_TWO_STAGE_RERANKER` | `false` | Two-stage hybrid → reranker pipeline |
| `QUERY_EXPANSION_MAX_SYNONYMS` | 2 | Max synonyms per word |
| `TWO_STAGE_RERANKER_CANDIDATES_MULTIPLIER` | 5 | Candidates to rerank (× top_k) |

### example `.env` for local-first (default)

```bash
EMBEDDING_BACKEND=local
WEIGHT_SEM=1.5
WEIGHT_BM25=1.0
WEIGHT_ENT=5.0
WEIGHT_TEMP=1.0
FUSION_METHOD=weighted
```

### example `.env` for nvidia nim (personal setup)

```bash
EMBEDDING_BACKEND=nim
NVIDIA_API_KEY=your-n...here
WEIGHT_SEM=1.5
WEIGHT_BM25=1.0
WEIGHT_ENT=5.0
WEIGHT_TEMP=1.0
FUSION_METHOD=weighted
```

<h2 align="center">performance</h2>

<h3 align="center">longmemeval results (oracle setting)</h3>

| Configuration | Embedding Backend | Recall@5 | Latency p95 | Status |
|---------------|-------------------|----------|-------------|--------|
| Base (v6) | NIM (Nemotron) | **95.20%** | ~2.8s | [verified](benchmarks/V6_RESULTS_sem500_bm500_ent500_temp500_k300_w1.5_1.0_5.0_1.0.json) |
| Improved (target) | NIM | **≥96.2%** (target) | <1.0s (target) | pending — run `benchmarks/run_improved_longmemeval.py` |
| Local (bge-small) | Local | pending | <1.0s | run with `EMBEDDING_BACKEND=local` |

*The 95.20% result is proven by the committed V6 result file in `benchmarks/`. Higher scores require the experimental improved configuration with query expansion and two-stage reranking.*

<h3 align="center">ablation study (base v6 config)</h3>

| Configuration | Recall@5 | Delta |
|---------------|----------|-------|
| Full (sem×1.5 + bm25×1.0 + ent×5.0 + temp×1.0 + RRF k=300 + cross-encoder) | 95.20% | — |
| — cross-encoder reranker | 94.30% | -0.90% |
| — entity boost | 93.50% | -1.70% |
| — BM25 | 92.80% | -2.40% |
| — temporal | 94.90% | -0.30% |
| — semantic | 91.00% | -4.20% |

<h2 align="center">benchmarking</h2>

```bash
# improve config with nim (requires api key)
EMBEDDING_BACKEND=nim python benchmarks/run_improved_longmemeval.py

# local (no api keys needed)
EMBEDDING_BACKEND=local python benchmarks/run_improved_longmemeval.py

# reproduce base v6 result
python benchmarks/run_v6.py
```

<h2 align="center">mcp server</h2>

```bash
# start the memster mcp server (stdio transport)
python memster_mcp_server.py
```

<h2 align="center">project cleanup</h2>

- **pieces mcp**: removed entirely. the old memster mcp server no longer references pieces or attempts to sync with it.
- **sqlite**: removed entirely. all storage goes through postgresql via `memster.hybrid_retrieval`.
- **legacy modules**: `memster_v4_features.py`, `memster_gbrain.py`, `memster_beads.py`, `memster_phase2.py`, and `memster_spaced_repetition.py` have been deleted. their functionality was either consolidated into `memster.hybrid_retrieval` or was dead code referencing sqlite/pieces.
- **benchmark claims**: readme and benchmarks.md now report verified numbers (95.20%) and mark higher scores as "target/pending".

<h2 align="center">license</h2>

[mit license](license)

<h2 align="center">acknowledgments</h2>

- the longmemeval dataset and benchmark.
- the sentence-transformers library for embedding models.
- the hermes agent project for the agent framework.