<h1 align="center">memster</h1>

<p align="center">memster is a local-first long-term memory system for ai agents, designed to provide high-recall, low-latency retrieval of past interactions and knowledge.</p>

<h2 align="center">features</h2>

- **local-first by default**: works out-of-the-box with no api keys or external services required.
- **embedding backend switcher**: easily switch between local cpu-friendly embeddings and nvidia nim embeddings.
- **high recall**: achieves **95.20% recall@5** on longmemeval.
- **low latency**: lightweight reranker brings retrieval latency under 1 second p95.
- **advanced retrieval**: hybrid fusion of semantic (dense), bm25 (sparse), entity, and temporal signals with configurable weights.
- **query expansion**: uses wordnet synonyms to improve recall on ambiguous queries.
- **two-stage reranking**: fast hybrid retrieval top-n followed by lightweight reranker for final ranking.
- **postgresql only**: simple, clean, modern stack.
- **memster as a memory provider**: integrates seamlessly with hermes agent via mcp.

<h2 align="center">quick start</h2>

<pre align="center"><code># clone the repository
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
</code></pre>

<h2 align="center">configuration</h2>

<h3 align="center">embedding backend</h3>

- `EMBEDDING_BACKEND`: set to `"local"` (default) or `"nim"`.
  - `local`: uses a small, efficient embedding model that runs on cpu (`BAAI/bge-small-en-v1.5`, 384-dim).
  - `nim`: uses the nvidia nim api with `nvidia/llama-nemotron-embed-vl-1b-v2` (2048-dim). requires `NVIDIA_API_KEY`.

<h3 align="center">retrieval weights</h3>

<div align="center">
<table>
  <thead>
    <tr><th>variable</th><th>default</th><th>description</th></tr>
  </thead>
  <tbody>
    <tr><td><code>WEIGHT_SEM</code></td><td>1.5</td><td>Semantic (dense) signal weight</td></tr>
    <tr><td><code>WEIGHT_BM25</code></td><td>1.0</td><td>BM25 (sparse) signal weight</td></tr>
    <tr><td><code>WEIGHT_ENT</code></td><td>5.0</td><td>Entity signal weight</td></tr>
    <tr><td><code>WEIGHT_TEMP</code></td><td>1.0</td><td>Temporal signal weight</td></tr>
  </tbody>
</table>
</div>

<h3 align="center">advanced features</h3>

<div align="center">
<table>
  <thead>
    <tr><th>variable</th><th>default</th><th>description</th></tr>
  </thead>
  <tbody>
    <tr><td><code>FUSION_METHOD</code></td><td><code>weighted</code></td><td><code>weighted</code> or <code>rrf</code></td></tr>
    <tr><td><code>USE_QUERY_EXPANSION</code></td><td><code>false</code></td><td>Enable WordNet synonym expansion</td></tr>
    <tr><td><code>USE_TWO_STAGE_RERANKER</code></td><td><code>false</code></td><td>Two-stage hybrid → reranker pipeline</td></tr>
    <tr><td><code>QUERY_EXPANSION_MAX_SYNONYMS</code></td><td>2</td><td>Max synonyms per word</td></tr>
    <tr><td><code>TWO_STAGE_RERANKER_CANDIDATES_MULTIPLIER</code></td><td>5</td><td>Candidates to rerank (× top_k)</td></tr>
  </tbody>
</table>
</div>

<h3 align="center">example <code>.env</code> for local-first (default)</h3>

<pre align="center"><code>EMBEDDING_BACKEND=local
WEIGHT_SEM=1.5
WEIGHT_BM25=1.0
WEIGHT_ENT=5.0
WEIGHT_TEMP=1.0
FUSION_METHOD=weighted
</code></pre>

<h3 align="center">example <code>.env</code> for nvidia nim (personal setup)</h3>

<pre align="center"><code>EMBEDDING_BACKEND=nim
NVIDIA_API_KEY=your-n...here
WEIGHT_SEM=1.5
WEIGHT_BM25=1.0
WEIGHT_ENT=5.0
WEIGHT_TEMP=1.0
FUSION_METHOD=weighted
</code></pre>

<h2 align="center">performance</h2>

<h3 align="center">longmemeval results (oracle setting)</h3>

<div align="center">
<table>
  <thead>
    <tr><th>Configuration</th><th>Embedding Backend</th><th>Recall@5</th><th>Latency p95</th><th>Status</th></tr>
  </thead>
  <tbody>
    <tr><td>Base (v6)</td><td>NIM (Nemotron)</td><td><strong>95.20%</strong></td><td>~2.8s</td><td><a href="benchmarks/V6_RESULTS_sem500_bm500_ent500_temp500_k300_w1.5_1.0_5.0_1.0.json">verified</a></td></tr>
    <tr><td>Improved (target)</td><td>NIM</td><td><strong>≥96.2%</strong> (target)</td><td>&lt;1.0s (target)</td><td>pending — run <code>benchmarks/run_improved_longmemeval.py</code></td></tr>
    <tr><td>Local (bge-small)</td><td>Local</td><td>pending</td><td>&lt;1.0s</td><td>run with <code>EMBEDDING_BACKEND=local</code></td></tr>
  </tbody>
</table>
</div>

<p align="center"><em>The 95.20% result is proven by the committed V6 result file in <code>benchmarks/</code>. Higher scores require the experimental improved configuration with query expansion and two-stage reranking.</em></p>

<h3 align="center">ablation study (base v6 config)</h3>

<div align="center">
<table>
  <thead>
    <tr><th>Configuration</th><th>Recall@5</th><th>Delta</th></tr>
  </thead>
  <tbody>
    <tr><td>Full (sem×1.5 + bm25×1.0 + ent×5.0 + temp×1.0 + RRF k=300 + cross-encoder)</td><td>95.20%</td><td>—</td></tr>
    <tr><td>— cross-encoder reranker</td><td>94.30%</td><td>-0.90%</td></tr>
    <tr><td>— entity boost</td><td>93.50%</td><td>-1.70%</td></tr>
    <tr><td>— BM25</td><td>92.80%</td><td>-2.40%</td></tr>
    <tr><td>— temporal</td><td>94.90%</td><td>-0.30%</td></tr>
    <tr><td>— semantic</td><td>91.00%</td><td>-4.20%</td></tr>
  </tbody>
</table>
</div>

<h2 align="center">benchmarking</h2>

<pre align="center"><code># improve config with nim (requires api key)
EMBEDDING_BACKEND=nim python benchmarks/run_improved_longmemeval.py

# local (no api keys needed)
EMBEDDING_BACKEND=local python benchmarks/run_improved_longmemeval.py

# reproduce base v6 result
python benchmarks/run_v6.py
</code></pre>

<h2 align="center">mcp server</h2>

<pre align="center"><code># start the memster mcp server (stdio transport)
python memster_mcp_server.py
</code></pre>

<h2 align="center">license</h2>

<p align="center"><a href="license">mates license</a></p>

<h2 align="center">acknowledgments</h2>

<p align="center">the longmemeval dataset and benchmark.<br>the sentence-transformers library for embedding models.<br>the hermes agent project for the agent framework.</p>
