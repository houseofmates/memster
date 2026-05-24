<h1 align="center">memster</h1>

a local-first, long-term memory system for hermes agent. it provides persistent memory across sessions, enabling the agent to learn from experience, recall past conversations, and build a deepening model of the user over time. built with postgresql and designed for durability, memster is the core of hermes' self-improving capabilities.

<h2 align="center">features</h2>

- **persistent memory** — stores memories in a postgresql database with full-text search (tsvector/gin) and vector embeddings for semantic recall
- **four-network model** — organizes memories into world, experience, opinion, and observation networks for structured recall
- **mcp integration** — exposes memory functions via model context protocol for seamless integration with hermes and other agents
- **auto importance scoring** — automatically scores memory importance (0.0-1.0) based on content signals like entities, errors, length, and action/outcome pairs
- **conflict detection** — detects semantic conflicts (opposite states like "service is up" vs "service is down") before inserting new memories, returning warnings without blocking
- **pg_trgm duplicate detection** — uses postgresql's pg_trgm extension for O(log n) near-duplicate detection instead of O(n^2) python comparison
- **dream system** — database-integrated dream system that processes memories during idle periods to reinforce learning
- **semantic enhancements** — includes 9 semantic intelligence improvements for better understanding and association
- **self-evolution capabilities** — uses memster's own data to evaluate and improve its memory processes
- **durability ops** — operational procedures for making memster amnesia-proof with backups, repair, and consistency checks
- **activity tracking** — complete local-first activity tracking and memory system for logging user interactions
- **memory provider integration** — can be used as a memory provider in hermes agent for enhanced context

<h3 align="center">v5 enhancements</h3>

- **hybrid retrieval engine** — multi-signal retrieval fusing semantic similarity, BM25 keyword scoring, entity-based boosting, and temporal-proximity boosting with configurable fusion weights and optional LLM reranking
- **rules-based entity extraction** — zero-llm-call entity and relationship extraction using regex/patterns (inspired by GBrain) with typed relationships (works_at, founded, invested_in, etc.)
- **verbatim storage layer** — option to store original conversation turns alongside summarized memory beads for exact recall when needed
- **sophisticated decay scoring** — multi-factor decay model with time decay, access frequency boost, reinforcement boost, network-specific curves, and importance-based pinning
- **two-tier caching** — l1 in-memory lru cache (5-min ttl) + l2 disk-based cache (1-hour ttl) with automatic invalidation
- **entity graph queries** — graph-style queries: who works at X? what did Y invest in? find connection paths, entity timelines
- **feedback loop / reinforcement learning** — positive/negative/neutral feedback that adjusts memory strength and retrieval ranking
- **delta compression** — store only diffs when updating memories, enabling version history and efficient storage
- **bi-temporal tracking** — tracks both event_time (when thing happened) and ingested_at (when stored) for richer temporal reasoning
- **configurable extraction modes** — choose extraction strategy: llm (summarization), verbatim (raw text), hybrid (both), or algorithmic (zero-llm heuristics)
- **privacy & forgetting** — gdpr-compliant operations: memster_forget, memster_forget_entity, memster_export, memster_purge
- **pluggable backend interface** — abstract basebackend with postgresql implementation; design allows swapping to sqlite or chromadb
- **observability** — prometheus metrics endpoint, structured logging with correlation ids, operation latency histograms, health checks

<h2 align="center">installation</h2>

memster is designed to run alongside hermes agent. it requires postgresql and python 3.11+.

<h3 align="center">prerequisites</h3>

- postgresql 14+ (required — sqlite is no longer supported)
- python 3.11 or higher
- psycopg2 (installed automatically with `pip install -e .[all]`)
- git

<h3 align="center">setup</h3>

1. clone the repository
   ```bash
   git clone https://github.com/houseofmates/memster.git
   cd memster
   ```

2. create the postgresql database
   ```bash
   sudo -u postgres createdb memster
   sudo -u postgres psql -c "CREATE USER memster WITH PASSWORD 'your_password';"
   sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE memster TO memster;"
   ```

3. install dependencies
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .[all]
   ```

4. configure the database
   - copy `.env.example` to `.env` and set `DATABASE_URL`
   ```bash
   cp .env.example .env
   # Edit .env: DATABASE_URL=postgresql://memster:***@localhost:5432/memster
   ```

5. start the memster mcp server
   ```bash
   python memster_mcp_server.py
   ```
   the server runs over stdio (mcp protocol).

<h2 align="center">usage</h2>

once the memster server is running, hermes agent can connect to it via the mcp integration. configure hermes to use memster as a memory provider in `config.yaml`:

```yaml
mcp_servers:
  memster:
    command: python
    args: ["/path/to/memster/memster_mcp_server.py"]
```

memster provides the following tools through mcp:

- `memster_curate` / `memster_remember` — store a new memory with auto-dedup, conflict detection, auto importance scoring, completeness analysis, and embeddings fallback notification
- `memster_query` — search memories with full-text search (tsvector)
- `memster_status` — get system status and memory count
- `memster_embeddings_status` — check embedding backend status, model/provider info, and NVIDIA NIM setup instructions
- `hybrid_search` — hybrid ranking combining vector similarity, full-text, and importance
- `find_duplicates` — find near-duplicate memories using pg_trgm for O(log n) detection
- `memster_hybrid_retrieve` — advanced hybrid retrieval with configurable signal weights and optional LLM reranking
- `memster_extract_entities` — extract entities and relationships from text using zero-LLM rules
- `memster_store_verbatim` / `memster_get_verbatim` — store and retrieve verbatim conversation turns
- `memster_reinforce` — boost memory strength via reinforcement learning
- `memster_feedback` — submit feedback on memories (positive/negative/neutral)
- `memster_memory_diff` — view differences between memory versions
- `memster_forget` / `memster_forget_entity` — GDPR-compliant memory deletion
- `memster_health` — comprehensive system health check
- `memster_metrics` — prometheus-formatted metrics endpoint
- and many more for dream system, activity tracking, spaced repetition, entity graph queries, etc.

<h3 align="center">new features</h3>

**auto importance scoring** — memories are automatically scored 0.0-1.0 based on:
- network type baseline (world: 0.6, experience: 0.5, opinion: 0.4, observation: 0.4)
- presence of specific entities (IPs, paths, ports, URLs) — combined +0.2
- error/failure keywords (+0.15)
- content length > 100 chars (+0.1)
- action + outcome pair detection (+0.15)

**conflict detection** — before inserting a memory, memster checks for semantic conflicts (opposite states) in the same network type. for example, if you store "service nginx is up", memster will warn if there's an existing memory saying "service nginx is down". the `memster_curate` and `memster_remember` tools return a `conflicts_detected` field with details. conflicts are returned as warnings — they never block insertion.

**embeddings fallback** — when nvidia nim embeddings are unavailable, tools return `"embeddings_unavailable": true` and `"fallback_mode": "keyword_search"` in their responses. the `memster_embeddings_status` tool reports the current backend status, model/provider info, and provides setup instructions for enabling nvidia nim.

<h2 align="center">architecture</h2>

memster consists of several core components:

- **memster_mcp_server.py** — the main mcp server that exposes memory functions (postgresql backend)
- **memster_beads.py** — defines the memory bead structure and networks
- **memster_gbrain.py** — the global brain that coordinates memory operations
- **memster_spaced_repetition.py** — implements spaced repetition for memory retention
- **dream_consolidation.py** — processes memories during dream cycles to reinforce learning
- **memster_v4_features.py** — contains the 9 semantic intelligence improvements
- **memster_phase2.py** — phase 2 enhancements for the memory system
- **memster/** — v5 enhancement modules:
  - hybrid_retrieval.py — multi-signal retrieval engine
  - entity_extraction.py — rules-based entity and relationship extraction
  - verbatim.py — verbatim conversation storage
  - decay.py — sophisticated decay scoring
  - cache.py — two-tier caching system
  - feedback.py — feedback loop and reinforcement learning
  - delta.py — delta compression and version history
  - privacy.py — gdpr-compliant forgetting and export
  - graph_queries.py — entity graph traversal and timeline queries
  - extraction.py — configurable extraction modes (llm/verbatim/hybrid/algorithmic)
  - observability.py — prometheus metrics, health checks, structured logging
  - integration.py — wires v5 modules into mcp server
  - backends/ — pluggable backend interface with postgresql implementation

the system uses a postgresql database with the following tables:
- `memories` — stores individual memory beads with tsvector full-text search and gin index
- `memory_edges` — graph edges connecting related memories
- `memory_embeddings` — nvidia nim vector embeddings for semantic search
- `entities` — extracted entities (ips, paths, service names, etc.)
- `memory_entities` — junction table linking memories to entities
- `entity_relationships` — typed relationships between entities (works_at, founded, etc.)
- `verbatim_conversations` — stored conversation turns for exact recall
- `memory_versions` — delta-compressed version history for memory updates
- `memory_feedback` — feedback history for reinforcement learning
- and more for sessions, tasks, narrative arcs, memory palaces, etc.

<h2 align="center">benchmarks</h2>

see `benchmarks/BENCHMARKS.md` for:
- benchmarking suite instructions
- synthetic test corpus generator
- retrieval performance comparisons (semantic-only vs hybrid vs hybrid+rerank)
- comparison to competitor systems (mem0, mempalace, supermemory, honcho)
- notes on running against locomo and longmemeval datasets

<h2 align="center">contributing</h2>

we welcome contributions! please read [contributing.md](contributing.md) for details on our code of conduct and the process for submitting pull requests.

<h2 align="center">license</h2>

[the mates license](license)

built by [house of mates](https://github.com/houseofmates).