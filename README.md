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
   # Edit .env: DATABASE_URL=postgresql://memster:your_password@localhost:5432/memster
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
- `memster_curate` / `memster_remember` — store a new memory with auto-dedup, importance scoring, and conflict detection
- `memster_query` — search memories with full-text search (tsvector)
- `memster_status` — get system status and memory count
- `memster_embeddings_status` — check embedding backend status and setup instructions
- `hybrid_search` — hybrid ranking combining vector similarity, full-text, and importance
- `find_duplicates` — find near-duplicate memories using pg_trgm
- and many more for dream system, activity tracking, spaced repetition, etc.

<h3 align="center">new features</h3>

**auto importance scoring** — memories are automatically scored 0.0-1.0 based on:
- network type baseline (world: 0.6, experience: 0.5, observation: 0.4)
- presence of specific entities (IPs, paths, ports, URLs)
- error/failure keywords
- content length
- action + outcome pair detection

**conflict detection** — before inserting a memory, memster checks for semantic conflicts in the same network type. for example, if you store "service nginx is up", memster will warn if there's an existing memory saying "service nginx is down". conflicts are returned as warnings — they never block insertion.

**embeddings status** — the `memster_embeddings_status` tool reports whether nvidia nim embeddings are available and provides setup instructions if not.

<h2 align="center">architecture</h2>

memster consists of several core components:

- **memster_mcp_server.py** — the main mcp server that exposes memory functions (postgresql backend)
- **memster_beads.py** — defines the memory bead structure and networks
- **memster_gbrain.py** — the global brain that coordinates memory operations
- **memster_spaced_repetition.py** — implements spaced repetition for memory retention
- **dream_consolidation.py** — processes memories during dream cycles to reinforce learning
- **memster_v4_features.py** — contains the 9 semantic intelligence improvements
- **memster_phase2.py** — phase 2 enhancements for the memory system

the system uses a postgresql database with the following tables:
- `memories` — stores individual memory beads with tsvector full-text search and gin index
- `memory_edges` — graph edges connecting related memories
- `memory_embeddings` — nvidia nim vector embeddings for semantic search
- `entities` — extracted entities (ips, paths, service names, etc.)
- and more for sessions, tasks, narrative arcs, memory palaces, etc.

<h2 align="center">contributing</h2>

we welcome contributions! please read [contributing.md](contributing.md) for details on our code of conduct and the process for submitting pull requests.

<h2 align="center">license</h2>

[the mates license](license)

built by [house of mates](https://github.com/houseofmates).