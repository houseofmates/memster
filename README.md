<h1 align="center">memster</h1>

a local-first, long-term memory system for hermes agent. it provides persistent memory across sessions, enabling the agent to learn from experience, recall past conversations, and build a deepening model of the user over time. built with postgresql and designed for durability, memster is the core of hermes' self-improving capabilities.

<h2 align="center">features</h2>

- **persistent memory** — stores memories in a postgresql database with full-text search and vector embeddings for semantic recall
- **four-network model** — organizes memories into world, experience, opinion, and observation networks for structured recall
- **mcp integration** — exposes memory functions via model context protocol for seamless integration with hermes and other agents
- **dream system** — database-integrated dream system that processes memories during idle periods to reinforce learning
- **semantic enhancements** — includes 9 semantic intelligence improvements for better understanding and association
- **self-evolution capabilities** — uses memster's own data to evaluate and improve its memory processes
- **durability ops** — operational procedures for making memster amnesia-proof with backups, repair, and consistency checks
- **activity tracking** — complete local-first activity tracking and memory system for logging user interactions
- **memory provider integration** — can be used as a memory provider in hermes agent for enhanced context

<h2 align="center">installation</h2>

memster is designed to run alongside hermes agent. it requires postgresql and python 3.11+.

<h3 align="center">prerequisites</h3>

- postgresql 13+ (or use docker)
- python 3.11 or higher
- git

<h3 align="center">setup</h3>

1. clone the repository
   ```bash
   git clone https://github.com/houseofmates/memster.git
   cd memster
   ```

2. install dependencies
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .[all]
   ```

3. configure the database
   - ensure postgresql is running
   - create a database and user for memster
   - copy `.env.example` to `.env` and fill in the connection details
   - run the schema migrations
     ```bash
     alembic upgrade head
     ```

4. start the memster mcp server
   ```bash
   python memster_mcp_server.py
   ```
   the server will run on http://localhost:8000 by default

<h2 align="center">usage</h2>

once the memster server is running, hermes agent can connect to it via the mcp integration. configure hermes to use memster as a memory provider in `config.yaml`:

```yaml
mcp_servers:
  memster:
    url: http://localhost:8000
```

memster provides the following tools through mcp:
- `memster_curate` — store a new memory
- `memster_remember` — retrieve memories by content
- `memster_query` — search memories with full-text search
- `memster_status` — get system status and memory count
- and more for dream system, activity tracking, etc.

<h2 align="center">architecture</h2>

memster consists of several core components:

- **memster_mcp_server.py** — the main mcp server that exposes memory functions
- **memster_beads.py** — defines the memory bead structure and networks
- **memster_gbrain.py** — the global brain that coordinates memory operations
- **memster_spaced_repetition.py** — implements spaced repetition for memory retention
- **dream_consolidation.py** — processes memories during dream cycles to reinforce learning
- **memster_v4_features.py** — contains the 9 semantic intelligence improvements
- **memster_phase2.py** — phase 2 enhancements for the memory system

the system uses a postgresql database with the following tables:
- `memories` — stores individual memory beads
- `memory_networks` — defines the four networks (world, experience, opinion, observation)
- `dream_cycles` — tracks dream system activity
- `activity_log` — logs user interactions for activity tracking

<h2 align="center">contributing</h2>

we welcome contributions! please read [contributing.md](contributing.md) for details on our code of conduct and the process for submitting pull requests.

<h2 align="center">license</h2>

[the mates license](license)

built by [house of mates](https://github.com/houseofmates).
