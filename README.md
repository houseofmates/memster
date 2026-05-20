# Memster

**Memster** is a persistent long-term memory system for AI assistants, exposed as a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server. It stores, retrieves, searches, and manages structured memories in a SQLite database, giving AI agents durable memory that survives across sessions.

Designed for the AI assistant *Hermes*, Memster goes far beyond simple note storage. It combines semantic search with vector embeddings, a memory graph, spaced repetition scheduling, ephemeral wisp memories, passive activity capture, narrative arc tracking, memory palaces, and community detection — all through a unified MCP tool interface.

---

## Features

### Core Memory Operations

- **Structured storage** — Memories are classified by `network_type`: `world` (facts about the environment), `experience` (things that happened), `opinion` (evaluations and preferences), `observation` (general observations).
- **Full-text search** — SQLite FTS5 virtual table indexes all memory content for fast keyword queries.
- **Importance and decay scoring** — Every memory carries an `importance` weight (0–1) and a `decay_score` (1.0 = fresh, fades over time). Retrieval ranks by `importance × decay_score`.
- **Temporal validity** — Memories support `valid_from` and `valid_to` bounds. Expired memories are flagged and can be surfaced separately.
- **Auto-deduplication** — Before inserting, content is normalized (lowercase, stripped, punctuation removed) and compared against existing memories to prevent duplicates.
- **Completeness checking** — Incoming memories are analyzed for missing specifics (IPs, paths, time references, outcomes) and completeness warnings are returned to the caller.
- **Automatic tier promotion** — Memories are tiered L0 (most important) through L3. The consolidation system automatically promotes frequently accessed, high-importance memories toward L0.

### Semantic Search and Embeddings

- **NVIDIA NIM vector embeddings** — When the `nvidia_nim_embeddings` module is available, every memory is embedded using `nvidia/llama-3.2-nv-embedqa-1b-v2` (2048-dimensional vectors) for semantic similarity search.
- **Hybrid search** — Combines 50% vector similarity score, 30% FTS rank, and 20% importance weight into a single hybrid relevance score.
- **Backfill** — A `backfill_embeddings` tool generates missing embeddings for existing memories in configurable batches.
- **Graceful fallback** — All semantic tools fall back to LIKE-based text search when embeddings are unavailable.

### Memory Graph

- **Typed edges** — `memory_edges` connects memories with typed relations: `related`, `co_occurrence`, `causal`, `temporal`, `contradiction`, `supports`, `refines`, `supercedes`.
- **Auto-linking** — When a new memory is stored, `auto_link_memory` automatically creates edges based on entity co-occurrence, temporal proximity, contradiction detection, and refinement patterns.
- **Graph rebuild** — `build_graph_edges` rescans all memories and rebuilds the edge set from entity overlaps.
- **Related memory retrieval** — `get_related_memories` traverses the graph to surface connected memories ordered by edge weight.

### GBrain: Auto-linking and Community Detection

`memster_gbrain.py` implements a standalone graph intelligence module:

- **Composite edge weight formula**: `w = 0.25 × jaccard(content) + 0.30 × entity_overlap + 0.20 × temporal_decay + 0.25 × embedding_cosine_sim`
- **Batch graph rebuild** — `gbrain_rebuild_graph` scans memories within a configurable time window and creates edges for all strongly related pairs using a sliding-window day-pair approach.
- **Incremental linking** — `gbrain_auto_link_single` links a single new memory against the recent corpus without rebuilding the full graph.
- **Community detection** — `gbrain_get_communities` uses Louvain community detection (with label propagation fallback) on the `memory_edges` graph to identify clusters of related memories.
- **Arc suggestions** — `gbrain_suggest_arcs` identifies connected components with 5+ members and proposes narrative arcs from them.
- **Edge inspection** — `gbrain_edge_info` returns the exact edge weight and shared entity count between any two memories.

### Spaced Repetition (SM-2++)

`memster_spaced_repetition.py` implements an SM-2 variant with importance weighting:

- **SM-2++ algorithm** — Reviews are scheduled using Ebbinghaus-style easiness factors, with intervals that grow exponentially for well-retained memories.
- **Importance modifier** — Low-importance memories review more frequently; high-importance ones get longer intervals.
- **Review quality** — Integer or half-integer quality scores 0–5. Scores below 3 reset the interval and increment the lapse counter.
- **Review history** — Each memory stores up to 50 recent review events (date, quality, interval, easiness factor) in a JSON column.
- **Batch review** — `batch_review` processes multiple memories in a single transaction with rollback on failure.
- **Retention prediction** — `predict_retention` estimates the probability of retention at a given future date based on current interval and easiness.
- **Auto-scheduling** — `auto_schedule_important` finds high-importance memories with no review schedule and initializes them.

### Beads-Inspired Features

`memster_beads.py` brings concepts from the [beads](https://github.com/gastownhall/beads) project:

- **Content-hash IDs** — Each memory gets a SHA-256–derived short hash ID (e.g. `ws-a3f2c1`) in addition to its integer row ID.
- **Dependency DAG** — `memory_dependencies` table tracks typed dependencies between memories: `blocks`, `parent_child`, `discovered_from`, `related`, `waits_for`.
- **Ready queue** — `get_ready_memories` returns only memories with no unresolved blocking dependencies, modeling a task-queue pattern.
- **Wisp memories** — Ephemeral memories with a TTL (default 24h). Wisps auto-expire and can be:
  - `squash_wisp` — promoted to permanent L2 memories
  - `burn_wisp` — deleted immediately without trace
  - `gc_wisps` — garbage collected in batch (dry-run or actual deletion)
- **AI compaction** — `compact_memory_ai` summarizes verbose memories using NVIDIA NIM (with heuristic sentence-extraction fallback). Only compacts if the result is 30%+ smaller.
- **Gates** — `set_memory_gate` marks a memory as requiring confirmation (`confirm`), timed review (`timer`), or cross-reference verification (`verify`) before being surfaced. `resolve_gate` approves or rejects.
- **Append-only audit log** — Every mutation (insert, update, dependency, wisp lifecycle, gate events) is appended to `~/memster/audit.jsonl` as a tamper-evident JSONL entry.
- **Workspace fingerprinting** — `compute_workspace_fingerprint` generates a stable 16-character hex ID from hostname, username, and data directory path.

### V4 Feature Module

`memster_v4_features.py` provides 10+ advanced capabilities:

- **Bayesian confidence scoring** — `update_confidence(memory_id, observation_result)` updates a Bayesian confidence field using a 1.2× confirm / 0.6× contradict factor. The multi-signal scorer combines Bayesian confidence, importance, access frequency, content specificity, recency, and pinned status.
- **Contradiction detection** — `detect_contradictions` finds memory pairs with conflicting content using negation mismatch analysis, antonym detection, and conflicting value detection (IPs, version numbers).
- **Staleness detection** — `detect_stale_memories` identifies memories past their `valid_to` date and flags semantically stale content.
- **Context packet assembly** — `assemble_context_packet` builds a token-budgeted context block from FTS + importance-ranked memories, with diversity injection (spreading across network types).
- **Wiki integration** — Full read/write sync with a filesystem-based Markdown wiki (`~/memster/wiki/`):
  - `wiki_search` — full-text search across all `.md` files
  - `wiki_read` — read a wiki page by slug
  - `wiki_list` — enumerate all pages with optional category filter
  - `wiki_sweep` — audit for orphan pages, broken links, and untagged content
  - `wiki_to_memster_sync` — extract key facts from wiki pages and store as memories
- **Proactive surfacing** — `check_proactive` detects when current context intersects with past memories and suggests them unprompted.
- **Working memory context** — `get_working_memory_context` retrieves the most recent memories for a session as an active context window.
- **Retrieval routing** — `route_retrieval` analyzes a query and selects the optimal strategy: `keyword` (short queries), `semantic` (conceptual queries), `temporal` (date-references), `graph` (relational queries), or `hybrid`.
- **Follow-up intent detection** — `infer_follow_up_intent` detects pronoun-heavy queries that reference prior conversation.
- **Temporal bounds management** — `set_temporal_bounds_v4` updates `valid_from`, `valid_to`, and `observed_at` for any memory.
- **Graceful forgetting** — `graceful_forget_step` applies access-weighted exponential decay to old L2 memories. Dream cycle integration via `run_dream_cycle_v4` processes pattern discovery and community suggestions.
- **zlib compression** — `compress_memory_v4` / `decompress_memory_v4` use a `ZLIB:` prefixed hex encoding to compress verbose memories, with size ratio reporting.

### Session and Cross-Session Context

- **Session wrapping** — `session_wrap` stores an end-of-session summary, memory count, detected topic keywords, and fronter identity. If no notes are provided, an auto-digest is generated from the most recent memories.
- **Cross-session context** — `get_cross_session_context` finds past sessions and memories discussing a given topic, enabling an AI to recall what happened in previous conversations.
- **Unified briefing** — `unified_briefing` pulls the most relevant memories, cross-session context, and SP fronter status in a single call optimized for prompt injection.
- **Narrative briefing** — `get_narrative_briefing` groups memories by network type into context threads.

### Narrative Arcs and Conclusions

- **Narrative arcs** — `create_narrative_arc` / `add_memory_to_arc` tracks storylines across many memories. Each arc has a type (`manual`, `auto`, `derived`) and status (`ongoing`, `completed`, `abandoned`).
- **Arc timeline** — `get_arc_timeline` returns arc memories ordered by event time with their assigned arc roles (`beginning`, `event`, `development`, `climax`, `resolution`).
- **Conclusions** — `derive_conclusion` stores synthesized conclusions drawn from multiple source memories. `validate_conclusion` confirms or rejects them as evidence evolves.

### Memory Palace

Spatial memory organization modeled on the method of loci:

- **Palaces** — `create_palace` creates a named memory palace with a description.
- **Rooms** — `add_room` adds 3D-positioned rooms to a palace (`position_x`, `position_y`, `position_z`).
- **Placement** — `place_memory_in_room` associates a memory with a specific room.
- **Walking** — `walk_palace` returns the full spatial map: palace → rooms → memories in each room.

### Task Channels

- **Tasks** — `create_task` creates named project channels (with optional release version tagging).
- **Assignment** — `assign_memory_to_task` links memories to a task for scoped retrieval.
- **Completion** — `complete_task` marks a task done with a timestamp snapshot.
- **Active tasks** — `get_active_tasks` returns all non-completed tasks ordered by priority.

### Federation

- **Peer registration** — `register_peer` registers a remote Memster instance for sync (URL, auth token, sync direction: `bidirectional`, `publish`, `subscribe`).
- **Peer listing** — `list_peers` returns all registered peers with status and last-sync timestamps.

### Simply Plural Integration

For plural systems, Memster integrates with [Simply Plural](https://simplepluralapp.com/):

- **Front tracking** — Every memory automatically receives the current fronter's UID and name at insert time (`fronter_uid`, `fronter_name` columns).
- **SP status** — `sp_status` returns the current front roster from the SP API.
- **Headmate roster** — `sp_members` lists all headmates including archived ones.
- **Front history ingestion** — `ingest_sp_history` pulls SP front history and stores each fronting event as an `observation` memory.
- **Fronter-boosted retrieval** — The briefing system applies a +0.15 relevance boost to memories from the current fronter.

### Passive Activity Capture

- **Window events** — Stores app focus events with duration tracking.
- **Clipboard events** — Logs clipboard copy/paste events with source app.
- **Screenshot OCR** — Stores OCR text extracted from screenshots for full-text search.
- **Noise filtering** — `filter_passive_capture` identifies and removes noise captures (layout errors, dev tool output, vague content) using regex heuristics.
- **Activity summary** — `get_activity_summary` generates a human-readable summary of the last N hours of app usage and clipboard history.

### Pieces MCP Sync

- **Workstream ingestion** — `memster_sync_pieces` connects to a local [Pieces for Developers](https://pieces.app/) MCP server and ingests code snippets and workstream assets as observation memories.

### Dream Consolidation

`dream_consolidation.py` is an offline consolidation script designed to run as a systemd timer:

- **Access-weighted decay** — Applies `decay_rate / (1 + log(access_count + 1))` decay to all L2 memories older than 7 days. Frequently accessed memories decay far more slowly.
- **Hot memory promotion** — Memories accessed more than 10 times are promoted to L0 (highest tier).
- **Decay floor** — Decay score never falls below 0.05, so no memory completely disappears from consolidation alone.

---

## Architecture

```
memster_mcp_server.py      (main — MCP server, all tool handlers)
├── memster_v4_features.py  (V4: contradiction detection, wiki sync, context assembly, etc.)
├── memster_beads.py        (beads: wisps, dependencies, gates, audit log, fingerprinting)
├── memster_spaced_repetition.py  (SM-2++ review scheduling)
├── memster_phase2.py       (stub: trust scoring / typed schema placeholders)
├── memster_gbrain.py       (graph auto-linking, community detection — standalone)
└── dream_consolidation.py  (offline decay / promotion — runs as systemd timer)
```

### Database Schema

All data is stored in a single SQLite database (default: `~/memster/memster_unified.db`, overridable via `MEMSTER_DB_PATH`).

| Table | Purpose |
|---|---|
| `memories` | Central memory store with all scoring columns |
| `memory_edges` | Graph edges between memories with typed relations and weights |
| `memory_embeddings` | NVIDIA NIM vector embeddings (separate from main table) |
| `entities` | Named entity registry (canonical names, types) |
| `memory_entities` | Memory-to-entity join table with confidence scores |
| `sessions` | Cross-session summaries with topic extraction |
| `window_events` | Passive app focus capture |
| `clipboard_events` | Clipboard history |
| `screenshot_events` | OCR text from screenshots |
| `memory_dependencies` | Beads dependency DAG |
| `tasks` / `task_memory_assignments` | Task channel organization |
| `narrative_arcs` / `arc_memories` | Story arc tracking |
| `memory_palaces` / `palace_rooms` / `room_memories` | Spatial organization |
| `conclusions` | Synthesized conclusions with confidence |
| `federation_peers` | Remote sync peer registry |
| `memories_fts` | FTS5 virtual table (auto-synced via triggers) |

### Startup Sequence

1. `init_database()` — Creates all tables and FTS triggers, runs V4 schema upgrades, initializes beads dependency tables and spaced repetition columns.
2. `asyncio.run(main())` — Launches the MCP stdio server.
3. All tool calls are dispatched through the `call_tool()` async function registered with the MCP server.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MEMSTER_DB_PATH` | `~/memster/memster_unified.db` | Override database path |
| `PIECES_MCP_URL` | `http://localhost:39310/model_context_protocol/2025-03-26/mcp` | Pieces MCP server URL |

---

## Installation

### Requirements

- Python 3.10+
- `mcp` — Model Context Protocol SDK
- `networkx` — Graph analytics (for gbrain)
- `numpy` — Numerical operations
- `scikit-learn` — Cosine similarity (for gbrain)

**Optional:**
- `nvidia_nim_embeddings` — NVIDIA NIM vector embedding integration
- `simply_plural_api` — Simply Plural front tracking
- `python-louvain` (`community`) — Louvain community detection (falls back to label propagation)

```bash
pip install mcp networkx numpy scikit-learn
```

### Running

```bash
python memster_mcp_server.py
```

The server communicates over stdio and is designed to be launched by an MCP-compatible client (Claude Desktop, Kilo, etc.).

### MCP Client Configuration

```json
{
  "mcpServers": {
    "memster": {
      "command": "python",
      "args": ["/path/to/memster_mcp_server.py"],
      "env": {
        "MEMSTER_DB_PATH": "/path/to/custom.db"
      }
    }
  }
}
```

### Dream Cycle (Systemd Timer)

```ini
# /etc/systemd/system/memster-dream.service
[Unit]
Description=Memster memory consolidation

[Service]
ExecStart=/usr/bin/python3 /path/to/dream_consolidation.py
User=%i

# /etc/systemd/system/memster-dream.timer
[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

---

## Tool Reference

### Activity & Passive Capture

| Tool | Description |
|---|---|
| `get_activity_summary` | Human-readable summary of recent app usage and clipboard |
| `get_recent_activity` | Structured activity data with app-time breakdown |
| `search_window_history` | Search app/window history by title |
| `search_clipboard` | Search clipboard history |
| `search_ocr_text` | Search OCR text from screenshots |
| `filter_passive_capture` | Clean up noise from passive capture data |

### Core Memory CRUD

| Tool | Description |
|---|---|
| `memster_remember` | Store a memory with completeness check, dedup, temporal detection, fronter tagging |
| `memster_query` | Full-text search with network type, date range, and limit filters |
| `remember_memory` | Simplified memory storage with auto-dedup and entity extraction |
| `query_memories` | Retrieve memories filtered by text, category, and tier |
| `update_memory` | Update content, category, or tier of an existing memory |
| `merge_memories` | Merge two memories, re-linking all graph edges to the survivor |
| `delete_memory` | Delete a memory and all its graph edges |
| `delete_by_query` | Delete all memories matching a text query |
| `remember_batch` | Store multiple memories in a single call with deduplication |
| `find_duplicates` | Find near-duplicate memory groups by content similarity |

### Semantic Search

| Tool | Description |
|---|---|
| `semantic_memory_search` | Vector similarity search using NVIDIA NIM embeddings |
| `hybrid_search` | Hybrid ranking: 50% vector + 30% FTS + 20% importance |
| `find_similar` | Find memories similar to a given memory ID |
| `backfill_embeddings` | Generate embeddings for memories that lack them |

### Memory Graph

| Tool | Description |
|---|---|
| `link_memories` | Create a typed edge between two memories |
| `get_related_memories` | Retrieve graph-connected memories ordered by edge weight |
| `extract_graph_relations` | Extract semantic relations from memory content |
| `build_graph_edges` | Rebuild all graph edges from entity overlaps |
| `query_by_entity` | Find memories referencing a named entity |
| `enrich_memory_lookup` | Extract entities from content and find related memories |

### Session & Briefing

| Tool | Description |
|---|---|
| `get_briefing` | SP-weighted session briefing of top memories |
| `get_narrative_briefing` | Narrative context threads grouped by network type |
| `unified_briefing` | Memster + SP + cross-session context in one call |
| `session_wrap` | End-of-session summary with auto-digest |
| `get_cross_session_context` | Find past sessions and memories on a topic |
| `memory_timeline` | Chronological memory timeline grouped by day |
| `check_proactive` | Detect relevant past memories for current context |
| `assemble_context_packet` | Token-budgeted context block for prompt injection |

### Maintenance & Health

| Tool | Description |
|---|---|
| `get_memory_health` | Database statistics: counts, average importance/decay |
| `sleep_consolidate` | Run access-weighted decay and tier promotion cycle |
| `detect_stale_memories` | Find memories past `valid_to` or semantically outdated |
| `detect_contradictions` | Find contradictory memory pairs |
| `check_memory_quality` | Quality score with issues and suggestions |
| `check_memory_completeness` | Detailed completeness analysis for a memory |
| `find_expired_memories` | Find memories past their `valid_to` date |

### Spaced Repetition

| Tool | Description |
|---|---|
| `schedule_review` | Schedule the next review using SM-2++ |
| `get_due_reviews` | Get memories due for review within N days |
| `batch_review` | Process multiple reviews in one transactional call |
| `predict_retention` | Estimate retention probability at a future date |
| `auto_schedule_important` | Initialize review schedules for high-importance memories |

### Beads Features

| Tool | Description |
|---|---|
| `create_wisp` | Create an ephemeral memory with TTL |
| `squash_wisp` | Promote a wisp to permanent memory |
| `burn_wisp` | Delete a wisp without trace |
| `gc_wisps` | Garbage collect expired wisps |
| `compact_memory_ai` | AI-compress a memory (30%+ savings threshold) |
| `add_dependency` | Add a typed dependency between two memories |
| `get_ready_memories` | Get unblocked memories with no open dependencies |
| `set_memory_gate` | Set a confirmation/timer/verify gate |
| `resolve_gate` | Approve or reject a pending gate |
| `audit_log` | Append a manual audit event |
| `query_audit_log` | Query the append-only audit log |
| `compute_workspace_fingerprint` | Get the stable workspace identity hash |

### Temporal

| Tool | Description |
|---|---|
| `set_temporal_bounds` | Set `valid_from`, `valid_to`, `observed_at` for a memory |
| `update_temporal_bounds` | Same as above (alias) |
| `update_confidence` | Bayesian confidence update based on observation |
| `score_memory_confidence` | Multi-signal confidence score breakdown |
| `track_provenance` | Update source and extraction timestamp |
| `record_retrieval_telemetry` | Log a retrieval event (increments access count) |

### Memory Palace

| Tool | Description |
|---|---|
| `create_palace` | Create a named memory palace |
| `add_room` | Add a 3D-positioned room to a palace |
| `place_memory_in_room` | Associate a memory with a room |
| `walk_palace` | Traverse a palace: rooms and their memories |

### Narrative Arcs

| Tool | Description |
|---|---|
| `create_narrative_arc` | Create a narrative arc (manual, auto, or derived) |
| `add_memory_to_arc` | Add a memory to an arc with a role |
| `get_arc_timeline` | Get the ordered timeline of an arc |
| `list_arcs` | List arcs with optional status and type filters |
| `complete_arc` | Mark an arc as completed |

### Conclusions

| Tool | Description |
|---|---|
| `derive_conclusion` | Store a synthesized conclusion from source memories |
| `validate_conclusion` | Confirm or reject a pending conclusion |
| `get_conclusions` | List conclusions with optional status filter |

### Task Channels

| Tool | Description |
|---|---|
| `create_task` | Create a named task channel |
| `complete_task` | Mark a task as completed |
| `assign_memory_to_task` | Associate a memory with a task |
| `get_task_memories` | Get all memories in a task channel |
| `get_active_tasks` | List all non-completed tasks by priority |

### Feedback

| Tool | Description |
|---|---|
| `rate_memory` | Submit feedback (helpful, irrelevant, wrong, outdated, promote, demote) |
| `get_memory_feedback_stats` | Retrieve importance/decay/access stats for a memory |

### Federation

| Tool | Description |
|---|---|
| `register_peer` | Register a remote Memster peer |
| `list_peers` | List all registered peers |
| `sync_with_peer` | Initiate a sync with a federation peer |

### Simply Plural

| Tool | Description |
|---|---|
| `sp_status` | Current front roster |
| `sp_members` | All headmates |
| `ingest_sp_history` | Store SP front history as memories |

### Wiki

| Tool | Description |
|---|---|
| `wiki_search` | Full-text search across wiki pages |
| `wiki_read` | Read a wiki page by slug |
| `wiki_list` | List all wiki pages |
| `wiki_sweep` | Audit wiki for orphans and broken links |
| `wiki_to_memster_sync` | Extract facts from wiki into memories |

### Pieces Integration

| Tool | Description |
|---|---|
| `memster_sync_pieces` | Ingest workstream snippets from Pieces MCP |

### Retrieval Intelligence

| Tool | Description |
|---|---|
| `route_retrieval` | Route a query to the optimal retrieval strategy |
| `infer_follow_up_intent` | Detect if a query is a follow-up to prior context |
| `get_working_memory_context` | Get recent memories as active working context |

---

## Module Summary

| File | Lines | Purpose |
|---|---|---|
| `memster_mcp_server.py` | ~3,850 | MCP server, all tool handlers, core helpers |
| `memster_v4_features.py` | 1,935 | Advanced V4 features (confidence, wiki, context, etc.) |
| `memster_beads.py` | 583 | Beads-inspired features (wisps, deps, gates, audit) |
| `memster_spaced_repetition.py` | 188 | SM-2++ review scheduling |
| `memster_gbrain.py` | 575 | Graph auto-linking and community detection |
| `memster_phase2.py` | 14 | Phase 2 feature stubs (trust scoring, typed schemas) |
| `dream_consolidation.py` | 82 | Offline consolidation daemon |
