#!/usr/bin/env python3
"""
Memster MCP Server

Clean MCP wrapper around the hybrid retrieval engine.
Provides Model Context Protocol tools for memory storage and retrieval.

Usage:
  python memster_mcp_server.py                    # stdio transport (for Hermes Agent)
  DATABASE_URL=... python memster_mcp_server.py   # custom DB URL

Requires: PostgreSQL, memster.hybrid_retrieval
No Pieces MCP dependency. Zero SQLite legacy code.
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras

logger = logging.getLogger("memster.mcp")

# ── Database ──────────────────────────────────────────────────────

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://house:@/memster?host=/run/postgresql&port=5433",
)


def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    return conn


def init_schema():
    """Create all required tables if they don't exist."""
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS fuzzystrmatch")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id              SERIAL PRIMARY KEY,
                content         TEXT NOT NULL,
                network_type    TEXT NOT NULL DEFAULT 'observation',
                t_event         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                t_recorded      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source          TEXT,
                conversation_id TEXT,
                embedding       TEXT,
                category        TEXT DEFAULT 'observation',
                tier            TEXT DEFAULT 'L2',
                memory_type     TEXT,
                importance      REAL DEFAULT 0.5,
                decay_score     REAL DEFAULT 1.0,
                access_count    INTEGER DEFAULT 0,
                fronter_uid     TEXT,
                fronter_name    TEXT,
                valid_from      TEXT,
                valid_to        TEXT,
                event_time      TIMESTAMP,
                search_vector   TSVECTOR GENERATED ALWAYS AS (
                    to_tsvector('english', content)
                ) STORED,
                local_embedding JSONB,
                CONSTRAINT memories_network_type_check
                    CHECK (network_type IN ('world','experience','opinion','observation'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_network_type ON memories(network_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_t_event ON memories(t_event)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation ON memories(conversation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fronter ON memories(fronter_uid)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_validity ON memories(valid_from, valid_to)")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_search
            ON memories USING GIN (search_vector)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_edges (
                id                SERIAL PRIMARY KEY,
                source_memory_id  INTEGER REFERENCES memories(id) ON DELETE CASCADE,
                target_memory_id  INTEGER REFERENCES memories(id) ON DELETE CASCADE,
                edge_type         TEXT DEFAULT 'related',
                weight            REAL DEFAULT 1.0,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_entity_data (
                memory_id   INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                entities    JSONB,
                importance  REAL DEFAULT 0.5
            )
        """)
        conn.commit()
        logger.info("Schema initialised")
    except Exception as e:
        logger.error(f"Schema init failed: {e}")
        raise
    finally:
        conn.close()


# ── MCP Server ────────────────────────────────────────────────────

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, CallToolRequest
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from memster.hybrid_retrieval import (
    HybridRetrievalEngine,
    get_backend_info,
    embed_text,
    is_embedding_available,
    HYBRID_RETRIEVAL_TOOL_DEF,
)

_engine: Optional[HybridRetrievalEngine] = None


def get_engine() -> HybridRetrievalEngine:
    global _engine
    if _engine is None:
        _engine = HybridRetrievalEngine(get_conn)
    return _engine


# ── Tool definitions ──────────────────────────────────────────────

TOOLS = [
    Tool(
        name="memster_store",
        description="Store a new memory.",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Memory content"},
                "network_type": {
                    "type": "string",
                    "enum": ["world", "experience", "opinion", "observation"],
                    "default": "observation",
                },
                "source": {"type": "string", "default": "mcp"},
                "importance": {"type": "number", "default": 0.5},
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="memster_retrieve",
        description="Retrieve memories using hybrid search.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="memster_delete",
        description="Delete a memory by ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "integer"},
            },
            "required": ["memory_id"],
        },
    ),
    Tool(
        name="memster_search",
        description="Full-text search across memories.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="memster_count",
        description="Count total memories.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="memster_health",
        description="Health check with backend info.",
        inputSchema={"type": "object", "properties": {}},
    ),
]

# Add the hybrid retrieval tool from hybrid_retrieval.py
TOOLS.append(
    Tool(
        name=HYBRID_RETRIEVAL_TOOL_DEF["name"],
        description=HYBRID_RETRIEVAL_TOOL_DEF["description"],
        inputSchema=HYBRID_RETRIEVAL_TOOL_DEF["inputSchema"],
    )
)


# ── Tool handlers ─────────────────────────────────────────────────

async def handle_store(args: Dict[str, Any]) -> List[TextContent]:
    content = args["content"]
    network_type = args.get("network_type", "observation")
    source = args.get("source", "mcp")
    importance = float(args.get("importance", 0.5))

    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO memories (content, network_type, source, importance, t_event)
               VALUES (%s, %s, %s, %s, %s) RETURNING id""",
            (content, network_type, source, importance, datetime.now()),
        )
        row = cursor.fetchone()
        memory_id = row["id"]

        # Auto-embed
        engine = get_engine()
        engine.embed_and_store(memory_id, content)

        conn.commit()
        return [TextContent(
            type="text",
            text=json.dumps({"stored": True, "id": memory_id}, indent=2),
        )]
    finally:
        conn.close()


async def handle_retrieve(args: Dict[str, Any]) -> List[TextContent]:
    query = args["query"]
    top_k = int(args.get("top_k", 10))
    engine = get_engine()
    results = engine.retrieve(query, top_k=top_k)
    return [TextContent(
        type="text",
        text=json.dumps({
            "query": query,
            "count": len(results),
            "results": results,
        }, indent=2, default=str),
    )]


async def handle_delete(args: Dict[str, Any]) -> List[TextContent]:
    memory_id = int(args["memory_id"])
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        return [TextContent(
            type="text",
            text=json.dumps({"deleted": deleted, "id": memory_id}),
        )]
    finally:
        conn.close()


async def handle_search(args: Dict[str, Any]) -> List[TextContent]:
    query = args["query"]
    limit = int(args.get("limit", 20))
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """SELECT id, content, network_type, category, importance, t_event
               FROM memories
               WHERE search_vector @@ plainto_tsquery('english', %s)
                  OR content ILIKE %s
               ORDER BY importance DESC, t_event DESC
               LIMIT %s""",
            (query, f"%{query}%", limit),
        )
        results = [dict(r) for r in cursor.fetchall()]
        return [TextContent(
            type="text",
            text=json.dumps({"query": query, "count": len(results), "results": results}, indent=2, default=str),
        )]
    finally:
        conn.close()


async def handle_count() -> List[TextContent]:
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM memories")
        count = cursor.fetchone()["count"]
        return [TextContent(
            type="text",
            text=json.dumps({"count": count}),
        )]
    finally:
        conn.close()


async def handle_health() -> List[TextContent]:
    try:
        conn = get_conn()
        conn.close()
        db_ok = True
    except Exception as e:
        db_ok = False
        db_error = str(e)

    return [TextContent(
        type="text",
        text=json.dumps({
            "status": "healthy" if db_ok else "unhealthy",
            "database": "postgresql",
            "backend": get_backend_info(),
            "embeddings_available": is_embedding_available(),
        }, indent=2),
    )]


# ── Routing ───────────────────────────────────────────────────────

HANDLERS = {
    "memster_store": handle_store,
    "memster_retrieve": handle_retrieve,
    "memster_delete": handle_delete,
    "memster_search": handle_search,
    "memster_count": handle_count,
    "memster_health": handle_health,
    "memster_hybrid_retrieve": None,  # handled inline below
}


async def handle_request(req: CallToolRequest) -> List[TextContent]:
    name = req.params.name
    args = req.params.arguments or {}

    if name == "memster_hybrid_retrieve":
        engine = get_engine()
        from memster.hybrid_retrieval import handle_hybrid_retrieve
        result_json = await handle_hybrid_retrieve(args, engine)
        return [TextContent(type="text", text=result_json)]

    handler = HANDLERS.get(name)
    if handler is None:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    return await handler(args)


# ── Main ──────────────────────────────────────────────────────────

async def main():
    init_schema()

    if not MCP_AVAILABLE:
        print("mcp package not available. Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)

    server = Server("memster")
    server.list_tools = lambda: TOOLS
    server.call_tool = handle_request

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import asyncio
    asyncio.run(main())