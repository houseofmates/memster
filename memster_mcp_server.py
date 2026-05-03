#!/usr/bin/env python3
"""
Memster MCP Server v3 Combined - Complete long-term memory system with Pieces integration.
Combines working base with all 47 tools from backup.
"""

import json
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib import request, error

# NVIDIA NIM Embeddings (frontier-quality 2048d vectors)
EMBEDDINGS_AVAILABLE = False

# Beads-inspired features (from github.com/gastownhall/beads)
BEADS_AVAILABLE = False
try:
    from memster_beads import *
    BEADS_AVAILABLE = True
except ImportError as e:
    BEADS_AVAILABLE = False
try:
    from nvidia_nim_embeddings import (
        embed_text, embed_batch, store_embedding, get_embedding,
        vector_search, backfill_embeddings, auto_embed_on_insert,
        cached_query_embed, cosine_similarity, is_available as embeddings_available
    )
    EMBEDDINGS_AVAILABLE = embeddings_available()
    if EMBEDDINGS_AVAILABLE:
        pass  # logger not yet available; will log after setup
except ImportError as e:
    EMBEDDINGS_AVAILABLE = False

# Configuration
DATABASE_PATH = os.path.expanduser("~/memster/memster_unified.db")
DB_PATH = DATABASE_PATH
MEMSTER_DIR = os.path.expanduser("~/memster")
PIECES_MCP_URL = "http://localhost:39310/model_context_protocol/2025-03-26/mcp"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("memster")

# MCP import with graceful fallback
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("warning: mcp package not available", file=sys.stderr)

# aiohttp import with graceful fallback
try:
    import aiohttp
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


# V4 feature module
try:
    from memster_v4_features import *
    V4_AVAILABLE = True
except ImportError:
    V4_AVAILABLE = False
    print("warning: v4 features not available", file=sys.stderr)
# Beads feature module
try:
    from memster_beads import *
    BEADS_AVAILABLE = True
except ImportError:
    BEADS_AVAILABLE = False
    print("warning: beads features not available", file=sys.stderr)


def init_database() -> None:
    if BEADS_AVAILABLE:
        beads_init = init_all_beads_features(DATABASE_PATH)
        logger.info(f"beads features: {beads_init}")
    """Initialize the SQLite database with required tables and indexes."""
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Main memories table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        network_type TEXT NOT NULL CHECK(network_type IN ('world', 'experience', 'opinion', 'observation')),
        t_event TIMESTAMP NOT NULL,
        t_recorded TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        source TEXT,
        conversation_id TEXT,
        embedding TEXT,
        category TEXT DEFAULT 'observation',
        tier TEXT DEFAULT 'L2',
        memory_type TEXT,
        importance REAL DEFAULT 0.5,
        decay_score REAL DEFAULT 1.0,
        access_count INTEGER DEFAULT 0,
        fronter_uid TEXT,
        fronter_name TEXT,
        valid_from TEXT,
        valid_to TEXT
    )
    """)

    # === MIGRATION: add new columns if they don't exist ===
    # These columns were added in a later version - add safely to existing DBs
    for col_def in [
        ("fronter_uid", "TEXT"),
        ("fronter_name", "TEXT"),
        ("valid_from", "TEXT"),
        ("valid_to", "TEXT"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE memories ADD COLUMN {col_def[0]} {col_def[1]}")
            logger.info(f"Added column {col_def[0]} to memories table")
        except sqlite3.OperationalError:
            pass  # column already exists

    # Indexes for common queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_network_type ON memories(network_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_t_event ON memories(t_event)")
    # idx_t_recorded skipped - column may not exist in older DB
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation ON memories(conversation_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fronter ON memories(fronter_uid)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_validity ON memories(valid_from, valid_to)")

    # FTS5 virtual table for full-text search
    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        content,
        network_type UNINDEXED,
        t_event UNINDEXED,
        source UNINDEXED,
        conversation_id UNINDEXED,
        content='memories',
        content_rowid='id'
    )
    """)

    # Triggers to keep FTS index in sync
    cursor.execute("""
    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, content, network_type, t_event, source, conversation_id)
        VALUES (new.id, new.content, new.network_type, new.t_event, new.source, new.conversation_id);
    END
    """)

    cursor.execute("""
    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, content, network_type, t_event, source, conversation_id)
        VALUES ('delete', old.id, old.content, old.network_type, old.t_event, old.source, old.conversation_id);
    END
    """)

    cursor.execute("""
    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, content, network_type, t_event, source, conversation_id)
        VALUES ('delete', old.id, old.content, old.network_type, old.t_event, old.source, old.conversation_id);
        INSERT INTO memories_fts(rowid, content, network_type, t_event, source, conversation_id)
        VALUES (new.id, new.content, new.network_type, new.t_event, new.source, new.conversation_id);
    END
    """)

    # Additional tables for advanced features
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memory_edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_memory_id INTEGER NOT NULL,
        target_memory_id INTEGER NOT NULL,
        relation_type TEXT DEFAULT 'related',
        weight REAL DEFAULT 0.5,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
        FOREIGN KEY (target_memory_id) REFERENCES memories(id) ON DELETE CASCADE
    )""")
    # idx_mem_edges_source skipped - column may not exist (old DB uses source_id not source_memory_id)
    # idx_mem_edges_target skipped - column may not exist (old DB uses target_id not target_memory_id)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memory_embeddings (
        memory_id INTEGER PRIMARY KEY,
        embedding TEXT,
        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        canonical_name TEXT NOT NULL,
        entity_type TEXT,
        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memory_entities (
        memory_id INTEGER NOT NULL,
        entity_id INTEGER NOT NULL,
        confidence REAL DEFAULT 1.0,
        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
        FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memster_links (
        wiki_slug TEXT NOT NULL,
        memory_id INTEGER NOT NULL,
        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
    )""")

    # Sessions table for cross-session context tracking
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT UNIQUE NOT NULL,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ended_at TIMESTAMP,
        summary TEXT,
        primary_fronter_uid TEXT,
        primary_fronter_name TEXT,
        memory_count INTEGER DEFAULT 0,
        topics TEXT,
        FOREIGN KEY (primary_fronter_uid) REFERENCES sp_members(uid)
    )""")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON sessions(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_started ON sessions(started_at)")

    # Activity tracking tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS window_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        window_class TEXT,
        window_title TEXT,
        timestamp TIMESTAMP,
        duration_seconds INTEGER DEFAULT 0
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS clipboard_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content_preview TEXT,
        source_app TEXT,
        timestamp TIMESTAMP
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS screenshot_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT,
        timestamp TIMESTAMP,
        ocr_text TEXT
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS screenshot_hashes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT,
        hash_value TEXT,
        timestamp TIMESTAMP
    )""")

    conn.commit()
    conn.close()

    # V4 schema upgrades (new columns + tables)
    if V4_AVAILABLE:
        try:
            upgrade_result = upgrade_database_v4(DATABASE_PATH)
            logger.info(f"V4 schema upgrade: {upgrade_result}")
        except Exception as e:
            logger.warning(f"V4 schema upgrade failed: {e}")

    # Beads schema integration
    if BEADS_AVAILABLE:
        try:
            init_dependency_tables(DATABASE_PATH)
            logger.info("Beads schema initialized (memory_dependencies, new memory columns)")
        except Exception as e:
            logger.warning(f"Beads schema init failed: {e}")

    logger.info(f"Database initialized at {DATABASE_PATH}")


def get_db_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# === normalization helpers ===

def normalize_text(text):
    """Normalize text for comparison: lowercase, strip, collapse whitespace, remove punctuation."""
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def word_overlap(a, b):
    """Jaccard similarity on word sets."""
    words_a = set(normalize_text(a).split())
    words_b = set(normalize_text(b).split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


# === passive capture filter ===

NOISE_PATTERNS = [
    r'^copied:\s*\n?(Layout was forced|ElementHandle|Timeout \d+ms|npm error)',
    r'^screen showed:.*File Edit Selection View',
    r'^copied:\s*\n?\[vite\]',
    r'^copied:\s*\n?(are there any improvements|what do you think)',
]

def is_noise_capture(content):
    """Check if passive capture content is noise."""
    if not content or len(content.strip()) < 10:
        return True
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
            return True
    printable = sum(1 for c in content if c.isprintable() or c in '\n\t')
    if len(content) > 0 and printable / len(content) < 0.7:
        return True
    return False


def extract_meaning_from_capture(content, source_type):
    """Try to extract structured meaning from raw passive capture."""
    if is_noise_capture(content):
        return None
    if content.startswith("copied:"):
        content = content[7:].strip()
    if content.startswith("screen showed:"):
        content = content[14:].strip()
    error_match = re.search(r'(error|Error|ERROR)[:\s]+(.+?)(?:\n|$)', content)
    if error_match:
        return f"error encountered: {error_match.group(2).strip()[:100]}"
    path_match = re.search(r'(/[\w/.-]+\.\w{1,5})', content)
    if path_match:
        return f"file referenced: {path_match.group(1)}"
    if len(content) < 200 and len(content.split()) > 3:
        return content.strip()
    return None


# === duplicate detection ===

def check_duplicate_before_insert(content, category):
    """Check if a memory with similar content already exists."""
    conn = get_db_connection()
    cursor = conn.cursor()
    normalized_new = normalize_text(content)
    cursor.execute("SELECT id, content FROM memories WHERE category = ?", (category or "general",))
    for row in cursor.fetchall():
        if normalize_text(row['content']) == normalized_new:
            conn.close()
            return row['id']
    cursor.execute("SELECT id, content FROM memories")
    for row in cursor.fetchall():
        if normalize_text(row['content']) == normalized_new:
            conn.close()
            return row['id']
    conn.close()
    return None


def find_near_duplicate_memories(similarity_threshold=0.8):
    """Find groups of duplicate or near-duplicate memories."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, content, category, tier FROM memories ORDER BY id")
    memories = [dict(r) for r in cursor.fetchall()]
    conn.close()

    visited = set()
    groups = []
    for i, mem_a in enumerate(memories):
        if mem_a['id'] in visited:
            continue
        group = [mem_a]
        for j, mem_b in enumerate(memories):
            if i == j or mem_b['id'] in visited:
                continue
            if normalize_text(mem_a['content']) == normalize_text(mem_b['content']):
                group.append(mem_b)
            elif (mem_a.get('category') == mem_b.get('category') and 
                  word_overlap(mem_a['content'], mem_b['content']) >= similarity_threshold):
                group.append(mem_b)
        if len(group) > 1:
            for m in group:
                visited.add(m['id'])
            groups.append(group)
    return groups


# === retrieval helpers ===

def get_memories_with_scoring(query_text=None, tier=None, category=None, max_results=10):
    """Retrieve memories with importance/decay scoring for ordering."""
    conn = get_db_connection()
    cursor = conn.cursor()

    where, params = [], []
    if tier:
        where.append("tier = ?")
        params.append(tier)
    if category:
        where.append("category = ?")
        params.append(category)
    if query_text:
        where.append("content LIKE ?")
        params.append("%" + query_text + "%")

    where_clause = " AND ".join(where) if where else "1=1"

    sql = f"""
    SELECT id, content, category, tier, importance, decay_score, access_count, 
        COALESCE(importance, 0.5) * COALESCE(decay_score, 1.0) as relevance_score
    FROM memories 
    WHERE {where_clause}
    ORDER BY relevance_score DESC, access_count DESC
    LIMIT ?
    """
    params.append(max_results)
    cursor.execute(sql, params)
    results = [dict(r) for r in cursor.fetchall()]
    for m in results:
        cursor.execute("UPDATE memories SET access_count = access_count + 1 WHERE id = ?", (m['id'],))
    conn.commit()
    conn.close()
    return results


# === activity helpers ===

def get_recent_activity(hours=1, limit=50):
    """Get recent activity data."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    cursor.execute("SELECT window_class, window_title, timestamp, duration_seconds FROM window_events WHERE timestamp > ? ORDER BY timestamp DESC LIMIT ?", 
                   (cutoff, limit))
    windows = [dict(r) for r in cursor.fetchall()]
    cursor.execute("""SELECT window_class, SUM(duration_seconds) as total_seconds, COUNT(*) as event_count 
                     FROM window_events WHERE timestamp > ? GROUP BY window_class ORDER BY total_seconds DESC""", (cutoff,))
    apps = [{"app": r["window_class"], "minutes": (r["total_seconds"] or 0) // 60} for r in cursor.fetchall()]
    cursor.execute("SELECT content_preview, source_app, timestamp FROM clipboard_events WHERE timestamp > ? ORDER BY timestamp DESC LIMIT ?", 
                   (cutoff, 20))
    clipboard = [dict(r) for r in cursor.fetchall()]
    cursor.execute("SELECT COUNT(*) as count FROM screenshot_events WHERE timestamp > ?", (cutoff,))
    screenshots = cursor.fetchone()["count"]
    conn.close()
    return {"hours": hours, "window_events": len(windows), "apps": apps[:10], "clipboard": clipboard[:5], "screenshots": screenshots}


def search_window_history(query, hours=24, limit=20):
    """Search window history."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    cursor.execute("""SELECT window_class, window_title, timestamp, duration_seconds 
                     FROM window_events WHERE timestamp > ? AND window_title LIKE ? 
                     ORDER BY timestamp DESC LIMIT ?""", (cutoff, "%" + query + "%", limit))
    results = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return {"query": query, "matches": len(results), "results": results}


def search_clipboard(query, hours=24, limit=10):
    """Search clipboard history."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    cursor.execute("""SELECT content_preview, source_app, timestamp FROM clipboard_events 
                     WHERE timestamp > ? AND content_preview LIKE ? ORDER BY timestamp DESC LIMIT ?""", 
                   (cutoff, "%" + query + "%", limit))
    results = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return {"query": query, "matches": len(results), "results": results}


def get_activity_summary_text(hours=2):
    """Get human-readable activity summary."""
    activity = get_recent_activity(hours=hours)
    lines = ["activity summary for last " + str(hours) + " hour(s):"]
    if activity["apps"]:
        lines.append("\napp usage:")
        for app in activity["apps"][:5]:
            lines.append(" - " + app.get("app", "unknown") + ": " + str(app.get("minutes", 0)) + " min")
    else:
        lines.append("\nno app activity recorded.")
    if activity["clipboard"]:
        lines.append("\nrecent clipboard:")
        for item in activity["clipboard"][:3]:
            preview = (item.get("content_preview") or "")[:50]
            lines.append(" - [" + item.get("source_app", "unknown") + "] " + preview + "...")
    lines.append("\nscreenshots: " + str(activity["screenshots"]))
    return "\n".join(lines)


def search_ocr_text(query, limit=10):
    """Search OCR text from screenshots."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, image_path, timestamp, ocr_text FROM screenshot_events WHERE ocr_text LIKE ? ORDER BY timestamp DESC LIMIT ?", 
                   ("%" + query + "%", limit))
    results = [{"id": r[0], "image_path": r[1], "timestamp": r[2], "ocr_preview": (r[3] or "")[:200]} for r in cursor.fetchall()]
    conn.close()
    return {"query": query, "matches": len(results), "results": results}


# === enhanced memory quality and completeness ===

# Patterns for detecting temporal references in content
TEMPORAL_PATTERNS = {
    "valid_from": [
        r"since\s+(\d{4}-\d{2}-\d{2}|\w+\s+\d+|\d+\s+\w+)",
        r"from\s+(\d{4}-\d{2}-\d{2}|\w+\s+\d+|\d+\s+\w+)",
        r"starting\s+(\w+\s+\d+)",
        r"as of\s+(\w+\s+\d+)",
        r"currently\s+running",
        r"active\s+since",
    ],
    "valid_to": [
        r"until\s+(\d{4}-\d{2}-\d{2}|\w+\s+\d+|\d+\s+\w+)",
        r"expires?\s+(\w+\s+\d+)",
        r"deprecated\s+since",
        r"no longer\s+(running|active|used)",
    ],
    "current": [
        r"right now",
        r"currently",
        r"at the moment",
        r"as of (today|now)",
    ],
    "past": [
        r"yesterday",
        r"last (week|month|year|night)",
        r"ago",
        r"previously",
        r"before",
    ],
}

# Patterns for detecting missing completeness
INCOMPLETENESS_PATTERNS = {
    "world": {
        "missing_time": [r"service (was|is)\s+down", r"it (works|broke)", r"error", r"issue"],
        "missing_specifics": [r"server", r"service", r"process", r"container"],
    },
    "experience": {
        "missing_outcome": [r"tried", r"worked on", r"started", r"found"],
        "missing_detail": [r"error", r"fix", r"solved", r"broke"],
    },
}

# Entity patterns that should be in memories
ENTITY_PATTERNS = {
    "ip": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
    "path": r"/[\w/.-]+\.\w{1,10}",
    "port": r":\d{2,5}",
    "url": r"https?://[^\s]+",
    "package": r"\w+-\w+(?:-\w+)*",
    "service": r"\b(?:server|container|docker|service|daemon|service)\b",
}


def detect_temporal_bounds(content: str) -> Dict[str, Optional[str]]:
    """Extract temporal validity bounds from content."""
    bounds = {"valid_from": None, "valid_to": None, "is_current": False}

    # Check for "current" patterns - implies no valid_to
    for pattern in TEMPORAL_PATTERNS.get("current", []):
        if re.search(pattern, content, re.IGNORECASE):
            bounds["is_current"] = True
            break

    # Check for past patterns - implies no valid_from (historical)
    for pattern in TEMPORAL_PATTERNS.get("past", []):
        if re.search(pattern, content, re.IGNORECASE):
            # If it's about the past, valid_from might be set by context
            break

    # Try to extract explicit valid_from
    for pattern in TEMPORAL_PATTERNS.get("valid_from", []):
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            bounds["valid_from"] = match.group(1) if match.groups() else "detected"
            break

    # Try to extract explicit valid_to
    for pattern in TEMPORAL_PATTERNS.get("valid_to", []):
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            bounds["valid_to"] = match.group(1) if match.groups() else "detected"
            break

    return bounds


def check_memory_completeness(content: str, network_type: str = "observation") -> Dict[str, Any]:
    """
    Check if a memory is complete enough to be useful.
    Returns dict with:
      - is_complete: bool
      - score: int (0-5)
      - issues: list of missing elements
      - suggestions: list of what to add
    """
    score = 0
    issues = []
    suggestions = []

    content_lower = content.lower()
    word_count = len(content.split())
    char_count = len(content)

    # 1. Basic length check
    if char_count < 10:
        issues.append("content_too_short")
        suggestions.append("add more detail (minimum ~30 chars)")
        score -= 2
    elif char_count < 50:
        issues.append("content_short")
        score -= 1
    else:
        score += 1

    # 2. Specific data detection (IPs, paths, URLs, ports)
    found_entities = []
    for etype, pattern in ENTITY_PATTERNS.items():
        if re.search(pattern, content):
            found_entities.append(etype)

    if not found_entities:
        if network_type == "world":
            issues.append("missing_specifics")
            suggestions.append("add specific identifiers (IPs, paths, ports, service names)")
        elif network_type == "experience":
            issues.append("missing_specifics")
            suggestions.append("add specific details (file paths, error messages, commands)")
    else:
        score += min(len(found_entities), 2)

    # 3. Action/outcome detection for experiences
    if network_type == "experience":
        has_outcome = any(kw in content_lower for kw in ["fixed", "resolved", "solved", "worked", "success", "done", "completed"])
        has_error = any(kw in content_lower for kw in ["error", "failed", "broke", "crash", "issue"])
        has_action = any(kw in content_lower for kw in ["tried", "ran", "executed", "built", "created", "modified"])

        if has_outcome:
            score += 1
        elif has_error and not has_action:
            issues.append("missing_action")
            suggestions.append("what action was taken to fix or investigate?")
        elif has_action and not (has_outcome or has_error):
            issues.append("missing_outcome")
            suggestions.append("what was the result of this action?")

    # 4. Check for time references
    has_time = re.search(r"\d{4}|\d+\s+(?:hour|day|week|month|year|minute|second)", content_lower)
    has_date = re.search(r"\d{4}-\d{2}-\d{2}|\w+\s+\d{1,2}|yesterday|today|now", content_lower)
    if has_time or has_date:
        score += 1
    elif network_type == "world" or network_type == "experience":
        issues.append("missing_time_reference")
        suggestions.append("add a time reference (date, 'yesterday', 'last week', etc)")

    # 5. Check for vagueness (generic words that could be about anything)
    vague_words = ["it", "that", "this", "something", "stuff", "things"]
    vague_ratio = sum(1 for w in content.split() if w.lower().strip(",.!?") in vague_words) / max(word_count, 1)
    if vague_ratio > 0.3:
        issues.append("too_vague")
        suggestions.append("replace generic words like 'it' or 'stuff' with specific nouns")

    # Normalize score to 0-5
    score = max(0, min(5, score))

    return {
        "is_complete": score >= 3 and len(issues) <= 1,
        "score": score,
        "issues": issues,
        "suggestions": suggestions,
        "found_entities": found_entities,
        "word_count": word_count,
        "char_count": char_count,
    }


def get_current_fronter() -> Dict[str, Optional[str]]:
    """Get the currently active fronter from Simply Plural."""
    # Try to import SP integration - if not available, return None
    try:
        from simply_plural_api import get_fronters
        result = get_fronters()
        if result and len(result) > 0:
            return {
                "uid": result[0].get("uid"),
                "name": result[0].get("name"),
                "is_primary": True,
            }
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Failed to get current fronter: {e}")
    return {"uid": None, "name": None, "is_primary": False}


def get_cross_session_context(topic: str, limit: int = 5, hours: int = 365) -> Dict[str, Any]:
    """
    Find past sessions related to a topic by searching memory summaries and topics.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cutoff = (datetime.now() - timedelta(days=hours)).isoformat()

    # Get sessions related to the topic
    cursor.execute("""
        SELECT id, session_id, summary, started_at, primary_fronter_name, topics, memory_count
        FROM sessions
        WHERE (topics LIKE ? OR summary LIKE ?)
          AND started_at > ?
        ORDER BY started_at DESC
        LIMIT ?
    """, (f"%{topic}%", f"%{topic}%", cutoff, limit))

    sessions = [dict(r) for r in cursor.fetchall()]

    # Also get individual memories from past sessions about this topic
    cursor.execute("""
        SELECT m.id, m.content, m.t_event, m.network_type, m.fronter_name,
               s.session_id, s.started_at
        FROM memories m
        JOIN sessions s ON m.conversation_id = s.session_id
        WHERE m.content LIKE ?
          AND m.conversation_id IS NOT NULL
          AND s.started_at > ?
        ORDER BY m.t_event DESC
        LIMIT ?
    """, (f"%{topic}%", cutoff, limit))

    related_memories = [dict(r) for r in cursor.fetchall()]
    conn.close()

    # Count unique sessions found
    session_ids = set(s["session_id"] for s in sessions)
    session_ids.update(m.get("session_id") for m in related_memories if m.get("session_id"))

    return {
        "topic": topic,
        "unique_sessions_found": len(session_ids),
        "sessions": sessions,
        "related_memories": related_memories,
        "context": f"found {len(session_ids)} past sessions discussing '{topic}'" if session_ids else None,
    }


def auto_generate_session_digest(session_id: str = None, limit: int = 20) -> str:
    """
    Generate a summary of recent memories, optionally scoped to a session.
    Used when session_wrap is called without notes.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    if session_id:
        cursor.execute("""
            SELECT content, network_type, t_event, fronter_name
            FROM memories
            WHERE conversation_id = ?
            ORDER BY t_event DESC
            LIMIT ?
        """, (session_id, limit))
    else:
        cursor.execute("""
            SELECT content, network_type, t_event, fronter_name
            FROM memories
            ORDER BY t_recorded DESC
            LIMIT ?
        """, (limit,))

    memories = [dict(r) for r in cursor.fetchall()]
    conn.close()

    if not memories:
        return "session with no recorded memories"

    # Group by network_type
    by_type = {}
    for m in memories:
        nt = m.get("network_type", "unknown")
        if nt not in by_type:
            by_type[nt] = []
        by_type[nt].append(m["content"][:100] + "..." if len(m["content"]) > 100 else m["content"])

    lines = ["session summary:"]
    for nt, contents in by_type.items():
        lines.append(f"\n{nt.upper()}:")
        for c in contents[:3]:  # max 3 per type
            lines.append(f"  - {c}")

    total = len(memories)
    if total > limit:
        lines.append(f"\n... and {total - limit} more memories")

    return "\n".join(lines)


def store_session_wrap(session_notes: str, conversation_id: str = None) -> Dict:
    """
    Store session wrap info: summary + memory count + topics.
    Creates or updates a session record.
    """
    import uuid as uuid_lib
    session_id = conversation_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid_lib.uuid4().hex[:8]}"

    conn = get_db_connection()
    cursor = conn.cursor()

    # Count memories for this session
    if conversation_id:
        cursor.execute("SELECT COUNT(*) FROM memories WHERE conversation_id = ?", (conversation_id,))
        memory_count = cursor.fetchone()[0] or 0
    else:
        memory_count = 0

    # Get primary fronter for this session
    fronter = get_current_fronter()

    # Extract potential topics from memory content (simple keyword extraction)
    topics = ""
    if conversation_id:
        cursor.execute("""
            SELECT content FROM memories
            WHERE conversation_id = ?
            ORDER BY t_event DESC LIMIT 10
        """, (conversation_id,))
        recent = [r[0] for r in cursor.fetchall()]
        if recent:
            # Extract potential topic keywords
            all_text = " ".join(recent).lower()
            topic_words = []
            for word in re.findall(r'\b[a-z]{4,}\b', all_text):
                if word not in ('this', 'that', 'with', 'from', 'have', 'been', 'were', 'they', 'their'):
                    topic_words.append(word)
            # Get top 5 most common words as topics
            from collections import Counter
            topics = ", ".join([w for w, _ in Counter(topic_words).most_common(5)])

    # Insert or replace session record
    cursor.execute("""
        INSERT OR REPLACE INTO sessions (session_id, summary, memory_count, topics,
                                          primary_fronter_uid, primary_fronter_name)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, session_notes, memory_count, topics,
          fronter.get("uid"), fronter.get("name")))

    # Update ended_at timestamp
    cursor.execute("UPDATE sessions SET ended_at = ? WHERE session_id = ?",
                   (datetime.now().isoformat(), session_id))

    conn.commit()
    conn.close()

    return {
        "session_id": session_id,
        "summary": session_notes,
        "memory_count": memory_count,
        "topics": topics,
        "fronter": fronter.get("name"),
        "wrapped": True,
    }


def get_briefing(max_items: int = 10) -> Dict:
    """
    Get session briefing with SP-weighted memory retrieval.
    Memories from the current fronter get boosted ranking.
    """
    fronter = get_current_fronter()
    fronter_uid = fronter.get("uid")
    fronter_name = fronter.get("name")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Get recent memories, boosting those from current fronter
    if fronter_uid:
        cursor.execute("""
            SELECT m.id, m.content, m.network_type, m.t_event, m.tier,
                   m.importance, m.decay_score, m.access_count, m.fronter_name,
                   CASE WHEN m.fronter_uid = ? THEN 0.15 ELSE 0.0 END as fronter_boost
            FROM memories m
            WHERE m.valid_to IS NULL OR m.valid_to > datetime('now')
            ORDER BY (COALESCE(m.importance, 0.5) * COALESCE(m.decay_score, 1.0) * (1 + m.access_count * 0.1) + fronter_boost) DESC
            LIMIT ?
        """, (fronter_uid, max_items))
    else:
        cursor.execute("""
            SELECT m.id, m.content, m.network_type, m.t_event, m.tier,
                   m.importance, m.decay_score, m.access_count, m.fronter_name,
                   0.0 as fronter_boost
            FROM memories m
            WHERE m.valid_to IS NULL OR m.valid_to > datetime('now')
            ORDER BY (COALESCE(m.importance, 0.5) * COALESCE(m.decay_score, 1.0) * (1 + m.access_count * 0.1)) DESC
            LIMIT ?
        """, (max_items,))

    memories = [dict(r) for r in cursor.fetchall()]

    # Also get wiki context
    cursor.execute("SELECT slug, title FROM wiki_pages ORDER BY RANDOM() LIMIT 3")
    wiki_snippets = [f"{r['title']} ({r['slug']})" for r in cursor.fetchall()]

    conn.close()

    return {
        "current_fronter": fronter_name,
        "current_fronter_uid": fronter_uid,
        "memory_count": len(memories),
        "recent_memories": memories,
        "wiki_context": wiki_snippets,
    }


def unified_briefing(context: str = "", max_items: int = 15) -> Dict:
    """
    Unified briefing combining memster + SP + wiki in one call.
    SP-weighted retrieval when context suggests specific fronter relevance.
    """
    briefing = get_briefing(max_items=max_items)

    # Add cross-session context if context suggests looking for related topics
    if context:
        context_lower = context.lower()
        # Extract potential topic words from context
        topic_words = re.findall(r'\b[a-z]{4,}\b', context_lower)
        if topic_words:
            # Use the most significant topic word for cross-session search
            cross = get_cross_session_context(topic_words[0], limit=3)
            if cross.get("unique_sessions_found", 0) > 0:
                briefing["cross_session_context"] = cross

    return briefing


def get_narrative_briefing(token_limit: int = 500) -> Dict:
    """
    Narrative briefing with richer context chains.
    Groups memories by network type and creates context threads.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    fronter = get_current_fronter()

    result = {
        "current_fronter": fronter.get("name"),
        "context_threads": [],
        "total_memories": 0,
    }

    for network_type in ["world", "experience", "opinion", "observation"]:
        cursor.execute("""
            SELECT id, content, t_event, tier, importance
            FROM memories
            WHERE network_type = ?
              AND (valid_to IS NULL OR valid_to > datetime('now'))
            ORDER BY importance DESC, t_event DESC
            LIMIT 5
        """, (network_type,))
        memories = [dict(r) for r in cursor.fetchall()]
        if memories:
            result["context_threads"].append({
                "network": network_type,
                "memories": memories,
            })
            result["total_memories"] += len(memories)

    conn.close()
    return result


def memory_timeline(days: int = 30) -> Dict:
    """Get chronological memory timeline grouped by day."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()

    cursor.execute("""
        SELECT id, content, network_type, t_event, t_recorded, fronter_name, tier
        FROM memories
        WHERE t_event >= ?
        ORDER BY t_event DESC
    """, (cutoff,))

    memories = [dict(r) for r in cursor.fetchall()]
    conn.close()

    # Group by date
    by_date = {}
    for m in memories:
        date_key = m.get("t_event", "")[:10]  # YYYY-MM-DD
        if date_key not in by_date:
            by_date[date_key] = []
        by_date[date_key].append(m)

    timeline = [{"date": d, "memories": ms} for d, ms in sorted(by_date.items(), reverse=True)]
    return {"days": days, "timeline": timeline, "total": len(memories)}


def sp_status() -> Dict:
    """Get current Simply Plural front status."""
    try:
        from simply_plural_api import get_fronters
        fronters = get_fronters()
        if fronters and len(fronters) > 0:
            return {
                "fronting": [{"uid": f.get("uid"), "name": f.get("name")} for f in fronters],
                "primary": fronters[0].get("name"),
                "count": len(fronters),
            }
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"SP status error: {e}")

    # Fallback - try the MCP approach
    try:
        fronter = get_current_fronter()
        if fronter.get("uid"):
            return {
                "fronting": [{"uid": fronter["uid"], "name": fronter["name"]}],
                "primary": fronter["name"],
                "count": 1,
            }
    except Exception:
        pass

    return {"fronting": [], "primary": None, "count": 0, "error": "SP not available"}


def sp_members(include_archived: bool = False) -> Dict:
    """Get all headmates from Simply Plural."""
    try:
        from simply_plural_api import get_members
        members = get_members(include_archived=include_archived)
        return {"members": members, "count": len(members) if members else 0}
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"SP members error: {e}")

    return {"members": [], "count": 0, "error": "SP not available"}


def ingest_sp_history(days: int = 30) -> Dict:
    """Pull SP front history and store as memories."""
    try:
        from simply_plural_api import get_front_history
        history = get_front_history(days=days)
        if not history:
            return {"ingested": 0, "message": "no SP history available"}

        conn = get_db_connection()
        cursor = conn.cursor()
        ingested = 0

        for entry in history:
            if not entry.get("name"):
                continue
            cursor.execute("""
                INSERT INTO memories (content, network_type, t_event, source, conversation_id)
                VALUES (?, 'observation', ?, ?, ?)
            """, (
                f"SP: {entry['name']} was fronting ({entry.get('duration', 'unknown duration')})",
                entry.get("timestamp", datetime.now().isoformat()),
                "simply_plural",
                f"sp_history_{entry.get('timestamp', '')}"
            ))
            ingested += 1

        conn.commit()
        conn.close()
        return {"ingested": ingested, "message": f"ingested {ingested} SP history entries"}
    except ImportError:
        return {"ingested": 0, "error": "SP API not available"}
    except Exception as e:
        return {"ingested": 0, "error": str(e)}


# === semantic search helpers ===

def hybrid_search(query: str, limit: int = 10, db_path: str = None) -> List[Dict]:
    """Hybrid ranking: 50% vector similarity + 30% FTS rank + 20% importance."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    # 1) vector search (top 30 candidates)
    vector_results = {}
    if EMBEDDINGS_AVAILABLE:
        try:
            sem = vector_search(query, limit=30, threshold=0.1, db_path=path)
            for r in sem:
                mid = r.get("id")
                if mid:
                    vector_results[mid] = r.get("similarity", 0.0)
        except Exception as e:
            logger.debug(f"vector search failed: {e}")

    # 2) FTS search (top 30 candidates)
    fts_results = {}
    try:
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'")
        if c.fetchone():
            c.execute("SELECT rowid, rank FROM memories_fts WHERE content MATCH ? ORDER BY rank LIMIT 30", (query,))
            for row in c.fetchall():
                fts_results[row["rowid"]] = abs(row["rank"]) if row["rank"] else 0
    except Exception:
        pass

    # 3) combine candidate IDs
    all_ids = set(vector_results.keys()) | set(fts_results.keys())
    if not all_ids:
        conn.close()
        # fallback to LIKE search
        c = conn.cursor()
        c.execute("""SELECT id, content, category, tier, COALESCE(importance, 0.5) as importance 
                     FROM memories WHERE content LIKE ? ORDER BY importance DESC LIMIT ?""",
                   (f"%{query}%", limit))
        results = [dict(r) for r in c.fetchall()]
        conn.close()
        return results

    # 4) get full memory data for candidates
    placeholders = ",".join("?" * len(all_ids))
    c = conn.cursor()
    c.execute(f"""SELECT id, content, category, tier, COALESCE(importance, 0.5) as importance 
                  FROM memories WHERE id IN ({placeholders})""", list(all_ids))
    memories = {r["id"]: dict(r) for r in c.fetchall()}

    # 5) hybrid scoring
    max_fts = max(fts_results.values()) if fts_results else 1.0
    scored = []
    for mid, mem in memories.items():
        vec_score = vector_results.get(mid, 0.0)
        fts_raw = fts_results.get(mid, 0.0)
        fts_score = fts_raw / max_fts if max_fts > 0 else 0.0
        imp_score = mem.get("importance", 0.5)

        hybrid = (0.5 * vec_score) + (0.3 * fts_score) + (0.2 * imp_score)
        mem["hybrid_score"] = round(hybrid, 4)
        mem["vec_score"] = round(vec_score, 4)
        mem["fts_score"] = round(fts_score, 4)
        scored.append(mem)

    scored.sort(key=lambda x: x["hybrid_score"], reverse=True)

    # bump access count
    for m in scored[:limit]:
        c.execute("UPDATE memories SET access_count = access_count + 1 WHERE id = ?", (m["id"],))
    conn.commit()
    conn.close()

    return scored[:limit]


def sleep_consolidate(db_path: str = None, batch_size: int = 100) -> Dict:
    """Consolidate old memories with access-weighted decay.

    Key insight: frequently accessed memories should decay MUCH slower.
    Decay rate = base_rate / (1 + log(access_count))
    This means: never-accessed = full decay, accessed 10x = ~30% of decay rate
    """
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    c = conn.cursor()
    now = datetime.now()
    results = {"decayed": 0, "promoted": 0, "summarized": 0}

    # Get memories eligible for decay
    cutoff = (now - timedelta(days=7)).isoformat()
    c.execute("""
        SELECT id, decay_score, access_count
        FROM memories
        WHERE tier = 'L2'
          AND (t_recorded IS NULL OR t_recorded < ?)
          AND decay_score > 0.1
    """, (cutoff,))

    eligible = c.fetchall()
    decayed = 0

    for row in eligible:
        mid, current_decay, access_count = row
        access_count = access_count or 0

        # Access-weighted decay: more access = slower decay
        # base decay per week = 5%, but reduces with access
        #   0 accesses: 0.05 (full decay)
        #   1 access:  0.05 / (1 + log(1)) = 0.05 / 1 = 0.05
        #   5 accesses: 0.05 / (1 + log(5)) = 0.05 / 2.6 = 0.019
        #   10 accesses: 0.05 / (1 + log(10)) = 0.05 / 3.3 = 0.015
        import math
        access_multiplier = 1.0 / (1 + math.log1p(max(0, access_count)))
        new_decay = current_decay * (1.0 - (0.05 * access_multiplier))
        new_decay = max(0.05, new_decay)  # floor at 0.05 so nothing disappears

        if new_decay < current_decay:
            c.execute("UPDATE memories SET decay_score = ? WHERE id = ?", (new_decay, mid))
            decayed += 1

    results["decayed"] = decayed

    # promote important L2 -> L1 (keep promotion thresholds the same)
    c.execute("""UPDATE memories SET tier = 'L1'
                 WHERE tier = 'L2' AND access_count >= 5 AND importance >= 0.7
                 AND decay_score >= 0.5""")
    results["promoted"] = c.rowcount

    # promote important L1 -> L0
    c.execute("""UPDATE memories SET tier = 'L0'
                 WHERE tier = 'L1' AND access_count >= 10 AND importance >= 0.8
                 AND decay_score >= 0.7""")
    results["promoted"] += c.rowcount

    conn.commit()
    conn.close()
    logger.info(f"sleep: decayed={results['decayed']} promoted={results['promoted']}")
    return results


def remember_batch(memories: List[Dict], db_path: str = None) -> Dict:
    """Batch insert multiple memories at once."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    c = conn.cursor()
    now = datetime.now().isoformat()

    created = []
    skipped = []

    for mem in memories:
        content = mem.get("content", "").strip()
        category = mem.get("category", "general")

        if not content:
            skipped.append({"content": content, "reason": "empty"})
            continue
        c.execute("SELECT id FROM memories WHERE content = ?", (content,))
        if c.fetchone():
            skipped.append({"content": content[:50], "reason": "duplicate"})
            continue

        c.execute("""INSERT INTO memories (content, category, t_event, t_recorded, tier, memory_type) 
                     VALUES (?, ?, ?, ?, ?, ?)""",
                   (content, category, now, now, "L2", category if category in ("world", "experience", "opinion", "observation") else "observation"))
        mid = c.lastrowid
        created.append(mid)

    conn.commit()
    conn.close()
    return {"created": len(created), "skipped": len(skipped), "ids": created}


# === post-insert hooks (simplified) ===

def run_post_insert_hooks(memory_id, content, category, tags=None):
    """Run post-insert processing."""
    results = {"entities": 0, "edges": 0, "importance": None, "wiki_pages": None}
    # Simplified - no external dependencies
    return results


# === MCP Server instance ===
if MCP_AVAILABLE:
    app = Server("memster_v3")


# === Tool Definitions ===


# V5 Features helper
def get_v5_features():
    """Get V5 features instance with database path."""
    from memster_v5_features import get_v5_features as v5_factory
    return v5_factory(DB_PATH)

TOOL_DEFINITIONS = [
    # Activity tracking tools
    Tool(
        name="get_activity_summary",
        description="Get a human-readable summary of what the user has been doing recently.",
        inputSchema={"type": "object", "properties": {"hours": {"type": "integer", "default": 2}}}
    ),
    Tool(
        name="get_recent_activity",
        description="Get structured activity data - apps used, clipboard, screenshots.",
        inputSchema={"type": "object", "properties": {"hours": {"type": "integer", "default": 1}}}
    ),
    Tool(
        name="search_window_history",
        description="Search what apps/windows the user was working in.",
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}, "hours": {"type": "integer", "default": 24}}, "required": ["query"]}
    ),
    Tool(
        name="search_clipboard",
        description="Search clipboard history for pasted/copied content.",
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}, "hours": {"type": "integer", "default": 24}}, "required": ["query"]}
    ),
    Tool(
        name="search_ocr_text",
        description="Search text extracted from screenshots via OCR.",
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 10}}, "required": ["query"]}
    ),

    # Memory CRUD tools
    Tool(
        name="memster_query",
        description="Search memories with filters for query string, network type, date range, and result limit.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string for full-text search"},
                "network_type": {"type": "string", "enum": ["world", "experience", "opinion", "observation"]},
                "date_from": {"type": "string"},
                "date_to": {"type": "string"},
                "limit": {"type": "integer", "default": 20}
            }
        }
    ),
    Tool(
        name="memster_remember",
        description="Store a new memory with auto-dedup, completeness checking, temporal detection, and SP fronter tagging.",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "network_type": {"type": "string", "enum": ["world", "experience", "opinion", "observation"]},
                "t_event": {"type": "string"},
                "source": {"type": "string"},
                "conversation_id": {"type": "string"},
                "embedding": {"type": "string"},
                "fronter_uid": {"type": "string"},
                "fronter_name": {"type": "string"}
            },
            "required": ["content", "network_type"]
        }
    ),
    Tool(
        name="memster_sync_pieces",
        description="Sync with Pieces MCP server to ingest workstream snippets.",
        inputSchema={"type": "object", "properties": {"limit": {"type": "integer", "default": 50}}}
    ),
    Tool(
        name="query_memories",
        description="Search memories with text/category/tier filters. Results ordered by importance and access frequency.",
        inputSchema={"type": "object", "properties": {
            "query_text": {"type": "string"},
            "category": {"type": "string"},
            "tier": {"type": "string"},
            "max_results": {"type": "integer", "default": 10}
        }}
    ),
    Tool(
        name="remember_memory",
        description="Store a new memory. Auto-dedup, auto-extract entities, auto-create graph edges.",
        inputSchema={"type": "object", "properties": {
            "content": {"type": "string"},
            "category": {"type": "string", "enum": ["world", "experience", "opinion", "observation"]},
            "tags": {"type": "array", "items": {"type": "string"}}
        }, "required": ["content"]}
    ),
    Tool(
        name="update_memory",
        description="Update an existing memory's content, category, or tier.",
        inputSchema={"type": "object", "properties": {
            "memory_id": {"type": "integer"},
            "content": {"type": "string"},
            "category": {"type": "string"},
            "tier": {"type": "string"}
        }, "required": ["memory_id"]}
    ),
    Tool(
        name="merge_memories",
        description="Merge two memories into one. Keeps the first as primary, absorbs second.",
        inputSchema={"type": "object", "properties": {
            "primary_id": {"type": "integer"},
            "secondary_id": {"type": "integer"},
            "merged_content": {"type": "string"}
        }, "required": ["primary_id", "secondary_id"]}
    ),
    Tool(
        name="delete_memory",
        description="Delete a memory by its ID.",
        inputSchema={"type": "object", "properties": {"memory_id": {"type": "integer"}}, "required": ["memory_id"]}
    ),
    Tool(
        name="delete_by_query",
        description="Delete memories matching a content query.",
        inputSchema={"type": "object", "properties": {
            "query_text": {"type": "string"},
            "exact_match": {"type": "boolean", "default": False}
        }, "required": ["query_text"]}
    ),
    Tool(
        name="find_duplicates",
        description="Find duplicate or near-duplicate memories.",
        inputSchema={"type": "object", "properties": {"similarity_threshold": {"type": "number", "default": 0.8}}}
    ),

    # Semantic search tools
    Tool(
        name="semantic_memory_search",
        description="Search memories using AI semantic similarity.",
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 5}}, "required": ["query"]}
    ),
    Tool(
        name="hybrid_search",
        description="Hybrid ranking: 50% vector similarity + 30% full-text + 20% importance.",
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 10}}, "required": ["query"]}
    ),
    Tool(
        name="remember_batch",
        description="Store multiple memories at once. Auto-dedup.",
        inputSchema={"type": "object", "properties": {
            "memories": {"type": "array", "items": {"type": "object", "properties": {
                "content": {"type": "string"},
                "category": {"type": "string"}
            }}}
        }, "required": ["memories"]}
    ),
        Tool(
        name="sleep_consolidate",
        description="Run memory consolidation cycle.",
        inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
        name="backfill_embeddings",
        description="Generate and store NVIDIA NIM vector embeddings for memories that don't have them yet. Required after upgrading to frontier embeddings.",
        inputSchema={"type": "object", "properties": {"batch_size": {"type": "integer", "default": 10}, "delay": {"type": "number", "default": 1.0}}}
        ),
        Tool(
        name="find_similar",
        description="Find memories similar to a specific memory by ID.",
        inputSchema={"type": "object", "properties": {"memory_id": {"type": "integer"}, "limit": {"type": "integer", "default": 5}}, "required": ["memory_id"]}
    ),

    # Memory graph tools
    Tool(
        name="get_related_memories",
        description="Get memories linked to a specific memory via graph edges.",
        inputSchema={"type": "object", "properties": {"memory_id": {"type": "integer"}, "limit": {"type": "integer", "default": 10}}, "required": ["memory_id"]}
    ),
    Tool(
        name="link_memories",
        description="Create a typed link between two memories.",
        inputSchema={"type": "object", "properties": {
            "source_id": {"type": "integer"},
            "target_id": {"type": "integer"},
            "link_type": {"type": "string", "default": "related"}
        }, "required": ["source_id", "target_id"]}
    ),

    # Briefing and surfacing tools
    Tool(
        name="get_briefing",
        description="Get a session briefing - surface relevant memories.",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="get_narrative_briefing",
        description="Get a narrative briefing with context threads.",
        inputSchema={"type": "object", "properties": {"token_limit": {"type": "integer", "default": 500}}}
    ),
    Tool(
        name="check_proactive",
        description="Check if there are relevant past memories for current context.",
        inputSchema={"type": "object", "properties": {"context": {"type": "string"}, "max_suggestions": {"type": "integer", "default": 3}}, "required": ["context"]}
    ),

    # Health and maintenance tools
    Tool(
        name="get_memory_health",
        description="Get health report of the memory system.",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="filter_passive_capture",
        description="Clean up passive capture noise.",
        inputSchema={"type": "object", "properties": {"dry_run": {"type": "boolean", "default": True}}}
    ),
    Tool(
        name="query_by_entity",
        description="Find memories by extracted entity (IP, path, project name, etc).",
        inputSchema={"type": "object", "properties": {
            "entity_name": {"type": "string"},
            "entity_type": {"type": "string"},
            "max_results": {"type": "integer", "default": 10}
        }, "required": ["entity_name"]}
    ),
    Tool(
        name="memory_timeline",
        description="Get chronological memory timeline.",
        inputSchema={"type": "object", "properties": {"days": {"type": "integer", "default": 30}}}
    ),
    Tool(
        name="session_wrap",
        description="End-of-session summary. Auto-generates digest from recent memories if no notes provided.",
        inputSchema={"type": "object", "properties": {
            "session_notes": {"type": "string"},
            "conversation_id": {"type": "string"}
        }}
    ),
    Tool(
        name="ingest_sp_history",
        description="Store Simply Plural front history as memories.",
        inputSchema={"type": "object", "properties": {"days": {"type": "integer", "default": 30}}}
    ),
    Tool(
        name="build_graph_edges",
        description="Rebuild memory graph edges from entity overlaps.",
        inputSchema={"type": "object", "properties": {}}
    ),

    # Wiki cross-system tools
    Tool(
        name="wiki_search",
        description="Search the wiki.",
        inputSchema={"type": "object", "properties": {
            "query": {"type": "string"},
            "category": {"type": "string"},
            "limit": {"type": "integer", "default": 5}
        }, "required": ["query"]}
    ),
    Tool(
        name="wiki_read",
        description="Read a wiki page by slug.",
        inputSchema={"type": "object", "properties": {"slug": {"type": "string"}}, "required": ["slug"]}
    ),
    Tool(
        name="wiki_list",
        description="List all wiki pages.",
        inputSchema={"type": "object", "properties": {"category": {"type": "string"}}}
    ),
    Tool(
        name="wiki_sweep",
        description="Audit wiki: find orphans, broken links, untagged pages.",
        inputSchema={"type": "object", "properties": {"category": {"type": "string"}}}
    ),

    # Simply Plural tools
    Tool(
        name="sp_status",
        description="Get Simply Plural status: current front, member count.",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="sp_members",
        description="List all headmates from Simply Plural.",
        inputSchema={"type": "object", "properties": {"include_archived": {"type": "boolean", "default": False}}}
    ),

    # Bridge tools
    Tool(
        name="unified_briefing",
        description="Pull context from memster + wiki + SP in one call.",
        inputSchema={"type": "object", "properties": {"context": {"type": "string"}, "max_items": {"type": "integer", "default": 15}}}
    ),
    Tool(
        name="wiki_to_memster_sync",
        description="Extract key facts from wiki pages into memster memories.",
        inputSchema={"type": "object", "properties": {"max_pages": {"type": "integer", "default": 20}}}
    ),
    Tool(
        name="enrich_memory_lookup",
        description="Check if content references known entities.",
        inputSchema={"type": "object", "properties": {
            "content": {"type": "string"},
            "category": {"type": "string"}
        }, "required": ["content"]}
    ),
    Tool(
        name="detect_stale_memories",
        description="Find memories that may be outdated.",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="check_memory_quality",
        description="Check if content is worth storing. Returns quality score, issues, and suggestions.",
        inputSchema={"type": "object", "properties": {
            "content": {"type": "string"},
            "category": {"type": "string"}
        }, "required": ["content"]}
    ),
    Tool(
        name="check_memory_completeness",
        description="Enhanced completeness check for memories. Checks for missing time refs, vagueness, missing entities.",
        inputSchema={"type": "object", "properties": {
            "content": {"type": "string"},
            "category": {"type": "string", "default": "observation"}
        }, "required": ["content"]}
    ),
    Tool(
        name="get_cross_session_context",
        description="Find past sessions related to a topic. Returns related sessions and memories.",
        inputSchema={"type": "object", "properties": {
            "topic": {"type": "string"},
            "limit": {"type": "integer", "default": 5},
            "hours": {"type": "integer", "default": 365}
        }, "required": ["topic"]}
    ),

    # Brainstack-inspired enhanced retrieval tools
    Tool(
        name="record_retrieval_telemetry",
        description="Record telemetry for memory retrieval events.",
        inputSchema={"type": "object", "properties": {
            "memory_id": {"type": "integer"},
            "match_served": {"type": "boolean", "default": False},
            "query_text": {"type": "string"}
        }, "required": ["memory_id", "query_text"]}
    ),
    Tool(
        name="update_temporal_bounds",
        description="Update temporal validity bounds for a memory.",
        inputSchema={"type": "object", "properties": {
            "memory_id": {"type": "integer"},
            "valid_from": {"type": "string"},
            "valid_to": {"type": "string"},
            "observed_at": {"type": "string"}
        }, "required": ["memory_id"]}
    ),
    Tool(
        name="route_retrieval",
        description="Analyze query and route to appropriate retrieval strategy.",
        inputSchema={"type": "object", "properties": {
            "query": {"type": "string"},
            "session_id": {"type": "string"},
            "conversation_id": {"type": "string"}
        }, "required": ["query", "session_id"]}
    ),
    Tool(
        name="track_provenance",
        description="Merge provenance info into memories.",
        inputSchema={"type": "object", "properties": {
            "memory_id": {"type": "integer"},
            "source": {"type": "string"},
            "source_type": {"type": "string"},
            "extracted_at": {"type": "string"}
        }, "required": ["memory_id"]}
    ),
    Tool(
        name="extract_graph_relations",
        description="Extract semantic graph relations from memory content.",
        inputSchema={"type": "object", "properties": {
            "memory_id": {"type": "integer"},
            "content": {"type": "string"}
        }, "required": ["memory_id"]}
    ),
    Tool(
        name="get_working_memory_context",
        description="Get current working memory context for a session.",
        inputSchema={"type": "object", "properties": {
            "session_id": {"type": "string"},
            "conversation_id": {"type": "string"},
            "limit": {"type": "integer", "default": 10}
        }, "required": ["session_id"]}
    ),
    Tool(
        name="infer_follow_up_intent",
        description="Detect if current query is a follow-up to previous conversation.",
        inputSchema={"type": "object", "properties": {
            "session_id": {"type": "string"},
            "conversation_id": {"type": "string"},
            "query": {"type": "string"}
        }, "required": ["session_id", "query"]}
    ),
    # === V5: Memory Feedback ===
    Tool(
        name="rate_memory",
        description="Rate a memory's usefulness. Feedback types: helpful, irrelevant, wrong, outdated, promote, demote.",
        inputSchema={"type": "object", "properties": {
            "memory_id": {"type": "integer"},
            "feedback_type": {"type": "string", "enum": ["helpful", "irrelevant", "wrong", "outdated", "promote", "demote"]},
            "context": {"type": "string"}
        }, "required": ["memory_id", "feedback_type"]}
    ),
    Tool(
        name="get_memory_feedback_stats",
        description="Get feedback statistics for a memory or overall system.",
        inputSchema={"type": "object", "properties": {
            "memory_id": {"type": "integer"}
        }}
    ),

    # === V5: Task Channels ===
    Tool(
        name="create_task",
        description="Create a new task/project channel for organizing memories.",
        inputSchema={"type": "object", "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "release_version": {"type": "string"}
        }, "required": ["name"]}
    ),
    Tool(
        name="complete_task",
        description="Mark a task as completed with memory snapshot.",
        inputSchema={"type": "object", "properties": {
            "task_id": {"type": "integer"}
        }, "required": ["task_id"]}
    ),
    Tool(
        name="assign_memory_to_task",
        description="Assign a memory to a specific task channel.",
        inputSchema={"type": "object", "properties": {
            "memory_id": {"type": "integer"},
            "task_id": {"type": "integer"}
        }, "required": ["memory_id", "task_id"]}
    ),
    Tool(
        name="get_task_memories",
        description="Get all memories assigned to a task.",
        inputSchema={"type": "object", "properties": {
            "task_id": {"type": "integer"}
        }, "required": ["task_id"]}
    ),

    # === V5: Memory Compression ===
    Tool(
        name="compress_memory",
        description="Compress a memory to save space. Original content is zlib compressed.",
        inputSchema={"type": "object", "properties": {
            "memory_id": {"type": "integer"}
        }, "required": ["memory_id"]}
    ),
    Tool(
        name="decompress_memory",
        description="Decompress a previously compressed memory.",
        inputSchema={"type": "object", "properties": {
            "memory_id": {"type": "integer"}
        }, "required": ["memory_id"]}
    ),

    # === V5: Memory Palace ===
    Tool(
        name="create_palace",
        description="Create a new memory palace for spatial memory organization.",
        inputSchema={"type": "object", "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"}
        }, "required": ["name"]}
    ),
    Tool(
        name="add_room",
        description="Add a room to a memory palace.",
        inputSchema={"type": "object", "properties": {
            "palace_id": {"type": "integer"},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "position_x": {"type": "number", "default": 0},
            "position_y": {"type": "number", "default": 0},
            "position_z": {"type": "number", "default": 0}
        }, "required": ["palace_id", "name"]}
    ),
    Tool(
        name="place_memory_in_room",
        description="Place a memory in a palace room.",
        inputSchema={"type": "object", "properties": {
            "room_id": {"type": "integer"},
            "memory_id": {"type": "integer"}
        }, "required": ["room_id", "memory_id"]}
    ),
    Tool(
        name="walk_palace",
        description="Walk through a memory palace and see all rooms and their memories.",
        inputSchema={"type": "object", "properties": {
            "palace_id": {"type": "integer"}
        }, "required": ["palace_id"]}
    ),

    # === V5: Narrative Arcs ===
    Tool(
        name="create_narrative_arc",
        description="Create a narrative arc to track a storyline across memories.",
        inputSchema={"type": "object", "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "arc_type": {"type": "string", "enum": ["auto", "manual", "derived"], "default": "manual"}
        }, "required": ["title"]}
    ),
    Tool(
        name="add_memory_to_arc",
        description="Add a memory to a narrative arc.",
        inputSchema={"type": "object", "properties": {
            "arc_id": {"type": "integer"},
            "memory_id": {"type": "integer"},
            "arc_role": {"type": "string", "enum": ["beginning", "event", "development", "climax", "resolution"], "default": "event"}
        }, "required": ["arc_id", "memory_id"]}
    ),
    Tool(
        name="get_arc_timeline",
        description="Get the timeline of memories in a narrative arc.",
        inputSchema={"type": "object", "properties": {
            "arc_id": {"type": "integer"}
        }, "required": ["arc_id"]}
    ),
    Tool(
        name="list_arcs",
        description="List narrative arcs with optional filtering.",
        inputSchema={"type": "object", "properties": {
            "status": {"type": "string", "enum": ["ongoing", "completed", "abandoned"]},
            "arc_type": {"type": "string"}
        }}
    ),
    Tool(
        name="complete_arc",
        description="Mark a narrative arc as completed.",
        inputSchema={"type": "object", "properties": {
            "arc_id": {"type": "integer"}
        }, "required": ["arc_id"]}
    ),

    # === V5: Conclusions ===
    Tool(
        name="derive_conclusion",
        description="Derive a conclusion from multiple memories.",
        inputSchema={"type": "object", "properties": {
            "content": {"type": "string"},
            "memory_ids": {"type": "array", "items": {"type": "integer"}},
            "confidence": {"type": "number", "default": 0.5}
        }, "required": ["content", "memory_ids"]}
    ),
    Tool(
        name="validate_conclusion",
        description="Validate or reject a conclusion.",
        inputSchema={"type": "object", "properties": {
            "conclusion_id": {"type": "integer"},
            "still_valid": {"type": "boolean"},
            "notes": {"type": "string"}
        }, "required": ["conclusion_id", "still_valid"]}
    ),
    Tool(
        name="get_conclusions",
        description="Get conclusions with optional filtering.",
        inputSchema={"type": "object", "properties": {
            "status": {"type": "string", "enum": ["pending", "confirmed", "rejected", "revised"]},
            "limit": {"type": "integer", "default": 10}
        }}
    ),

    # === V5: Federation ===
    Tool(
        name="register_peer",
        description="Register a federation peer for memory sync.",
        inputSchema={"type": "object", "properties": {
            "peer_url": {"type": "string"},
            "peer_name": {"type": "string"},
            "auth_token": {"type": "string"},
            "sync_direction": {"type": "string", "enum": ["bidirectional", "publish", "subscribe"], "default": "bidirectional"}
        }, "required": ["peer_url"]}
    ),
    Tool(
        name="list_peers",
        description="List all registered federation peers.",
        inputSchema={"type": "object", "properties": {
            "status": {"type": "string", "enum": ["active", "inactive", "error"]}
        }}
    ),
    Tool(
        name="sync_with_peer",
        description="Sync memories with a federation peer.",
        inputSchema={"type": "object", "properties": {
            "peer_id": {"type": "integer"},
            "direction": {"type": "string", "enum": ["bidirectional", "push", "pull"], "default": "bidirectional"}
        }, "required": ["peer_id"]}
    ),

# === Tier 5: Advanced Features ===
    Tool(
        name="set_temporal_bounds",
        description="Set temporal validity bounds for a memory (valid_from, valid_to, observed_at).",
        inputSchema={"type": "object", "properties": {
            "memory_id": {"type": "integer"},
            "valid_from": {"type": "string"},
            "valid_to": {"type": "string"},
            "observed_at": {"type": "string"}
        }, "required": ["memory_id"]}
    ),
    Tool(
        name="find_expired_memories",
        description="Find memories past their valid_to date.",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="detect_contradictions",
        description="Detect contradictory memory pairs via keyword overlap and negation matching.",
        inputSchema={"type": "object", "properties": {
            "threshold": {"type": "number", "default": 0.6}
        }}
    ),
    Tool(
        name="update_confidence",
        description="Update bayesian confidence based on observation result (true=confirmed, false=contradicted).",
        inputSchema={"type": "object", "properties": {
            "memory_id": {"type": "integer"},
            "observation_result": {"type": "boolean"}
        }, "required": ["memory_id", "observation_result"]}
    ),
    Tool(
        name="score_memory_confidence",
        description="Get detailed multi-signal confidence score for a memory.",
        inputSchema={"type": "object", "properties": {
            "memory_id": {"type": "integer"}
        }, "required": ["memory_id"]}
    ),
    Tool(
        name="assemble_context_packet",
        description="Assemble a token-budgeted context packet for prompt injection from recent, core, relevant memories.",
        inputSchema={"type": "object", "properties": {
            "query": {"type": "string", "default": ""},
            "max_tokens": {"type": "integer", "default": 2000},
            "context_type": {"type": "string", "default": "auto"}
        }}
    ),
    Tool(
        name="get_active_tasks",
        description="Get all non-completed tasks ordered by priority.",
        inputSchema={"type": "object", "properties": {
            "limit": {"type": "integer", "default": 50}
        }}
    ),
    # Beads features
    Tool(
        name="create_wisp",
        description="Create an ephemeral wisp memory with optional TTL (hours) and tags. Wisps auto-expire.",
        inputSchema={"type": "object", "properties": {"content": {"type": "string"}, "category": {"type": "string", "default": "observation"}, "ttl_hours": {"type": "integer", "default": 24}, "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["content"]}
    ),
    Tool(
        name="squash_wisp",
        description="Compress a wisp memory into a regular memory and delete the wisp.",
        inputSchema={"type": "object", "properties": {"memory_id": {"type": "integer"}}, "required": ["memory_id"]}
    ),
    Tool(
        name="burn_wisp",
        description="Immediately delete a wisp memory (mark as deleted).",
        inputSchema={"type": "object", "properties": {"memory_id": {"type": "integer"}}, "required": ["memory_id"]}
    ),
    Tool(
        name="gc_wisps",
        description="Garbage-collect expired wisps (dry-run or actual deletion).",
        inputSchema={"type": "object", "properties": {"dry_run": {"type": "boolean", "default": True}}}
    ),
    Tool(
        name="compact_memory_ai",
        description="AI-assisted memory compaction: summarize and consolidate a memory's content.",
        inputSchema={"type": "object", "properties": {"memory_id": {"type": "integer"}, "model": {"type": "string"}}}
    ),
    Tool(
        name="add_dependency",
        description="Add a dependency edge between two memories (source -> target).",
        inputSchema={"type": "object", "properties": {"source_id": {"type": "integer"}, "target_id": {"type": "integer"}, "dep_type": {"type": "string", "default": "related"}}}
    ),
    Tool(
        name="get_ready_memories",
        description="Get memories that are ready (unblocked) for processing, ordered by priority.",
        inputSchema={"type": "object", "properties": {"limit": {"type": "integer", "default": 20}}}
    ),
    Tool(
        name="audit_log",
        description="Append an audit log entry for traceability.",
        inputSchema={"type": "object", "properties": {"kind": {"type": "string"}, "data": {"type": "object"}, "actor": {"type": "string", "default": "hermes"}}}
    ),
    Tool(
        name="query_audit_log",
        description="Query the audit log for past actions.",
        inputSchema={"type": "object", "properties": {"kind": {"type": "string"}, "since_hours": {"type": "integer", "default": 24}, "limit": {"type": "integer", "default": 50}}}
    ),
    Tool(
        name="set_memory_gate",
        description="Set a confirmation gate on a memory (e.g., require approval before certain actions).",
        inputSchema={"type": "object", "properties": {"memory_id": {"type": "integer"}, "gate_type": {"type": "string", "default": "confirm"}, "approvers": {"type": "array", "items": {"type": "string"}}}}
    ),
    Tool(
        name="resolve_gate",
        description="Resolve a memory gate (approve or reject).",
        inputSchema={"type": "object", "properties": {"memory_id": {"type": "integer"}, "approved": {"type": "boolean", "default": True}}}
    ),
    Tool(
        name="compute_workspace_fingerprint",
        description="Compute a fingerprint of the current workspace for change detection.",
        inputSchema={"type": "object", "properties": {}}
    ),
]


if MCP_AVAILABLE:
    @app.list_tools()
    async def list_tools() -> List[Tool]:
        """List available Memster tools."""
        return TOOL_DEFINITIONS


# === Tool Handlers ===

async def handle_query(args: Dict[str, Any]) -> List[TextContent]:
    """Handle memster_query tool."""
    query = args.get("query", "").strip()
    network_type = args.get("network_type")
    date_from = args.get("date_from")
    date_to = args.get("date_to")
    limit = args.get("limit", 20)

    conn = get_db_connection()
    cursor = conn.cursor()

    params: List[Any] = []
    where_clauses: List[str] = []

    if query:
        where_clauses.append("memories_fts MATCH ?")
        params.append(query)

    if network_type:
        where_clauses.append("m.network_type = ?")
        params.append(network_type)

    if date_from:
        where_clauses.append("m.t_event >= ?")
        params.append(date_from)

    if date_to:
        where_clauses.append("m.t_event <= ?")
        params.append(date_to)

    if query:
        sql = """
        SELECT m.*, rank
        FROM memories m
        JOIN memories_fts fts ON m.id = fts.rowid
        WHERE {where_clause}
        ORDER BY rank, m.t_event DESC
        LIMIT ?
        """.format(where_clause=" AND ".join(where_clauses))
    else:
        sql = """
        SELECT m.*, 0 as rank
        FROM memories m
        WHERE {where_clause}
        ORDER BY m.t_event DESC
        LIMIT ?
        """.format(where_clause=" AND ".join(where_clauses) if where_clauses else "1=1")

    params.append(limit)

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    memories = []
    for row in rows:
        memories.append({
            "id": row["id"],
            "content": row["content"],
            "network_type": row["network_type"],
            "t_event": row["t_event"],
            "t_recorded": row["t_recorded"],
            "source": row["source"],
            "conversation_id": row["conversation_id"],
            "rank": row["rank"] if query else None
        })

    result = {"count": len(memories), "memories": memories}
    logger.info(f"Query returned {len(memories)} results")
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_remember(args: Dict[str, Any]) -> List[TextContent]:
    """Handle memster_remember tool with completeness checking, dedup, and temporal detection."""
    content = args.get("content", "").strip()
    network_type = args.get("network_type")
    t_event = args.get("t_event") or datetime.now().isoformat()
    source = args.get("source", "mcp")
    conversation_id = args.get("conversation_id")
    embedding = args.get("embedding")

    if not content:
        raise ValueError("Content is required")
    if network_type not in ("world", "experience", "opinion", "observation"):
        raise ValueError("Invalid network_type")

    # === ENHANCEMENT 1: check for duplicate before insert ===
    duplicate_id = check_duplicate_before_insert(content, network_type)
    if duplicate_id:
        logger.info(f"Duplicate memory detected: {duplicate_id}")
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "duplicate": True,
            "existing_memory_id": duplicate_id,
            "message": "similar memory already exists"
        }, indent=2))]

    # === ENHANCEMENT 2: check completeness and warn if incomplete ===
    completeness = check_memory_completeness(content, network_type)
    warnings = []
    if not completeness["is_complete"]:
        if completeness["suggestions"]:
            warnings = completeness["suggestions"]
            logger.debug(f"Memory completeness warnings: {warnings}")

    # === ENHANCEMENT 3: detect temporal bounds ===
    temporal_bounds = detect_temporal_bounds(content)

    # === ENHANCEMENT 4: get current fronter and tag memory ===
    fronter = get_current_fronter()
    fronter_uid = args.get("fronter_uid") or fronter.get("uid")
    fronter_name = args.get("fronter_name") or fronter.get("name")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO memories (content, network_type, t_event, source, conversation_id, embedding,
                             fronter_uid, fronter_name, valid_from, valid_to)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (content, network_type, t_event, source, conversation_id, embedding,
          fronter_uid, fronter_name, temporal_bounds.get("valid_from"), temporal_bounds.get("valid_to")))

    memory_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # Auto-embed with NVIDIA NIM
    if EMBEDDINGS_AVAILABLE and not embedding:
        try:
            auto_embed_on_insert(memory_id, content, DB_PATH)
        except Exception as e:
            logger.debug(f"auto-embed failed for memory {memory_id}: {e}")

    logger.info(f"Stored memory {memory_id} in {network_type} network" +
                (f" (fronter: {fronter_name})" if fronter_name else ""))

    result = {
        "success": True,
        "id": memory_id,
        "network_type": network_type,
        "fronter": fronter_name,
        "temporal_bounds": temporal_bounds,
    }
    if warnings:
        result["completeness_warnings"] = warnings
        result["completeness_score"] = completeness["score"]

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_sync_pieces(args: Dict[str, Any]) -> List[TextContent]:
    """Handle memster_sync_pieces tool."""
    limit = args.get("limit", 50)

    request_body = json.dumps({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "get_snippets",
            "arguments": {"limit": limit}
        },
        "id": 1
    }).encode("utf-8")

    try:
        req = request.Request(
            PIECES_MCP_URL,
            data=request_body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            method="POST"
        )

        with request.urlopen(req, timeout=10) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))
    except error.URLError as e:
        logger.error(f"Failed to connect to Pieces MCP: {e}")
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "error": f"Pieces MCP connection failed: {e}"
        }, indent=2))]
    except Exception as e:
        logger.error(f"Pieces sync error: {e}")
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2))]

    snippets = []
    if response_data and "result" in response_data:
        result = response_data["result"]
        if isinstance(result, list):
            snippets = result
        elif isinstance(result, dict) and "content" in result:
            snippets = [result]

    conn = get_db_connection()
    cursor = conn.cursor()
    ingested_count = 0

    for snippet in snippets:
        if isinstance(snippet, dict):
            content = snippet.get("content", snippet.get("text", "") or snippet.get("name", ""))
            source = snippet.get("source", "pieces")
            created = snippet.get("created") or datetime.now().isoformat()
        else:
            content = str(snippet)
            source = "pieces"
            created = datetime.now().isoformat()

        if not content or len(content.strip()) < 10:
            continue

        cursor.execute("""
            INSERT INTO memories (content, network_type, t_event, source, conversation_id)
            VALUES (?, 'observation', ?, ?, ?)
        """, (content.strip(), created, source, "pieces_sync"))
        ingested_count += 1

    conn.commit()
    conn.close()

    logger.info(f"Synced {ingested_count} snippets from Pieces MCP")

    result = {
        "success": True,
        "ingested": ingested_count,
        "total_available": len(snippets)
    }
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


# === Main Call Tool Handler ===

if MCP_AVAILABLE:
    @app.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls."""
        # === Activity tools ===
        if name == "get_activity_summary":
            return [TextContent(type="text", text=json.dumps({"summary": get_activity_summary_text(hours=arguments.get("hours", 2))}, indent=2))]

        elif name == "get_recent_activity":
            return [TextContent(type="text", text=json.dumps(get_recent_activity(hours=arguments.get("hours", 1)), indent=2))]

        elif name == "search_window_history":
            return [TextContent(type="text", text=json.dumps(search_window_history(
                query=arguments.get("query", ""), hours=arguments.get("hours", 24)), indent=2))]

        elif name == "search_clipboard":
            return [TextContent(type="text", text=json.dumps(search_clipboard(
                query=arguments.get("query", ""), hours=arguments.get("hours", 24)), indent=2))]

        elif name == "search_ocr_text":
            return [TextContent(type="text", text=json.dumps(search_ocr_text(
                query=arguments.get("query", ""), limit=arguments.get("limit", 10)), indent=2))]

        # === Core Memster tools ===
        elif name == "memster_query":
            return await handle_query(arguments)

        elif name == "memster_remember":
            return await handle_remember(arguments)

        elif name == "memster_sync_pieces":
            return await handle_sync_pieces(arguments)

        # === Memory CRUD ===
        elif name == "query_memories":
            results = get_memories_with_scoring(
                query_text=arguments.get("query_text"),
                tier=arguments.get("tier"),
                category=arguments.get("category"),
                max_results=arguments.get("max_results", 10)
            )
            return [TextContent(type="text", text=json.dumps({"count": len(results), "memories": results}, indent=2))]

        elif name == "remember_memory":
            content = arguments.get("content", "").strip()
            category = arguments.get("category", "general")
            tags = arguments.get("tags", [])

            if not content:
                raise ValueError("content required")

            # dedup check
            existing_id = check_duplicate_before_insert(content, category)
            if existing_id:
                return [TextContent(type="text", text=json.dumps({
                    "duplicate": True, "existing_id": existing_id,
                    "message": "similar memory already exists"
                }, indent=2))]

            # insert
            conn = get_db_connection()
            now = datetime.now().isoformat()
            cursor = conn.cursor()
            network_type = category if category in ("world", "experience", "opinion", "observation") else "observation"
            cursor.execute(
            "INSERT INTO memories (content, category, network_type, t_event, t_recorded, tier, memory_type) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (content, category, network_type, now, now, "L2", network_type)
            )
            mid = cursor.lastrowid
            conn.commit()
            conn.close()

            hooks = run_post_insert_hooks(mid, content, category, tags)

            # Auto-embed with NVIDIA NIM (async-friendly, won't block on failure)
            if EMBEDDINGS_AVAILABLE:
                try:
                    auto_embed_on_insert(mid, content, DB_PATH)
                except Exception as e:
                    logger.debug(f"auto-embed failed for memory {mid}: {e}")

            return [TextContent(type="text", text=json.dumps({
                "created": True, "id": mid, "category": category,
                "entities_extracted": hooks["entities"],
                "edges_created": hooks["edges"],
                "embedded": EMBEDDINGS_AVAILABLE
            }, indent=2))]

        elif name == "update_memory":
            memory_id = arguments.get("memory_id")
            if not memory_id:
                raise ValueError("memory_id required")

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            existing = cursor.fetchone()
            if not existing:
                conn.close()
                return [TextContent(type="text", text=json.dumps({"error": f"memory {memory_id} not found"}, indent=2))]

            updates, params = [], []
            new_content = arguments.get("content")
            if new_content:
                updates.append("content = ?")
                params.append(new_content.strip())
            new_category = arguments.get("category")
            if new_category:
                updates.append("category = ?")
                params.append(new_category)
            new_tier = arguments.get("tier")
            if new_tier:
                updates.append("tier = ?")
                params.append(new_tier)

            if not updates:
                conn.close()
                return [TextContent(type="text", text=json.dumps({"error": "no fields to update"}, indent=2))]

            updates.append("t_recorded = ?")
            params.append(datetime.now().isoformat())
            params.append(memory_id)

            cursor.execute(f"UPDATE memories SET {', '.join(updates)} WHERE id = ?", params)
            conn.commit()
            conn.close()

            # Re-embed if content changed
            if new_content and EMBEDDINGS_AVAILABLE:
                try:
                    auto_embed_on_insert(memory_id, new_content.strip(), DB_PATH)
                except Exception as e:
                    logger.debug(f"re-embed failed for memory {memory_id}: {e}")

            return [TextContent(type="text", text=json.dumps({
                "updated": True, "id": memory_id,
                "fields_updated": [k.replace(" = ?", "") for k in updates if k != "t_recorded = ?"]
            }, indent=2))]

        elif name == "merge_memories":
            primary_id = arguments.get("primary_id")
            secondary_id = arguments.get("secondary_id")
            merged_content = arguments.get("merged_content")

            if not primary_id or not secondary_id:
                raise ValueError("both primary_id and secondary_id required")

            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM memories WHERE id = ?", (primary_id,))
            primary = cursor.fetchone()
            cursor.execute("SELECT * FROM memories WHERE id = ?", (secondary_id,))
            secondary = cursor.fetchone()

            if not primary or not secondary:
                conn.close()
                return [TextContent(type="text", text=json.dumps({"error": "one or both memories not found"}, indent=2))]

            # use provided merged content or combine
            if not merged_content:
                merged_content = primary["content"]
                if secondary["content"] not in primary["content"]:
                    merged_content += " | " + secondary["content"]

            # update primary
            cursor.execute("UPDATE memories SET content = ?, t_recorded = ? WHERE id = ?",
                (merged_content, datetime.now().isoformat(), primary_id))

            # re-link edges from secondary to primary
            cursor.execute("UPDATE memory_edges SET source_memory_id = ? WHERE source_memory_id = ?", (primary_id, secondary_id))
            cursor.execute("UPDATE memory_edges SET target_memory_id = ? WHERE target_memory_id = ?", (primary_id, secondary_id))

            # delete secondary
            cursor.execute("DELETE FROM memories WHERE id = ?", (secondary_id,))

            conn.commit()
            conn.close()

            return [TextContent(type="text", text=json.dumps({
                "merged": True,
                "primary_id": primary_id,
                "deleted_id": secondary_id,
                "merged_content": merged_content[:200]
            }, indent=2))]

        elif name == "delete_memory":
            memory_id = arguments.get("memory_id")
            if not memory_id:
                raise ValueError("memory_id required")
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, content FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                return [TextContent(type="text", text=json.dumps({"error": f"memory {memory_id} not found"}, indent=2))]
            deleted_content = row["content"]
            cursor.execute("DELETE FROM memory_edges WHERE source_memory_id = ? OR target_memory_id = ?", (memory_id, memory_id))
            cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            conn.close()
            return [TextContent(type="text", text=json.dumps({"deleted": True, "id": memory_id, "content_preview": deleted_content[:100]}, indent=2))]

        elif name == "delete_by_query":
            query_text = arguments.get("query_text", "").strip()
            exact = arguments.get("exact_match", False)
            if not query_text:
                raise ValueError("query_text required")
            conn = get_db_connection()
            cursor = conn.cursor()
            if exact:
                cursor.execute("SELECT id FROM memories WHERE content = ?", (query_text,))
            else:
                cursor.execute("SELECT id FROM memories WHERE content LIKE ?", ("%" + query_text + "%",))
            ids = [r["id"] for r in cursor.fetchall()]
            for mid in ids:
                cursor.execute("DELETE FROM memory_edges WHERE source_memory_id = ? OR target_memory_id = ?", (mid, mid))
                cursor.execute("DELETE FROM memories WHERE id = ?", (mid,))
            conn.commit()
            conn.close()
            return [TextContent(type="text", text=json.dumps({"deleted": len(ids), "ids": ids}, indent=2))]

        elif name == "find_duplicates":
            groups = find_near_duplicate_memories(arguments.get("similarity_threshold", 0.8))
            result = [{"count": len(g), "memories": [{"id": m["id"], "content": m["content"][:100], "category": m.get("category")} for m in g]} for g in groups]
            return [TextContent(type="text", text=json.dumps({"duplicate_groups": len(result), "groups": result}, indent=2))]



        # === Semantic search ===
        elif name == "semantic_memory_search":
            try:
                if EMBEDDINGS_AVAILABLE:
                    # Use real vector search
                    sem_results = vector_search(arguments.get("query"), limit=arguments.get("limit", 5), threshold=0.1, db_path=DB_PATH)
                    # Enrich with memory content
                    if sem_results:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        ids = [r["id"] for r in sem_results]
                        placeholders = ",".join("?" * len(ids))
                        cursor.execute(f"SELECT id, content, category, tier, importance, decay_score, access_count FROM memories WHERE id IN ({placeholders})", ids)
                        mem_map = {r["id"]: dict(r) for r in cursor.fetchall()}
                        conn.close()
                        enriched = []
                        for r in sem_results:
                            mem = mem_map.get(r["id"], {})
                            enriched.append({**mem, "similarity": r["similarity"], "id": r["id"]})
                            return [TextContent(type="text", text=json.dumps({"query": arguments.get("query"), "count": len(enriched), "results": enriched, "embedding_model": "nvidia/llama-3.2-nv-embedqa-1b-v2"}, indent=2))]
                            # Fallback to text search
                            results = get_memories_with_scoring(query_text=arguments.get("query"), max_results=arguments.get("limit", 5))
                            return [TextContent(type="text", text=json.dumps({"query": arguments.get("query"), "count": len(results), "results": results, "note": "using text search fallback (embeddings module not loaded)"}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "hybrid_search":
            try:
                results = hybrid_search(arguments.get("query", ""), limit=arguments.get("limit", 10))
                result_dict = {"query": arguments.get("query"), "count": len(results), "results": results}
                if EMBEDDINGS_AVAILABLE:
                    result_dict["embedding_model"] = "nvidia/llama-3.2-nv-embedqa-1b-v2"
                    return [TextContent(type="text", text=json.dumps(result_dict, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "remember_batch":
            try:
                mems = arguments.get("memories", [])
                result = remember_batch(mems)
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "sleep_consolidate":
            try:
                result = sleep_consolidate()
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "backfill_embeddings":
            try:
                if EMBEDDINGS_AVAILABLE:
                    result = backfill_embeddings(DB_PATH, batch_size=arguments.get("batch_size", 10), delay=arguments.get("delay", 1.0))
                    return [TextContent(type="text", text=json.dumps(result, indent=2))]
                else:
                    return [TextContent(type="text", text=json.dumps({"error": "NVIDIA NIM embeddings not available — check API key"}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "find_similar":
            try:
                memory_id = arguments.get("memory_id")
                if not memory_id:
                    return [TextContent(type="text", text=json.dumps({"error": "memory_id required"}, indent=2))]
                    # Try vector-based similarity first
                    if EMBEDDINGS_AVAILABLE:
                        src_emb = get_embedding(memory_id, DB_PATH)
                        if src_emb:
                            all_embeds = __import__("nvidia_nim_embeddings").get_all_embeddings(DB_PATH)
                            scored = []
                            for mid, emb_vec in all_embeds.items():
                                if mid == memory_id:
                                    continue
                                    sim = cosine_similarity(src_emb, emb_vec)
                                    if sim > 0.1:
                                        scored.append({"id": mid, "similarity": round(sim, 4)})
                                        scored.sort(key=lambda x: x["similarity"], reverse=True)
                                        # Enrich with content
                                        if scored:
                                            conn = get_db_connection()
                                            c = conn.cursor()
                                            ids = [s["id"] for s in scored[:arguments.get("limit", 5)]]
                                            placeholders = ",".join("?" * len(ids))
                                            c.execute(f"SELECT id, content, category FROM memories WHERE id IN ({placeholders})", ids)
                                            mem_map = {r["id"]: dict(r) for r in c.fetchall()}
                                            conn.close()
                                            enriched = [{**mem_map.get(s["id"], {}), "id": s["id"], "similarity": s["similarity"]} for s in scored[:arguments.get("limit", 5)]]
                                            return [TextContent(type="text", text=json.dumps({"memory_id": memory_id, "similar": enriched, "method": "vector_similarity"}, indent=2))]
                                            # Fallback: find memories with same category
                                            conn = get_db_connection()
                                            c = conn.cursor()
                                            c.execute("SELECT category FROM memories WHERE id = ?", (memory_id,))
                                            row = c.fetchone()
                                            if row and row["category"]:
                                                c.execute("SELECT id, content, category FROM memories WHERE category = ? AND id != ? LIMIT ?",
                                                (row["category"], memory_id, arguments.get("limit", 5)))
                                                similar = [dict(r) for r in c.fetchall()]
                                            else:
                                                similar = []
                                                conn.close()
                                                return [TextContent(type="text", text=json.dumps({"memory_id": memory_id, "similar": similar, "note": "using category-based fallback"}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "get_related_memories":
            try:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("""SELECT m.id, m.content, m.category, me.weight
                FROM memory_edges me
                JOIN memories m ON me.target_memory_id = m.id
                WHERE me.source_memory_id = ?
                ORDER BY me.weight DESC LIMIT ?""",
                (arguments.get("memory_id"), arguments.get("limit", 10)))
                related = [{"id": r["id"], "content": r["content"], "category": r["category"], "weight": r["weight"]} for r in c.fetchall()]
                conn.close()
                return [TextContent(type="text", text=json.dumps({"memory_id": arguments.get("memory_id"), "related": related}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "link_memories":
            try:
                source_id = arguments.get("source_id")
                target_id = arguments.get("target_id")
                link_type = arguments.get("link_type", "related")
                if not source_id or not target_id:
                    return [TextContent(type="text", text=json.dumps({"error": "source_id and target_id required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("INSERT OR REPLACE INTO memory_edges (source_memory_id, target_memory_id, link_type, weight) VALUES (?, ?, ?, 1.0)",
                    (source_id, target_id, link_type))
                    c.execute("INSERT OR REPLACE INTO memory_edges (source_memory_id, target_memory_id, link_type, weight) VALUES (?, ?, ?, 1.0)",
                    (target_id, source_id, link_type))
                    conn.commit()
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"linked": True, "source": source_id, "target": target_id, "type": link_type}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "merge_memories":
            try:
                primary_id = arguments.get("primary_id")
                secondary_id = arguments.get("secondary_id")
                merged_content = arguments.get("merged_content", "")
                if not primary_id or not secondary_id:
                    return [TextContent(type="text", text=json.dumps({"error": "primary_id and secondary_id required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("SELECT content, category FROM memories WHERE id = ?", (primary_id,))
                    primary = c.fetchone()
                    c.execute("SELECT content, category FROM memories WHERE id = ?", (secondary_id,))
                    secondary = c.fetchone()
                    if not primary or not secondary:
                        conn.close()
                        return [TextContent(type="text", text=json.dumps({"error": "one or both memories not found"}, indent=2))]
                        if not merged_content:
                            merged_content = primary["content"] + "\n\n" + secondary["content"]
                            c.execute("UPDATE memories SET content = ? WHERE id = ?", (merged_content, primary_id))
                            c.execute("DELETE FROM memories WHERE id = ?", (secondary_id,))
                            c.execute("UPDATE memory_edges SET source_memory_id = ? WHERE source_memory_id = ?", (primary_id, secondary_id))
                            c.execute("UPDATE memory_edges SET target_memory_id = ? WHERE target_memory_id = ?", (primary_id, secondary_id))
                            conn.commit()
                            conn.close()
                            return [TextContent(type="text", text=json.dumps({"merged": True, "primary": primary_id, "absorbed": secondary_id}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "extract_graph_relations":
            try:
                memory_id = arguments.get("memory_id")
                content = arguments.get("content", "")
                if not content:
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("SELECT content FROM memories WHERE id = ?", (memory_id,))
                    row = c.fetchone()
                    conn.close()
                    if row:
                        content = row["content"]
                    else:
                        return [TextContent(type="text", text=json.dumps({"error": "memory not found and no content provided"}, indent=2))]
                        result = extract_graph_relations(memory_id, content)
                        return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "build_graph_edges":
            try:
                if V4_AVAILABLE:
                    result = build_graph_edges_v4()
                else:
                    result = {"edges_rebuilt": 0, "note": "v4 features not loaded"}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "query_by_entity":
            try:
                entity_name = arguments.get("entity_name")
                entity_type = arguments.get("entity_type")
                if not entity_name:
                    return [TextContent(type="text", text=json.dumps({"error": "entity_name required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    if entity_type:
                        c.execute("SELECT DISTINCT m.id, m.content, m.category, me.entity_type FROM memory_entities me JOIN memories m ON me.memory_id = m.id WHERE me.entity_name LIKE ? AND me.entity_type = ? ORDER BY m.importance DESC LIMIT ?",
                        (f"%{entity_name}%", entity_type, arguments.get("max_results", 10)))
                    else:
                        c.execute("SELECT DISTINCT m.id, m.content, m.category, me.entity_type FROM memory_entities me JOIN memories m ON me.memory_id = m.id WHERE me.entity_name LIKE ? ORDER BY m.importance DESC LIMIT ?",
                        (f"%{entity_name}%", arguments.get("max_results", 10)))
                        results = [{"id": r["id"], "content": r["content"][:200], "category": r["category"], "entity_type": r["entity_type"]} for r in c.fetchall()]
                        conn.close()
                        return [TextContent(type="text", text=json.dumps({"entity": entity_name, "type": entity_type, "count": len(results), "results": results}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "check_proactive":
            try:
                context = arguments.get("context", "")
                if V4_AVAILABLE:
                    result = check_proactive_v4(context, max_suggestions=arguments.get("max_suggestions", 3))
                else:
                    # Fallback: simple keyword search
                    results = get_memories_with_scoring(query_text=context, max_results=arguments.get("max_suggestions", 3))
                    result = {"context": context, "suggestions": results, "note": "basic fallback (no proactive scoring)"}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "enrich_memory_lookup":
            try:
                content = arguments.get("content", "")
                category = arguments.get("category", "")
                result = {"content_preview": content[:200], "category": category, "entities_found": [], "related_memories": [], "related_wiki": []}
                # Extract potential entities from content (basic regex)
                entities = set()
                for pattern, etype in [(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'ip'),
                (r'/[a-zA-Z][^\s,]*', 'path'),
                (r'\.(250|233)\b', 'host'),
                (r'signal|discord|telegram|whatsapp', 'platform')]:
                    for m in re.finditer(pattern, content):
                        entities.add((m.group(), etype))
                        result["entities_found"] = [{"name": e[0], "type": e[1]} for e in entities]
                        # Search for related memories
                        conn = get_db_connection()
                        c = conn.cursor()
                        for ename, _ in entities:
                            c.execute("SELECT id, content FROM memories WHERE content LIKE ? LIMIT 3", (f"%{ename}%",))
                            for row in c.fetchall():
                                result["related_memories"].append({"id": row["id"], "content": row["content"][:100]})
                                conn.close()
                                # Search wiki
                                try:
                                    for ename, _ in entities:
                                        wiki_results = wiki_search_func(ename, limit=2)
                                        for wr in wiki_results:
                                            result["related_wiki"].append({"slug": wr.get("slug", ""), "title": wr.get("title", "")})
                                except Exception:
                                    pass
                                    result["note"] = "basic entity extraction (full enrich requires embeddings module)"
                                    return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "check_memory_quality":
            # Replaced with new completeness checker
            try:
                content = arguments.get("content", "")
                category = arguments.get("category", "")
                completeness = check_memory_completeness(content, category or "observation")
                quality = "high" if completeness["score"] >= 4 else "medium" if completeness["score"] >= 2 else "low"
                return [TextContent(type="text", text=json.dumps({
                    "quality": quality,
                    "score": completeness["score"],
                    "is_complete": completeness["is_complete"],
                    "issues": completeness["issues"],
                    "suggestions": completeness["suggestions"],
                    "found_entities": completeness["found_entities"],
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "check_memory_completeness":
            # New enhanced completeness check
            try:
                content = arguments.get("content", "")
                category = arguments.get("category", "observation")
                completeness = check_memory_completeness(content, category)
                return [TextContent(type="text", text=json.dumps(completeness, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "get_cross_session_context":
            # Find past sessions related to a topic
            try:
                topic = arguments.get("topic", "")
                limit = arguments.get("limit", 5)
                hours = arguments.get("hours", 365)
                result = get_cross_session_context(topic, limit, hours)
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "track_retrieval_telemetry":
            try:
                memory_id = arguments.get("memory_id")
                query_text = arguments.get("query_text", "")
                match_served = arguments.get("match_served", False)
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("UPDATE memories SET access_count = access_count + 1 WHERE id = ?", (memory_id,))
                conn.commit()
                conn.close()
                return [TextContent(type="text", text=json.dumps({"tracked": True, "memory_id": memory_id}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "score_memory_confidence":
            try:
                memory_id = arguments.get("memory_id")
                if not memory_id:
                    return [TextContent(type="text", text=json.dumps({"error": "memory_id required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("SELECT id, content, category, importance, decay_score, access_count, created_at FROM memories WHERE id = ?", (memory_id,))
                    row = c.fetchone()
                    if not row:
                        conn.close()
                        return [TextContent(type="text", text=json.dumps({"error": "memory not found"}, indent=2))]
                        # Multi-signal confidence
                        confidence = 0.5
                        # Importance signal
                        if row["importance"] and row["importance"] > 0.5:
                            confidence += 0.1
                            # Access frequency signal
                            if row["access_count"] and row["access_count"] > 3:
                                confidence += 0.1
                                # Decay signal
                                if row["decay_score"] and row["decay_score"] < 0.3:
                                    confidence += 0.1
                                    # Specificity signal
                                    content = row["content"] or ""
                                    if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', content):
                                        confidence += 0.1
                                        if re.search(r'/[a-zA-Z]', content):
                                            confidence += 0.05
                                            confidence = min(confidence, 1.0)
                                            conn.close()
                                            return [TextContent(type="text", text=json.dumps({"memory_id": memory_id, "confidence": round(confidence, 3), "signals": {"importance": row["importance"], "access_count": row["access_count"], "decay_score": row["decay_score"]}}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "record_retrieval_telemetry":
            try:
                memory_id = arguments.get("memory_id")
                query_text = arguments.get("query_text", "")
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("UPDATE memories SET access_count = access_count + 1 WHERE id = ?", (memory_id,))
                conn.commit()
                conn.close()
                return [TextContent(type="text", text=json.dumps({"recorded": True}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "route_retrieval":
            try:
                query = arguments.get("query", "")
                session_id = arguments.get("session_id", "")
                conversation_id = arguments.get("conversation_id", "")
                result = {"query": query, "strategy": "hybrid", "reason": "default routing"}
                if len(query.split()) <= 2:
                    result["strategy"] = "keyword"
                    result["reason"] = "short query favors keyword search"
                elif any(kw in query.lower() for kw in ["how", "why", "what if", "explain"]):
                    result["strategy"] = "semantic"
                    result["reason"] = "conceptual query favors semantic search"
                    return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "infer_follow_up_intent":
            try:
                session_id = arguments.get("session_id", "")
                conversation_id = arguments.get("conversation_id", "")
                query = arguments.get("query", "")
                result = {"is_follow_up": False, "confidence": 0.0, "reason": "no prior context match"}
                # Simple heuristic: pronouns and references suggest follow-up
                follow_indicators = ["that", "this", "it", "the same", "earlier", "before", "previous", "again"]
                if any(ind in query.lower() for ind in follow_indicators):
                    result = {"is_follow_up": True, "confidence": 0.6, "reason": "contains referential language"}
                    return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "get_working_memory_context":
            try:
                session_id = arguments.get("session_id", "")
                conversation_id = arguments.get("conversation_id", "")
                limit = arguments.get("limit", 10)
                # Get recent memories as working context
                results = get_memories_with_scoring(max_results=limit)
                return [TextContent(type="text", text=json.dumps({"session_id": session_id, "context_count": len(results), "memories": results}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "update_memory":
            try:
                memory_id = arguments.get("memory_id")
                if not memory_id:
                    return [TextContent(type="text", text=json.dumps({"error": "memory_id required"}, indent=2))]
                    new_content = arguments.get("content")
                    new_category = arguments.get("category")
                    new_tier = arguments.get("tier")
                    updates = []
                    params = []
                    if new_content:
                        updates.append("content = ?")
                        params.append(new_content)
                        if new_category:
                            updates.append("category = ?")
                            params.append(new_category)
                            if new_tier:
                                updates.append("tier = ?")
                                params.append(new_tier)
                                if not updates:
                                    return [TextContent(type="text", text=json.dumps({"error": "no fields to update"}, indent=2))]
                                    params.append(memory_id)
                                    conn = get_db_connection()
                                    cursor = conn.cursor()
                                    cursor.execute(f"UPDATE memories SET {', '.join(updates)} WHERE id = ?", params)
                                    conn.commit()
                                    conn.close()
                                    # Re-embed if content changed
                                    if new_content and EMBEDDINGS_AVAILABLE:
                                        try:
                                            auto_embed_on_insert(memory_id, new_content, DB_PATH)
                                        except Exception:
                                            pass
                                            return [TextContent(type="text", text=json.dumps({"updated": True, "id": memory_id, "fields": [u.split()[0] for u in updates]}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "delete_memory":
            try:
                memory_id = arguments.get("memory_id")
                if not memory_id:
                    return [TextContent(type="text", text=json.dumps({"error": "memory_id required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                    c.execute("DELETE FROM memory_entities WHERE memory_id = ?", (memory_id,))
                    c.execute("DELETE FROM memory_edges WHERE source_memory_id = ? OR target_memory_id = ?", (memory_id, memory_id))
                    c.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,))
                    conn.commit()
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"deleted": True, "id": memory_id}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "delete_by_query":
            try:
                query_text = arguments.get("query_text")
                exact = arguments.get("exact_match", False)
                conn = get_db_connection()
                c = conn.cursor()
                if exact:
                    c.execute("SELECT id FROM memories WHERE content = ?", (query_text,))
                else:
                    c.execute("SELECT id FROM memories WHERE content LIKE ?", (f"%{query_text}%",))
                    ids = [r["id"] for r in c.fetchall()]
                    for mid in ids:
                        c.execute("DELETE FROM memories WHERE id = ?", (mid,))
                        c.execute("DELETE FROM memory_entities WHERE memory_id = ?", (mid,))
                        c.execute("DELETE FROM memory_edges WHERE source_memory_id = ? OR target_memory_id = ?", (mid, mid))
                        c.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (mid,))
                        conn.commit()
                        conn.close()
                        return [TextContent(type="text", text=json.dumps({"deleted_count": len(ids), "ids": ids}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "filter_passive_capture":
            try:
                dry_run = arguments.get("dry_run", True)
                if V4_AVAILABLE:
                    result = filter_passive_capture_v4(dry_run=dry_run)
                else:
                    # Fallback: basic noise detection (clip/ocr/screenshot memories)
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("SELECT id, content, source FROM memories WHERE source IN ('clipboard', 'ocr', 'screenshot') ORDER BY id DESC LIMIT 50")
                    rows = [dict(r) for r in c.fetchall()]
                    conn.close()
                    if dry_run:
                        result = {"would_delete": len(rows), "sample_ids": [r["id"] for r in rows[:10]], "dry_run": True, "note": "basic fallback"}
                    else:
                        for r in rows:
                            delete_memory_by_id(r["id"])
                        result = {"deleted": len(rows), "note": "basic fallback"}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "find_duplicates":
            try:
                threshold = arguments.get("similarity_threshold", 0.8)
                if V4_AVAILABLE:
                    result = find_duplicates_v4(threshold=threshold)
                else:
                    dupes = find_near_duplicate_memories(similarity_threshold=threshold)
                    result = {"duplicates": dupes, "count": len(dupes), "note": "using basic near-duplicate fallback"}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "detect_stale_memories":
            try:
                if V4_AVAILABLE:
                    result = detect_stale_memories_v4()
                else:
                    # Fallback: find expired valid_to memories
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("SELECT id, content, valid_to FROM memories WHERE valid_to IS NOT NULL AND valid_to < date('now')")
                    rows = [dict(r) for r in c.fetchall()]
                    conn.close()
                    result = {"stale_memories": rows, "count": len(rows), "note": "basic fallback (no semantic staleness)"}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "detect_contradictions":
            try:
                if V4_AVAILABLE:
                    result = detect_contradictions_v4(threshold=arguments.get("threshold", 0.6))
                else:
                    result = {"contradictions": [], "count": 0, "note": "v4 features not loaded"}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "rate_memory":
            try:
                memory_id = arguments.get("memory_id")
                feedback_type = arguments.get("feedback_type")
                context = arguments.get("context", "")
                if not memory_id or not feedback_type:
                    return [TextContent(type="text", text=json.dumps({"error": "memory_id and feedback_type required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    if feedback_type == "promote":
                        c.execute("UPDATE memories SET importance = MIN(importance + 0.2, 1.0) WHERE id = ?", (memory_id,))
                    elif feedback_type == "demote":
                        c.execute("UPDATE memories SET importance = MAX(importance - 0.2, 0.0) WHERE id = ?", (memory_id,))
                    elif feedback_type == "helpful":
                        c.execute("UPDATE memories SET access_count = access_count + 1 WHERE id = ?", (memory_id,))
                    elif feedback_type == "irrelevant" or feedback_type == "wrong" or feedback_type == "outdated":
                        c.execute("UPDATE memories SET decay_score = MIN(decay_score + 0.3, 1.0) WHERE id = ?", (memory_id,))
                        conn.commit()
                        conn.close()
                        return [TextContent(type="text", text=json.dumps({"rated": True, "memory_id": memory_id, "feedback": feedback_type}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "get_memory_feedback_stats":
            try:
                memory_id = arguments.get("memory_id")
                conn = get_db_connection()
                c = conn.cursor()
                if memory_id:
                    c.execute("SELECT id, importance, decay_score, access_count FROM memories WHERE id = ?", (memory_id,))
                    row = c.fetchone()
                    conn.close()
                    if row:
                        return [TextContent(type="text", text=json.dumps({"memory_id": row["id"], "importance": row["importance"], "decay_score": row["decay_score"], "access_count": row["access_count"]}, indent=2))]
                    else:
                        return [TextContent(type="text", text=json.dumps({"error": "memory not found"}, indent=2))]
                else:
                    c.execute("SELECT AVG(importance) as avg_imp, AVG(decay_score) as avg_decay, AVG(access_count) as avg_acc FROM memories")
                    row = c.fetchone()
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"avg_importance": row["avg_imp"] if row else 0, "avg_decay": row["avg_decay"] if row else 0, "avg_access_count": row["avg_acc"] if row else 0}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "create_task":
            try:
                name = arguments.get("name")
                description = arguments.get("description", "")
                if not name:
                    return [TextContent(type="text", text=json.dumps({"error": "name required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("INSERT INTO tasks (name, description, status, priority, release_version) VALUES (?, ?, 'active', 5, ?)",
                    (name, description, arguments.get("release_version", "")))
                    task_id = c.lastrowid
                    conn.commit()
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"created": True, "id": task_id, "name": name}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "complete_task":
            try:
                task_id = arguments.get("task_id")
                if not task_id:
                    return [TextContent(type="text", text=json.dumps({"error": "task_id required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("UPDATE tasks SET status = 'completed', completed_at = CURRENT_TIMESTAMP WHERE id = ?", (task_id,))
                    conn.commit()
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"completed": True, "id": task_id}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "assign_memory_to_task":
            try:
                memory_id = arguments.get("memory_id")
                task_id = arguments.get("task_id")
                if not memory_id or not task_id:
                    return [TextContent(type="text", text=json.dumps({"error": "memory_id and task_id required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("INSERT OR REPLACE INTO task_memory_assignments (task_id, memory_id) VALUES (?, ?)", (task_id, memory_id))
                    conn.commit()
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"assigned": True, "memory_id": memory_id, "task_id": task_id}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "get_task_memories":
            try:
                task_id = arguments.get("task_id")
                if not task_id:
                    return [TextContent(type="text", text=json.dumps({"error": "task_id required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("SELECT m.id, m.content, m.category FROM memories m JOIN task_memory_assignments tma ON m.id = tma.memory_id WHERE tma.task_id = ?", (task_id,))
                    results = [{"id": r["id"], "content": r["content"][:200], "category": r["category"]} for r in c.fetchall()]
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"task_id": task_id, "memories": results}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "get_active_tasks":
            try:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("SELECT id, name, description, status, priority FROM tasks WHERE status != 'completed' ORDER BY priority DESC LIMIT ?", (arguments.get("limit", 50),))
                results = [{"id": r["id"], "name": r["name"], "description": r["description"], "status": r["status"], "priority": r["priority"]} for r in c.fetchall()]
                conn.close()
                return [TextContent(type="text", text=json.dumps({"count": len(results), "tasks": results}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]


        # === Beads-inspired tool handlers ===
        elif name == "add_dependency":
            sid = int(args.get("source_id", 0))
            tid = int(args.get("target_id", 0))
            dt = args.get("dep_type", "related")
            result = add_dependency(DATABASE_PATH, sid, tid, dt)
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "get_ready":
            result = get_ready_memories(DATABASE_PATH, int(args.get("limit", 20)))
            return [TextContent(type="text", text=json.dumps(result, default=str))]

        elif name == "create_wisp":
            result = create_wisp(DATABASE_PATH, args["content"], args.get("category", "observation"), int(args.get("ttl_hours", 24)), args.get("tags", []))
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "squash_wisp":
            result = squash_wisp(DATABASE_PATH, int(args["memory_id"]))
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "burn_wisp":
            result = burn_wisp(DATABASE_PATH, int(args["memory_id"]))
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "gc_wisps":
            result = gc_wisps(DATABASE_PATH, bool(args.get("dry_run", True)))
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "compact_memory_ai":
            result = compact_memory_ai(DATABASE_PATH, int(args["memory_id"]), args.get("model"))
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "set_memory_gate":
            result = set_memory_gate(DATABASE_PATH, int(args["memory_id"]), args.get("gate_type", "confirm"), args.get("approvers", []))
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "resolve_gate":
            result = resolve_gate(DATABASE_PATH, int(args["memory_id"]), bool(args.get("approved", True)))
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "query_audit_log":
            result = query_audit_log(args.get("kind"), int(args.get("since_hours", 24)), int(args.get("limit", 50)))
            return [TextContent(type="text", text=json.dumps(result, default=str))]

        elif name == "get_workspace_fingerprint":
            result = {"fingerprint": compute_workspace_fingerprint()}
            return [TextContent(type="text", text=json.dumps(result))]
        elif name == "compress_memory":
            try:
                memory_id = arguments.get("memory_id")
                if not memory_id:
                    return [TextContent(type="text", text=json.dumps({"error": "memory_id required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("SELECT content FROM memories WHERE id = ?", (memory_id,))
                    row = c.fetchone()
                    if not row:
                        conn.close()
                        return [TextContent(type="text", text=json.dumps({"error": "memory not found"}, indent=2))]
                        import zlib
                        compressed = zlib.compress(row["content"].encode())
                        c.execute("UPDATE memories SET content = ? WHERE id = ?", (compressed.decode('latin-1'), memory_id))
                        conn.commit()
                        conn.close()
                        return [TextContent(type="text", text=json.dumps({"compressed": True, "id": memory_id, "original_size": len(row["content"]), "compressed_size": len(compressed)}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "decompress_memory":
            try:
                memory_id = arguments.get("memory_id")
                if not memory_id:
                    return [TextContent(type="text", text=json.dumps({"error": "memory_id required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("SELECT content FROM memories WHERE id = ?", (memory_id,))
                    row = c.fetchone()
                    if not row:
                        conn.close()
                        return [TextContent(type="text", text=json.dumps({"error": "memory not found"}, indent=2))]
                        import zlib
                        try:
                            decompressed = zlib.decompress(row["content"].encode('latin-1')).decode()
                            c.execute("UPDATE memories SET content = ? WHERE id = ?", (decompressed, memory_id))
                            conn.commit()
                            conn.close()
                            return [TextContent(type="text", text=json.dumps({"decompressed": True, "id": memory_id, "size": len(decompressed)}, indent=2))]
                        except zlib.error:
                            conn.close()
                            return [TextContent(type="text", text=json.dumps({"error": "content is not compressed", "id": memory_id}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "create_palace":
            try:
                palace_name = arguments.get("name")
                description = arguments.get("description", "")
                if not palace_name:
                    return [TextContent(type="text", text=json.dumps({"error": "name required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("INSERT INTO memory_palaces (name, description) VALUES (?, ?)", (palace_name, description))
                    palace_id = c.lastrowid
                    conn.commit()
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"created": True, "id": palace_id, "name": palace_name}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "add_room":
            try:
                palace_id = arguments.get("palace_id")
                room_name = arguments.get("name")
                if not palace_id or not room_name:
                    return [TextContent(type="text", text=json.dumps({"error": "palace_id and name required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("INSERT INTO palace_rooms (palace_id, name, description, position_x, position_y, position_z) VALUES (?, ?, ?, ?, ?, ?)",
                    (palace_id, room_name, arguments.get("description", ""), arguments.get("position_x", 0), arguments.get("position_y", 0), arguments.get("position_z", 0)))
                    room_id = c.lastrowid
                    conn.commit()
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"created": True, "id": room_id, "name": room_name, "palace_id": palace_id}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "place_memory_in_room":
            try:
                room_id = arguments.get("room_id")
                memory_id = arguments.get("memory_id")
                if not room_id or not memory_id:
                    return [TextContent(type="text", text=json.dumps({"error": "room_id and memory_id required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("INSERT OR REPLACE INTO room_memories (room_id, memory_id) VALUES (?, ?)", (room_id, memory_id))
                    conn.commit()
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"placed": True, "room_id": room_id, "memory_id": memory_id}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "walk_palace":
            try:
                palace_id = arguments.get("palace_id")
                if not palace_id:
                    return [TextContent(type="text", text=json.dumps({"error": "palace_id required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("SELECT id, name, description FROM memory_palaces WHERE id = ?", (palace_id,))
                    palace = c.fetchone()
                    if not palace:
                        conn.close()
                        return [TextContent(type="text", text=json.dumps({"error": "palace not found"}, indent=2))]
                        c.execute("SELECT id, name, description, position_x, position_y, position_z FROM palace_rooms WHERE palace_id = ?", (palace_id,))
                        rooms = []
                        for room in c.fetchall():
                            c.execute("SELECT m.id, m.content FROM memories m JOIN room_memories rm ON m.id = rm.memory_id WHERE rm.room_id = ?", (room["id"],))
                            room_mems = [{"id": r["id"], "content": r["content"][:100]} for r in c.fetchall()]
                            rooms.append({"id": room["id"], "name": room["name"], "description": room["description"], "position": {"x": room["position_x"], "y": room["position_y"], "z": room["position_z"]}, "memories": room_mems})
                            conn.close()
                            return [TextContent(type="text", text=json.dumps({"palace": {"id": palace["id"], "name": palace["name"], "description": palace["description"]}, "rooms": rooms}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "create_narrative_arc":
            try:
                title = arguments.get("title")
                if not title:
                    return [TextContent(type="text", text=json.dumps({"error": "title required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("INSERT INTO narrative_arcs (title, description, arc_type, status) VALUES (?, ?, ?, 'ongoing')",
                    (title, arguments.get("description", ""), arguments.get("arc_type", "manual")))
                    arc_id = c.lastrowid
                    conn.commit()
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"created": True, "id": arc_id, "title": title}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "add_memory_to_arc":
            try:
                arc_id = arguments.get("arc_id")
                memory_id = arguments.get("memory_id")
                if not arc_id or not memory_id:
                    return [TextContent(type="text", text=json.dumps({"error": "arc_id and memory_id required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("INSERT INTO arc_memories (arc_id, memory_id, arc_role) VALUES (?, ?, ?)",
                    (arc_id, memory_id, arguments.get("arc_role", "event")))
                    conn.commit()
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"added": True, "arc_id": arc_id, "memory_id": memory_id}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "get_arc_timeline":
            try:
                arc_id = arguments.get("arc_id")
                if not arc_id:
                    return [TextContent(type="text", text=json.dumps({"error": "arc_id required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("SELECT m.id, m.content, m.created_at, am.arc_role FROM memories m JOIN arc_memories am ON m.id = am.memory_id WHERE am.arc_id = ? ORDER BY m.created_at", (arc_id,))
                    results = [{"id": r["id"], "content": r["content"][:200], "created_at": r["created_at"], "arc_role": r["arc_role"]} for r in c.fetchall()]
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"arc_id": arc_id, "timeline": results}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "list_arcs":
            try:
                conn = get_db_connection()
                c = conn.cursor()
                query = "SELECT id, title, description, arc_type, status FROM narrative_arcs"
                params = []
                if arguments.get("status"):
                    query += " WHERE status = ?"
                    params.append(arguments["status"])
                    if arguments.get("arc_type"):
                        query += " WHERE arc_type = ?"
                        params.append(arguments["arc_type"])
                        c.execute(query, params)
                        results = [{"id": r["id"], "title": r["title"], "description": r["description"], "arc_type": r["arc_type"], "status": r["status"]} for r in c.fetchall()]
                        conn.close()
                        return [TextContent(type="text", text=json.dumps({"count": len(results), "arcs": results}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "complete_arc":
            try:
                arc_id = arguments.get("arc_id")
                if not arc_id:
                    return [TextContent(type="text", text=json.dumps({"error": "arc_id required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("UPDATE narrative_arcs SET status = 'completed' WHERE id = ?", (arc_id,))
                    conn.commit()
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"completed": True, "id": arc_id}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "derive_conclusion":
            try:
                content = arguments.get("content")
                memory_ids = arguments.get("memory_ids", [])
                if not content:
                    return [TextContent(type="text", text=json.dumps({"error": "content required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("INSERT INTO conclusions (content, confidence, status, source_memory_ids) VALUES (?, ?, 'pending', ?)",
                    (content, arguments.get("confidence", 0.5), json.dumps(memory_ids)))
                    conclusion_id = c.lastrowid
                    conn.commit()
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"created": True, "id": conclusion_id, "content": content}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "validate_conclusion":
            try:
                conclusion_id = arguments.get("conclusion_id")
                if not conclusion_id:
                    return [TextContent(type="text", text=json.dumps({"error": "conclusion_id required"}, indent=2))]
                    still_valid = arguments.get("still_valid", True)
                    conn = get_db_connection()
                    c = conn.cursor()
                    status = "confirmed" if still_valid else "rejected"
                    c.execute("UPDATE conclusions SET status = ?, notes = ? WHERE id = ?", (status, arguments.get("notes", ""), conclusion_id))
                    conn.commit()
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"validated": True, "id": conclusion_id, "status": status}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "get_conclusions":
            try:
                conn = get_db_connection()
                c = conn.cursor()
                query = "SELECT id, content, confidence, status, created_at FROM conclusions"
                params = []
                if arguments.get("status"):
                    query += " WHERE status = ?"
                    params.append(arguments["status"])
                    query += " ORDER BY created_at DESC LIMIT ?"
                    params.append(arguments.get("limit", 10))
                    c.execute(query, params)
                    results = [{"id": r["id"], "content": r["content"], "confidence": r["confidence"], "status": r["status"], "created_at": r["created_at"]} for r in c.fetchall()]
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"count": len(results), "conclusions": results}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "register_peer":
            try:
                peer_url = arguments.get("peer_url")
                if not peer_url:
                    return [TextContent(type="text", text=json.dumps({"error": "peer_url required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("INSERT INTO federation_peers (peer_url, peer_name, auth_token, sync_direction, status) VALUES (?, ?, ?, ?, 'active')",
                    (peer_url, arguments.get("peer_name", ""), arguments.get("auth_token", ""), arguments.get("sync_direction", "bidirectional")))
                    peer_id = c.lastrowid
                    conn.commit()
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"registered": True, "id": peer_id, "url": peer_url}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "list_peers":
            try:
                conn = get_db_connection()
                c = conn.cursor()
                query = "SELECT id, peer_url, peer_name, sync_direction, status, last_sync FROM federation_peers"
                params = []
                if arguments.get("status"):
                    query += " WHERE status = ?"
                    params.append(arguments["status"])
                    c.execute(query, params)
                    results = [{"id": r["id"], "url": r["peer_url"], "name": r["peer_name"], "direction": r["sync_direction"], "status": r["status"], "last_sync": r["last_sync"]} for r in c.fetchall()]
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"count": len(results), "peers": results}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "sync_with_peer":
            try:
                peer_id = arguments.get("peer_id")
                if not peer_id:
                    return [TextContent(type="text", text=json.dumps({"error": "peer_id required"}, indent=2))]
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("SELECT peer_url, auth_token, sync_direction FROM federation_peers WHERE id = ?", (peer_id,))
                    peer = c.fetchone()
                    conn.close()
                    if not peer:
                        return [TextContent(type="text", text=json.dumps({"error": "peer not found"}, indent=2))]
                        return [TextContent(type="text", text=json.dumps({"sync_result": "stub", "peer_id": peer_id, "direction": arguments.get("direction", peer["sync_direction"])}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "set_temporal_bounds":
            try:
                memory_id = arguments.get("memory_id")
                if not memory_id:
                    return [TextContent(type="text", text=json.dumps({"error": "memory_id required"}, indent=2))]
                conn = get_db_connection()
                c = conn.cursor()
                updates = []
                params = []
                if arguments.get("valid_from"):
                    updates.append("valid_from = ?")
                    params.append(arguments["valid_from"])
                if arguments.get("valid_to"):
                    updates.append("valid_to = ?")
                    params.append(arguments["valid_to"])
                if arguments.get("observed_at"):
                    updates.append("t_event = ?")
                    params.append(arguments["observed_at"])
                if updates:
                    params.append(memory_id)
                    c.execute(f"UPDATE memories SET {', '.join(updates)} WHERE id = ?", params)
                    conn.commit()
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"updated": True, "id": memory_id}, indent=2))]
                else:
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"updated": False, "reason": "no bounds provided"}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "update_temporal_bounds":
            try:
                memory_id = arguments.get("memory_id")
                if not memory_id:
                    return [TextContent(type="text", text=json.dumps({"error": "memory_id required"}, indent=2))]
                conn = get_db_connection()
                c = conn.cursor()
                updates = []
                params = []
                if arguments.get("valid_from"):
                    updates.append("valid_from = ?")
                    params.append(arguments["valid_from"])
                if arguments.get("valid_to"):
                    updates.append("valid_to = ?")
                    params.append(arguments["valid_to"])
                if arguments.get("observed_at"):
                    updates.append("t_event = ?")
                    params.append(arguments["observed_at"])
                if updates:
                    params.append(memory_id)
                    c.execute(f"UPDATE memories SET {', '.join(updates)} WHERE id = ?", params)
                    conn.commit()
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"updated": True, "id": memory_id}, indent=2))]
                else:
                    conn.close()
                    return [TextContent(type="text", text=json.dumps({"updated": False, "reason": "no bounds provided"}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "find_expired_memories":
            try:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("SELECT id, content, valid_to FROM memories WHERE valid_to IS NOT NULL AND valid_to < date('now')")
                results = [{"id": r["id"], "content": r["content"][:100], "valid_to": r["valid_to"]} for r in c.fetchall()]
                conn.close()
                return [TextContent(type="text", text=json.dumps({"count": len(results), "expired": results}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "update_confidence":
            try:
                memory_id = arguments.get("memory_id")
                if memory_id is None:
                    return [TextContent(type="text", text=json.dumps({"error": "memory_id required"}, indent=2))]
                    observation = arguments.get("observation_result", True)
                    conn = get_db_connection()
                    c = conn.cursor()
                    if observation:
                        c.execute("UPDATE memories SET importance = MIN(importance + 0.1, 1.0) WHERE id = ?", (memory_id,))
                    else:
                        c.execute("UPDATE memories SET importance = MAX(importance - 0.2, 0.0), decay_score = MIN(decay_score + 0.15, 1.0) WHERE id = ?", (memory_id,))
                        conn.commit()
                        conn.close()
                        return [TextContent(type="text", text=json.dumps({"updated": True, "id": memory_id, "observation_confirmed": observation}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "get_memory_health":
            try:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("SELECT COUNT(*) as total, AVG(importance) as avg_imp, AVG(decay_score) as avg_decay FROM memories")
                row = c.fetchone()
                c.execute("SELECT COUNT(*) as cnt FROM memory_embeddings")
                emb_count = c.fetchone()["cnt"]
                c.execute("SELECT COUNT(*) as cnt FROM memory_entities")
                entity_count = c.fetchone()["cnt"]
                c.execute("SELECT COUNT(*) as cnt FROM memory_edges")
                edge_count = c.fetchone()["cnt"]
                conn.close()
                return [TextContent(type="text", text=json.dumps({"total_memories": row["total"], "avg_importance": round(row["avg_imp"] or 0, 3), "avg_decay": round(row["avg_decay"] or 0, 3), "embeddings": emb_count, "entities": entity_count, "edges": edge_count}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "get_briefing":
            try:
                result = get_briefing()
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "get_narrative_briefing":
            try:
                result = get_narrative_briefing(token_limit=arguments.get("token_limit", 500))
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "unified_briefing":
            try:
                result = unified_briefing(context=arguments.get("context", ""), max_items=arguments.get("max_items", 15))
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "get_activity_summary":
            try:
                result = get_activity_summary(hours=arguments.get("hours", 2))
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "memory_timeline":
            try:
                result = memory_timeline(days=arguments.get("days", 30))
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "session_wrap":
            try:
                session_notes = arguments.get("session_notes", "")
                conversation_id = arguments.get("conversation_id")
                # Auto-generate digest if no notes provided
                if not session_notes:
                    session_notes = auto_generate_session_digest(
                        session_id=conversation_id,
                        limit=20
                    )
                result = store_session_wrap(
                    session_notes=session_notes,
                    conversation_id=conversation_id
                )
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "track_provenance":
            try:
                memory_id = arguments.get("memory_id")
                if not memory_id:
                    return [TextContent(type="text", text=json.dumps({"error": "memory_id required"}, indent=2))]
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("UPDATE memories SET source = ?, t_event = ? WHERE id = ?",
                          (arguments.get("source", ""), arguments.get("extracted_at", ""), memory_id))
                conn.commit()
                conn.close()
                return [TextContent(type="text", text=json.dumps({"updated": True, "id": memory_id}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "ingest_sp_history":
            try:
                result = ingest_sp_history(days=arguments.get("days", 30))
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "sp_status":
            try:
                result = sp_status()
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "sp_members":
            try:
                result = sp_members(include_archived=arguments.get("include_archived", False))
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "wiki_search":
            try:
                if V4_AVAILABLE:
                    result = wiki_search_v4(arguments.get("query", ""), category=arguments.get("category"), limit=arguments.get("limit", 5))
                else:
                    result = wiki_search_func(arguments.get("query", ""), category=arguments.get("category"), limit=arguments.get("limit", 5))
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "wiki_read":
            try:
                if V4_AVAILABLE:
                    result = wiki_read_v4(arguments.get("slug", ""))
                else:
                    result = wiki_read_func(arguments.get("slug", ""))
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "wiki_list":
            try:
                if V4_AVAILABLE:
                    result = wiki_list_v4(category=arguments.get("category"), limit=arguments.get("limit", 50))
                else:
                    result = wiki_list_func(category=arguments.get("category"), limit=arguments.get("limit", 50))
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "wiki_sweep":
            try:
                if V4_AVAILABLE:
                    result = wiki_sweep_v4(category=arguments.get("category"))
                else:
                    result = wiki_sweep_func(category=arguments.get("category"))
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "wiki_to_memster_sync":
            try:
                if V4_AVAILABLE:
                    result = wiki_to_memster_sync_v4(max_pages=arguments.get("max_pages", 20))
                else:
                    result = {"synced": 0, "note": "v4 features not loaded"}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "assemble_context_packet":
            try:
                if V4_AVAILABLE:
                    result = assemble_context_packet_v4(query=arguments.get("query", ""), max_tokens=arguments.get("max_tokens", 2000), context_type=arguments.get("context_type", "auto"))
                else:
                    results = get_memories_with_scoring(query_text=arguments.get("query", ""), max_results=10)
                    result = {"query": arguments.get("query", ""), "token_budget": arguments.get("max_tokens", 2000), "memories": results, "tokens_used": sum(len(r.get("content", "")) // 4 for r in results), "note": "basic fallback (no diversity injection)"}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "score_memory_confidence":
            try:
                memory_id = arguments.get("memory_id")
                if V4_AVAILABLE:
                    result = score_memory_confidence_v4(memory_id)
                else:
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("SELECT confidence_score FROM memories WHERE id = ?", (memory_id,))
                    row = c.fetchone()
                    conn.close()
                    result = {"memory_id": memory_id, "confidence": row["confidence_score"] if row else None, "note": "basic fallback"}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "compress_memory":
            try:
                memory_id = arguments.get("memory_id")
                if V4_AVAILABLE:
                    result = compress_memory_v4(memory_id)
                else:
                    # Basic zlib compression
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("SELECT content FROM memories WHERE id = ?", (memory_id,))
                    row = c.fetchone()
                    if row:
                        original = row["content"]
                        compressed = zlib.compress(original.encode()).hex()
                        c.execute("UPDATE memories SET content = ? WHERE id = ?", (compressed, memory_id))
                        conn.commit()
                    conn.close()
                    result = {"memory_id": memory_id, "compressed": True, "note": "basic zlib fallback"}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "decompress_memory":
            try:
                memory_id = arguments.get("memory_id")
                if V4_AVAILABLE:
                    result = decompress_memory_v4(memory_id)
                else:
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("SELECT content FROM memories WHERE id = ?", (memory_id,))
                    row = c.fetchone()
                    if row:
                        try:
                            decompressed = zlib.decompress(bytes.fromhex(row["content"])).decode()
                            c.execute("UPDATE memories SET content = ? WHERE id = ?", (decompressed, memory_id))
                            conn.commit()
                            result = {"memory_id": memory_id, "decompressed": True}
                        except (ValueError, zlib.error):
                            result = {"memory_id": memory_id, "decompressed": False, "note": "content not compressed"}
                    conn.close()
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "validate_conclusion":
            try:
                conclusion_id = arguments.get("conclusion_id")
                still_valid = arguments.get("still_valid")
                notes = arguments.get("notes", "")
                if V4_AVAILABLE:
                    result = validate_conclusion_v4(conclusion_id, still_valid, notes)
                else:
                    conn = get_db_connection()
                    c = conn.cursor()
                    status = "confirmed" if still_valid else "rejected"
                    c.execute("UPDATE conclusions SET status = ?, notes = ? WHERE id = ?", (status, notes, conclusion_id))
                    conn.commit()
                    conn.close()
                    result = {"conclusion_id": conclusion_id, "status": status, "note": "basic fallback"}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "rate_memory":
            try:
                memory_id = arguments.get("memory_id")
                feedback_type = arguments.get("feedback_type")
                feedback_context = arguments.get("context", "")
                if V4_AVAILABLE:
                    result = rate_memory_v4(memory_id, feedback_type, feedback_context)
                else:
                    # Basic: adjust importance based on feedback
                    conn = get_db_connection()
                    c = conn.cursor()
                    delta = {"helpful": 0.1, "promote": 0.2, "irrelevant": -0.1, "wrong": -0.2, "outdated": -0.15, "demote": -0.2}.get(feedback_type, 0)
                    c.execute("UPDATE memories SET importance = MAX(0, MIN(1, importance + ?)) WHERE id = ?", (delta, memory_id))
                    conn.commit()
                    c.execute("SELECT importance FROM memories WHERE id = ?", (memory_id,))
                    row = c.fetchone()
                    conn.close()
                    result = {"memory_id": memory_id, "feedback": feedback_type, "new_importance": row["importance"] if row else None, "note": "basic fallback"}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "get_conclusions":
            try:
                if V4_AVAILABLE:
                    result = get_conclusions_v4(status=arguments.get("status"), limit=arguments.get("limit", 10))
                else:
                    conn = get_db_connection()
                    c = conn.cursor()
                    query = "SELECT * FROM conclusions"
                    params = []
                    if arguments.get("status"):
                        query += " WHERE status = ?"
                        params.append(arguments["status"])
                    query += " ORDER BY confidence DESC LIMIT ?"
                    params.append(arguments.get("limit", 10))
                    c.execute(query, params)
                    result = {"conclusions": [dict(r) for r in c.fetchall()]}
                    conn.close()
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "get_memory_feedback_stats":
            try:
                if V4_AVAILABLE:
                    result = get_memory_feedback_stats_v4(memory_id=arguments.get("memory_id"))
                else:
                    result = {"stats": {}, "note": "v4 features not loaded"}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "get_cross_session_context":
            try:
                if V4_AVAILABLE:
                    result = get_cross_session_context_v4(topic=arguments.get("topic"), limit=arguments.get("limit", 5), hours=arguments.get("hours", 365))
                else:
                    # Fallback: search memories for the topic
                    results = get_memories_with_scoring(query_text=arguments.get("topic"), max_results=arguments.get("limit", 5))
                    result = {"topic": arguments.get("topic"), "related_memories": results, "note": "basic fallback (no session context)"}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "get_working_memory_context":
            try:
                if V4_AVAILABLE:
                    result = get_working_memory_context_v4(session_id=arguments.get("session_id"), conversation_id=arguments.get("conversation_id"), limit=arguments.get("limit", 10))
                else:
                    # Fallback: recent memories
                    results = get_memories_with_scoring(query_text="", max_results=arguments.get("limit", 10))
                    result = {"session_id": arguments.get("session_id"), "recent_memories": results, "note": "basic fallback"}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "infer_follow_up_intent":
            try:
                if V4_AVAILABLE:
                    result = infer_follow_up_intent_v4(session_id=arguments.get("session_id"), conversation_id=arguments.get("conversation_id"), query=arguments.get("query"))
                else:
                    result = {"is_follow_up": False, "note": "v4 features not loaded"}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "route_retrieval":
            try:
                if V4_AVAILABLE:
                    result = route_retrieval_v4(query=arguments.get("query"), session_id=arguments.get("session_id"), conversation_id=arguments.get("conversation_id"))
                else:
                    result = {"strategy": "hybrid", "query": arguments.get("query"), "note": "basic fallback"}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "record_retrieval_telemetry":
            try:
                if V4_AVAILABLE:
                    result = record_retrieval_telemetry_v4(memory_id=arguments.get("memory_id"), query_text=arguments.get("query_text"), match_served=arguments.get("match_served", False))
                else:
                    result = {"recorded": False, "note": "v4 features not loaded"}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "set_temporal_bounds":
            try:
                memory_id = arguments.get("memory_id")
                if V4_AVAILABLE:
                    result = set_temporal_bounds_v4(memory_id, valid_from=arguments.get("valid_from"), valid_to=arguments.get("valid_to"), observed_at=arguments.get("observed_at"))
                else:
                    conn = get_db_connection()
                    c = conn.cursor()
                    updates = []
                    params = []
                    for field in ["valid_from", "valid_to", "observed_at"]:
                        if arguments.get(field):
                            updates.append(f"{field} = ?")
                            params.append(arguments[field])
                    if updates:
                        c.execute(f"UPDATE memories SET {', '.join(updates)} WHERE id = ?", params + [memory_id])
                        conn.commit()
                    conn.close()
                    result = {"memory_id": memory_id, "updated": True, "note": "basic fallback"}
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]



        elif name == "create_wisp":
            if not BEADS_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "beads features not loaded"}, indent=2))]
            try:
                content = arguments.get("content", "").strip()
                if not content:
                    raise ValueError("content required")
                category = arguments.get("category", "observation")
                ttl_hours = arguments.get("ttl_hours", 24)
                tags = arguments.get("tags", [])
                result = create_wisp(DB_PATH, content, category, ttl_hours, tags)
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "squash_wisp":
            if not BEADS_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "beads features not loaded"}, indent=2))]
            try:
                memory_id = arguments.get("memory_id")
                if memory_id is None:
                    raise ValueError("memory_id required")
                result = squash_wisp(DB_PATH, memory_id)
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "burn_wisp":
            if not BEADS_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "beads features not loaded"}, indent=2))]
            try:
                memory_id = arguments.get("memory_id")
                if memory_id is None:
                    raise ValueError("memory_id required")
                result = burn_wisp(DB_PATH, memory_id)
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "gc_wisps":
            if not BEADS_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "beads features not loaded"}, indent=2))]
            try:
                dry_run = arguments.get("dry_run", True)
                result = gc_wisps(DB_PATH, dry_run=dry_run)
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "compact_memory_ai":
            if not BEADS_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "beads features not loaded"}, indent=2))]
            try:
                memory_id = arguments.get("memory_id")
                if memory_id is None:
                    raise ValueError("memory_id required")
                model = arguments.get("model")
                result = compact_memory_ai(DB_PATH, memory_id, model)
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "add_dependency":
            if not BEADS_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "beads features not loaded"}, indent=2))]
            try:
                source_id = arguments.get("source_id")
                target_id = arguments.get("target_id")
                if source_id is None or target_id is None:
                    raise ValueError("source_id and target_id required")
                dep_type = arguments.get("dep_type", "related")
                result = add_dependency(DB_PATH, source_id, target_id, dep_type)
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "get_ready_memories":
            if not BEADS_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "beads features not loaded"}, indent=2))]
            try:
                limit = arguments.get("limit", 20)
                result = get_ready_memories(DB_PATH, limit)
                return [TextContent(type="text", text=json.dumps({"memories": result}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "audit_log":
            if not BEADS_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "beads features not loaded"}, indent=2))]
            try:
                kind = arguments.get("kind")
                if not kind:
                    raise ValueError("kind required")
                data = arguments.get("data")
                actor = arguments.get("actor", "hermes")
                entry_path = audit_log(kind, data, actor)
                return [TextContent(type="text", text=json.dumps({"logged": True, "path": entry_path}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "query_audit_log":
            if not BEADS_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "beads features not loaded"}, indent=2))]
            try:
                kind = arguments.get("kind")
                since_hours = arguments.get("since_hours", 24)
                limit = arguments.get("limit", 50)
                result = query_audit_log(kind, since_hours, limit)
                return [TextContent(type="text", text=json.dumps({"entries": result}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "set_memory_gate":
            if not BEADS_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "beads features not loaded"}, indent=2))]
            try:
                memory_id = arguments.get("memory_id")
                if memory_id is None:
                    raise ValueError("memory_id required")
                gate_type = arguments.get("gate_type", "confirm")
                approvers = arguments.get("approvers")
                result = set_memory_gate(DB_PATH, memory_id, gate_type, approvers)
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "resolve_gate":
            if not BEADS_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "beads features not loaded"}, indent=2))]
            try:
                memory_id = arguments.get("memory_id")
                if memory_id is None:
                    raise ValueError("memory_id required")
                approved = arguments.get("approved", True)
                result = resolve_gate(DB_PATH, memory_id, approved)
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "compute_workspace_fingerprint":
            if not BEADS_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "beads features not loaded"}, indent=2))]
            try:
                result = compute_workspace_fingerprint()
                return [TextContent(type="text", text=json.dumps({"fingerprint": result}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
        else:
            raise ValueError(f"Unknown tool: {name}")




# === Main ===

async def main():
    """Run the Memster MCP server over stdio."""
    init_database()
    logger.info("Memster MCP server starting")

    async with stdio_server() as (read, write):
        await app.run(
            read,
            write,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    if MCP_AVAILABLE:
        import asyncio
        asyncio.run(main())
    else:
        print("mcp package not available")
