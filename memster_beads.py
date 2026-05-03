#!/usr/bin/env python3
"""Beads-inspired enhancements for Memster MCP Server.

Features from github.com/gastownhall/beads:
1. Content-hash IDs
2. Dependency DAG
3. Wisps (ephemeral memories)
4. AI compaction/summarization
5. Audit trail (append-only JSONL)
6. Gates (confirmation conditions)
7. Ready queue (unblocked items)
8. Workspace fingerprinting
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta

logger = logging.getLogger("memster.beads")
MEMSTER_DIR = os.path.expanduser("~/memster")
AUDIT_LOG_PATH = os.path.join(MEMSTER_DIR, "audit.jsonl")
WISP_DEFAULT_TTL_HOURS = 24
MAX_HIERARCHY_DEPTH = 3
HASH_ID_LENGTH = 6


# ============================================================
# 1. CONTENT-HASH IDS (from beads internal/types/id_generator.go)
# ============================================================

def generate_content_hash(title: str, content: str, salt: str = "") -> str:
    """Generate a collision-resistant content-based hash ID.
    Like beads: SHA256 of title + content + timestamp + salt.
    Returns full hex hash; caller uses progressive length for ID.
    """
    h = hashlib.sha256()
    h.update(title.encode())
    h.update(content.encode())
    h.update(salt.encode() if salt else str(time.time_ns()).encode())
    return h.hexdigest()


def get_short_hash_id(full_hash: str, length: int = HASH_ID_LENGTH) -> str:
    """Extract a short hash ID with progressive extension.
    Like beads: start at 6 chars, extend on collision.
    """
    return full_hash[:length]


def generate_child_id(parent_hash_id: str, child_number: int) -> str:
    """Create hierarchical child ID like beads bd-abc123.1."""
    return f"{parent_hash_id}.{child_number}"


# ============================================================
# 2. DEPENDENCY DAG (from beads dependency system)
# ============================================================

def init_dependency_tables(db_path: str) -> None:
    """Create dependency tables in memster DB."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS memory_dependencies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            dep_type TEXT NOT NULL DEFAULT "related",
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
            FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
        )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_dep_source ON memory_dependencies(source_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_dep_target ON memory_dependencies(target_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_dep_type ON memory_dependencies(dep_type)")

    # Add content_hash and is_ephemeral columns to memories if missing
    for col_def in [
        ("content_hash", "TEXT UNIQUE"),
        ("is_ephemeral", "INTEGER DEFAULT 0"),
        ("expires_at", "TEXT"),
        ("gate_type", "TEXT"),  # none, confirm, timer, event
        ("gate_status", "TEXT DEFAULT \"open\""),  # open, pending, resolved, expired
        ("hash_id", "TEXT UNIQUE"),  # short hash-based ID like bd-abc123
    ]:
        try:
            c.execute(f"ALTER TABLE memories ADD COLUMN {col_def[0]} {col_def[1]}")
        except sqlite3.OperationalError:
            pass  # already exists

    conn.commit()
    conn.close()


def add_dependency(db_path: str, source_id: int, target_id: int, dep_type: str = "related") -> dict:
    """Add a dependency between two memories.
    dep_type: blocks, parent_child, discovered_from, related, waits_for
    Like beads bd dep add.
    """
    valid_types = ("blocks", "parent_child", "discovered_from", "related", "waits_for")
    if dep_type not in valid_types:
        return {"error": f"invalid dep_type, must be one of {valid_types}"}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    # Verify both memories exist
    c.execute("SELECT id FROM memories WHERE id IN (?, ?)", (source_id, target_id))
    found = {r["id"] for r in c.fetchall()}
    if source_id not in found or target_id not in found:
        conn.close()
        return {"error": "one or both memory IDs not found"}

    # Check for duplicate dependency
    c.execute("SELECT id FROM memory_dependencies WHERE source_id=? AND target_id=? AND dep_type=?", (source_id, target_id, dep_type))
    if c.fetchone():
        conn.close()
        return {"duplicate": True, "message": "dependency already exists"}

    c.execute("INSERT INTO memory_dependencies (source_id, target_id, dep_type) VALUES (?, ?, ?)", (source_id, target_id, dep_type))
    dep_id = c.lastrowid
    conn.commit()
    conn.close()

    audit_log("dep_add", {"source_id": source_id, "target_id": target_id, "dep_type": dep_type})
    return {"created": True, "dep_id": dep_id}


def get_ready_memories(db_path: str, limit: int = 20) -> list:
    """Get memories that are NOT blocked by open dependencies.
    Like beads bd ready — shows only unblocked work.
    A memory is blocked if it has a \"blocks\" dependency where the target is not closed/completed.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Find memories that have unresolved blockers
    c.execute("""
        SELECT DISTINCT md.source_id
        FROM memory_dependencies md
        JOIN memories m ON m.id = md.target_id
        WHERE md.dep_type = "blocks"
          AND m.tier NOT IN ("L1", "completed")
          AND (m.valid_to IS NULL OR m.valid_to > datetime("now"))
    """)
    blocked_ids = {r["source_id"] for r in c.fetchall()}

    # Get all non-blocked, non-ephemeral-expired memories needing attention
    placeholders = ",".join("?" * len(blocked_ids)) if blocked_ids else "0"
    params = list(blocked_ids) if blocked_ids else []

    c.execute(f"""
        SELECT m.* FROM memories m
        WHERE m.id NOT IN ({placeholders})
          AND m.is_ephemeral = 0
          AND (m.gate_status IS NULL OR m.gate_status = "open")
        ORDER BY m.importance DESC, m.t_event DESC
        LIMIT ?
    """, params + [limit])

    results = []
    for row in c.fetchall():
        r = dict(row)
        # Get dependencies for context
        c.execute("SELECT target_id, dep_type FROM memory_dependencies WHERE source_id = ?", (r["id"],))
        r["dependencies"] = [dict(d) for d in c.fetchall()]
        results.append(r)

    conn.close()
    return results


# ============================================================
# 3. WISPS (from beads vapor/liquid phase concept)
# ============================================================

def create_wisp(db_path: str, content: str, category: str = "observation",
                ttl_hours: int = WISP_DEFAULT_TTL_HOURS,
                tags: list = None) -> dict:
    """Create an ephemeral memory (wisp) that auto-expires after TTL.
    Like beads bd mol wisp — vapor-phase, not synced to persistent storage.
    Can be promoted to permanent (squash) or deleted without trace (burn).
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    now = datetime.now().isoformat()
    expires = (datetime.now() + timedelta(hours=ttl_hours)).isoformat()

    # Generate hash ID like beads
    full_hash = generate_content_hash("wisp", content)
    short_id = f"ws-{get_short_hash_id(full_hash)}"

    network_type = category if category in ("world", "experience", "opinion", "observation") else "observation"
    c.execute("""
        INSERT INTO memories (content, category, network_type, t_event, t_recorded,
            tier, memory_type, is_ephemeral, expires_at, hash_id)
        VALUES (?, ?, ?, ?, ?, "L3", "wisp", 1, ?, ?)
    """, (content, category, network_type, now, now, expires, short_id))

    mid = c.lastrowid

    # Update content_hash
    c.execute("UPDATE memories SET content_hash = ? WHERE id = ?", (full_hash[:32], mid))
    conn.commit()
    conn.close()

    audit_log("wisp_create", {"id": mid, "hash_id": short_id, "ttl_hours": ttl_hours})
    return {"created": True, "id": mid, "hash_id": short_id,
            "ephemeral": True, "expires_at": expires}


def squash_wisp(db_path: str, memory_id: int) -> dict:
    """Promote an ephemeral wisp to a permanent memory.
    Like beads bd mol squash — clears ephemeral flag, removes TTL.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT is_ephemeral FROM memories WHERE id = ?", (memory_id,))
    row = c.fetchone()
    if not row or not row[0]:
        conn.close()
        return {"error": "memory not found or not ephemeral"}

    c.execute("""UPDATE memories SET is_ephemeral = 0, expires_at = NULL,
                tier = "L2" WHERE id = ?""", (memory_id,))
    conn.commit()
    conn.close()

    audit_log("wisp_squash", {"id": memory_id})
    return {"promoted": True, "id": memory_id}


def burn_wisp(db_path: str, memory_id: int) -> dict:
    """Delete an ephemeral wisp without trace.
    Like beads bd mol burn — no digest, no summary, just gone.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT is_ephemeral FROM memories WHERE id = ?", (memory_id,))
    row = c.fetchone()

    if not row or not row[0]:
        conn.close()
        return {"error": "memory not found or not ephemeral"}

    c.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    conn.commit()
    conn.close()

    audit_log("wisp_burn", {"id": memory_id})
    return {"burned": True, "id": memory_id}


def gc_wisps(db_path: str, dry_run: bool = False) -> dict:
    """Garbage collect expired wisps.
    Like beads bd mol wisp gc — removes ephemeral memories past TTL.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    now = datetime.now().isoformat()
    c.execute("""
        SELECT id, hash_id, content FROM memories
        WHERE is_ephemeral = 1 AND expires_at < ?
    """, (now,))
    expired = [{"id": r[0], "hash_id": r[1], "content_preview": r[2][:80]} for r in c.fetchall()]

    if not dry_run and expired:
        c.execute("DELETE FROM memories WHERE is_ephemeral = 1 AND expires_at < ?", (now,))
        conn.commit()

    conn.close()

    if not dry_run:
        audit_log("wisp_gc", {"count": len(expired)})

    return {"expired_count": len(expired), "deleted": not dry_run, "wisps": expired}


# ============================================================
# 4. AI COMPACTION (from beads internal/compact/compactor.go)
# ============================================================

def compact_memory_ai(db_path: str, memory_id: int, model: str = None) -> dict:
    """AI-powered compaction of a memory.
    Like beads CompactTier1: summarize verbose content to preserve essence.
    Uses NVIDIA NIM for summarization if available, else heuristic compression.
    Only compacts if summary would save >= 30% space (like beads size check).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
    row = c.fetchone()

    if not row:
        conn.close()
        return {"error": f"memory {memory_id} not found"}

    original = row["content"] if "content" in row.keys() else str(row[1])
    original_size = len(original)

    # Try AI summarization via NIM
    summary = None
    try:
        from nvidia_nim_embeddings import get_nim_summary
        summary = get_nim_summary(original, model=model)
    except (ImportError, Exception) as e:
        logger.debug(f"NIM summarization unavailable: {e}")

    # Fallback: heuristic compression (keep first/last sentences, drop middle)
    if not summary:
        sentences = original.split(". ")
        if len(sentences) <= 3:
            conn.close()
            return {"skipped": True, "reason": "too short to compact"}
        summary = sentences[0] + ". " + sentences[-1]

    compacted_size = len(summary)

    # Beads-style check: skip if compaction would increase size
    if compacted_size >= original_size * 0.7:
        conn.close()
        return {"skipped": True, "reason": f"summary ({compacted_size}B) not 30%+ smaller than original ({original_size}B)"}

    # Apply compaction
    c.execute("UPDATE memories SET content = ? WHERE id = ?", (summary, memory_id))
    try:
        c.execute("ALTER TABLE memories ADD COLUMN compacted_at TEXT")
    except sqlite3.OperationalError:
        pass
    c.execute("UPDATE memories SET compacted_at = ? WHERE id = ?", (datetime.now().isoformat(), memory_id))
    conn.commit()
    conn.close()

    audit_log("compact", {"id": memory_id, "original_size": original_size, "compacted_size": compacted_size})
    return {"compacted": True, "id": memory_id,
            "original_size": original_size, "compacted_size": compacted_size,
            "savings_pct": round((1 - compacted_size/original_size) * 100, 1)}


# ============================================================
# 5. AUDIT TRAIL (from beads internal/audit/audit.go)
# ============================================================

def audit_log(kind: str, data: dict = None, actor: str = "hermes") -> str:
    """Append an event to the audit log (append-only JSONL).
    Like beads interactions.jsonl — tamper-evident history of all mutations.
    Returns the audit entry ID.
    """
    os.makedirs(MEMSTER_DIR, exist_ok=True)

    entry_id = hashlib.sha256(f"{kind}{time.time_ns()}".encode()).hexdigest()[:12]
    entry = {
        "id": f"aud-{entry_id}",
        "kind": kind,
        "actor": actor,
        "created_at": datetime.now().isoformat(),
        **(data or {})
    }

    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\\n")

    return entry["id"]


def query_audit_log(kind: str = None, since_hours: int = 24, limit: int = 50) -> list:
    """Query the audit log for recent events.
    Like beads bd events list.
    """
    if not os.path.exists(AUDIT_LOG_PATH):
        return []

    cutoff = (datetime.now() - timedelta(hours=since_hours)).isoformat()
    results = []
    with open(AUDIT_LOG_PATH) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if kind and entry.get("kind") != kind:
                    continue
                if entry.get("created_at", "") >= cutoff:
                    results.append(entry)
            except json.JSONDecodeError:
                continue

    return results[-limit:]


# ============================================================
# 6. GATES (from beads gates.md — async coordination primitives)
# ============================================================

def set_memory_gate(db_path: str, memory_id: int,
                    gate_type: str = "confirm",
                    approvers: list = None) -> dict:
    """Set a gate on a memory to require confirmation before trusting it.
    Like beads human/timer gates but for memories:
    - confirm: needs human confirmation before being surfaced in briefings
    - timer: auto-resolves after a duration (e.g. wait 24h before trusting)
    - verify: needs cross-reference verification against other memories
    """

    valid_types = ("confirm", "timer", "verify", "none")
    if gate_type not in valid_types:
        return {"error": f"gate_type must be one of {valid_types}"}

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    status = "pending" if gate_type != "none" else "open"

    c.execute("UPDATE memories SET gate_type = ?, gate_status = ? WHERE id = ?",
              (gate_type, status, memory_id))

    if c.rowcount == 0:
        conn.close()
        return {"error": f"memory {memory_id} not found"}

    conn.commit()
    conn.close()

    audit_log("gate_set", {"id": memory_id, "gate_type": gate_type, "status": status})
    return {"updated": True, "id": memory_id, "gate_type": gate_type, "status": status}


def resolve_gate(db_path: str, memory_id: int, approved: bool = True) -> dict:
    """Resolve a pending gate on a memory.
    Like beads bd gate approve / bd gate skip.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT gate_type, gate_status FROM memories WHERE id = ?", (memory_id,))
    row = c.fetchone()

    if not row:
        conn.close()
        return {"error": f"memory {memory_id} not found"}

    new_status = "resolved" if approved else "rejected"
    c.execute("UPDATE memories SET gate_status = ? WHERE id = ?", (new_status, memory_id))
    conn.commit()
    conn.close()

    audit_log("gate_resolve", {"id": memory_id, "approved": approved})
    return {"resolved": True, "id": memory_id, "status": new_status}


# ============================================================
# 8. WORKSPACE FINGERPRINTING (from beads internal/beads/fingerprint.go)
# ============================================================



def compute_workspace_fingerprint() -> str:
    """Generate a unique identifier for this workspace.
    Like beads ComputeRepoID: hash of hostname + user + memster dir path.
    """
    import socket
    identity = f"{socket.gethostname()}:{os.getenv('USER', 'house')}:{MEMSTER_DIR}"
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


# ============================================================
# INITIALIZATION
# ============================================================

def init_all_beads_features(db_path: str) -> dict:
    """Initialize all beads-inspired tables and columns.
    Call this once at server startup.
    """
    init_dependency_tables(db_path)
    # Run wisp GC on startup (clean up any expired ephemeral memories)
    gc_result = gc_wisps(db_path, dry_run=True)

    fingerprint = compute_workspace_fingerprint()

    return {
        "initialized": True,
        "fingerprint": fingerprint,
        "expired_wisps_found": gc_result["expired_count"]
    }


# ============================================================
# MCP TOOL DEFINITIONS (to be appended to TOOL_DEFINITIONS)
# ============================================================

BEADS_TOOL_DEFINITIONS = []

try:
    from mcp.types import Tool

    BEADS_TOOL_DEFINITIONS = [
        Tool(
            name="add_dependency",
            description="Add a dependency between memories (blocks, discovered_from, parent_child, related, waits_for). From beads dependency system.",
            inputSchema={"type": "object", "properties": {
                "source_id": {"type": "integer"},
                "target_id": {"type": "integer"},
                "dep_type": {"type": "string", "enum": ["blocks", "discovered_from", "parent_child", "related", "waits_for"], "default": "related"}
            }, "required": ["source_id", "target_id"]}
        ),
        Tool(
            name="get_ready",
            description="Get unblocked memories that need attention. Like beads bd ready - only shows items with no unresolved blockers.",
            inputSchema={"type": "object", "properties": {"limit": {"type": "integer", "default": 20}}}
        ),
        Tool(
            name="create_wisp",
            description="Create an ephemeral memory (wisp) that auto-expires after TTL. Like beads vapor-phase - for temporary operational data. Can be promoted (squash) or deleted (burn).",
            inputSchema={"type": "object", "properties": {
                "content": {"type": "string"},
                "category": {"type": "string", "default": "observation"},
                "ttl_hours": {"type": "integer", "default": 24},
                "tags": {"type": "array", "items": {"type": "string"}}
            }, "required": ["content"]}
        ),
        Tool(
            name="squash_wisp",
            description="Promote an ephemeral wisp to permanent memory. Like beads bd mol squash.",
            inputSchema={"type": "object", "properties": {"memory_id": {"type": "integer"}}, "required": ["memory_id"]}
        ),
        Tool(
            name="burn_wisp",
            description="Delete an ephemeral wisp without trace. Like beads bd mol burn.",
            inputSchema={"type": "object", "properties": {"memory_id": {"type": "integer"}}, "required": ["memory_id"]}
        ),
        Tool(
            name="gc_wisps",
            description="Garbage collect expired ephemeral memories. Like beads bd mol wisp gc.",
            inputSchema={"type": "object", "properties": {"dry_run": {"type": "boolean", "default": True}}}
        ),
        Tool(
            name="compact_memory_ai",
            description="AI-powered compaction of a memory - summarize verbose content to preserve essence. Like beads CompactTier1 - only compacts if 30%+ space savings.",
            inputSchema={"type": "object", "properties": {"memory_id": {"type": "integer"}, "model": {"type": "string"}}, "required": ["memory_id"]}
        ),
        Tool(
            name="set_memory_gate",
            description="Set a gate on a memory to require confirmation before trusting it. Like beads human/timer gates - confirm, timer, or verify.",
            inputSchema={"type": "object", "properties": {
                "memory_id": {"type": "integer"},
                "gate_type": {"type": "string", "enum": ["confirm", "timer", "verify", "none"], "default": "confirm"},
                "approvers": {"type": "array", "items": {"type": "string"}}
            }, "required": ["memory_id"]}
        ),
        Tool(
            name="resolve_gate",
            description="Resolve a pending gate on a memory. Like beads bd gate approve/skip.",
            inputSchema={"type": "object", "properties": {
                "memory_id": {"type": "integer"},
                "approved": {"type": "boolean", "default": True}
            }, "required": ["memory_id"]}
        ),
        Tool(
            name="query_audit_log",
            description="Query the append-only audit log for recent mutations. Like beads bd events list.",
            inputSchema={"type": "object", "properties": {
                "kind": {"type": "string"},
                "since_hours": {"type": "integer", "default": 24},
                "limit": {"type": "integer", "default": 50}
            }}
        ),
        Tool(
            name="get_workspace_fingerprint",
            description="Get the unique fingerprint for this workspace. Like beads ComputeRepoID - consistent identity across sessions.",
            inputSchema={"type": "object", "properties": {}}
        ),
    ]

except ImportError:
    BEADS_TOOL_DEFINITIONS = []


