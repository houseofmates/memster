#!/usr/bin/env python3
"""
memster MCP server v3 - enhanced with entity extraction, memory graph,
semantic search, proactive surfacing, importance scoring, update/merge
"""

import json
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# add memster dir to path for local imports
sys.path.insert(0, os.path.expanduser("~/memster"))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("warning: mcp package not available", file=sys.stderr)

DB_PATH = os.path.expanduser("~/memster/memster_core.db")
MEMSTER_DIR = os.path.expanduser("~/memster")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
logger = logging.getLogger("memster")

# === module imports with graceful fallback ===

ENTITY_EXTRACTION_AVAILABLE = False
try:
    from ner_extractor import extract_and_store as ner_extract_and_store, extract_entities_from_text, init_entities_table
    # auto-init entities table
    try:
        init_entities_table(DB_PATH)
    except Exception:
        pass
    ENTITY_EXTRACTION_AVAILABLE = True
    logger.info("ner_extractor loaded")
except ImportError as e:
    logger.warning(f"ner_extractor not available: {e}")

MEMORY_GRAPH_AVAILABLE = False
try:
    from memory_graph import auto_create_edges, create_edge, find_candidate_edges
    # auto-init memory_edges table
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""CREATE TABLE IF NOT EXISTS memory_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_memory_id INTEGER NOT NULL,
            target_memory_id INTEGER NOT NULL,
            relation_type TEXT DEFAULT 'related',
            weight REAL DEFAULT 0.5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
            FOREIGN KEY (target_memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_edges_source ON memory_edges(source_memory_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_edges_target ON memory_edges(target_memory_id)")
        conn.commit()
        conn.close()
    except Exception:
        pass
    MEMORY_GRAPH_AVAILABLE = True
    logger.info("memory_graph loaded")
except ImportError as e:
    logger.warning(f"memory_graph not available: {e}")

MEMORY_LINKS_AVAILABLE = False
try:
    from memory_links import link_memories, get_related_memories
    MEMORY_LINKS_AVAILABLE = True
    logger.info("memory_links loaded")
except ImportError as e:
    logger.warning(f"memory_links not available: {e}")

SEMANTIC_SEARCH_AVAILABLE = False
try:
    from semantic_search import search_memories as semantic_search_memories, find_similar
    SEMANTIC_SEARCH_AVAILABLE = True
    logger.info("semantic_search loaded")
except ImportError as e:
    logger.warning(f"semantic_search not available: {e}")

PROACTIVE_AVAILABLE = False
try:
    from proactive_surfacer import ProactiveSurfacer, monitor_and_trigger
    PROACTIVE_AVAILABLE = True
    logger.info("proactive_surfacer loaded")
except ImportError as e:
    logger.warning(f"proactive_surfacer not available: {e}")

BRIEFING_AVAILABLE = False
try:
    from briefing_v2 import get_narrative_briefing, get_context_thread
    BRIEFING_AVAILABLE = True
    logger.info("briefing_v2 loaded")
except ImportError as e:
    logger.warning(f"briefing_v2 not available: {e}")

CONTEXT_SURFACER_AVAILABLE = False
try:
    from context_surfacer import surface_context
    CONTEXT_SURFACER_AVAILABLE = True
    logger.info("context_surfacer loaded")
except ImportError as e:
    logger.warning(f"context_surfacer not available: {e}")

IMPORTANCE_AVAILABLE = False
try:
    from importance_scoring import compute_importance_score, update_memory_importance, batch_update_importance
    IMPORTANCE_AVAILABLE = True
    logger.info("importance_scoring loaded")
except ImportError as e:
    logger.warning(f"importance_scoring not available: {e}")

DECAY_AVAILABLE = False
try:
    from decay import DecayManager
    DECAY_AVAILABLE = True
    logger.info("decay loaded")
except ImportError as e:
    logger.warning(f"decay not available: {e}")

ACTR_AVAILABLE = False
try:
    from actr_scoring import compute_activation, retrieval_probability
    ACTR_AVAILABLE = True
    logger.info("actr_scoring loaded")
except ImportError as e:
    logger.warning(f"actr_scoring not available: {e}")

DEDUP_AVAILABLE = False
try:
    from dedup import Deduplicator, add_with_dedup
    DEDUP_AVAILABLE = True
    logger.info("dedup loaded")
except ImportError as e:
    logger.warning(f"dedup not available: {e}")

CONTRADICTION_AVAILABLE = False
try:
    from contradiction_detector import detect_contradictions
    CONTRADICTION_AVAILABLE = True
    logger.info("contradiction_detector loaded")
except ImportError as e:
    logger.warning(f"contradiction_detector not available: {e}")

# === Brainstack-inspired features ===

TEMPORAL_VALIDITY_AVAILABLE = False
try:
    from temporal_validity import check_validity
    TEMPORAL_VALIDITY_AVAILABLE = True
    logger.info("temporal_validity loaded")
except ImportError as e:
    logger.warning(f"temporal_validity not available: {e}")

RETRIEVAL_ROUTING_AVAILABLE = False
try:
    from retrieval_routing import RouteConfig, match_intent, RetrievalRouter
    RETRIEVAL_ROUTING_AVAILABLE = True
    logger.info("retrieval_routing loaded")
except ImportError as e:
    logger.warning(f"retrieval_routing not available: {e}")

PROVENANCE_AVAILABLE = False
try:
    from provenance_utils import (
        compute_similarity,
        text_fingerprint,
        find_similar_memories,
        merge_provenance,
        SimilarityResult,
        check_memory_deduplication,
    )
    PROVENANCE_AVAILABLE = True
    logger.info("provenance_utils loaded")
except ImportError as e:
    logger.warning(f"provenance_utils not available: {e}")

GRAPH_EXTRACTION_AVAILABLE = False
try:
    from graph_extraction import extract_relations, extract_entities
    GRAPH_EXTRACTION_AVAILABLE = True
    logger.info("graph_extraction loaded")
except ImportError as e:
    logger.warning(f"graph_extraction not available: {e}")

RETRIEVAL_TELEMETRY_AVAILABLE = False
try:
    from retrieval_telemetry import read_telemetry, apply_telemetry, RetrievalTelemetry
    RETRIEVAL_TELEMETRY_AVAILABLE = True
    logger.info("retrieval_telemetry loaded")
except ImportError as e:
    logger.warning(f"retrieval_telemetry not available: {e}")

import functools

@functools.lru_cache(maxsize=256)
def _cached_encode(query: str):
    """LRU-cached query embedding. avoids recomputing repeated queries."""
    try:
        from semantic_search import encode_text
        return encode_text(query)
    except:
        return None


def hybrid_search(query: str, limit: int = 10, db_path: str = None) -> List[Dict]:
    """
    hybrid ranking: 50% vector similarity + 30% FTS rank + 20% importance.
    inspired by mnemosyne's hybrid retrieval.
    """
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    
    # 1) vector search (top 30 candidates)
    vector_results = {}
    try:
        from semantic_search import search_memories as _semantic_search
        sem = _semantic_search(query, limit=30, threshold=0.1, db_path=path)
        for r in sem:
            mid = r.get("id") or r.get("memory_id")
            if mid:
                vector_results[mid] = r.get("similarity", 0.0)
    except Exception:
        pass
    
    # 2) FTS search (top 30 candidates)
    fts_results = {}
    try:
        c = conn.cursor()
        # check if FTS table exists
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
        c.execute("SELECT id, content, category, tier, COALESCE(importance, 0.5) as importance FROM memories WHERE content LIKE ? ORDER BY importance DESC LIMIT ?",
                  (f"%{query}%", limit))
        results = [dict(r) for r in c.fetchall()]
        conn.close()
        return results
    
    # 4) get full memory data for candidates
    placeholders = ",".join("?" * len(all_ids))
    c = conn.cursor()
    c.execute(f"SELECT id, content, category, tier, COALESCE(importance, 0.5) as importance, COALESCE(decay_score, 1.0) as decay_score, access_count FROM memories WHERE id IN ({placeholders})",
              list(all_ids))
    memories = {r["id"]: dict(r) for r in c.fetchall()}
    
    # 5) hybrid scoring: 50% vector + 30% FTS + 20% importance
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
    """
    sleep cycle: consolidate old low-tier memories.
    - L2 memories older than 7 days with low access → decay importance
    - groups of similar L2 memories → summarize into L1
    - very old untouched L1 → promote to L0 if important enough
    inspired by mnemosyne's sleep/consolidation.
    """
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    now = datetime.now()
    
    results = {"decayed": 0, "promoted": 0, "summarized": 0}
    
    # 1) decay old L2 memories
    cutoff = (now - timedelta(days=7)).isoformat()
    c.execute("""UPDATE memories SET decay_score = decay_score * 0.95
        WHERE tier = 'L2' AND t_recorded < ? AND access_count < 3
        AND decay_score > 0.1""", (cutoff,))
    results["decayed"] = c.changes
    
    # 2) promote important L2 → L1 if they've been accessed enough
    c.execute("""UPDATE memories SET tier = 'L1'
        WHERE tier = 'L2' AND access_count >= 5 AND importance >= 0.7
        AND decay_score >= 0.5""")
    results["promoted"] = c.changes
    
    # 3) promote important L1 → L0
    c.execute("""UPDATE memories SET tier = 'L0'
        WHERE tier = 'L1' AND access_count >= 10 AND importance >= 0.8
        AND decay_score >= 0.7""")
    results["promoted"] += c.changes
    
    conn.commit()
    conn.close()
    
    logger.info(f"sleep: decayed={results['decayed']} promoted={results['promoted']}")
    return results


def remember_batch(memories: List[Dict], db_path: str = None) -> Dict:
    """
    batch insert multiple memories at once.
    each dict: {content, category, tags?}
    inspired by mnemosyne's remember_batch.
    """
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
        
        # dedup check (quick)
        c.execute("SELECT id FROM memories WHERE content = ?", (content,))
        if c.fetchone():
            skipped.append({"content": content[:50], "reason": "duplicate"})
            continue
        
        c.execute("INSERT INTO memories (content, category, t_event, t_recorded, tier, memory_type) VALUES (?, ?, ?, ?, ?, ?)",
                  (content, category, now, now, "L2", category if category in ("world", "experience", "opinion", "observation") else "observation"))
        mid = c.lastrowid
        created.append(mid)
    
    conn.commit()
    
    # run hooks on created memories (in background, non-blocking)
    for mid in created:
        try:
            c.execute("SELECT content, category FROM memories WHERE id = ?", (mid,))
            row = c.fetchone()
            if row:
                run_post_insert_hooks(mid, row["content"], row["category"])
        except Exception:
            pass
    
    conn.commit()
    conn.close()
    
    return {"created": len(created), "skipped": len(skipped), "ids": created}


# === core db helpers ===

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# === normalization helpers ===

def normalize_text(text):
    """Normalize text for comparison: lowercase, strip, collapse whitespace, remove punctuation."""
    if not text:
        return ""
    text = text.lower().strip()
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
    """Check if passive capture content is noise (raw errors, screen dumps, etc)."""
    if not content or len(content.strip()) < 10:
        return True
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
            return True
    # raw OCR dumps are usually low-printable-ratio
    printable = sum(1 for c in content if c.isprintable() or c in '\n\t')
    if len(content) > 0 and printable / len(content) < 0.7:
        return True
    return False


def extract_meaning_from_capture(content, source_type):
    """Try to extract structured meaning from raw passive capture. Returns cleaned content or None."""
    if is_noise_capture(content):
        return None
    
    # strip "copied: " prefix
    if content.startswith("copied:"):
        content = content[7:].strip()
    
    # strip "screen showed: " prefix
    if content.startswith("screen showed:"):
        content = content[14:].strip()
    
    # if it's an error, extract the error type
    error_match = re.search(r'(error|Error|ERROR)[:\s]+(.+?)(?:\n|$)', content)
    if error_match:
        return f"error encountered: {error_match.group(2).strip()[:100]}"
    
    # if it's a file path, store that
    path_match = re.search(r'(/[\w/.-]+\.\w{1,5})', content)
    if path_match:
        return f"file referenced: {path_match.group(1)}"
    
    # if short enough and looks like useful content, return cleaned
    if len(content) < 200 and len(content.split()) > 3:
        return content.strip()
    
    return None


# === wiki integration ===

WIKI_AVAILABLE = False
try:
    sys.path.insert(0, MEMSTER_DIR)
    from wiki_mcp_server import WIKI_DIR as _WIKI_DIR, DB_PATH as _WIKI_DB, page_path as _wiki_page_path, build_frontmatter as _wiki_front, index_page as _wiki_index, get_db as _wiki_get_db
    WIKI_AVAILABLE = True
    logger.info("wiki integration loaded")
except ImportError as e:
    logger.warning(f"wiki integration not available: {e}")

BRIDGE_AVAILABLE = False
try:
    from memster_bridge import (
        unified_briefing,
        wiki_to_memster,
        enrich_memory,
        check_memory_quality,
        detect_stale_memories,
        get_working_memory_context,
        add_to_working_memory,
        clear_working_memory,
        infer_follow_up_intent,
    )
    BRIDGE_AVAILABLE = True
    logger.info("bridge layer loaded")
except ImportError as e:
    logger.warning(f"bridge not available: {e}")


def sync_memory_to_wiki(memory_id, content, category):
    """Check if a memory should update an existing wiki page. Returns wiki_slug if linked."""
    if not WIKI_AVAILABLE:
        return None
    
    try:
        conn = _wiki_get_db()
        c = conn.cursor()
        
        # search for wiki pages that mention key terms from the memory
        words = set(content.lower().split())
        # get project/location-specific terms
        key_terms = [w for w in words if len(w) > 3 and w not in ('that', 'this', 'with', 'from', 'have', 'been', 'will', 'they', 'their', 'there', 'about', 'which', 'when', 'what', 'your', 'more', 'some', 'into', 'just', 'also', 'than', 'them', 'very', 'only')]
        
        if not key_terms:
            conn.close()
            return None
        
        # search wiki pages for matching terms
        matched_slugs = set()
        for term in key_terms[:5]:
            c.execute("SELECT slug FROM pages_fts WHERE content MATCH ? LIMIT 3", (term,))
            for row in c.fetchall():
                matched_slugs.add(row["slug"])
        
        if not matched_slugs:
            conn.close()
            return None
        
        # link memory to all matched pages
        for slug in matched_slugs:
            c.execute("INSERT OR IGNORE INTO memster_links (wiki_slug, memory_id) VALUES (?, ?)",
                      (slug, memory_id))
        
        conn.commit()
        conn.close()
        return list(matched_slugs) if matched_slugs else None
    
    except Exception as e:
        logger.warning(f"wiki sync failed for memory {memory_id}: {e}")
        return None


# === post-insert hooks ===

def run_post_insert_hooks(memory_id, content, category, tags=None):
    """Run all post-insert processing: entities, edges, importance, wiki sync."""
    results = {"entities": 0, "edges": 0, "importance": None, "wiki_pages": None}
    
    # entity extraction
    if ENTITY_EXTRACTION_AVAILABLE:
        try:
            entities = ner_extract_and_store(memory_id, content, DB_PATH)
            results["entities"] = len(entities) if entities else 0
        except Exception as e:
            logger.warning(f"entity extraction failed for memory {memory_id}: {e}")
    
    # edge creation
    if MEMORY_GRAPH_AVAILABLE:
        try:
            edges = auto_create_edges(memory_id, content, category or "general", tags or [], DB_PATH)
            results["edges"] = len(edges) if edges else 0
        except Exception as e:
            logger.warning(f"edge creation failed for memory {memory_id}: {e}")
    
    # importance scoring
    if IMPORTANCE_AVAILABLE:
        try:
            score = update_memory_importance(memory_id, DB_PATH)
            results["importance"] = score
        except Exception as e:
            logger.warning(f"importance scoring failed for memory {memory_id}: {e}")
    
    # wiki sync
    if WIKI_AVAILABLE:
        try:
            wiki_pages = sync_memory_to_wiki(memory_id, content, category)
            results["wiki_pages"] = wiki_pages
        except Exception as e:
            logger.warning(f"wiki sync failed for memory {memory_id}: {e}")
    
    return results


# === duplicate detection ===

def check_duplicate_before_insert(content, category):
    """Check if a memory with similar content already exists. Returns existing ID or None."""
    conn = get_db_connection()
    cursor = conn.cursor()
    normalized_new = normalize_text(content)
    
    # check same category first
    cursor.execute("SELECT id, content FROM memories WHERE category = ?", (category or "general",))
    for row in cursor.fetchall():
        if normalize_text(row['content']) == normalized_new:
            conn.close()
            return row['id']
    
    # then all categories
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
    
    # use COALESCE to handle null importance/decay_score
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
    
    # bump access count for returned memories
    for m in results:
        cursor.execute("UPDATE memories SET access_count = access_count + 1 WHERE id = ?", (m['id'],))
    conn.commit()
    conn.close()
    
    return results


# === activity helpers ===

def get_recent_activity(hours=1, limit=50):
    conn = get_db_connection()
    cursor = conn.cursor()
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    cursor.execute("SELECT window_class, window_title, timestamp, duration_seconds FROM window_events WHERE timestamp > ? ORDER BY timestamp DESC LIMIT ?", (cutoff, limit))
    windows = [dict(r) for r in cursor.fetchall()]
    cursor.execute("SELECT window_class, SUM(duration_seconds) as total_seconds, COUNT(*) as event_count FROM window_events WHERE timestamp > ? GROUP BY window_class ORDER BY total_seconds DESC", (cutoff,))
    apps = [{"app": r["window_class"], "minutes": (r["total_seconds"] or 0) // 60} for r in cursor.fetchall()]
    cursor.execute("SELECT content_preview, source_app, timestamp FROM clipboard_events WHERE timestamp > ? ORDER BY timestamp DESC LIMIT ?", (cutoff, 20))
    clipboard = [dict(r) for r in cursor.fetchall()]
    cursor.execute("SELECT COUNT(*) as count FROM screenshot_events WHERE timestamp > ?", (cutoff,))
    screenshots = cursor.fetchone()["count"]
    conn.close()
    return {"hours": hours, "window_events": len(windows), "apps": apps[:10], "clipboard": clipboard[:5], "screenshots": screenshots}


def search_window_history(query, hours=24, limit=20):
    conn = get_db_connection()
    cursor = conn.cursor()
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    cursor.execute("SELECT window_class, window_title, timestamp, duration_seconds FROM window_events WHERE timestamp > ? AND window_title LIKE ? ORDER BY timestamp DESC LIMIT ?", (cutoff, "%" + query + "%", limit))
    results = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return {"query": query, "matches": len(results), "results": results}


def search_clipboard(query, hours=24, limit=10):
    conn = get_db_connection()
    cursor = conn.cursor()
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    cursor.execute("SELECT content_preview, source_app, timestamp FROM clipboard_events WHERE timestamp > ? AND content_preview LIKE ? ORDER BY timestamp DESC LIMIT ?", (cutoff, "%" + query + "%", limit))
    results = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return {"query": query, "matches": len(results), "results": results}


def get_activity_summary_text(hours=2):
    activity = get_recent_activity(hours=hours)
    lines = ["activity summary for last " + str(hours) + " hour(s):"]
    if activity["apps"]:
        lines.append("\napp usage:")
        for app in activity["apps"][:5]:
            lines.append("  - " + app.get("app", "unknown") + ": " + str(app.get("minutes", 0)) + " min")
    else:
        lines.append("\nno app activity recorded.")
    if activity["clipboard"]:
        lines.append("\nrecent clipboard:")
        for item in activity["clipboard"][:3]:
            preview = (item.get("content_preview") or "")[:50]
            lines.append("  - [" + item.get("source_app", "unknown") + "] " + preview + "...")
    lines.append("\nscreenshots: " + str(activity["screenshots"]))
    return "\n".join(lines)


def search_ocr_text(query, limit=10):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, image_path, timestamp, ocr_text FROM screenshot_events WHERE ocr_text LIKE ? ORDER BY timestamp DESC LIMIT ?", ("%" + query + "%", limit))
    results = [{"id": r[0], "image_path": r[1], "timestamp": r[2], "ocr_preview": (r[3] or "")[:200]} for r in cursor.fetchall()]
    conn.close()
    return {"query": query, "matches": len(results), "results": results}


# === MCP server ===

if MCP_AVAILABLE:
    app = Server("memster_v3")

    @app.list_tools()
    async def list_tools():
        return [
            # existing tools
            Tool(name="get_activity_summary", description="Get a human-readable summary of what the user has been doing recently.",
                inputSchema={"type": "object", "properties": {"hours": {"type": "integer", "default": 2}}}),
            Tool(name="get_recent_activity", description="Get structured activity data - apps used, clipboard, screenshots.",
                inputSchema={"type": "object", "properties": {"hours": {"type": "integer", "default": 1}}}),
            Tool(name="search_window_history", description="Search what apps/windows the user was working in.",
                inputSchema={"type": "object", "properties": {"query": {"type": "string"}, "hours": {"type": "integer", "default": 24}}, "required": ["query"]}),
            Tool(name="search_clipboard", description="Search clipboard history for pasted/copied content.",
                inputSchema={"type": "object", "properties": {"query": {"type": "string"}, "hours": {"type": "integer", "default": 24}}, "required": ["query"]}),
            Tool(name="search_ocr_text", description="Search text extracted from screenshots via OCR.",
                inputSchema={"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 10}}, "required": ["query"]}),
            
            # memory CRUD
            Tool(name="query_memories", description="Search memories with text/category/tier filters. Results ordered by importance and access frequency.",
                inputSchema={"type": "object", "properties": {
                    "query_text": {"type": "string"},
                    "category": {"type": "string"},
                    "tier": {"type": "string"},
                    "max_results": {"type": "integer", "default": 10}}}),
            Tool(name="remember_memory", description="Store a new memory. Auto-dedup, auto-extract entities, auto-create graph edges, auto-score importance.",
                inputSchema={"type": "object", "properties": {
                    "content": {"type": "string"},
                    "category": {"type": "string", "enum": ["world", "experience", "opinion", "observation"]},
                    "tags": {"type": "array", "items": {"type": "string"}}},
                    "required": ["content"]}),
            Tool(name="update_memory", description="Update an existing memory's content, category, or tier.",
                inputSchema={"type": "object", "properties": {
                    "memory_id": {"type": "integer"},
                    "content": {"type": "string"},
                    "category": {"type": "string"},
                    "tier": {"type": "string"}},
                    "required": ["memory_id"]}),
            Tool(name="merge_memories", description="Merge two memories into one. Keeps the first as primary, absorbs second, deletes second.",
                inputSchema={"type": "object", "properties": {
                    "primary_id": {"type": "integer"},
                    "secondary_id": {"type": "integer"},
                    "merged_content": {"type": "string"}},
                    "required": ["primary_id", "secondary_id"]}),
            Tool(name="delete_memory", description="Delete a memory by its ID.",
                inputSchema={"type": "object", "properties": {"memory_id": {"type": "integer"}}, "required": ["memory_id"]}),
            Tool(name="delete_by_query", description="Delete memories matching a content query.",
                inputSchema={"type": "object", "properties": {
                    "query_text": {"type": "string"},
                    "exact_match": {"type": "boolean", "default": False}},
                    "required": ["query_text"]}),
            Tool(name="find_duplicates", description="Find duplicate or near-duplicate memories.",
                inputSchema={"type": "object", "properties": {"similarity_threshold": {"type": "number", "default": 0.8}}}),
            
            # semantic search
            Tool(name="semantic_memory_search", description="Search memories using AI semantic similarity. Finds conceptually related memories even without exact keyword match.",
                inputSchema={"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 5}}, "required": ["query"]}),
            Tool(name="hybrid_search", description="Search with hybrid ranking: 50% vector similarity + 30% full-text + 20% importance. Best retrieval quality.",
                inputSchema={"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 10}}, "required": ["query"]}),
            Tool(name="remember_batch", description="Store multiple memories at once. Auto-dedup. Each item: {content, category}.",
                inputSchema={"type": "object", "properties": {
                    "memories": {"type": "array", "items": {"type": "object", "properties": {
                        "content": {"type": "string"},
                        "category": {"type": "string"}}}}},
                    "required": ["memories"]}),
            Tool(name="sleep_consolidate", description="Run memory consolidation cycle: decay old unused memories, promote frequently accessed ones to higher tiers.",
                inputSchema={"type": "object", "properties": {}}),
            Tool(name="find_similar", description="Find memories similar to a specific memory by ID.",
                inputSchema={"type": "object", "properties": {"memory_id": {"type": "integer"}, "limit": {"type": "integer", "default": 5}}, "required": ["memory_id"]}),
            
            # memory graph
            Tool(name="get_related_memories", description="Get memories linked to a specific memory via graph edges.",
                inputSchema={"type": "object", "properties": {"memory_id": {"type": "integer"}, "limit": {"type": "integer", "default": 10}}, "required": ["memory_id"]}),
            Tool(name="link_memories", description="Create a typed link between two memories.",
                inputSchema={"type": "object", "properties": {
                    "source_id": {"type": "integer"},
                    "target_id": {"type": "integer"},
                    "link_type": {"type": "string", "default": "related"}},
                    "required": ["source_id", "target_id"]}),
            
            # briefing and surfacing
            Tool(name="get_briefing", description="Get a session briefing - surface relevant memories based on current context.",
                inputSchema={"type": "object", "properties": {}}),
            Tool(name="get_narrative_briefing", description="Get a narrative briefing with context threads and chains.",
                inputSchema={"type": "object", "properties": {"token_limit": {"type": "integer", "default": 500}}}),
            Tool(name="check_proactive", description="Check if there are relevant past memories for current context.",
                inputSchema={"type": "object", "properties": {"context": {"type": "string"}, "max_suggestions": {"type": "integer", "default": 3}}, "required": ["context"]}),
            
            # === health and maintenance ===
            Tool(name="get_memory_health", description="Get health report of the memory system.",
                inputSchema={"type": "object", "properties": {}}),
            Tool(name="filter_passive_capture", description="Clean up passive capture noise. Remove clipboard/screenshot entries that are raw dumps.",
                inputSchema={"type": "object", "properties": {"dry_run": {"type": "boolean", "default": True}}}),
            Tool(name="query_by_entity", description="Find memories by extracted entity (IP, path, project name, command, etc).",
                inputSchema={"type": "object", "properties": {
                    "entity_name": {"type": "string"},
                    "entity_type": {"type": "string"},
                    "max_results": {"type": "integer", "default": 10}},
                    "required": ["entity_name"]}),
            Tool(name="memory_timeline", description="Get chronological memory timeline. Shows when things happened, grouped by day.",
                inputSchema={"type": "object", "properties": {"days": {"type": "integer", "default": 30}}}),
            Tool(name="session_wrap", description="End-of-session summary. Identifies key decisions, action items, and what to remember.",
                inputSchema={"type": "object", "properties": {"session_notes": {"type": "string"}}}),
            Tool(name="ingest_sp_history", description="Store Simply Plural front history as memories. Captures who was fronting when.",
                inputSchema={"type": "object", "properties": {"days": {"type": "integer", "default": 30}}}),
            Tool(name="build_graph_edges", description="Rebuild memory graph edges from entity overlaps. Run after bulk imports.",
                inputSchema={"type": "object", "properties": {}}),
            
            # === wiki cross-system tools ===
            Tool(name="wiki_search", description="Search the wiki (personal knowledge base). Returns matching pages with snippets.",
                inputSchema={"type": "object", "properties": {
                    "query": {"type": "string"},
                    "category": {"type": "string"},
                    "limit": {"type": "integer", "default": 5}},
                    "required": ["query"]}),
            Tool(name="wiki_read", description="Read a wiki page by slug.",
                inputSchema={"type": "object", "properties": {"slug": {"type": "string"}}, "required": ["slug"]}),
            Tool(name="wiki_list", description="List all wiki pages, optionally filtered by category.",
                inputSchema={"type": "object", "properties": {"category": {"type": "string"}}}),
            Tool(name="wiki_sweep", description="Audit wiki: find orphans, broken links, untagged pages, missing memster links.",
                inputSchema={"type": "object", "properties": {"category": {"type": "string"}}}),
            
            # === simply plural tools ===
            Tool(name="sp_status", description="Get Simply Plural status: current front, member count, connection state.",
                inputSchema={"type": "object", "properties": {}}),
            Tool(name="sp_members", description="List all headmates from Simply Plural. Filter to active (non-archived) only.",
                inputSchema={"type": "object", "properties": {"include_archived": {"type": "boolean", "default": False}}}),
            
 # === bridge tools (memster + wiki + SP unified) ===
 Tool(name="unified_briefing", description="Pull context from memster + wiki + SP in one call. Use at session start or when you need full context.",
 inputSchema={"type": "object", "properties": {"context": {"type": "string"}, "max_items": {"type": "integer", "default": 15}}}),
 Tool(name="wiki_to_memster_sync", description="Extract key facts from wiki pages into memster memories. Makes wiki content searchable via memster.",
 inputSchema={"type": "object", "properties": {"max_pages": {"type": "integer", "default": 20}}}),
 Tool(name="enrich_memory_lookup", description="Check if content references known entities. Returns wiki pages, related entities, and related memories.",
 inputSchema={"type": "object", "properties": {
 "content": {"type": "string"},
 "category": {"type": "string"}},
 "required": ["content"]}),
 Tool(name="detect_stale_memories", description="Find memories that may be outdated, contradictory, or untouched for 30+ days.",
 inputSchema={"type": "object", "properties": {}}),
 Tool(name="check_memory_quality", description="Check if content is worth storing before committing. Returns quality score and recommendation.",
 inputSchema={"type": "object", "properties": {
 "content": {"type": "string"},
 "category": {"type": "string"}},
 "required": ["content"]}),
 
 # === brainstack-inspired enhanced retrieval tools ===
 Tool(name="record_retrieval_telemetry", description="Record telemetry for memory retrieval events (served_count, match_served_count, last_served_at, query_text)",
 inputSchema={"type": "object", "properties": {
 "memory_id": {"type": "integer"},
 "match_served": {"type": "boolean", "default": False},
 "query_text": {"type": "string"}},
 "required": ["memory_id", "query_text"]}),
 Tool(name="update_temporal_bounds", description="Update temporal validity bounds for a memory (valid_from, valid_to, observed_at). Used for time-sensitive facts.",
 inputSchema={"type": "object", "properties": {
 "memory_id": {"type": "integer"},
 "valid_from": {"type": "string", "description": "ISO datetime string when memory becomes valid"},
 "valid_to": {"type": "string", "description": "ISO datetime string when memory expires (null for no expiration)"},
 "observed_at": {"type": "string", "description": "ISO datetime when this was observed/recorded"}},
 "required": ["memory_id"]}),
 Tool(name="route_retrieval", description="Analyze query and route to appropriate retrieval strategy (high-stakes, bulk, temporal, semantic, budget, none)",
 inputSchema={"type": "object", "properties": {
 "query": {"type": "string"},
 "session_id": {"type": "string"},
 "conversation_id": {"type": "string"}},
 "required": ["query", "session_id"]}),
Tool(name="track_provenance", description="Merge provenance info into memories and detect duplicate sources.",
    inputSchema={"type": "object", "properties": {
        "memory_id": {"type": "integer"},
        "source": {"type": "string", "description": "Source identifier (e.g., screenshot-123, clipboard-456)"},
        "source_type": {"type": "string", "description": "Type: screenshot, clipboard, ocr, manual, etc"},
        "extracted_at": {"type": "string", "description": "ISO datetime when extracted"}},
    "required": ["memory_id"]}),
Tool(name="extract_graph_relations", description="Extract semantic graph relations (works_on, is_at, has_status, etc) from memory content.",
    inputSchema={"type": "object", "properties": {
        "memory_id": {"type": "integer"},
        "content": {"type": "string", "description": "Memory content to analyze (optional, will fetch if not provided)"}},
    "required": ["memory_id"]}),
 Tool(name="get_working_memory_context", description="Get current working memory context for a session including tracked entities and relationship chains.",
 inputSchema={"type": "object", "properties": {
 "session_id": {"type": "string"},
 "conversation_id": {"type": "string"},
 "limit": {"type": "integer", "default": 10}},
 "required": ["session_id"]}),
 Tool(name="infer_follow_up_intent", description="Detect if current query is a follow-up to previous conversation and infer missing context.",
 inputSchema={"type": "object", "properties": {
 "session_id": {"type": "string"},
 "conversation_id": {"type": "string"},
 "query": {"type": "string"}},
"required": ["session_id", "query"]}),
]

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]):
    try:
        # === activity tools ===
        if name == "get_activity_summary":
            return [TextContent(type="text", text=json.dumps({"summary": get_activity_summary_text(hours=arguments.get("hours", 2))}, indent=2))]
            
        elif name == "get_recent_activity":
            return [TextContent(type="text", text=json.dumps(get_recent_activity(hours=arguments.get("hours", 1)), indent=2))]
            
        elif name == "search_window_history":
            return [TextContent(type="text", text=json.dumps(search_window_history(query=arguments.get("query", ""), hours=arguments.get("hours", 24)), indent=2))]
            
        elif name == "search_clipboard":
            return [TextContent(type="text", text=json.dumps(search_clipboard(query=arguments.get("query", ""), hours=arguments.get("hours", 24)), indent=2))]
            
        elif name == "search_ocr_text":
            return [TextContent(type="text", text=json.dumps(search_ocr_text(query=arguments.get("query", ""), limit=arguments.get("limit", 10)), indent=2))]
            
        # === memory CRUD ===
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
                
            # quality gate
            if BRIDGE_AVAILABLE:
                quality = check_memory_quality(content, category)
                if not quality["should_store"]:
                    return [TextContent(type="text", text=json.dumps({
                        "rejected": True,
                        "quality": quality["quality"],
                        "reason": quality["reason"],
                        "content_preview": content[:80]
                    }, indent=2))]
                
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
            cursor.execute(
                "INSERT INTO memories (content, category, t_event, t_recorded, tier, memory_type) VALUES (?, ?, ?, ?, ?, ?)",
                (content, category, now, now, "L2", category if category in ("world", "experience", "opinion", "observation") else "observation")
            )
            mid = cursor.lastrowid
                
            # also add to dedup index if available
            if DEDUP_AVAILABLE:
                try:
                    add_with_dedup(content, mid)
                except Exception:
                    pass
                
            conn.commit()
            conn.close()
                
            # post-insert hooks
            hooks = run_post_insert_hooks(mid, content, category, tags)
                
            return [TextContent(type="text", text=json.dumps({
                "created": True, "id": mid, "category": category,
                "entities_extracted": hooks["entities"],
                "edges_created": hooks["edges"],
                "importance_score": hooks["importance"]
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
                
            # re-run hooks if content changed
            hooks = {}
            if new_content:
                hooks = run_post_insert_hooks(memory_id, new_content, new_category or existing["category"])
                
            return [TextContent(type="text", text=json.dumps({
                "updated": True, "id": memory_id,
                "fields_updated": [k.replace(" = ?", "") for k in updates if k != "t_recorded = ?"],
                "hooks": hooks if hooks else None
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
            cursor.execute("UPDATE edges SET source_id = ? WHERE source_id = ?", (primary_id, secondary_id))
            cursor.execute("UPDATE edges SET target_id = ? WHERE target_id = ?", (primary_id, secondary_id))
                
            # delete secondary
            cursor.execute("DELETE FROM memories WHERE id = ?", (secondary_id,))
                
            conn.commit()
            conn.close()
                
            # re-run hooks on merged content
            hooks = run_post_insert_hooks(primary_id, merged_content, primary["category"])
                
            return [TextContent(type="text", text=json.dumps({
                "merged": True,
                "primary_id": primary_id,
                "deleted_id": secondary_id,
                "merged_content": merged_content[:200],
                "hooks": hooks
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
            # clean up edges
            cursor.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (memory_id, memory_id))
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
                cursor.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (mid, mid))
                cursor.execute("DELETE FROM memories WHERE id = ?", (mid,))
            conn.commit()
            conn.close()
            return [TextContent(type="text", text=json.dumps({"deleted": len(ids), "ids": ids}, indent=2))]
            
        elif name == "find_duplicates":
            groups = find_near_duplicate_memories(arguments.get("similarity_threshold", 0.8))
            result = [{"count": len(g), "memories": [{"id": m["id"], "content": m["content"][:100], "category": m.get("category")} for m in g]} for g in groups]
            return [TextContent(type="text", text=json.dumps({"duplicate_groups": len(result), "groups": result}, indent=2))]
            
        # === semantic search ===
        elif name == "semantic_memory_search":
            if not SEMANTIC_SEARCH_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "semantic search not available"}, indent=2))]
            try:
                results = semantic_search_memories(arguments.get("query", ""), limit=arguments.get("limit", 5), db_path=DB_PATH)
                return [TextContent(type="text", text=json.dumps({"query": arguments.get("query"), "results": results}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        elif name == "hybrid_search":
            try:
                results = hybrid_search(arguments.get("query", ""), limit=arguments.get("limit", 10))
                return [TextContent(type="text", text=json.dumps({"query": arguments.get("query"), "count": len(results), "results": results}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        elif name == "remember_batch":
            try:
                mems = arguments.get("memories", [])
                if not mems:
                    return [TextContent(type="text", text=json.dumps({"error": "memories array required"}, indent=2))]
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
            
        elif name == "find_similar":
            if not SEMANTIC_SEARCH_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "semantic search not available"}, indent=2))]
            try:
                results = find_similar(arguments.get("memory_id"), db_path=DB_PATH, limit=arguments.get("limit", 5))
                return [TextContent(type="text", text=json.dumps({"memory_id": arguments.get("memory_id"), "similar": results}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        # === memory graph ===
        elif name == "get_related_memories":
            if not MEMORY_LINKS_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "memory links not available"}, indent=2))]
            try:
                related = get_related_memories(arguments.get("memory_id"))
                return [TextContent(type="text", text=json.dumps({"memory_id": arguments.get("memory_id"), "related": related[:arguments.get("limit", 10)]}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        elif name == "link_memories":
            if not MEMORY_LINKS_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "memory links not available"}, indent=2))]
            try:
                ok = link_memories(arguments.get("source_id"), arguments.get("target_id"), arguments.get("link_type", "related"))
                return [TextContent(type="text", text=json.dumps({"linked": ok, "source": arguments.get("source_id"), "target": arguments.get("target_id")}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        # === briefing and surfacing ===
        elif name == "get_briefing":
            if CONTEXT_SURFACER_AVAILABLE:
                try:
                    result = surface_context(db_path=DB_PATH)
                    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
                except Exception as e:
                    return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            # fallback
            results = get_memories_with_scoring(max_results=5)
            briefing = "recent memories:\n" + "\n".join("- " + m["content"][:80] for m in results)
            return [TextContent(type="text", text=json.dumps({"briefing": briefing}, indent=2))]
            
        elif name == "get_narrative_briefing":
            if not BRIEFING_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "narrative briefing not available"}, indent=2))]
            try:
                result = get_narrative_briefing(arguments.get("token_limit", 500), DB_PATH)
                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        elif name == "check_proactive":
            if not PROACTIVE_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "proactive surfacing not available"}, indent=2))]
            try:
                result = monitor_and_trigger(arguments.get("context", ""), DB_PATH)
                if not result.get("should_surface"):
                    surfacer = ProactiveSurfacer(DB_PATH)
                    related = surfacer.find_related_past_work(arguments.get("context", ""), arguments.get("max_suggestions", 3))
                    if related:
                        result = {"should_surface": True, "related_memories": related, "message": f"found {len(related)} related memories"}
                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        # === health and maintenance ===
        elif name == "get_memory_health":
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT tier, COUNT(*) as c FROM memories GROUP BY UPPER(tier)")
            tiers = {r["tier"]: r["c"] for r in cursor.fetchall()}
            cursor.execute("SELECT COUNT(*) as total FROM memories")
            total = cursor.fetchone()["total"]
            cursor.execute("SELECT COUNT(*) as c FROM memory_embeddings")
            indexed = cursor.fetchone()["c"]
            cursor.execute("SELECT COUNT(*) as c FROM edges")
            edges = cursor.fetchone()["c"]
            cursor.execute("SELECT COUNT(*) as c FROM extraction")
            extractions = cursor.fetchone()["c"]
            cursor.execute("SELECT COUNT(*) as c FROM screenshot_events")
            shots = cursor.fetchone()["c"]
            cursor.execute("SELECT COUNT(*) as c FROM screenshot_hashes")
            hashed = cursor.fetchone()["c"]
            # avg importance
            cursor.execute("SELECT AVG(COALESCE(importance, 0.5)) as avg_imp FROM memories")
            avg_imp = cursor.fetchone()["avg_imp"]
            conn.close()
            report = {
                "total_memories": total, "tiers": tiers,
                "index_coverage": round(indexed / total * 100, 1) if total else 0,
                "graph_edges": edges, "extracted_entities": extractions,
                "avg_importance": round(avg_imp, 3) if avg_imp else 0,
                "screenshots": shots, "screenshots_hashed": hashed,
                "modules_loaded": {
                    "entity_extraction": ENTITY_EXTRACTION_AVAILABLE,
                    "memory_graph": MEMORY_GRAPH_AVAILABLE,
                    "semantic_search": SEMANTIC_SEARCH_AVAILABLE,
                    "proactive": PROACTIVE_AVAILABLE,
                    "briefing": BRIEFING_AVAILABLE,
                    "importance": IMPORTANCE_AVAILABLE,
                    "decay": DECAY_AVAILABLE,
                    "actr": ACTR_AVAILABLE,
                }
            }
            return [TextContent(type="text", text=json.dumps(report, indent=2))]
            
        elif name == "filter_passive_capture":
            dry_run = arguments.get("dry_run", True)
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, content, category FROM memories WHERE category IN ('clipboard', 'screenshot')")
            to_delete = []
            for row in cursor.fetchall():
                if is_noise_capture(row["content"]):
                    to_delete.append({"id": row["id"], "preview": row["content"][:80]})
                
            if not dry_run:
                for item in to_delete:
                    cursor.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (item["id"], item["id"]))
                    cursor.execute("DELETE FROM memories WHERE id = ?", (item["id"],))
                conn.commit()
                
            conn.close()
            return [TextContent(type="text", text=json.dumps({
                "dry_run": dry_run,
                "noise_found": len(to_delete),
                "items": to_delete[:20],
                "action": "would delete" if dry_run else "deleted"
            }, indent=2))]
            
        elif name == "query_by_entity":
            entity_name = arguments.get("entity_name", "")
            entity_type = arguments.get("entity_type")
            max_results = arguments.get("max_results", 10)
            if not entity_name:
                return [TextContent(type="text", text=json.dumps({"error": "entity_name required"}, indent=2))]
            try:
                conn = get_db_connection()
                c = conn.cursor()
                if entity_type:
                    c.execute("""SELECT m.id, m.content, m.category, m.tier, e.canonical_name, e.entity_type
                        FROM memory_entities me
                        JOIN entities e ON me.entity_id = e.id
                        JOIN memories m ON me.memory_id = m.id
                        WHERE e.canonical_name LIKE ? AND e.entity_type = ?
                        ORDER BY me.confidence DESC LIMIT ?""",
                        (f"%{entity_name}%", entity_type, max_results))
                else:
                    c.execute("""SELECT m.id, m.content, m.category, m.tier, e.canonical_name, e.entity_type
                        FROM memory_entities me
                        JOIN entities e ON me.entity_id = e.id
                        JOIN memories m ON me.memory_id = m.id
                        WHERE e.canonical_name LIKE ?
                        ORDER BY me.confidence DESC LIMIT ?""",
                        (f"%{entity_name}%", max_results))
                results = [{"memory_id": r["id"], "content": r["content"][:200], "category": r["category"],
                            "entity": r["canonical_name"], "entity_type": r["entity_type"]} for r in c.fetchall()]
                conn.close()
                return [TextContent(type="text", text=json.dumps({"entity": entity_name, "count": len(results), "results": results}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        elif name == "memory_timeline":
            days = arguments.get("days", 30)
            try:
                conn = get_db_connection()
                c = conn.cursor()
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                c.execute("""SELECT DATE(t_recorded) as day, COUNT(*) as count, 
                    GROUP_CONCAT(category) as categories
                    FROM memories WHERE t_recorded > ?
                    GROUP BY DATE(t_recorded) ORDER BY day DESC""", (cutoff,))
                timeline = [{"date": r["day"], "count": r["count"], "categories": r["categories"]} for r in c.fetchall()]
                # also get total
                c.execute("SELECT COUNT(*) FROM memories")
                total = c.fetchone()[0]
                conn.close()
                return [TextContent(type="text", text=json.dumps({"days": days, "total_memories": total, "timeline": timeline}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        elif name == "session_wrap":
            session_notes = arguments.get("session_notes", "")
            try:
                conn = get_db_connection()
                c = conn.cursor()
                # get memories from this session (last hour)
                cutoff = (datetime.now() - timedelta(hours=2)).isoformat()
                c.execute("SELECT id, content, category FROM memories WHERE t_recorded > ? ORDER BY t_recorded", (cutoff,))
                recent = [dict(r) for r in c.fetchall()]
                conn.close()
                    
                result = {
                    "session_memories": len(recent),
                    "memories": [{"id": m["id"], "content": m["content"][:100], "category": m["category"]} for m in recent],
                    "session_notes": session_notes,
                }
                    
                # if session_notes provided, store as a memory
                if session_notes:
                    conn2 = get_db_connection()
                    c2 = conn2.cursor()
                    now_str = datetime.now().isoformat()
                    c2.execute("INSERT INTO memories (content, category, t_event, t_recorded, tier, memory_type) VALUES (?, 'experience', ?, ?, 'L1', 'experience')",
                               (f"session wrap: {session_notes}", now_str, now_str))
                    result["wrap_memory_id"] = c2.lastrowid
                    conn2.commit()
                    conn2.close()
                    
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        elif name == "ingest_sp_history":
            days = arguments.get("days", 30)
            try:
                import sp_integration
                client = sp_integration.get_client()
                client.initialize()
                members = client.get_members()
                member_map = {m["id"]: m for m in members}
                    
                history = sp_integration.fetch_front_history(limit=days)
                    
                conn = get_db_connection()
                c = conn.cursor()
                now_str = datetime.now().isoformat()
                ingested = 0
                    
                for entry in history:
                    m = member_map.get(entry["member_id"], {})
                    name = m.get("name", "unknown")
                    color = m.get("color", "#888888")
                    ts = entry.get("timestamp", 0)
                        
                    # convert timestamp
                    if isinstance(ts, (int, float)) and ts > 1000000000000:
                        from datetime import datetime as dt
                        dt_obj = dt.fromtimestamp(ts / 1000)
                        time_str = dt_obj.isoformat()
                    else:
                        time_str = str(ts)
                        
                    content = f"{name} was fronting (color: {color})"
                        
                    # dedup
                    c.execute("SELECT id FROM memories WHERE content = ? AND t_event = ?", (content, time_str))
                    if c.fetchone():
                        continue
                        
                    c.execute("INSERT INTO memories (content, category, t_event, t_recorded, tier, memory_type) VALUES (?, 'experience', ?, ?, 'L2', 'experience')",
                              (content, time_str, now_str))
                    ingested += 1
                    
                conn.commit()
                conn.close()
                    
                return [TextContent(type="text", text=json.dumps({"ingested": ingested, "total_history": len(history)}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        elif name == "build_graph_edges":
            try:
                conn = get_db_connection()
                c = conn.cursor()
                    
                # find memories that share entities
                c.execute("""SELECT me1.memory_id as m1, me2.memory_id as m2
                    FROM memory_entities me1
                    JOIN memory_entities me2 ON me1.entity_id = me2.entity_id AND me1.memory_id < me2.memory_id
                    GROUP BY me1.memory_id, me2.memory_id""")
                pairs = c.fetchall()
                    
                edges_created = 0
                for p in pairs:
                    c.execute("INSERT OR IGNORE INTO memory_edges (source_memory_id, target_memory_id, relation_type, weight) VALUES (?, ?, 'shared_entity', 0.5)",
                              (p["m1"], p["m2"]))
                    if c.changes:
                        edges_created += 1
                    
                conn.commit()
                    
                # count total
                c.execute("SELECT COUNT(*) FROM memory_edges")
                total = c.fetchone()[0]
                conn.close()
                    
                return [TextContent(type="text", text=json.dumps({"edges_created": edges_created, "total_edges": total}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        # === wiki cross-system tools ===
        elif name == "wiki_search":
            if not WIKI_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "wiki not available"}, indent=2))]
            try:
                conn = _wiki_get_db()
                c = conn.cursor()
                query = arguments.get("query", "")
                category = arguments.get("category")
                limit = arguments.get("limit", 5)
                    
                if category:
                    c.execute("""SELECT p.slug, p.title, p.category, snippet(pages_fts, 3, '>>>', '<<<', '...', 32) as snippet
                        FROM pages_fts fts JOIN pages p ON fts.slug = p.slug
                        WHERE pages_fts MATCH ? AND fts.category = ? ORDER BY rank LIMIT ?""", (query, category, limit))
                else:
                    c.execute("""SELECT p.slug, p.title, p.category, snippet(pages_fts, 3, '>>>', '<<<', '...', 32) as snippet
                        FROM pages_fts fts JOIN pages p ON fts.slug = p.slug
                        WHERE pages_fts MATCH ? ORDER BY rank LIMIT ?""", (query, limit))
                results = [dict(r) for r in c.fetchall()]
                conn.close()
                return [TextContent(type="text", text=json.dumps({"query": query, "count": len(results), "results": results}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        elif name == "wiki_read":
            if not WIKI_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "wiki not available"}, indent=2))]
            try:
                from wiki_mcp_server import read_page
                page = read_page(arguments.get("slug", ""))
                if not page:
                    return [TextContent(type="text", text=json.dumps({"error": f"page not found"}, indent=2))]
                return [TextContent(type="text", text=json.dumps(page, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        elif name == "wiki_list":
            if not WIKI_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "wiki not available"}, indent=2))]
            try:
                conn = _wiki_get_db()
                c = conn.cursor()
                cat = arguments.get("category")
                if cat:
                    c.execute("SELECT slug, title, category, word_count, updated FROM pages WHERE category = ? ORDER BY updated DESC", (cat,))
                else:
                    c.execute("SELECT slug, title, category, word_count, updated FROM pages ORDER BY category, slug")
                results = [dict(r) for r in c.fetchall()]
                conn.close()
                return [TextContent(type="text", text=json.dumps({"count": len(results), "pages": results}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        elif name == "wiki_sweep":
            if not WIKI_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "wiki not available"}, indent=2))]
            try:
                conn = _wiki_get_db()
                c = conn.cursor()
                issues = []
                    
                # orphans
                c.execute("SELECT slug, title FROM pages WHERE link_count = 0 AND slug NOT IN (SELECT target_slug FROM links)")
                orphans = [dict(r) for r in c.fetchall()]
                if orphans:
                    issues.append({"type": "orphans", "count": len(orphans), "pages": orphans})
                    
                # broken links
                c.execute("SELECT DISTINCT l.source_slug, l.target_slug FROM links l LEFT JOIN pages p ON l.target_slug = p.slug WHERE p.slug IS NULL")
                broken = [{"source": r["source_slug"], "target": r["target_slug"]} for r in c.fetchall()]
                if broken:
                    issues.append({"type": "broken_links", "count": len(broken), "links": broken})
                    
                # untagged
                c.execute("SELECT slug, title FROM pages WHERE slug NOT IN (SELECT slug FROM tags)")
                untagged = [dict(r) for r in c.fetchall()]
                if untagged:
                    issues.append({"type": "untagged", "count": len(untagged), "pages": untagged})
                    
                c.execute("SELECT COUNT(*) FROM pages")
                total = c.fetchone()[0]
                conn.close()
                    
                return [TextContent(type="text", text=json.dumps({"total_pages": total, "issues": issues}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        # === simply plural tools ===
        elif name == "sp_status":
            try:
                import sp_integration
                client = sp_integration.get_client()
                connected = client.initialize()
                front = client.get_front()
                members = client.get_members()
                active = [m for m in members if not m.get("archived")]
                return [TextContent(type="text", text=json.dumps({
                    "connected": connected,
                    "current_front": front,
                    "total_members": len(members),
                    "active_members": len(active),
                    "polling": client._polling,
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        elif name == "sp_members":
            try:
                import sp_integration
                client = sp_integration.get_client()
                client.initialize()
                members = client.get_members()
                include_archived = arguments.get("include_archived", False)
                if not include_archived:
                    members = [m for m in members if not m.get("archived")]
                # simplify output
                result = [{"name": m["name"], "color": m["color"], "pronouns": m.get("pronouns", ""), "id": m["id"]} for m in members]
                return [TextContent(type="text", text=json.dumps({"count": len(result), "members": result}, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        # === bridge tools ===
        elif name == "unified_briefing":
            if not BRIDGE_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "bridge not available"}, indent=2))]
            try:
                result = unified_briefing(
                    context=arguments.get("context", ""),
                    max_items=arguments.get("max_items", 15))
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        elif name == "wiki_to_memster_sync":
            if not BRIDGE_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "bridge not available"}, indent=2))]
            try:
                result = wiki_to_memster(max_pages=arguments.get("max_pages", 20))
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        elif name == "enrich_memory_lookup":
            if not BRIDGE_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "bridge not available"}, indent=2))]
            try:
                result = enrich_memory(
                    content=arguments.get("content", ""),
                    category=arguments.get("category", "general"))
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            
        elif name == "detect_stale_memories":
            if not BRIDGE_AVAILABLE:
                return [TextContent(type="text", text=json.dumps({"error": "bridge not available"}, indent=2))]
            try:
                result = detect_stale_memories()
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
    
        elif name == "check_memory_quality":
if not BRIDGE_AVAILABLE:
return [TextContent(type="text", text=json.dumps({"error": "bridge not available"}, indent=2))]
try:
result = check_memory_quality(
content=arguments.get("content", ""),
category=arguments.get("category", "general"),
)
return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
    return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "get_working_memory_context":
    if not BRIDGE_AVAILABLE:
        return [TextContent(type="text", text=json.dumps({"error": "bridge not available"}, indent=2))]
    try:
        result = get_working_memory_context(
            session_id=arguments.get("session_id", "default"),
            conversation_id=arguments.get("conversation_id", "default"),
            limit=arguments.get("limit", 10)
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "update_temporal_bounds":
try:
from temporal_validity import TemporalValidityTracker
        tv = TemporalValidityTracker()
result = tv.set_temporal_bounds(
memory_id=arguments.get("memory_id"),
valid_from=arguments.get("valid_from"),
valid_to=arguments.get("valid_to"),
observed_at=arguments.get("observed_at")
)
return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "route_retrieval":
try:
from retrieval_routing import RetrievalRouter
        router = RetrievalRouter()
result = router.route_query(
query=arguments.get("query", ""),
session_id=arguments.get("session_id", ""),
conversation_id=arguments.get("conversation_id")
)
return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
            except Exception as e:
return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "track_provenance":
try:
from provenance_utils import ProvenanceTracker
tracker = ProvenanceTracker()
result = tracker.add_source(
memory_id=arguments.get("memory_id"),
source=arguments.get("source"),
source_type=arguments.get("source_type"),
extracted_at=arguments.get("extracted_at")
)
return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

        elif name == "extract_graph_relations":
try:
from graph_extraction import GraphExtractor
extractor = GraphExtractor()
memory_id = arguments.get("memory_id")
content = arguments.get("content")
if not content:
conn = get_db_connection()
cursor = conn.cursor()
cursor.execute("SELECT content FROM memories WHERE id = ?", (memory_id,))
row = cursor.fetchone()
conn.close()
if row:
content = row["content"]
else:
return [TextContent(type="text", text=json.dumps({"error": "memory not found"}, indent=2))]
result = extractor.extract_relations(content)
return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]


        elif name == "infer_follow_up_intent":
try:
from working_memory_policy import WorkingMemoryPolicy
policy = WorkingMemoryPolicy()
result = policy.infer_follow_up_intent(
session_id=arguments.get("session_id", ""),
conversation_id=arguments.get("conversation_id"),
query=arguments.get("query", "")
)
        
            except Exception as e:
            logger.error("tool error: " + str(e))
            return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]


if __name__ == "__main__":
    if MCP_AVAILABLE:
        async def main():
            logger.info("memster MCP server v3 starting (enhanced)")
            async with stdio_server() as (read, write):
                await app.run(read, write, app.create_initialization_options())
        import asyncio
        asyncio.run(main())
    else:
        print("mcp package not available")
