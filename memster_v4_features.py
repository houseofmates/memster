"""
Memster V4 Feature Module - 10 new features for the memory system.
Loaded by memster_mcp_server.py as an import.
"""

import json
import logging
import os
import re
import sqlite3
import zlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("memster.v4")

# Database connection helper
DEFAULT_DB_PATH = os.path.expanduser("~/memster/memster_unified.db")

def get_db(db_path: str = None):
    """Get a database connection with row factory."""
    conn = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# === Feature 1: Bayesian Confidence Scoring ===

def bayesian_update(memory_id: int, observation_result: bool, db_path: str = None) -> Dict:
    """Bayesian confidence update. Confirm: P *= 1.2 (cap 1.0). Contradict: P *= 0.6 (floor 0.0)."""
    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT confidence_score, observation_log FROM memories WHERE id = ?", (memory_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return {"error": f"memory {memory_id} not found"}

    old_confidence = row["confidence_score"] if row["confidence_score"] is not None else 0.5
    log = json.loads(row["observation_log"] or "[]")

    if observation_result:
        new_confidence = min(old_confidence * 1.2, 1.0)
        log.append({"t": datetime.now().isoformat(), "result": True, "old": old_confidence, "new": new_confidence})
    else:
        new_confidence = max(old_confidence * 0.6, 0.0)
        log.append({"t": datetime.now().isoformat(), "result": False, "old": old_confidence, "new": new_confidence})

    c.execute("UPDATE memories SET confidence_score = ?, observation_log = ? WHERE id = ?",
              (new_confidence, json.dumps(log[-50:]), memory_id))  # Keep last 50 observations
    conn.commit()
    conn.close()

    return {
        "updated": True,
        "memory_id": memory_id,
        "old_confidence": round(old_confidence, 4),
        "new_confidence": round(new_confidence, 4),
        "observation_confirmed": observation_result
    }


def score_memory_confidence_v4(memory_id: int, db_path: str = None) -> Dict:
    """Multi-signal confidence scoring for a memory."""
    conn = get_db()
    c = conn.cursor()

    c.execute("""SELECT id, content, category, network_type, importance, decay_score,
                 access_count, confidence_score, observation_log, valid_from, valid_to, pinned
                 FROM memories WHERE id = ?""", (memory_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return {"error": f"memory {memory_id} not found"}

    # Signal 1: Bayesian confidence (if observations exist)
    bayesian = row["confidence_score"] if row["confidence_score"] is not None else 0.5
    obs_log = json.loads(row["observation_log"] or "[]")
    obs_count = len(obs_log)

    # Signal 2: Importance weight
    importance = row["importance"] or 0.5

    # Signal 3: Access frequency (more accesses = more verified)
    access = row["access_count"] or 0
    access_signal = min(access / 10.0, 1.0)

    # Signal 4: Specificity (concrete details = more trustworthy)
    content = row["content"] or ""
    specificity = 0.0
    if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', content):
        specificity += 0.15
    if re.search(r'/[a-zA-Z][^\s,]{3,}', content):
        specificity += 0.10
    if re.search(r'\b\d+\.\d+\b', content):
        specificity += 0.10
    if re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d', content, re.I):
        specificity += 0.10
    if len(content.split()) > 10:
        specificity += 0.05

    # Signal 5: Recency (newer = slightly more trusted if no observations)
    decay = row["decay_score"] if row["decay_score"] is not None else 1.0
    recency_signal = 1.0 - decay

    # Signal 6: Pinned memories are trusted
    pinned = row["pinned"] if row["pinned"] is not None else 0
    pinned_signal = 0.2 if pinned else 0.0

    # Composite: weighted average
    # Bayesian gets 40% weight if we have observations, else importance takes over
    if obs_count > 0:
        composite = (bayesian * 0.40) + (importance * 0.20) + (access_signal * 0.15) + (specificity * 0.10) + (recency_signal * 0.10) + (pinned_signal * 0.05)
    else:
        composite = (importance * 0.35) + (0.5 * 0.15) + (access_signal * 0.20) + (specificity * 0.15) + (recency_signal * 0.10) + (pinned_signal * 0.05)

    composite = min(max(composite, 0.0), 1.0)

    # Temporal validity check
    now = datetime.now().isoformat()
    temporal_status = "valid"
    if row["valid_to"] and row["valid_to"] < now:
        temporal_status = "expired"
        composite *= 0.5  # Halve confidence for expired memories

    conn.close()

    return {
        "memory_id": memory_id,
        "composite_confidence": round(composite, 4),
        "signals": {
            "bayesian": round(bayesian, 4),
            "importance": round(importance, 4),
            "access_frequency": round(access_signal, 4),
            "specificity": round(specificity, 4),
            "recency": round(recency_signal, 4),
            "pinned_boost": round(pinned_signal, 4)
        },
        "observation_count": obs_count,
        "temporal_status": temporal_status
    }


# === Feature 2: Semantic Contradiction Detection ===

NEGATION_WORDS = {"not", "no", "never", "isn't", "aren't", "wasn't", "weren't",
                  "doesn't", "don't", "didn't", "won't", "wouldn't", "can't",
                  "cannot", "shouldn't", "never", "neither", "nor", "nothing", "nowhere"}

ANTONYM_PAIRS = [
    ("enable", "disable"), ("start", "stop"), ("up", "down"), ("left", "right"),
    ("on", "off"), ("true", "false"), ("yes", "no"), ("add", "remove"),
    ("install", "uninstall"), ("open", "close"), ("connect", "disconnect"),
    ("increase", "decrease"), ("create", "delete"), ("begin", "end"),
    ("accept", "reject"), ("allow", "deny"), ("push", "pull"),
    ("attach", "detach"), ("mount", "unmount"), ("load", "unload"),
    ("fixed", "broken"), ("working", "broken"), ("alive", "dead"),
    ("running", "stopped"), ("active", "inactive"), ("online", "offline"),
]


def _extract_key_facts(content: str) -> List[str]:
    """Extract key factual claims from a memory."""
    sentences = re.split(r'[.!?]\s*', content)
    facts = []
    for s in sentences:
        s = s.strip()
        if len(s) < 5:
            continue
        # Skip meta statements
        if any(s.lower().startswith(p) for p in ["i ", "we ", "maybe", "perhaps"]):
            continue
        facts.append(s)
    return facts


def _check_pair_contradiction(text_a: str, text_b: str) -> Optional[Dict]:
    """Check if two memory texts contain a contradiction."""
    norm_a = normalize_text_v4(text_a)
    norm_b = normalize_text_v4(text_b)
    words_a = set(norm_a.split())
    words_b = set(norm_b.split())

    # Need significant word overlap to be about the same topic
    overlap = words_a & words_b
    union = words_a | words_b
    if not union:
        return None
    jaccard = len(overlap) / len(union)
    if jaccard < 0.3:
        return None

    # Check 1: One has negation the other doesn't
    neg_a = words_a & NEGATION_WORDS
    neg_b = words_b & NEGATION_WORDS
    if neg_a and not neg_b and jaccard > 0.4:
        # A negates something B affirms
        shared_content = overlap - NEGATION_WORDS
        if len(shared_content) >= 2:
            return {"type": "negation_mismatch", "confidence": min(jaccard + 0.1, 0.95),
                    "detail": f"one negates: {shared_content}"}
    if neg_b and not neg_a and jaccard > 0.4:
        shared_content = overlap - NEGATION_WORDS
        if len(shared_content) >= 2:
            return {"type": "negation_mismatch", "confidence": min(jaccard + 0.1, 0.95),
                    "detail": f"one negates: {shared_content}"}

    # Check 2: Antonym pairs
    for w1, w2 in ANTONYM_PAIRS:
        if w1 in norm_a and w2 in norm_b:
            return {"type": "antonym_conflict", "confidence": jaccard + 0.15,
                    "detail": f"{w1} vs {w2}"}
        if w2 in norm_a and w1 in norm_b:
            return {"type": "antonym_conflict", "confidence": jaccard + 0.15,
                    "detail": f"{w2} vs {w1}"}

    # Check 3: Conflicting values (IPs, versions, numbers)
    ips_a = set(re.findall(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', text_a))
    ips_b = set(re.findall(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', text_b))
    if ips_a and ips_b and ips_a != ips_b and jaccard > 0.25:
        return {"type": "conflicting_values", "confidence": 0.8,
                "detail": f"IP mismatch: {ips_a} vs {ips_b}"}

    versions_a = set(re.findall(r'v?(\d+\.\d+(?:\.\d+)?)', text_a))
    versions_b = set(re.findall(r'v?(\d+\.\d+(?:\.\d+)?)', text_b))
    if versions_a and versions_b and versions_a != versions_b and jaccard > 0.25:
        return {"type": "conflicting_values", "confidence": 0.7,
                "detail": f"version mismatch: {versions_a} vs {versions_b}"}

    return None


def detect_contradictions_v4(threshold: float = 0.5, db_path: str = None) -> Dict:
    """Detect contradictory memory pairs."""

    conn = get_db()
    c = conn.cursor()

    # Get memories grouped by category for efficient comparison
    c.execute("SELECT id, content, category, network_type FROM memories ORDER BY category, id")
    all_memories = [dict(r) for r in c.fetchall()]
    conn.close()

    contradictions = []
    checked = 0

    # Compare within same category first (O(n^2) within category, much smaller)
    from collections import defaultdict
    by_category = defaultdict(list)
    for m in all_memories:
        cat = m.get("category") or m.get("network_type") or "unknown"
        by_category[cat].append(m)

    for cat, mems in by_category.items():
        for i in range(len(mems)):
            for j in range(i + 1, min(i + 50, len(mems))):  # Limit comparisons per category
                checked += 1
                result = _check_pair_contradiction(mems[i]["content"], mems[j]["content"])
                if result and result["confidence"] >= threshold:
                    contradictions.append({
                        "memory_a": {"id": mems[i]["id"], "content": mems[i]["content"][:150]},
                        "memory_b": {"id": mems[j]["id"], "content": mems[j]["content"][:150]},
                        "contradiction_type": result["type"],
                        "confidence": round(result["confidence"], 3),
                        "detail": result["detail"]
                    })

    return {
        "contradictions_found": len(contradictions),
        "pairs_checked": checked,
        "contradictions": contradictions[:20]  # Cap output
    }


# === Feature 3: Enhanced Memory Linking ===

ENHANCED_RELATION_TYPES = ["related", "co_occurrence", "causal", "temporal",
                           "contradiction", "supports", "refines", "supercedes"]


def auto_link_memory(memory_id: int, content: str, category: str,
                     conversation_id: str = None, t_event: str = None,
                     db_path: str = None) -> Dict:
    """Auto-create edges when a new memory is stored."""

    conn = get_db()
    c = conn.cursor()
    edges_created = 0

    # 1. Co-occurrence: find memories sharing entities
    entities = _extract_entities_simple(content)
    for entity in entities:
        c.execute("SELECT id FROM memories WHERE id != ? AND content LIKE ? LIMIT 10",
                  (memory_id, f"%{entity}%"))
        for row in c.fetchall():
            target_id = row[0]
            if target_id != memory_id:
                c.execute("""INSERT OR IGNORE INTO memory_edges
                    (source_memory_id, target_memory_id, relation_type, weight)
                    VALUES (?, ?, 'co_occurrence', 0.3)""",
                    (memory_id, target_id))
                edges_created += 1

    # 2. Temporal: find memories from same session/time window
    if conversation_id:
        c.execute("""SELECT id FROM memories
                     WHERE conversation_id = ? AND id != ?
                     ORDER BY t_event DESC LIMIT 5""",
                  (conversation_id, memory_id))
        for row in c.fetchall():
            c.execute("""INSERT OR IGNORE INTO memory_edges
                (source_memory_id, target_memory_id, relation_type, weight)
                VALUES (?, ?, 'temporal', 0.4)""",
                (memory_id, row[0]))
            edges_created += 1

    # 3. Contradiction: check against recent memories
    c.execute("""SELECT id, content FROM memories
                 WHERE id != ? AND category = ?
                 ORDER BY t_event DESC LIMIT 20""",
              (memory_id, category))
    for row in c.fetchall():
        contradiction = _check_pair_contradiction(content, row[1])
        if contradiction and contradiction["confidence"] >= 0.5:
            c.execute("""INSERT OR IGNORE INTO memory_edges
                (source_memory_id, target_memory_id, relation_type, weight)
                VALUES (?, ?, 'contradiction', ?)""",
                (memory_id, row[0], contradiction["confidence"]))
            edges_created += 1

    # 4. Supports/refines: check if new memory elaborates on existing one
    norm_content = normalize_text_v4(content)
    c.execute("""SELECT id, content FROM memories
                 WHERE id != ? AND category = ?
                 ORDER BY importance DESC LIMIT 10""",
              (memory_id, category))
    for row in c.fetchall():
        norm_existing = normalize_text_v4(row[1])
        existing_words = set(norm_existing.split())
        new_words = set(norm_content.split())
        if existing_words and existing_words.issubset(new_words) and len(existing_words) >= 3:
            c.execute("""INSERT OR IGNORE INTO memory_edges
                (source_memory_id, target_memory_id, relation_type, weight)
                VALUES (?, ?, 'refines', 0.6)""",
                (memory_id, row[0]))
            edges_created += 1

    conn.commit()
    conn.close()

    return {"auto_linked": True, "edges_created": edges_created}


def _extract_entities_simple(content: str) -> List[str]:
    """Extract likely entities (IPs, paths, hostnames, proper nouns) from content."""
    entities = []
    # IPs
    entities.extend(re.findall(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', content))
    # Paths
    entities.extend(re.findall(r'(?:~|/)[a-zA-Z][\w/.-]+', content))
    # Docker/container names
    entities.extend(re.findall(r'\b[\w-]+-(?:server|container|backend|frontend|service|daemon)\b', content))
    # Domain patterns
    entities.extend(re.findall(r'\b[\w-]+\.[\w.-]+\.\w{2,}\b', content))
    # Remove duplicates, limit
    return list(set(entities))[:10]


# === Feature 4: Context-Aware Retrieval Reranking ===

def assemble_context_packet_v4(query: str = "", max_tokens: int = 2000,
                                context_type: str = "auto", db_path: str = None) -> Dict:
    """Assemble a token-budgeted context packet with diversity injection."""

    conn = get_db()
    c = conn.cursor()

    # Step 1: Get candidate memories
    candidates = []

    if query:
        # FTS search
        try:
            c.execute("""SELECT m.*, rank
                        FROM memories m
                        JOIN memories_fts fts ON m.id = fts.rowid
                        WHERE memories_fts MATCH ?
                        ORDER BY rank, m.importance DESC
                        LIMIT 30""", (query,))
            for row in c.fetchall():
                candidates.append(dict(row))
        except Exception:
            pass

        # Also get recent important ones as fallback
        if len(candidates) < 5:
            c.execute("""SELECT * FROM memories
                        WHERE importance >= 0.5
                        ORDER BY t_event DESC LIMIT 20""")
            for row in c.fetchall():
                if not any(r["id"] == dict(row)["id"] for r in candidates):
                    candidates.append(dict(row))
    else:
        # No query: get recent + important
        c.execute("""SELECT * FROM memories
                    ORDER BY importance * (1.0 - COALESCE(decay_score, 0)) DESC, t_event DESC
                    LIMIT 30""")
        candidates = [dict(r) for r in c.fetchall()]

    conn.close()

    if not candidates:
        return {"query": query, "token_budget": max_tokens, "memories": [],
                "tokens_used": 0, "note": "no candidates found"}

    # Step 2: Score and rerank with diversity
    scored = []
    seen_entities = set()
    seen_categories = set()
    category_counts = {}

    for mem in candidates:
        # Base score: importance * (1 - decay) * access_bonus
        base_score = (mem.get("importance") or 0.5) * (1.0 - (mem.get("decay_score") or 0))
        access_bonus = min((mem.get("access_count") or 0) / 20.0, 0.2)
        confidence_bonus = (mem.get("confidence_score") or 0.5) * 0.1
        score = base_score + access_bonus + confidence_bonus

        # Diversity penalty: if same entity appeared 3+ times, downweight
        mem_entities = set(_extract_entities_simple(mem.get("content", "")))
        entity_overlap = len(mem_entities & seen_entities)
        if entity_overlap >= 3:
            score *= 0.5
        elif entity_overlap >= 2:
            score *= 0.7

        # Category diversity: if top results are all same category, swap one
        cat = mem.get("category") or mem.get("network_type") or "unknown"
        cat_count = category_counts.get(cat, 0)
        if cat_count >= 3:
            score *= 0.6  # Penalize 4th+ result from same category

        # Pinned boost
        if mem.get("pinned"):
            score += 0.3

        scored.append({**mem, "retrieval_score": round(score, 4)})

        # Track for diversity
        seen_entities.update(mem_entities)
        category_counts[cat] = cat_count + 1

    # Sort by score
    scored.sort(key=lambda x: x["retrieval_score"], reverse=True)

    # Step 3: Fit within token budget
    selected = []
    tokens_used = 0
    # Rough estimate: 1 token ~= 4 chars
    for mem in scored:
        content = mem.get("content", "")
        est_tokens = len(content) // 4
        if tokens_used + est_tokens > max_tokens:
            continue
        selected.append({
            "id": mem["id"],
            "content": content,
            "category": mem.get("category") or mem.get("network_type"),
            "confidence": round(mem.get("confidence_score") or 0.5, 3),
            "retrieval_score": mem["retrieval_score"]
        })
        tokens_used += est_tokens

    return {
        "query": query,
        "context_type": context_type,
        "token_budget": max_tokens,
        "tokens_used": tokens_used,
        "memory_count": len(selected),
        "memories": selected
    }


# === Feature 5: Memory Expiry & Active Staleness ===

def detect_stale_memories_v4(db_path: str = None) -> Dict:
    """Find memories that may be outdated using multiple heuristics."""

    conn = get_db()
    c = conn.cursor()

    now = datetime.now().isoformat()
    stale = []

    # 1. Temporally expired (valid_to passed)
    c.execute("""SELECT id, content, valid_to, category FROM memories
                 WHERE valid_to IS NOT NULL AND valid_to < ?""", (now,))
    for row in c.fetchall():
        stale.append({"id": row["id"], "content": row["content"][:100],
                      "reason": "valid_to_expired", "valid_to": row["valid_to"]})

    # 2. High decay + low importance + low access = likely stale
    c.execute("""SELECT id, content, decay_score, importance, access_count FROM memories
                 WHERE decay_score > 0.8 AND importance < 0.3 AND access_count < 2
                 LIMIT 50""")
    for row in c.fetchall():
        if not any(s["id"] == row["id"] for s in stale):
            stale.append({"id": row["id"], "content": row["content"][:100],
                          "reason": "high_decay_low_importance",
                          "decay_score": row["decay_score"]})

    # 3. Check for superseded memories (newer memory with same entity but different value)
    c.execute("""SELECT m1.id, m1.content, m2.id as newer_id, m2.content as newer_content
                 FROM memories m1
                 JOIN memories m2 ON m1.category = m2.category
                 WHERE m1.id < m2.id AND m1.t_event < m2.t_event
                 AND m1.network_type IN ('world', 'observation')
                 AND m2.network_type IN ('world', 'observation')
                 LIMIT 50""")
    seen_pairs = set()
    for row in c.fetchall():
        pair_key = (row["id"], row["newer_id"])
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        contradiction = _check_pair_contradiction(row["content"], row["newer_content"])
        if contradiction and contradiction["confidence"] >= 0.6:
            if not any(s["id"] == row["id"] for s in stale):
                stale.append({"id": row["id"], "content": row["content"][:100],
                              "reason": "superseded_by_newer",
                              "superseded_by": row["newer_id"],
                              "conflict_type": contradiction["type"]})

    conn.close()
    return {"stale_count": len(stale), "stale_memories": stale[:30]}


# === Feature 6: Auto Compression in Consolidation ===

def compress_old_memories(db_path: str = None, days_threshold: int = 60,
                          min_group_size: int = 3) -> Dict:
    """Compress groups of old unused memories into summaries."""

    conn = get_db()
    c = conn.cursor()

    cutoff = (datetime.now() - timedelta(days=days_threshold)).isoformat()

    # Find old, unaccessed, non-pinned L2/L3 memories
    c.execute("""SELECT id, content, category, network_type, t_event
                 FROM memories
                 WHERE t_event < ? AND access_count = 0
                 AND (pinned IS NULL OR pinned = 0)
                 AND tier IN ('L2', 'L3')
                 ORDER BY category, t_event""",
              (cutoff,))
    old_memories = [dict(r) for r in c.fetchall()]

    if not old_memories:
        conn.close()
        return {"compressed_groups": 0, "memories_removed": 0, "note": "no candidates"}

    # Group by category
    from collections import defaultdict
    by_category = defaultdict(list)
    for m in old_memories:
        cat = m.get("category") or m.get("network_type") or "unknown"
        by_category[cat].append(m)

    compressed_groups = 0
    total_removed = 0

    for cat, mems in by_category.items():
        if len(mems) < min_group_size:
            continue

        # Compress this group: create a summary memory
        group_ids = [m["id"] for m in mems]
        group_content_parts = [m["content"] for m in mems[:10]]  # Limit source material

        # Simple compression: extract key entities and create condensed text
        all_entities = set()
        for content in group_content_parts:
            all_entities.update(_extract_entities_simple(content))

        # Build compressed content
        summary = f"[compressed from {len(mems)} memories, {cat}] "
        summary += "; ".join(set(m["content"][:80] for m in mems[:8]))

        # Store compressed memory
        now = datetime.now().isoformat()
        c.execute("""INSERT INTO memories
                     (content, network_type, t_event, t_recorded, category, tier,
                      importance, decay_score, confidence_score)
                     VALUES (?, 'observation', ?, ?, ?, 'L3', 0.3, 0.0, 0.5)""",
                  (summary, now, now, cat))
        new_id = c.lastrowid

        # Track in compressed_memories table
        for mid in group_ids:
            c.execute("""INSERT OR IGNORE INTO compressed_memories
                        (compressed_memory_id, original_memory_id, original_content)
                        VALUES (?, ?, ?)""",
                      (new_id, mid, next((m["content"] for m in mems if m["id"] == mid), "")))

        # Delete originals (they're preserved in compressed_memories)
        placeholders = ",".join("?" * len(group_ids))
        c.execute(f"DELETE FROM memory_edges WHERE source_memory_id IN ({placeholders})", group_ids)
        c.execute(f"DELETE FROM memory_edges WHERE target_memory_id IN ({placeholders})", group_ids)
        c.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", group_ids)

        compressed_groups += 1
        total_removed += len(group_ids)

    conn.commit()
    conn.close()

    return {
        "compressed_groups": compressed_groups,
        "memories_removed": total_removed,
        "days_threshold": days_threshold
    }


# === Feature 7: Bidirectional Wiki-Memster Sync ===

WIKI_DIR = os.path.expanduser("~/.hermes/wiki")


def _parse_wiki_frontmatter(content: str) -> Tuple[Dict, str]:
    """Parse YAML frontmatter from wiki markdown."""
    meta = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            # Simple YAML parsing (no pyyaml needed)
            for line in parts[1].strip().split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    meta[key.strip()] = val.strip().strip('"').strip("'")
            body = parts[2].strip()
    return meta, body


def wiki_search_v4(query: str, category: str = None, limit: int = 5) -> List[Dict]:
    """Search wiki pages by content."""
    if not os.path.isdir(WIKI_DIR):
        return []

    results = []
    query_lower = query.lower()

    for fname in os.listdir(WIKI_DIR):
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(WIKI_DIR, fname)
        try:
            with open(fpath, 'r') as f:
                content = f.read()
            meta, body = _parse_wiki_frontmatter(content)

            # Category filter
            if category and meta.get("category", "").lower() != category.lower():
                continue

            # Search in body and title
            title = meta.get("title", fname.replace(".md", ""))
            if query_lower in body.lower() or query_lower in title.lower():
                slug = meta.get("slug", fname.replace(".md", ""))
                results.append({
                    "slug": slug,
                    "title": title,
                    "category": meta.get("category", ""),
                    "snippet": body[:200],
                    "word_count": len(body.split())
                })
        except Exception:
            continue

        if len(results) >= limit:
            break

    return results


def wiki_read_v4(slug: str) -> Optional[Dict]:
    """Read a wiki page by slug."""
    if not os.path.isdir(WIKI_DIR):
        return None

    # Try exact filename, then search
    fpath = os.path.join(WIKI_DIR, f"{slug}.md")
    if not os.path.exists(fpath):
        # Search for matching slug in frontmatter
        for fname in os.listdir(WIKI_DIR):
            if not fname.endswith(".md"):
                continue
            try:
                with open(os.path.join(WIKI_DIR, fname), 'r') as f:
                    content = f.read()
                meta, _ = _parse_wiki_frontmatter(content)
                if meta.get("slug") == slug:
                    fpath = os.path.join(WIKI_DIR, fname)
                    break
            except Exception:
                continue

    if not os.path.exists(fpath):
        return None

    with open(fpath, 'r') as f:
        content = f.read()
    meta, body = _parse_wiki_frontmatter(content)

    # Extract wikilinks
    wikilinks = re.findall(r'\[\[([^\]]+)\]\]', body)

    return {
        "slug": meta.get("slug", slug),
        "title": meta.get("title", slug),
        "category": meta.get("category", ""),
        "tags": meta.get("tags", "").split(",") if meta.get("tags") else [],
        "content": body,
        "wikilinks": wikilinks,
        "word_count": len(body.split())
    }


def wiki_list_v4(category: str = None, limit: int = 50) -> List[Dict]:
    """List wiki pages."""
    if not os.path.isdir(WIKI_DIR):
        return []

    results = []
    for fname in sorted(os.listdir(WIKI_DIR)):
        if not fname.endswith(".md"):
            continue
        try:
            with open(os.path.join(WIKI_DIR, fname), 'r') as f:
                content = f.read()
            meta, body = _parse_wiki_frontmatter(content)

            if category and meta.get("category", "").lower() != category.lower():
                continue

            slug = meta.get("slug", fname.replace(".md", ""))
            results.append({
                "slug": slug,
                "title": meta.get("title", slug),
                "category": meta.get("category", ""),
                "word_count": len(body.split())
            })
        except Exception:
            continue

        if len(results) >= limit:
            break

    return results


def wiki_sweep_v4(category: str = None) -> Dict:
    """Audit wiki: find orphans, broken links, untagged pages."""
    if not os.path.isdir(WIKI_DIR):
        return {"error": "wiki directory not found"}

    all_pages = {}
    all_links = set()
    orphans = []
    broken_links = []
    untagged = []

    for fname in os.listdir(WIKI_DIR):
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(WIKI_DIR, fname)
        try:
            with open(fpath, 'r') as f:
                content = f.read()
            meta, body = _parse_wiki_frontmatter(content)
            slug = meta.get("slug", fname.replace(".md", ""))
            all_pages[slug] = True

            # Check if tagged
            if not meta.get("tags"):
                untagged.append(slug)

            # Extract wikilinks
            links = re.findall(r'\[\[([^\]]+)\]\]', body)
            for link in links:
                all_links.add((slug, link))
        except Exception:
            continue

    # Find broken links (links pointing to non-existent pages)
    for source_slug, target_slug in all_links:
        if target_slug not in all_pages:
            broken_links.append({"from": source_slug, "to": target_slug})

    # Find orphans (pages with no links in or out)
    linked_slugs = set()
    for source, target in all_links:
        linked_slugs.add(source)
        linked_slugs.add(target)

    for slug in all_pages:
        if slug not in linked_slugs:
            orphans.append(slug)

    return {
        "total_pages": len(all_pages),
        "orphans": orphans,
        "broken_links": broken_links,
        "untagged": untagged
    }


def wiki_to_memster_sync_v4(max_pages: int = 20, db_path: str = None) -> Dict:
    """Extract key facts from wiki pages into memster memories."""

    if not os.path.isdir(WIKI_DIR):
        return {"synced": 0, "error": "wiki directory not found"}

    conn = get_db()
    c = conn.cursor()
    synced = 0

    for fname in sorted(os.listdir(WIKI_DIR))[:max_pages]:
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(WIKI_DIR, fname)
        try:
            with open(fpath, 'r') as f:
                content = f.read()
            meta, body = _parse_wiki_frontmatter(content)
            slug = meta.get("slug", fname.replace(".md", ""))
            title = meta.get("title", slug)

            # Check if already synced
            c.execute("SELECT id FROM memories WHERE source = ? AND content LIKE ?",
                      ("wiki_sync", f"%{slug}%"))
            if c.fetchone():
                continue

            # Extract key facts (first 500 chars as a summary)
            summary = f"[wiki:{slug}] {title}: {body[:500]}"
            now = datetime.now().isoformat()

            c.execute("""INSERT INTO memories
                        (content, network_type, t_event, source, category, tier, importance)
                        VALUES (?, 'world', ?, 'wiki_sync', 'infrastructure', 'L2', 0.7)""",
                      (summary, now))
            synced += 1

            # Create bidirectional link
            mid = c.lastrowid
            c.execute("INSERT OR IGNORE INTO memster_links (wiki_slug, memory_id) VALUES (?, ?)",
                      (slug, mid))
        except Exception as e:
            logger.debug(f"wiki sync error for {fname}: {e}")
            continue

    conn.commit()
    conn.close()
    return {"synced": synced, "max_pages": max_pages}


def memster_to_wiki_sync_v4(min_memories: int = 3, db_path: str = None) -> Dict:
    """When 3+ memories reference same topic, generate wiki page draft."""

    conn = get_db()
    c = conn.cursor()

    # Find entity clusters (entities mentioned in 3+ memories)
    c.execute("""SELECT entity_name, COUNT(*) as cnt, GROUP_CONCAT(memory_id) as mids
                 FROM memory_entities
                 GROUP BY entity_name
                 HAVING cnt >= ?
                 ORDER BY cnt DESC LIMIT 20""",
              (min_memories,))

    pages_generated = 0
    for row in c.fetchall():
        entity = row["entity_name"]
        memory_ids = [int(x) for x in str(row["mids"]).split(",")]

        # Get the memory contents
        placeholders = ",".join("?" * len(memory_ids))
        c.execute(f"""SELECT id, content FROM memories
                      WHERE id IN ({placeholders})""", memory_ids)
        mems = [dict(r) for r in c.fetchall()]

        if len(mems) < min_memories:
            continue

        # Generate wiki page
        slug = entity.lower().replace(" ", "-").replace(".", "_")
        title = entity
        body_parts = [f"# {title}\n\n"]
        body_parts.append(f"Auto-generated from {len(mems)} related memories.\n\n")

        for m in mems:
            body_parts.append(f"- {m['content'][:200]}\n")

        body = "".join(body_parts)

        # Write wiki page
        wiki_path = os.path.join(WIKI_DIR, f"{slug}.md")
        os.makedirs(WIKI_DIR, exist_ok=True)

        frontmatter = f"---\nslug: {slug}\ntitle: {title}\ncategory: notes\ntags: auto-generated,{entity}\n---\n"
        with open(wiki_path, 'w') as f:
            f.write(frontmatter + body)

        pages_generated += 1

    conn.close()
    return {"pages_generated": pages_generated, "min_memories": min_memories}


# === Feature 8: Structured Composable Search ===

def composed_search_v4(query: Dict, db_path: str = None) -> Dict:
    """Execute a structured composable search query."""

    conn = get_db()
    c = conn.cursor()

    where_clauses = []
    params = []

    # must_contain: all terms must appear in content
    must_contain = query.get("must_contain", [])
    for term in must_contain:
        where_clauses.append("content LIKE ?")
        params.append(f"%{term}%")

    # must_not_contain: none of these terms should appear
    must_not = query.get("must_not_contain", [])
    for term in must_not:
        where_clauses.append("content NOT LIKE ?")
        params.append(f"%{term}%")

    # category filter
    if query.get("category"):
        where_clauses.append("category = ?")
        params.append(query["category"])

    # network_type filter
    if query.get("network_type"):
        where_clauses.append("network_type = ?")
        params.append(query["network_type"])

    # date range
    date_range = query.get("date_range", {})
    if date_range.get("from"):
        where_clauses.append("t_event >= ?")
        params.append(date_range["from"])
    if date_range.get("to"):
        where_clauses.append("t_event <= ?")
        params.append(date_range["to"])

    # min confidence
    if query.get("min_confidence") is not None:
        where_clauses.append("confidence_score >= ?")
        params.append(query["min_confidence"])

    # min importance
    if query.get("min_importance") is not None:
        where_clauses.append("importance >= ?")
        params.append(query["min_importance"])

    # entity filter
    if query.get("entity_filter"):
        where_clauses.append("content LIKE ?")
        params.append(f"%{query['entity_filter']}%")

    # pinned only
    if query.get("pinned_only"):
        where_clauses.append("pinned = 1")

    # valid only (not expired)
    if query.get("valid_only"):
        where_clauses.append("(valid_to IS NULL OR valid_to > datetime('now'))")

    limit = query.get("limit", 20)
    params.append(limit)

    where_str = " AND ".join(where_clauses) if where_clauses else "1=1"

    c.execute(f"""SELECT id, content, category, network_type, importance,
                  decay_score, confidence_score, access_count, t_event, pinned
                  FROM memories
                  WHERE {where_str}
                  ORDER BY importance DESC, t_event DESC
                  LIMIT ?""", params)

    results = []
    for row in c.fetchall():
        results.append({
            "id": row["id"],
            "content": row["content"][:300],
            "category": row["category"],
            "importance": row["importance"],
            "confidence": round(row["confidence_score"] or 0.5, 3),
            "t_event": row["t_event"],
            "pinned": bool(row["pinned"])
        })

    conn.close()

    return {
        "query": query,
        "result_count": len(results),
        "results": results,
        "filters_applied": len(where_clauses)
    }


# === Feature 9: Graceful Forgetting ===

def reinforce_memory(memory_id: int, db_path: str = None) -> Dict:
    """Reinforce a memory by resetting its decay score (on access)."""

    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT decay_score FROM memories WHERE id = ?", (memory_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return {"error": f"memory {memory_id} not found"}

    old_decay = row[0] if row[0] is not None else 0.0
    new_decay = 0.0  # Reset to full freshness

    c.execute("UPDATE memories SET decay_score = ?, access_count = access_count + 1 WHERE id = ?",
              (new_decay, memory_id))

    # Track in decay_trace
    c.execute("""INSERT INTO decay_trace (memory_id, old_score, new_score, reason, timestamp)
                 VALUES (?, ?, ?, 'reinforce', ?)""",
              (memory_id, old_decay, new_decay, datetime.now().isoformat()))

    conn.commit()
    conn.close()

    return {"reinforced": True, "id": memory_id, "old_decay": old_decay, "new_decay": new_decay}


def pin_memory(memory_id: int, pin: bool = True, db_path: str = None) -> Dict:
    """Pin or unpin a memory. Pinned memories never decay."""

    conn = get_db()
    c = conn.cursor()
    c.execute("UPDATE memories SET pinned = ? WHERE id = ?", (1 if pin else 0, memory_id))
    conn.commit()
    conn.close()

    return {"pinned": pin, "memory_id": memory_id}


def graceful_forget_step(db_path: str = None, decay_rate: float = 0.02,
                         compress_threshold: float = 0.1) -> Dict:
    """One step of graceful forgetting: decay, trace, compress, then clean."""

    conn = get_db()
    c = conn.cursor()

    now = datetime.now().isoformat()
    decayed = 0
    compressed = 0
    deleted = 0

    # Decay all non-pinned memories
    c.execute("""SELECT id, decay_score, importance FROM memories
                 WHERE pinned = 0 OR pinned IS NULL""")
    memories = [dict(r) for r in c.fetchall()]

    for mem in memories:
        old_decay = mem["decay_score"] or 0.0
        # Decay rate scales with age since last access
        new_decay = min(old_decay + decay_rate, 1.0)

        c.execute("UPDATE memories SET decay_score = ? WHERE id = ?",
                  (new_decay, mem["id"]))

        # Trace
        c.execute("""INSERT INTO decay_trace (memory_id, old_score, new_score, reason, timestamp)
                     VALUES (?, ?, ?, 'decay', ?)""",
                  (mem["id"], old_decay, new_decay, now))
        decayed += 1

    # Compress memories below threshold (not delete outright)
    c.execute("""SELECT id, content FROM memories
                 WHERE decay_score >= ? AND (pinned = 0 OR pinned IS NULL)
                 AND importance < 0.3""",
              (compress_threshold,))
    to_compress = [dict(r) for r in c.fetchall()]

    for mem in to_compress:
        try:
            compressed_data = zlib.compress(mem["content"].encode())
            c.execute("UPDATE memories SET content = ? WHERE id = ?",
                      (compressed_data.decode('latin-1'), mem["id"]))
            compressed += 1
        except Exception:
            pass

    # Delete memories that are both compressed AND very low importance
    c.execute("""DELETE FROM memories
                 WHERE decay_score >= 0.95 AND importance < 0.1
                 AND (pinned = 0 OR pinned IS NULL)""")
    deleted = c.rowcount

    conn.commit()
    conn.close()

    return {
        "decayed": decayed,
        "compressed": compressed,
        "deleted": deleted,
        "decay_rate": decay_rate,
        "compress_threshold": compress_threshold
    }


# === Feature 10: Dream System ===

def run_dream_cycle_v4(db_path: str = None, intensity: str = "normal") -> Dict:
    """Run a dream cycle for offline consolidation and pattern discovery."""

    conn = get_db()
    c = conn.cursor()

    now = datetime.now().isoformat()
    dream_run_id = None
    discoveries = []
    candidate_conclusions = []

    # Record dream run
    c.execute("""INSERT INTO dream_runs (started_at, intensity, status)
                 VALUES (?, ?, 'running')""", (now, intensity))
    dream_run_id = c.lastrowid

    # Phase 1: Random retrieval
    sample_size = {"light": 10, "normal": 20, "deep": 50}.get(intensity, 20)
    c.execute("""SELECT id, content, category, network_type FROM memories
                 ORDER BY RANDOM() LIMIT ?""", (sample_size,))
    random_sample = [dict(r) for r in c.fetchall()]

    # Phase 2: Find unexpected connections
    # Group by entity overlap across different categories
    entity_map = {}  # entity -> list of memory_ids
    for mem in random_sample:
        entities = _extract_entities_simple(mem["content"])
        for entity in entities:
            if entity not in entity_map:
                entity_map[entity] = []
            entity_map[entity].append(mem["id"])

    # Cross-category connections are the interesting ones
    for entity, mem_ids in entity_map.items():
        if len(mem_ids) < 2:
            continue
        # Get categories of these memories
        placeholders = ",".join("?" * len(mem_ids))
        c.execute(f"""SELECT DISTINCT category, network_type FROM memories
                      WHERE id IN ({placeholders})""", mem_ids)
        categories = set()
        for r in c.fetchall():
            categories.add(r["category"] or r["network_type"] or "unknown")

        if len(categories) >= 2:
            discovery = {
                "type": "cross_category_connection",
                "entity": entity,
                "memory_ids": mem_ids,
                "categories": list(categories),
                "insight": f"'{entity}' appears across {len(categories)} categories: {', '.join(categories)}"
            }
            discoveries.append(discovery)

            # Store discovery
            c.execute("""INSERT INTO dream_discoveries
                        (dream_run_id, discovery_type, entity, insight, memory_ids, confidence)
                        VALUES (?, 'cross_category', ?, ?, ?, 0.7)""",
                      (dream_run_id, entity, discovery["insight"],
                       json.dumps(mem_ids)))

    # Phase 3: Temporal pattern discovery
    # Find memories that were stored close together across different sessions
    c.execute("""SELECT id, content, t_event, conversation_id FROM memories
                 WHERE t_event > datetime('now', '-30 days')
                 ORDER BY t_event LIMIT 50""")
    recent = [dict(r) for r in c.fetchall()]

    # Group by session
    session_map = {}
    for mem in recent:
        sid = mem["conversation_id"] or "unknown"
        if sid not in session_map:
            session_map[sid] = []
        session_map[sid].append(mem)

    # Find recurring topics across sessions
    topic_sessions = {}
    for sid, mems in session_map.items():
        for mem in mems:
            entities = _extract_entities_simple(mem["content"])
            for entity in entities:
                if entity not in topic_sessions:
                    topic_sessions[entity] = set()
                topic_sessions[entity].add(sid)

    for entity, sessions in topic_sessions.items():
        if len(sessions) >= 2:
            discovery = {
                "type": "recurring_topic",
                "entity": entity,
                "session_count": len(sessions),
                "insight": f"'{entity}' appeared across {len(sessions)} sessions - may be important"
            }
            discoveries.append(discovery)
            c.execute("""INSERT INTO dream_discoveries
                        (dream_run_id, discovery_type, entity, insight, memory_ids, confidence)
                        VALUES (?, 'recurring_topic', ?, ?, '[]', 0.5)""",
                      (dream_run_id, entity, discovery["insight"]))

    # Phase 4: Candidate conclusions from clusters
    # Find memories with same entity that might form a conclusion
    for entity, mem_ids in entity_map.items():
        if len(mem_ids) >= 3:
            placeholders = ",".join("?" * len(mem_ids))
            c.execute(f"""SELECT content FROM memories WHERE id IN ({placeholders})""", mem_ids)
            contents = [r[0][:150] for r in c.fetchall()]
            candidate = {
                "type": "candidate_conclusion",
                "entity": entity,
                "source_count": len(mem_ids),
                "sources": contents[:5],
                "suggested_conclusion": f"patterns around '{entity}' from {len(mem_ids)} memories warrant a conclusion"
            }
            candidate_conclusions.append(candidate)

    # Update dream run status
    c.execute("""UPDATE dream_runs SET completed_at = ?, status = 'completed',
                 discoveries_count = ?, conclusions_count = ?
                 WHERE id = ?""",
              (datetime.now().isoformat(), len(discoveries),
               len(candidate_conclusions), dream_run_id))

    conn.commit()
    conn.close()

    return {
        "dream_run_id": dream_run_id,
        "intensity": intensity,
        "random_sample_size": len(random_sample),
        "discoveries": discoveries[:15],
        "candidate_conclusions": candidate_conclusions[:5]
    }


# === Utility Functions ===

def normalize_text_v4(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def check_proactive_v4(context: str, max_suggestions: int = 3, db_path: str = None) -> Dict:
    """Check if there are relevant past memories for current context."""

    conn = get_db()
    c = conn.cursor()

    suggestions = []

    # Extract entities from context
    entities = _extract_entities_simple(context)

    for entity in entities[:5]:
        c.execute("""SELECT id, content, importance, confidence_score FROM memories
                     WHERE content LIKE ?
                     ORDER BY importance DESC LIMIT ?""",
                  (f"%{entity}%", max_suggestions))
        for row in c.fetchall():
            suggestions.append({
                "memory_id": row["id"],
                "content": row["content"][:150],
                "relevance": "entity_match",
                "entity": entity,
                "importance": row["importance"]
            })

    # Also check for temporal relevance (recent related work)
    if len(context) > 10:
        # Simple keyword match
        words = context.lower().split()[:5]
        for word in words:
            if len(word) < 4:
                continue
            c.execute("""SELECT id, content FROM memories
                         WHERE content LIKE ?
                         AND t_event > datetime('now', '-7 days')
                         LIMIT 2""",
                      (f"%{word}%",))
            for row in c.fetchall():
                if not any(s["memory_id"] == row["id"] for s in suggestions):
                    suggestions.append({
                        "memory_id": row["id"],
                        "content": row["content"][:150],
                        "relevance": "recent_keyword",
                        "keyword": word
                    })

    conn.close()

    # Deduplicate and limit
    seen = set()
    unique = []
    for s in suggestions:
        if s["memory_id"] not in seen:
            seen.add(s["memory_id"])
            unique.append(s)

    return {
        "context_preview": context[:100],
        "suggestions_found": len(unique),
        "suggestions": unique[:max_suggestions * 2]
    }


def build_graph_edges_v4(db_path: str = None) -> Dict:
    """Rebuild memory graph edges from entity overlaps."""

    conn = get_db()
    c = conn.cursor()

    # Clear existing auto-generated edges
    c.execute("DELETE FROM memory_edges WHERE relation_type = 'co_occurrence'")

    # Get all entity-memory mappings via join with entities table
    c.execute("SELECT me.memory_id, e.canonical_name FROM memory_entities me JOIN entities e ON me.entity_id = e.id")
    entity_map = {}
    for row in c.fetchall():
        entity = row[1]
        mid = row[0]
        if entity not in entity_map:
            entity_map[entity] = []
        entity_map[entity].append(mid)

    edges_created = 0
    for entity, mids in entity_map.items():
        # Create edges between all memories sharing this entity
        for i in range(len(mids)):
            for j in range(i + 1, min(i + 10, len(mids))):
                if mids[i] != mids[j]:
                    c.execute("""INSERT OR IGNORE INTO memory_edges
                                (source_memory_id, target_memory_id, relation_type, weight)
                                VALUES (?, ?, 'co_occurrence', 0.3)""",
                              (mids[i], mids[j]))
                    edges_created += 1

    conn.commit()
    conn.close()

    return {"edges_rebuilt": edges_created, "entities_processed": len(entity_map)}


def find_duplicates_v4(similarity_threshold: float = 0.8, db_path: str = None) -> Dict:
    """Find duplicate or near-duplicate memories."""

    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT id, content, category FROM memories ORDER BY id")
    all_memories = [dict(r) for r in c.fetchall()]
    conn.close()

    groups = []
    seen = set()

    for i in range(len(all_memories)):
        if all_memories[i]["id"] in seen:
            continue
        group = [all_memories[i]]
        norm_i = normalize_text_v4(all_memories[i]["content"])
        words_i = set(norm_i.split())

        for j in range(i + 1, min(i + 100, len(all_memories))):
            if all_memories[j]["id"] in seen:
                continue

            norm_j = normalize_text_v4(all_memories[j]["content"])
            words_j = set(norm_j.split())

            if not words_i or not words_j:
                continue

            jaccard = len(words_i & words_j) / len(words_i | words_j)
            if jaccard >= similarity_threshold:
                group.append(all_memories[j])
                seen.add(all_memories[j]["id"])

        if len(group) > 1:
            seen.add(all_memories[i]["id"])
            groups.append(group)

    return {
        "duplicate_groups": len(groups),
        "groups": [{"count": len(g), "memories": [{"id": m["id"], "content": m["content"][:100]} for m in g]}
                   for g in groups[:20]]
    }


def filter_passive_capture_v4(dry_run: bool = True, db_path: str = None) -> Dict:
    """Clean up passive capture noise from clipboard/screenshot sources."""

    conn = get_db()
    c = conn.cursor()

    noise_patterns = [
        r'^copied:\s*\n?(Layout was forced|ElementHandle|Timeout \d+ms|npm error)',
        r'^screen showed:.*File Edit Selection View',
        r'^copied:\s*\n?\[vite\]',
        r'^copied:\s*\n?(are there any improvements|what do you think)',
    ]

    candidates = []
    c.execute("""SELECT id, content, source FROM memories
                 WHERE source IN ('clipboard', 'screenshot', 'passive', 'activity_daemon')
                 OR content LIKE 'copied:%'
                 OR content LIKE 'screen showed:%'""")

    for row in c.fetchall():
        content = row["content"] or ""
        is_noise = False

        # Check noise patterns
        for pattern in noise_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                is_noise = True
                break

        # Check if content is too short or unprintable
        if len(content.strip()) < 10:
            is_noise = True

        printable = sum(1 for ch in content if ch.isprintable() or ch in '\n\t')
        if len(content) > 0 and printable / len(content) < 0.8:
            is_noise = True

        if is_noise:
            candidates.append({"id": row["id"], "content": content[:80], "source": row["source"]})

    deleted = 0
    if not dry_run:
        for c_mem in candidates:
            c.execute("DELETE FROM memories WHERE id = ?", (c_mem["id"],))
            c.execute("DELETE FROM memory_edges WHERE source_memory_id = ? OR target_memory_id = ?",
                      (c_mem["id"], c_mem["id"]))
            deleted += 1
        conn.commit()

    conn.close()

    return {
        "noise_candidates": len(candidates),
        "deleted": deleted,
        "dry_run": dry_run,
        "candidates": candidates[:20]
    }


def extract_graph_relations_v4(memory_id: int, content: str = None,
                                db_path: str = None) -> Dict:
    """Extract semantic graph relations from memory content."""

    if not content:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT content FROM memories WHERE id = ?", (memory_id,))
        row = c.fetchone()
        conn.close()
        if row:
            content = row["content"]
        else:
            return {"error": "memory not found and no content provided"}

    relations = []

    # Extract entity relations: "X runs on Y", "X uses Y", "X depends on Y"
    patterns = [
        (r'(\S+)\s+(?:runs|running)\s+on\s+(\S+)', 'runs_on'),
        (r'(\S+)\s+(?:uses|using)\s+(\S+)', 'uses'),
        (r'(\S+)\s+(?:depends?\s+on)\s+(\S+)', 'depends_on'),
        (r'(\S+)\s+(?:connects?\s+to)\s+(\S+)', 'connects_to'),
        (r'(\S+)\s+(?:is\s+on)\s+(\S+)', 'located_on'),
        (r'(\S+)\s+(?:hosts?)\s+(\S+)', 'hosts'),
        (r'(\S+)\s+(?:replaced?|superseded?)\s+(\S+)', 'replaces'),
    ]

    for pattern, rel_type in patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for m in matches:
            relations.append({
                "subject": m.group(1),
                "predicate": rel_type,
                "object": m.group(2)
            })

    # Also extract simple entities
    entities = _extract_entities_simple(content)

    return {
        "memory_id": memory_id,
        "relations_found": len(relations),
        "relations": relations,
        "entities": entities
    }


# === DB Schema Upgrades ===

def upgrade_database_v4(db_path: str = None) -> Dict:
    """Add V4 schema: new columns, new tables."""

    conn = get_db()
    c = conn.cursor()
    upgrades = []

    # New columns on memories table
    for col_def in [
        ("confidence_score", "REAL DEFAULT 0.5"),
        ("observation_log", "TEXT DEFAULT '[]'"),
        ("pinned", "INTEGER DEFAULT 0"),
        ("observed_at", "TEXT"),
    ]:
        try:
            c.execute(f"ALTER TABLE memories ADD COLUMN {col_def[0]} {col_def[1]}")
            upgrades.append(f"added column memories.{col_def[0]}")
        except sqlite3.OperationalError:
            pass

    # New tables
    new_tables = {
        "compressed_memories": """
            CREATE TABLE IF NOT EXISTS compressed_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                compressed_memory_id INTEGER NOT NULL,
                original_memory_id INTEGER NOT NULL,
                original_content TEXT,
                compressed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (compressed_memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )""",
        "decay_trace": """
            CREATE TABLE IF NOT EXISTS decay_trace (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                old_score REAL,
                new_score REAL,
                reason TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )""",
        "dream_runs": """
            CREATE TABLE IF NOT EXISTS dream_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                intensity TEXT DEFAULT 'normal',
                status TEXT DEFAULT 'running',
                discoveries_count INTEGER DEFAULT 0,
                conclusions_count INTEGER DEFAULT 0
            )""",
        "dream_discoveries": """
            CREATE TABLE IF NOT EXISTS dream_discoveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dream_run_id INTEGER,
                discovery_type TEXT,
                entity TEXT,
                insight TEXT,
                memory_ids TEXT,
                confidence REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dream_run_id) REFERENCES dream_runs(id)
            )""",
        "tasks": """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'active',
                priority INTEGER DEFAULT 5,
                release_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )""",
        "task_memory_assignments": """
            CREATE TABLE IF NOT EXISTS task_memory_assignments (
                task_id INTEGER NOT NULL,
                memory_id INTEGER NOT NULL,
                PRIMARY KEY (task_id, memory_id),
                FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )""",
        "conclusions": """
            CREATE TABLE IF NOT EXISTS conclusions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                status TEXT DEFAULT 'pending',
                source_memory_ids TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
        "narrative_arcs": """
            CREATE TABLE IF NOT EXISTS narrative_arcs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                arc_type TEXT DEFAULT 'manual',
                status TEXT DEFAULT 'ongoing',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
        "arc_memories": """
            CREATE TABLE IF NOT EXISTS arc_memories (
                arc_id INTEGER NOT NULL,
                memory_id INTEGER NOT NULL,
                arc_role TEXT DEFAULT 'event',
                PRIMARY KEY (arc_id, memory_id),
                FOREIGN KEY (arc_id) REFERENCES narrative_arcs(id) ON DELETE CASCADE,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )""",
        "memory_palaces": """
            CREATE TABLE IF NOT EXISTS memory_palaces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
        "palace_rooms": """
            CREATE TABLE IF NOT EXISTS palace_rooms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                palace_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                position_x REAL DEFAULT 0,
                position_y REAL DEFAULT 0,
                position_z REAL DEFAULT 0,
                FOREIGN KEY (palace_id) REFERENCES memory_palaces(id) ON DELETE CASCADE
            )""",
        "room_memories": """
            CREATE TABLE IF NOT EXISTS room_memories (
                room_id INTEGER NOT NULL,
                memory_id INTEGER NOT NULL,
                PRIMARY KEY (room_id, memory_id),
                FOREIGN KEY (room_id) REFERENCES palace_rooms(id) ON DELETE CASCADE,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )""",
        "federation_peers": """
            CREATE TABLE IF NOT EXISTS federation_peers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                peer_url TEXT NOT NULL,
                peer_name TEXT,
                auth_token TEXT,
                sync_direction TEXT DEFAULT 'bidirectional',
                status TEXT DEFAULT 'active',
                last_sync TIMESTAMP
            )""",
        "memory_feedback": """
            CREATE TABLE IF NOT EXISTS memory_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                feedback_type TEXT NOT NULL,
                context TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )""",
    }

    for table_name, ddl in new_tables.items():
        try:
            c.execute(ddl)
            upgrades.append(f"ensured table {table_name}")
        except sqlite3.OperationalError as e:
            upgrades.append(f"table {table_name}: {e}")

    # Add indexes
    for idx in [
        "CREATE INDEX IF NOT EXISTS idx_decay_trace_memory ON decay_trace(memory_id)",
        "CREATE INDEX IF NOT EXISTS idx_dream_discoveries_run ON dream_discoveries(dream_run_id)",
        "CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence_score)",
        "CREATE INDEX IF NOT EXISTS idx_memories_pinned ON memories(pinned)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
    ]:
        try:
            c.execute(idx)
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()

    return {"upgrades_applied": len(upgrades), "details": upgrades}


# ============================================================
# Additional V4 handler functions (referenced by main server)
# ============================================================

def compress_memory_v4(memory_id: int) -> dict:
    """Compress a memory's content using zlib."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT content FROM memories WHERE id = ?", (memory_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return {"memory_id": memory_id, "error": "not found"}
    original = row["content"]
    if original.startswith("ZLIB:"):
        conn.close()
        return {"memory_id": memory_id, "already_compressed": True}
    compressed = "ZLIB:" + zlib.compress(original.encode()).hex()
    original_size = len(original)
    compressed_size = len(compressed)
    c.execute("UPDATE memories SET content = ? WHERE id = ?", (compressed, memory_id))
    conn.commit()
    conn.close()
    return {
        "memory_id": memory_id,
        "compressed": True,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "ratio": round(compressed_size / original_size, 2) if original_size > 0 else 0
    }


def decompress_memory_v4(memory_id: int) -> dict:
    """Decompress a previously compressed memory."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT content FROM memories WHERE id = ?", (memory_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return {"memory_id": memory_id, "error": "not found"}
    content = row["content"]
    if not content.startswith("ZLIB:"):
        conn.close()
        return {"memory_id": memory_id, "decompressed": False, "note": "not compressed"}
    hex_data = content[5:]
    decompressed = zlib.decompress(bytes.fromhex(hex_data)).decode()
    c.execute("UPDATE memories SET content = ? WHERE id = ?", (decompressed, memory_id))
    conn.commit()
    conn.close()
    return {"memory_id": memory_id, "decompressed": True, "size": len(decompressed)}


def validate_conclusion_v4(conclusion_id: int, still_valid: bool, notes: str = "") -> dict:
    """Validate or reject a conclusion."""
    conn = get_db()
    c = conn.cursor()
    status = "confirmed" if still_valid else "rejected"
    c.execute("UPDATE conclusions SET status = ?, notes = COALESCE(?, notes) WHERE id = ?",
              (status, notes, conclusion_id))
    conn.commit()
    conn.close()
    return {"conclusion_id": conclusion_id, "status": status, "notes": notes}


def rate_memory_v4(memory_id: int, feedback_type: str, context: str = "") -> dict:
    """Rate a memory and adjust its properties."""
    conn = get_db()
    c = conn.cursor()
    # Map feedback to importance delta
    importance_delta = {
        "helpful": 0.1, "promote": 0.2,
        "irrelevant": -0.1, "wrong": -0.2,
        "outdated": -0.15, "demote": -0.2
    }.get(feedback_type, 0)
    # Apply
    c.execute("UPDATE memories SET importance = MAX(0, MIN(1, importance + ?)) WHERE id = ?",
              (importance_delta, memory_id))
    # Update confidence if applicable
    if feedback_type in ("helpful", "promote"):
        c.execute("UPDATE memories SET confidence_score = MIN(1.0, COALESCE(confidence_score, 0.5) + 0.05) WHERE id = ?", (memory_id,))
    elif feedback_type in ("wrong", "outdated"):
        c.execute("UPDATE memories SET confidence_score = MAX(0.0, COALESCE(confidence_score, 0.5) - 0.1) WHERE id = ?", (memory_id,))
    conn.commit()
    c.execute("SELECT importance, confidence_score FROM memories WHERE id = ?", (memory_id,))
    row = c.fetchone()
    conn.close()
    return {
        "memory_id": memory_id,
        "feedback": feedback_type,
        "new_importance": row["importance"] if row else None,
        "new_confidence": row["confidence_score"] if row else None
    }


def get_conclusions_v4(status: str = None, limit: int = 10) -> dict:
    """Get conclusions with optional filtering."""
    conn = get_db()
    c = conn.cursor()
    query = "SELECT * FROM conclusions"
    params = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY confidence DESC LIMIT ?"
    params.append(limit)
    try:
        c.execute(query, params)
        conclusions = [dict(r) for r in c.fetchall()]
    except sqlite3.OperationalError:
        conclusions = []
    conn.close()
    return {"conclusions": conclusions, "count": len(conclusions)}


def get_memory_feedback_stats_v4(memory_id: int = None) -> dict:
    """Get feedback statistics."""
    conn = get_db()
    c = conn.cursor()
    if memory_id:
        c.execute("SELECT importance, confidence_score, access_count, decay_score FROM memories WHERE id = ?", (memory_id,))
        row = c.fetchone()
        conn.close()
        return {"memory_id": memory_id, "stats": dict(row) if row else {}}
    else:
        c.execute("SELECT AVG(importance) as avg_importance, AVG(confidence_score) as avg_confidence, COUNT(*) as total FROM memories")
        row = c.fetchone()
        conn.close()
        return {"overall_stats": dict(row) if row else {}}


def get_cross_session_context_v4(topic: str, limit: int = 5, hours: int = 365) -> dict:
    """Find past sessions related to a topic."""
    conn = get_db()
    c = conn.cursor()
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    # Search sessions by topic/summary
    c.execute("SELECT * FROM sessions WHERE started_at > ? ORDER BY started_at DESC LIMIT ?",
              (cutoff, limit))
    sessions = [dict(r) for r in c.fetchall()]
    # Also get related memories via FTS
    memory_results = []
    if topic:
        try:
            c.execute("SELECT * FROM memories WHERE id IN (SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?) ORDER BY importance DESC LIMIT ?",
                      (topic, limit))
            memory_results = [dict(r) for r in c.fetchall()]
        except sqlite3.OperationalError:
            pass
    conn.close()
    return {"topic": topic, "related_sessions": sessions, "related_memories": memory_results}


def get_working_memory_context_v4(session_id: str, conversation_id: str = "", limit: int = 10) -> dict:
    """Get current working memory context for a session."""
    conn = get_db()
    c = conn.cursor()
    # Get recent memories from this session
    if conversation_id:
        c.execute("SELECT * FROM memories WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                  (conversation_id, limit))
    else:
        c.execute("SELECT * FROM memories ORDER BY id DESC LIMIT ?", (limit,))
    memories = [dict(r) for r in c.fetchall()]
    conn.close()
    return {"session_id": session_id, "working_memories": memories, "count": len(memories)}


def infer_follow_up_intent_v4(session_id: str, query: str, conversation_id: str = "") -> dict:
    """Detect if query is a follow-up to previous conversation."""
    # Simple heuristic: check for pronouns, references to previous topics
    followup_indicators = ["it", "that", "this", "those", "the same", "earlier", "before", "previously", "still", "also", "what about"]
    query_lower = query.lower()
    is_followup = any(ind in query_lower.split() for ind in followup_indicators)
    return {"is_follow_up": is_followup, "confidence": 0.7 if is_followup else 0.2, "session_id": session_id}


def route_retrieval_v4(query: str, session_id: str, conversation_id: str = "") -> dict:
    """Analyze query and route to appropriate retrieval strategy."""
    query_lower = query.lower()
    # Simple routing heuristics
    if any(w in query_lower for w in ["when", "time", "date", "recently", "lately", "today", "yesterday"]):
        strategy = "temporal"
    elif any(w in query_lower for w in ["similar", "like", "related", "connected"]):
        strategy = "graph"
    elif any(w in query_lower for w in ["what is", "define", "explain", "how does"]):
        strategy = "semantic"
    elif len(query.split()) <= 2:
        strategy = "keyword"
    else:
        strategy = "hybrid"
    return {"strategy": strategy, "query": query, "session_id": session_id}


def record_retrieval_telemetry_v4(memory_id: int, query_text: str, match_served: bool = False) -> dict:
    """Record telemetry for memory retrieval events."""
    conn = get_db()
    c = conn.cursor()
    # Increment access count
    c.execute("UPDATE memories SET access_count = access_count + 1 WHERE id = ?", (memory_id,))
    conn.commit()
    conn.close()
    return {"recorded": True, "memory_id": memory_id, "match_served": match_served}


def set_temporal_bounds_v4(memory_id: int, valid_from: str = None, valid_to: str = None, observed_at: str = None) -> dict:
    """Set temporal validity bounds for a memory."""
    conn = get_db()
    c = conn.cursor()
    updates = []
    params = []
    for field, value in [("valid_from", valid_from), ("valid_to", valid_to), ("observed_at", observed_at)]:
        if value:
            updates.append(f"{field} = ?")
            params.append(value)
    if updates:
        c.execute(f"UPDATE memories SET {', '.join(updates)} WHERE id = ?", params + [memory_id])
        conn.commit()
    conn.close()
    return {"memory_id": memory_id, "updated": len(updates), "fields": [u.split("=")[0].strip() for u in updates]}
