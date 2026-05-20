#!/usr/bin/env python3
"""
Memster gbrain — auto-linking + community detection

Phase 3 feature: automatically construct semantic graph edges between memories
using co-occurrence, entity overlap, temporal proximity, and embedding similarity.
Also detects communities and suggests narrative arcs.

Design:
  - rebuild_graph(days_back, min_weight) — batch scan all memories
  - link_memory(memory_id) — incremental for one new memory
  - detect_communities() — community detection via NetworkX
  - suggest_arcs(community_id) — generate arc proposals

Weight formula:
  w = α·content_jaccard + β·entity_overlap_norm + γ·temporal_decay + δ·embedding_sim

  where α=0.25, β=0.30, γ=0.20, δ=0.25 (tunable)
"""

import json
import os
import re
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple

import networkx as nx
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

# Local imports — will be relative when loaded as module inside ~/memster
try:
    from memster_phase2 import get_db
except ImportError:
    # fallback for standalone testing
    def get_db(db_path=None):
        path = db_path or os.path.expanduser("~/.memster/memster_core.db")
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn

DEFAULT_DB_PATH = os.path.expanduser("~/.memster/memster_core.db")

# Weighting coefficients
ALPHA_CONTENT = 0.25
BETA_ENTITY = 0.30
GAMMA_TEMPORAL = 0.20
DELTA_EMBED = 0.25

# Temporal decay half-life in days
TAU_DAYS = 7.0

# Threshold for creating an edge
MIN_EDGE_WEIGHT = 0.25

# Minimum shared entities to count entity overlap
MIN_SHARED_ENTITIES = 1


# ============================================================
# helpers
# ============================================================

def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def _word_set(text: str) -> Set[str]:
    return set(_normalize_text(text).split())

def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def _temporal_decay(days_diff: float) -> float:
    """exponential decay: exp(-Δt / τ)"""
    return np.exp(-days_diff / TAU_DAYS)

def _cosine_sim(vec_a: Optional[str], vec_b: Optional[str]) -> float:
    """compare two embedding strings (JSON array or hex?)"""
    if not vec_a or not vec_b:
        return 0.0
    try:
        # embeddings stored as JSON list of floats
        a = np.array(json.loads(vec_a), dtype=float)
        b = np.array(json.loads(vec_b), dtype=float)
        if a.shape != b.shape:
            return 0.0
        return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])
    except Exception:
        return 0.0

def _get_entities_for_memory(cursor, memory_id: int) -> Set[str]:
    """fetch entity canonical names for a memory."""
    cursor.execute("""
        SELECT e.canonical_name
        FROM memory_entities me
        JOIN entities e ON me.entity_id = e.id
        WHERE me.memory_id = ?
    """, (memory_id,))
    return {row["canonical_name"] for row in cursor.fetchall()}


# ============================================================
# batch rebuild
# ============================================================

def rebuild_graph(days_back: int = 90, min_weight: float = MIN_EDGE_WEIGHT) -> Dict[str, Any]:
    """
    Scan memories within time window and create memory_edges for strong pairs.
    Returns stats: memories_scanned, edges_created, edges_skipped, errors.
    """
    conn = get_db()
    c = conn.cursor()

    # Get candidate memories within window, with embeddings
    cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()
    c.execute("""
        SELECT id, content, embedding, t_event, importance, confidence_score
        FROM memories
        WHERE t_event >= ?
        ORDER BY t_event
    """, (cutoff,))
    memories = [dict(r) for r in c.fetchall()]
    stats = {"memories_scanned": len(memories), "edges_created": 0, "edges_skipped": 0, "duplicates": 0}

    # Precompute word sets and entity sets
    print(f"[gbrain] preprocessing {len(memories)} memories...")
    for mem in memories:
        mem["words"] = _word_set(mem["content"])

    # For entity lookup, cache per memory id
    entity_cache = {}
    for mem in memories:
        e_set = _get_entities_for_memory(c, mem["id"])
        entity_cache[mem["id"]] = e_set

    # Candidate pair selection: we'll use sliding window by time + FTS co-occurrence
    # Simple: compare each memory to next N within 7 days — O(n^2) but limited window
    # Optimize: batch by day, then within-day pairs only
    from collections import defaultdict
    by_day = defaultdict(list)
    for mem in memories:
        day = mem["t_event"][:10] if mem["t_event"] else "unknown"
        by_day[day].append(mem)

    print(f"[gbrain] scanning pairs across {len(by_day)} days...")
    edges_to_create = []
    errors = []

    days_list = sorted(by_day.keys())
    for i, day in enumerate(days_list):
        day_mems = by_day[day]
        # Intra-day pairs
        for j in range(len(day_mems)):
            for k in range(j+1, len(day_mems)):
                m1, m2 = day_mems[j], day_mems[k]
                try:
                    weight = _compute_edge_weight(m1, m2, entity_cache)
                    if weight >= min_weight:
                        edges_to_create.append((m1["id"], m2["id"], weight))
                except Exception as e:
                    errors.append(f"{m1['id']}-{m2['id']}: {e}")

        # Inter-day: compare to next 6 days (τ range)
        lookahead_days = 6
        for future_day in days_list[i+1:i+1+lookahead_days]:
            future_mems = by_day[future_day]
            for m1 in day_mems:
                for m2 in future_mems:
                    try:
                        weight = _compute_edge_weight(m1, m2, entity_cache)
                        if weight >= min_weight:
                            edges_to_create.append((m1["id"], m2["id"], weight))
                    except Exception as e:
                        errors.append(f"{m1['id']}-{m2['id']}: {e}")

    print(f"[gbrain] edges candidate: {len(edges_to_create)}")
    # Deduplicate (a,b) vs (b,a) — keep higher weight
    edge_dict = {}
    for id1, id2, w in edges_to_create:
        key = tuple(sorted((id1, id2)))
        if key not in edge_dict or w > edge_dict[key][2]:
            edge_dict[key] = (key[0], key[1], w)

    # Insert with duplicate check against memory_edges
    print(f"[gbrain] inserting {len(edge_dict)} unique edges...")
    for src, tgt, w in edge_dict.values():
        try:
            c.execute("""
                INSERT OR IGNORE INTO memory_edges (source_memory_id, target_memory_id, relation_type, weight, created_at)
                VALUES (?, ?, 'related', ?, ?)
            """, (src, tgt, w, datetime.now().isoformat()))
            if c.rowcount > 0:
                stats["edges_created"] += 1
            else:
                stats["duplicates"] += 1
        except Exception as e:
            stats["edges_skipped"] += 1
            errors.append(f"insert {src}-{tgt}: {e}")

    conn.commit()
    conn.close()

    stats["errors"] = errors[:100]  # truncate
    stats["unique_pairs"] = len(edge_dict)
    return stats


def _compute_edge_weight(m1: Dict, m2: Dict, entity_cache: Dict[int, Set[str]]) -> float:
    """compute composite weight between two memories."""
    # content jaccard
    jac = _jaccard(m1["words"], m2["words"])

    # entity overlap (normalized by min unique entities)
    e1 = entity_cache.get(m1["id"], set())
    e2 = entity_cache.get(m2["id"], set())
    shared = len(e1 & e2)
    entity_score = (shared / min(len(e1), len(e2))) if (e1 and e2 and shared) else 0.0

    # temporal decay
    try:
        t1 = datetime.fromisoformat(m1["t_event"])
        t2 = datetime.fromisoformat(m2["t_event"])
        days = abs((t1 - t2).total_seconds()) / 86400
    except Exception:
        days = 0.0
    temp = _temporal_decay(days)

    # embedding similarity
    emb = _cosine_sim(m1.get("embedding"), m2.get("embedding"))

    # weighted sum
    weight = ALPHA_CONTENT * jac + BETA_ENTITY * entity_score + GAMMA_TEMPORAL * temp + DELTA_EMBED * emb
    return round(float(weight), 4)


# ============================================================
# community detection
# ============================================================

def detect_communities(rebuild: bool = False) -> Dict[str, Any]:
    """
    Run community detection on the memory_edges graph.
    Returns communities with central nodes and arc suggestions.
    """
    conn = get_db()
    c = conn.cursor()

    # Build graph from edges above threshold
    c.execute("SELECT source_memory_id, target_memory_id, weight FROM memory_edges WHERE weight >= ?", (MIN_EDGE_WEIGHT,))
    edges = [(row[0], row[1], row[2]) for row in c.fetchall()]

    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    print(f"[gbrain] graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Use Louvain community detection (requires pip install python-louvain? networkx has label_propagation)
    try:
        import networkx.algorithms.community as nx_comm
        # Try Louvain if available
        try:
            import community as community_louvain  # python-louvain package
            communities = community_louvain.best_partition(G)
            print("[gbrain] using Louvain communities")
        except ImportError:
            print("[gbrain] Louvain not available, using label propagation")
            communities_generator = nx_comm.label_propagation_communities(G)
            communities = {}
            for i, comm in enumerate(communities_generator):
                for node in comm:
                    communities[node] = i
    except Exception as e:
        print(f"[gbrain] community detection failed: {e}")
        conn.close()
        return {"error": str(e), "communities": []}

    # Organize by community id
    from collections import defaultdict
    comm_members = defaultdict(list)
    for node, cid in communities.items():
        comm_members[int(cid)].append(node)

    # Compute central node per community by weighted degree
    comm_info = {}
    for cid, members in comm_members.items():
        if len(members) < 3:
            continue  # skip tiny communities
        # subgraph degree
        subG = G.subgraph(members)
        degrees = dict(subG.degree(weight='weight'))
        central = max(members, key=lambda n: degrees.get(n, 0))
        # fetch a title hint: top entities in these memories
        titles = _suggest_arc_title(conn, members[:10])
        comm_info[cid] = {
            "member_count": len(members),
            "central_memory_id": central,
            "suggested_title": titles,
            "sample_members": members[:5]
        }

    conn.close()
    return {
        "total_communities": len(comm_info),
        "communities": comm_info,
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges()
    }


def _suggest_arc_title(conn, memory_ids: List[int]) -> str:
    """generate a human-readable arc title from top memory contents."""
    if not memory_ids:
        return "Untitled"
    placeholders = ",".join("?" * len(memory_ids))
    c = conn.cursor()
    c.execute(f"""
        SELECT content FROM memories WHERE id IN ({placeholders}) ORDER BY importance DESC LIMIT 3
    """, memory_ids)
    rows = c.fetchall()
    words = []
    for r in rows:
        content = r[0] or ""
        # take first 3 significant words longer than 4 chars
        for w in content.split():
            if len(w) > 4 and w.lower() not in ('the','and','for','from','with','this','that','these','those','have','been','were','was','are','not'):
                words.append(w)
            if len(words) >= 3:
                break
        if len(words) >= 3:
            break
    return " ".join(words[:3]) if words else "Cluster"


# ============================================================
# arc suggestion creation
# ============================================================

def suggest_arcs_from_community(community_id: int, min_members: int = 5) -> Dict[str, Any]:
    """
    Create a narrative_arc entry for the given community.
    Returns arc id and details.
    """
    conn = get_db()
    c = conn.cursor()

    # Get members from community detection run (requires fresh communities)
    # We'll store community_id in memory_edges? Better: recompute here.
    # For V1, just accept a list of memory IDs as arguments?
    # Let's change API: suggest_arcs takes memory_ids list instead.
    raise NotImplementedError("use suggest_arcs_for_memories(memory_ids)")


def suggest_arcs_for_memories(memory_ids: List[int], title_hint: str = None) -> Dict[str, Any]:
    """
    Create a narrative arc from a list of memory IDs.
    """
    if len(memory_ids) < 3:
        return {"error": "need at least 3 memories"}

    conn = get_db()
    c = conn.cursor()

    # Generate title
    if title_hint:
        title = title_hint[:120]
    else:
        title = _suggest_arc_title(conn, memory_ids[:10])

    c.execute("""
        INSERT INTO narrative_arcs (title, description, arc_type, status, created_at, updated_at)
        VALUES (?, ?, 'auto', 'suggested', ?, ?)
    """, (title, f"Auto-generated from {len(memory_ids)} memories", datetime.now().isoformat(), datetime.now().isoformat()))
    arc_id = c.lastrowid

    # Link memories
    for mem_id in memory_ids:
        c.execute("""
            INSERT OR IGNORE INTO arc_memories (arc_id, memory_id, relevance_score, position_order)
            VALUES (?, ?, 0.7, ?)
        """, (arc_id, mem_id, 0))  # position 0 = auto order by date later

    conn.commit()
    conn.close()

    return {"created": True, "arc_id": arc_id, "title": title, "memory_count": len(memory_ids)}


# ============================================================
# incremental linking for a single new memory
# ============================================================

def link_memory(memory_id: int, window_days: int = 30, min_weight: float = MIN_EDGE_WEIGHT) -> Dict[str, Any]:
    """
    Link a single memory against recent corpus.
    Called after a memory insert to auto-establish edges.
    """
    conn = get_db()
    c = conn.cursor()

    # Fetch the new memory
    c.execute("SELECT id, content, embedding, t_event, importance FROM memories WHERE id = ?", (memory_id,))
    m1 = dict(c.fetchone() or ())
    if not m1:
        conn.close()
        return {"error": "memory not found"}

    m1["words"] = _word_set(m1["content"])
    e1 = _get_entities_for_memory(c, m1["id"])
    m1["entities"] = e1

    # Candidate pool: recent memories within window, excluding self
    cutoff = (datetime.now() - timedelta(days=window_days)).isoformat()
    c.execute("""
        SELECT id, content, embedding, t_event, importance
        FROM memories
        WHERE t_event >= ? AND id != ?
    """, (cutoff, memory_id))
    candidates = [dict(r) for r in c.fetchall()]

    created = 0
    skipped = 0
    for m2 in candidates:
        m2["words"] = _word_set(m2["content"])
        m2["entities"] = _get_entities_for_memory(c, m2["id"])
        try:
            weight = _compute_edge_weight(m1, m2, {memory_id: e1, m2["id"]: m2["entities"]})
            if weight >= min_weight:
                # insert undirected edge (store both directions? memory_edges is directed by schema but we treat as undirected)
                # store source=memory_id target=m2
                c.execute("""
                    INSERT OR IGNORE INTO memory_edges (source_memory_id, target_memory_id, relation_type, weight, created_at)
                    VALUES (?, ?, 'related', ?, ?)
                """, (memory_id, m2["id"], weight, datetime.now().isoformat()))
                if c.rowcount > 0:
                    created += 1
                else:
                    skipped += 1
        except Exception as e:
            skipped += 1

    conn.commit()
    conn.close()

    return {"memory_id": memory_id, "edges_created": created, "edges_skipped": skipped}


# ============================================================
# tool entry points
# ============================================================

def gbrain_rebuild(args: Dict[str, Any]) -> List[Dict]:
    """Tool: full graph rebuild."""
    days = int(args.get("days_back", 90))
    min_w = float(args.get("min_weight", MIN_EDGE_WEIGHT))
    stats = rebuild_graph(days_back=days, min_weight=min_w)
    # After rebuilding, optionally detect communities
    comm = detect_communities()
    stats["communities"] = comm.get("total_communities", 0)
    return [{"type": "text", "text": json.dumps(stats, indent=2)}]


def gbrain_suggest_arcs(args: Dict[str, Any]) -> List[Dict]:
    """Tool: suggest arcs from recent communities."""
    # For now, query high-degree nodes grouped by community from memory_edges
    # Simple heuristic: find connected components and pick biggest
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT source_memory_id, target_memory_id FROM memory_edges WHERE weight >= ?", (MIN_EDGE_WEIGHT,))
    edges = [(r[0], r[1]) for r in c.fetchall()]

    if not edges:
        conn.close()
        return [{"type": "text", "text": json.dumps({"error": "No edges found. Run gbrain_rebuild_graph first."})}]

    G = nx.Graph()
    G.add_edges_from(edges)
    comps = list(nx.connected_components(G))
    # Filter to components with > 5 nodes
    big_comps = [c for c in comps if len(c) >= 5]
    suggestions = []
    for comp in big_comps[:5]:  # top 5
        mem_list = list(comp)[:20]  # cap
        result = suggest_arcs_for_memories(mem_list)
        suggestions.append(result)

    conn.close()
    return [{"type": "text", "text": json.dumps({"suggestions": suggestions, "communities_scanned": len(big_comps)}, indent=2)}]


def gbrain_get_communities(args: Dict[str, Any]) -> List[Dict]:
    """Return community membership for a memory or all."""
    memory_id = args.get("memory_id")
    # Easiest: recompute on demand (cheap for <10k edges)
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT source_memory_id, target_memory_id FROM memory_edges WHERE weight >= ?", (MIN_EDGE_WEIGHT,))
    edges = [(r[0], r[1]) for r in c.fetchall()]
    G = nx.Graph()
    G.add_edges_from(edges)

    try:
        import community as community_louvain
        comm_map = community_louvain.best_partition(G)
    except ImportError:
        from networkx.algorithms.community import label_propagation_communities
        comms = list(label_propagation_communities(G))
        comm_map = {}
        for i, comm in enumerate(comms):
            for node in comm:
                comm_map[node] = i

    if memory_id:
        cid = comm_map.get(int(memory_id))
        result = {"memory_id": memory_id, "community_id": cid}
    else:
        # group by community
        from collections import defaultdict
        groups = defaultdict(list)
        for node, cid in comm_map.items():
            groups[int(cid)].append(node)
        result = {"communities": {str(k): v[:20] for k,v in groups.items()}}
    conn.close()
    return [{"type": "text", "text": json.dumps(result, indent=2)}]


def gbrain_auto_link_single(args: Dict[str, Any]) -> List[Dict]:
    """Tool: link one memory incrementally."""
    memory_id = int(args.get("memory_id"))
    window = int(args.get("window_days", 30))
    result = link_memory(memory_id, window_days=window)
    return [{"type": "text", "text": json.dumps(result, indent=2)}]


def gbrain_edge_info(args: Dict[str, Any]) -> List[Dict]:
    """Tool: get edge weight and shared entities between two memories."""
    a = int(args["memory_id_a"])
    b = int(args["memory_id_b"])
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT weight FROM memory_edges WHERE (source_memory_id=? AND target_memory_id=?) OR (source_memory_id=? AND target_memory_id=?)",
              (a, b, b, a))
    row = c.fetchone()
    weight = row["weight"] if row else 0.0
    c.execute("SELECT COUNT(*) as cnt FROM memory_entities WHERE memory_id IN (?,?) GROUP BY memory_id", (a, b))
    # shared entities
    c.execute("""
        SELECT COUNT(*) as shared
        FROM memory_entities me1
        JOIN memory_entities me2 ON me1.entity_id = me2.entity_id
        WHERE me1.memory_id = ? AND me2.memory_id = ?
    """, (a, b))
    shared_row = c.fetchone()
    shared = shared_row["shared"] if shared_row else 0
    conn.close()
    return [{"type": "text", "text": json.dumps({"memory_id_a": a, "memory_id_b": b, "weight": weight, "shared_entities": shared}, indent=2)}]


# Registry
GBRAIN_TOOLS = [
    "gbrain_rebuild_graph",
    "gbrain_suggest_arcs",
    "gbrain_get_communities",
    "gbrain_auto_link_single",
    "gbrain_edge_info",
]
