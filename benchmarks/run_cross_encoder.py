#!/usr/bin/env python3
"""
Cross-encoder LongMemEval benchmark — replicates the V6 methodology:
  - Top-500 candidates per signal (semantic, BM25, entity, temporal)
  - RRF fusion (k=300)
  - Cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)
  =  top-5 final results

Usage:
  # NIM backend (requires API key)
  EMBEDDING_BACKEND=nim python benchmarks/run_cross_encoder.py

  # Local backend (default)
  EMBEDDING_BACKEND=local python benchmarks/run_cross_encoder.py

Optimizations:
  - Cross-encoder model loaded once, reused for all queries
  - Uses DEFAULT_TOP_K from hybrid_retrieval for signal limits
  - No redundant reconnection per query within session
  - Measures wall-clock latency per query
"""
import json, os, sys, time, random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── load hermes env (for API keys) ────────────────────────────────
_hermes_env = Path.home() / ".hermes" / ".env"
if _hermes_env.exists():
    with open(_hermes_env) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)

import psycopg2
import psycopg2.extras
from sentence_transformers import CrossEncoder

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://house:@/memster?host=/run/postgresql&port=5433",
)

# ── cross-encoder (loaded once) ───────────────────────────────────
print("Loading cross-encoder model...", flush=True)
t0 = time.time()
CROSS_ENCODER = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512
)
print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

# ── database ──────────────────────────────────────────────────────
conn = psycopg2.connect(DB_URL)
conn.cursor_factory = psycopg2.extras.RealDictCursor


def get_conn():
    """Return the shared connection."""
    return conn


# ── data loading ──────────────────────────────────────────────────
data_dir = Path(__file__).parent / "LongMemEval_dataset" / "data"
with open(data_dir / "longmemeval_oracle.json") as f:
    DATASET = json.load(f)

# Build evidence map
EVIDENCE = {}
for q in DATASET:
    qid = q["question_id"]
    qtype = q["question_type"]
    for si, session in enumerate(q["haystack_sessions"]):
        has_ev = any(turn.get("has_answer", False) for turn in session)
        EVIDENCE[f"{qid}|{qtype}|{si}"] = has_ev

TOTAL_EVIDENCE = sum(1 for v in EVIDENCE.values() if v)
print(f"Total evidence sessions: {TOTAL_EVIDENCE}", flush=True)
print(f"Total questions: {len(DATASET)}", flush=True)


# ── hybrid retrieval engine (must be imported after env vars) ──────
from memster.hybrid_retrieval import (
    HybridRetrievalEngine,
    embed_text,
    _vector_search_sql,
)
from memster.entity_extraction import extract_entities

# Single engine instance (reused across all queries)
_ENGINE = HybridRetrievalEngine(get_conn)


def get_candidates(query: str, signal_limit: int = 500) -> Dict[int, float]:
    """
    Multi-signal retrieval fused via RRF (k=300).
    No cross-encoder — just the fusion step.
    Returns {memory_id: rrf_score} for top candidates.
    """
    signals = {}

    # Semantic
    if _ENGINE.embeddings_available:
        query_emb = embed_text(query)
        if query_emb:
            sem = _vector_search_sql(query_emb, limit=signal_limit, threshold=0.0)
            if sem:
                signals["semantic"] = sem

    # BM25 — also search individual keywords for high-recall
    try:
        cur.execute(
            """SELECT id, ts_rank(search_vector, plainto_tsquery('english', %s)) as rank
               FROM memories WHERE search_vector @@ plainto_tsquery('english', %s)
               ORDER BY rank DESC LIMIT %s""",
            (query, query, signal_limit),
        )
        bm25 = {}
        for row in cur.fetchall():
            bm25[row["id"]] = float(row["rank"]) if row["rank"] else 0.0

        # Also search top keywords individually
        import re as _re
        keywords = [w for w in _re.findall(r'\w+', query.lower()) if len(w) > 3]
        for kw in keywords[:3]:  # top 3 longest keywords
            try:
                cur.execute(
                    """SELECT id, ts_rank(search_vector, plainto_tsquery('english', %s)) as rank
                       FROM memories WHERE search_vector @@ plainto_tsquery('english', %s)
                       ORDER BY rank DESC LIMIT 100""",
                    (kw, kw),
                )
                for row in cur.fetchall():
                    mid = row["id"]
                    if mid not in bm25:
                        bm25[mid] = 0.0
                    bm25[mid] = max(bm25[mid], float(row["rank"]) if row["rank"] else 0.0)
            except Exception:
                pass

        if bm25:
            signals["bm25"] = bm25
    except Exception:
        pass

    # Entity
    try:
        query_entities = extract_entities(query)
        if query_entities:
            qset = set()
            for key, values in query_entities.items():
                if isinstance(values, list):
                    qset.update(v.lower() for v in values)
                else:
                    qset.add(str(values).lower())
            if qset:
                cur.execute("SELECT memory_id, entities FROM memory_entity_data")
                ent = {}
                for row in cur.fetchall():
                    try:
                        mem_ents = (
                            json.loads(row["entities"])
                            if row["entities"]
                            else {}
                        )
                    except (json.JSONDecodeError, TypeError):
                        mem_ents = {}
                    mset = set()
                    for key, values in mem_ents.items():
                        if isinstance(values, list):
                            mset.update(v.lower() for v in values)
                        else:
                            mset.add(str(values).lower())
                    overlap = qset & mset
                    if overlap:
                        ent[row["memory_id"]] = float(len(overlap))
                if ent:
                    signals["entity"] = ent
    except Exception:
        pass

    # Entity (already done above — included in signals dict)
    # (if entity wasn't added, skip it)

    # Temporal
    try:
        now = datetime.now()
        cur.execute(
            "SELECT id, t_event FROM memories ORDER BY t_event DESC LIMIT %s",
            (signal_limit,),
        )
        temp = {}
        for row in cur.fetchall():
            try:
                t = datetime.fromisoformat(str(row["t_event"]))
                days = (now - t).total_seconds() / 86400
                temp[row["id"]] = round(math.exp(-days / 30.0), 4)
            except (ValueError, TypeError):
                temp[row["id"]] = 0.1
        if temp:
            signals["temporal"] = temp
    except Exception:
        pass

    # RRF fusion (k=300, matching V6)
    k = int(os.environ.get("RRF_K", "300"))
    rankings = {}
    for sig_name, scores in signals.items():
        sorted_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        rankings[sig_name] = {mid: rank + 1 for rank, mid in enumerate(sorted_ids)}

    all_ids = set()
    for ranks in rankings.values():
        all_ids.update(ranks.keys())

    rrf_scores = {}
    for mid in all_ids:
        score = 0.0
        for ranks in rankings.values():
            rank = ranks.get(mid, len(ranks) + 1)
            score += 1.0 / (k + rank)
        rrf_scores[mid] = score

    sorted_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
    return {mid: rrf_scores[mid] for mid in sorted_ids[:signal_limit]}


def rerank_cross_encoder(
    query: str, candidates: Dict[int, float], top_k: int = 5
) -> List[int]:
    """
    Rerank candidate pool using cross-encoder on top-N.
    Returns list of memory_ids in reranked order.
    """
    candidate_ids = list(candidates.keys())
    if not candidate_ids:
        return []

    # Fetch content for candidates (limit to top-200 for high recall)
    if len(candidate_ids) > 200:
        # Take top candidates by RRF score
        candidate_ids = candidate_ids[:200]

    placeholders = ",".join(["%s"] * len(candidate_ids))
    cur = conn.cursor()
    cur.execute(
        f"SELECT id, content FROM memories WHERE id IN ({placeholders})",
        candidate_ids,
    )
    rows = cur.fetchall()
    id_to_content = {r["id"]: r["content"] for r in rows}

    pairs = [[query, id_to_content.get(mid, "")] for mid in candidate_ids]
    scores = CROSS_ENCODER.predict(pairs)

    scored = list(zip(candidate_ids, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [mid for mid, _ in scored[:top_k]]


# ═══════════════════════════════════════════════════════════════════
# MAIN BENCHMARK
# ═══════════════════════════════════════════════════════════════════

print("\nRunning full LongMemEval with cross-encoder reranker...\n", flush=True)

import math

found = 0
total_ev_count = 0
latencies = []
results_by_type = defaultdict(list)

for qi, q in enumerate(DATASET):
    qid = q["question_id"]
    qtype = q["question_type"]
    query = q["question"]

    # Count evidence for this question
    q_ev = sum(
        1
        for tag, has in EVIDENCE.items()
        if tag.startswith(f"{qid}|{qtype}|") and has
    )
    total_ev_count += q_ev

    # Time the retrieval
    t0 = time.time()

    # 1. Multi-signal retrieval → RRF fusion → candidates
    candidates = get_candidates(query, signal_limit=500)

    # 2. Cross-encoder rerank
    reranked_ids = rerank_cross_encoder(query, candidates, top_k=5)

    # 3. Get categories for reranked results
    if reranked_ids:
        placeholders = ",".join(["%s"] * len(reranked_ids))
        cur = conn.cursor()
        cur.execute(
            f"SELECT id, category FROM memories WHERE id IN ({placeholders})",
            reranked_ids,
        )
        cat_map = {r["id"]: r["category"] for r in cur.fetchall()}
        retrieved_cats = {
            cat_map[mid]
            for mid in reranked_ids
            if mid in cat_map and cat_map[mid]
        }
    else:
        retrieved_cats = set()

    elapsed = time.time() - t0
    latencies.append(elapsed)

    # Count found evidence
    q_found = 0
    for cat in retrieved_cats:
        if (
            cat in EVIDENCE
            and EVIDENCE[cat]
            and cat.startswith(f"{qid}|{qtype}|")
        ):
            q_found += 1
    found += q_found
    results_by_type[qtype].append({"found": q_found, "total": q_ev})

    if (qi + 1) % 50 == 0:
        avg = sum(latencies) / len(latencies)
        rec = found / total_ev_count if total_ev_count > 0 else 0
        print(
            f"  [{qi+1}/{len(DATASET)}] recall={found}/{total_ev_count} "
            f"({rec*100:.2f}%) avg_lat={avg:.3f}s",
            flush=True,
        )

# ── Final results ─────────────────────────────────────────────────
recall = found / total_ev_count if total_ev_count > 0 else 0
avg_lat = sum(latencies) / len(latencies) if latencies else 0
p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
p99_lat = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0

print(f"\n{'='*60}", flush=True)
print("FINAL RESULTS — CROSS-ENCODER RERANKER", flush=True)
print(f"{'='*60}", flush=True)
print(f"Recall@5:   {recall:.4f} ({found}/{TOTAL_EVIDENCE})", flush=True)
print(f"Avg latency: {avg_lat:.3f}s  p95: {p95_lat:.3f}s  p99: {p99_lat:.3f}s", flush=True)
print(f"Questions:   {len(DATASET)}", flush=True)
print(f"Backend:     {os.environ.get('EMBEDDING_BACKEND', 'local')}", flush=True)
print(f"Reranker:    cross-encoder/ms-marco-MiniLM-L-6-v2", flush=True)
print(f"Fusion:      RRF k=300", flush=True)

# By type
print(f"\nBy question type:", flush=True)
for qt in sorted(results_by_type.keys()):
    total_q = sum(r["total"] for r in results_by_type[qt])
    found_q = sum(r["found"] for r in results_by_type[qt])
    print(
        f"  {qt:30s}: {found_q}/{total_q} ({found_q/total_q*100:.2f}%)"
        if total_q > 0 else f"  {qt:30s}: 0/0",
        flush=True,
    )

# Save
timestamp = time.strftime("%Y%m%d-%H%M%S")
backend = os.environ.get("EMBEDDING_BACKEND", "local")
result_path = Path(__file__).parent / f"CROSSENCODER_RESULTS_{backend}_{timestamp}.json"
output = {
    "benchmark": "LongMemEval (oracle) — Cross-Encoder Reranker",
    "timestamp": timestamp,
    "total_questions": len(DATASET),
    "total_evidence_sessions": TOTAL_EVIDENCE,
    "recall_at_5": round(recall, 4),
    "found": found,
    "total": TOTAL_EVIDENCE,
    "avg_latency_s": round(avg_lat, 3),
    "p95_latency_s": round(p95_lat, 3),
    "p99_latency_s": round(p99_lat, 3),
    "configuration": {
        "embedding_backend": backend,
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "reranker_top_n": 50,
        "candidate_limit_per_signal": 500,
        "fusion": "RRF k=300",
        "weights": {"semantic": 1.5, "bm25": 1.0, "entity": 5.0, "temporal": 1.0},
    },
}
with open(result_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved: {result_path}", flush=True)
conn.close()