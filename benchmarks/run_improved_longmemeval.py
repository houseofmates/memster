#!/usr/bin/env python3
"""
LongMemEval evaluation using the improved HybridRetrievalEngine.

Features:
  - Embedding backend switcher (local/nim)
  - Lightweight reranker (mxbai-rerank-xsmall-v1)
  - Query expansion (WordNet synonyms)
  - Two-stage reranking (hybrid -> top-N -> reranker -> top-K)
  - Weighted or RRF fusion

Usage:
  EMBEDDING_BACKEND=local python benchmarks/run_improved_longmemeval.py
  EMBEDDING_BACKEND=nim   python benchmarks/run_improved_longmemeval.py

Environment:
  WEIGHT_SEM, WEIGHT_BM25, WEIGHT_ENT, WEIGHT_TEMP
  FUSION_METHOD (weighted|rrf)
  USE_QUERY_EXPANSION (true|false)
  USE_TWO_STAGE_RERANKER (true|false)
  QUERY_EXPANSION_MAX_SYNONYMS
  TWO_STAGE_RERANKER_CANDIDATES_MULTIPLIER
  USE_LIGHTWEIGHT_RERANKER (true|false)
"""
import json, os, sys, time, random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://house:@/memster?host=/run/postgresql&port=5433",
)


def get_conn():
    conn = psycopg2.connect(DB_URL)
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    return conn


def load_dataset(dataset_type="oracle"):
    """Load LongMemEval dataset."""
    data_dir = (
        Path(__file__).parent.parent
        / "benchmarks"
        / "LongMemEval_dataset"
        / "data"
    )
    with open(data_dir / f"longmemeval_{dataset_type}.json") as f:
        return json.load(f)


def chunk_session(session, chunk_size=4, overlap=2):
    """Split a multi-turn session into overlapping chunks of turns.
    
    Each chunk is a list of consecutive turns. Overlap ensures answer turns
    aren't split across chunk boundaries.
    """
    chunks = []
    i = 0
    while i < len(session):
        chunk = session[i:i + chunk_size]
        chunks.append(chunk)
        i += (chunk_size - overlap)
    return chunks


def session_to_text(turns):
    """Convert a list of turns to a single text blob."""
    return "\n".join(
        f"[{t.get('role', '')}]: {t.get('content', '')}"
        for t in turns
    ).strip()


def batch_store_all(dataset):
    """Store all LongMemEval sessions in the database as chunked memory beads."""
    conn = get_conn()
    cur = conn.cursor()

    # Check if chunked data already exists
    cur.execute("SELECT COUNT(*) FROM memories WHERE source = 'longmemeval'")
    existing = cur.fetchone()["count"]
    if existing > 0:
        print(f"  {existing} memories already stored, skipping re-insertion")
        conn.close()
        return existing

    now = datetime.now().isoformat()
    total = 0
    for qi, question in enumerate(dataset):
        qid = question["question_id"]
        qtype = question["question_type"]
        for si, session in enumerate(question["haystack_sessions"]):
            # Chunk the session into smaller pieces
            chunks = chunk_session(session, chunk_size=4, overlap=2)
            for ci, chunk in enumerate(chunks):
                session_text = session_to_text(chunk)
                if len(session_text) < 10:
                    continue
                t_event = question.get("haystack_dates", [now])[
                    min(si, len(question.get("haystack_dates", [now])) - 1)
                ]
                # Category includes chunk index so we can still identify the original session
                category_tag = f"{qid}|{qtype}|{si}|chunk{ci}"
                try:
                    cur.execute(
                        """INSERT INTO memories (content, network_type, source, t_event, t_recorded, category, tier, importance)
                           VALUES (%s, 'experience', 'longmemeval', %s, %s, %s, 'L3', 0.7)""",
                        (session_text, t_event, now, category_tag),
                    )
                    total += 1
                except Exception:
                    pass
    conn.commit()
    conn.close()
    print(f"  Stored {total} chunked memories")
    return total


def build_evidence(dataset):
    """Build evidence mapping: category_tag -> has_answer (bool).
    Works with chunked categories: {qid}|{qtype}|{si}|chunk{ci}
    """
    evidence = {}
    # Track which sessions have answers (for counting unique sessions)
    session_evidence = {}  # {qid}|{qtype}|{si} -> bool
    for q in dataset:
        qid = q["question_id"]
        qtype = q["question_type"]
        for si, session in enumerate(q["haystack_sessions"]):
            has_ev = any(turn.get("has_answer", False) for turn in session)
            session_key = f"{qid}|{qtype}|{si}"
            session_evidence[session_key] = has_ev
            # Create evidence entries for all chunks of this session
            chunks = chunk_session(session, chunk_size=4, overlap=2)
            for ci in range(len(chunks)):
                evidence[f"{qid}|{qtype}|{si}|chunk{ci}"] = has_ev
    return evidence, session_evidence


def backfill_local_embeddings(engine):
    """Backfill local embeddings for memories that don't have them."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM memories WHERE source='longmemeval' AND local_embedding IS NULL"
    )
    missing = cur.fetchone()["count"]
    if missing == 0:
        conn.close()
        return 0

    print(f"\n  Backfilling {missing} missing local embeddings...")
    cur.execute(
        "SELECT id, content FROM memories WHERE source='longmemeval' AND local_embedding IS NULL"
    )
    rows = cur.fetchall()
    done = 0
    for row in rows:
        emb = engine.embed_and_store(row["id"], row["content"])
        if emb:
            done += 1
        if done % 100 == 0:
            conn.commit()
    conn.commit()
    conn.close()
    print(f"  Backfilled {done} embeddings")
    return done


# Import the improved hybrid retrieval engine
from memster.hybrid_retrieval import HybridRetrievalEngine


def compute_recall_at_k(
    engine, question, evidence, qid, qtype, k=5
) -> bool:
    """Check if at least one evidence session is found in top-k results."""
    results = engine.retrieve(
        query=question["question"],
        top_k=k,
        semantic_weight=float(os.environ.get("WEIGHT_SEM", "1.5")),
        bm25_weight=float(os.environ.get("WEIGHT_BM25", "1.0")),
        entity_weight=float(os.environ.get("WEIGHT_ENT", "5.0")),
        temporal_weight=float(os.environ.get("WEIGHT_TEMP", "1.0")),
        rerank_with_llm=False,
        fusion_method=os.environ.get("FUSION_METHOD", "weighted"),
    )

    retrieved_cats = {r["category"] for r in results if "category" in r}

    for cat in retrieved_cats:
        if (
            cat in evidence
            and evidence[cat]
            and cat.startswith(f"{qid}|{qtype}|")
        ):
            return True
    return False


def run_longmemeval_benchmark(
    engine,
    dataset,
    evidence,
    total_evidence_count,
    top_k=5,
    seed=None,
):
    """Run the full LongMemEval benchmark and return metrics."""
    if seed is not None:
        random.seed(seed)

    print(f"\nRunning evaluation (top_k={top_k})...")

    latencies = []
    total_found = 0
    qtype_results = defaultdict(list)
    question_results = []

    for qi, question in enumerate(dataset):
        qid = question["question_id"]
        qtype = question["question_type"]

        # Count unique evidence SESSIONS for this question
        seen_sessions = set()
        for tag, has in evidence.items():
            if has and tag.startswith(f"{qid}|{qtype}|"):
                # Extract session key: {qid}|{qtype}|{si}
                parts = tag.split("|")
                session_key = f"{parts[0]}|{parts[1]}|{parts[2]}"
                seen_sessions.add(session_key)
        q_evidence_count = len(seen_sessions)

        # Measure retrieval with timing
        start = time.time()
        results = engine.retrieve(
            query=question["question"],
            top_k=top_k,
            semantic_weight=float(os.environ.get("WEIGHT_SEM", "1.5")),
            bm25_weight=float(os.environ.get("WEIGHT_BM25", "1.0")),
            entity_weight=float(os.environ.get("WEIGHT_ENT", "5.0")),
            temporal_weight=float(os.environ.get("WEIGHT_TEMP", "1.0")),
            rerank_with_llm=False,
            fusion_method=os.environ.get("FUSION_METHOD", "weighted"),
        )
        elapsed = time.time() - start
        latencies.append(elapsed)

        # Check which unique evidence SESSIONS were found
        retrieved_cats = {r["category"] for r in results if "category" in r}
        found_sessions = set()
        for cat in retrieved_cats:
            if cat in evidence and evidence[cat] and cat.startswith(f"{qid}|{qtype}|"):
                # Extract session key from chunk category
                parts = cat.split("|")
                session_key = f"{parts[0]}|{parts[1]}|{parts[2]}"
                found_sessions.add(session_key)
        found_evidence = len(found_sessions)

        total_found += found_evidence
        session_recall = (
            round(found_evidence / q_evidence_count, 4)
            if q_evidence_count > 0
            else 0
        )

        qtype_results[qtype].append(
            {
                "qid": qid,
                "session_recall": session_recall,
                "found": found_evidence,
                "total": q_evidence_count,
            }
        )

        question_results.append(
            {
                "qid": qid,
                "qtype": qtype,
                "found": found_evidence,
                "total": q_evidence_count,
                "session_recall": session_recall,
                "latency": round(elapsed, 3),
            }
        )

        if qi % 50 == 0 and qi > 0:
            print(
                f"  [{qi}/{len(dataset)}] recall={total_found}/{total_evidence_count} "
                f"({total_found/total_evidence_count*100:.1f}%) "
                f"avg_lat={sum(latencies)/len(latencies):.3f}s"
            )

    # Compute final metrics
    recall_at_k = (
        total_found / total_evidence_count
        if total_evidence_count > 0
        else 0
    )
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0

    print(f"\n  FINAL: Recall@{top_k}={recall_at_k:.4f} "
          f"({total_found}/{total_evidence_count})")
    print(f"  Avg latency: {avg_latency:.3f}s  p95: {p95_latency:.3f}s  p99: {p99_latency:.3f}s")

    # Per-type breakdown
    print("\n  By Question Type:")
    stats = {}
    for qt, scores in sorted(qtype_results.items()):
        sr = sum(s["session_recall"] for s in scores) / len(scores) if scores else 0.0
        stats[qt] = {"count": len(scores), "session_recall": round(sr, 4)}
        total_q = sum(s["total"] for s in scores)
        found_q = sum(s["found"] for s in scores)
        raw_recall = found_q / total_q if total_q > 0 else 0
        print(f"    {qt:30s}: {raw_recall:.4f} ({found_q}/{total_q})")

    return {
        "recall_at_k": recall_at_k,
        "top_k": top_k,
        "found": total_found,
        "total": total_evidence_count,
        "avg_latency_s": round(avg_latency, 3),
        "p95_latency_s": round(p95_latency, 3),
        "p99_latency_s": round(p99_latency, 3),
        "by_type": stats,
        "question_results": question_results,
    }


def main():
    print("=" * 60)
    print("LONGMEMEVAL — Improved Hybrid Retrieval Engine v2.0")
    print("=" * 60)

    backend = os.environ.get("EMBEDDING_BACKEND", "local")
    top_k = int(os.environ.get("TOP_K", "5"))

    # Get multiple seeds
    seeds_str = os.environ.get("SEEDS", "42")
    seeds = [int(s.strip()) for s in seeds_str.split(",")]

    # Load dataset
    print("\n[1/5] Loading LongMemEval dataset...")
    dataset = load_dataset("oracle")
    print(f"  Loaded {len(dataset)} questions")

    # Store all sessions if needed
    print("\n[2/5] Preparing database...")
    total_stored = batch_store_all(dataset)

    # Build evidence map
    print("\n[3/5] Building evidence map...")
    evidence, session_evidence = build_evidence(dataset)
    total_unique_sessions = sum(1 for v in session_evidence.values() if v)
    total_evidence_chunks = sum(1 for v in evidence.values() if v)
    print(f"  Total unique evidence sessions: {total_unique_sessions}")
    print(f"  Total evidence chunks (with overlap): {total_evidence_chunks}")

    # Initialize engine
    print("\n[4/5] Initializing HybridRetrievalEngine...")
    engine = HybridRetrievalEngine(get_conn)
    print(f"  Embedding backend: {engine.embedding_backend}")
    print(f"  Embeddings available: {engine.embeddings_available}")
    print(f"  Lightweight reranker: {engine.lightweight_reranker is not None}")
    print(f"  Query expansion: {engine.use_query_expansion}")
    print(f"  Two-stage reranking: {engine.use_two_stage_reranker}")
    print(f"  Fusion: {os.environ.get('FUSION_METHOD', 'weighted')}")
    print(f"  Weights: sem={os.environ.get('WEIGHT_SEM', '1.5')} "
          f"bm25={os.environ.get('WEIGHT_BM25', '1.0')} "
          f"ent={os.environ.get('WEIGHT_ENT', '5.0')} "
          f"temp={os.environ.get('WEIGHT_TEMP', '1.0')}")

    if not engine.embeddings_available:
        print("ERROR: No embedding backend available.")
        return None

    # Backfill missing local embeddings
    if engine.embedding_backend == "local":
        n_backfilled = backfill_local_embeddings(engine)
        if n_backfilled > 0:
            print(f"  Backfilled {n_backfilled} missing embeddings")

    # Run benchmark for each seed
    print(f"\n[5/5] Running benchmark (seeds={seeds})...")

    all_results = []
    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        result = run_longmemeval_benchmark(
            engine, dataset, evidence, total_unique_sessions, top_k, seed
        )
        all_results.append(result)

    # Average results
    avg_recall = sum(r["recall_at_k"] for r in all_results) / len(all_results)
    avg_lat = sum(r["avg_latency_s"] for r in all_results) / len(all_results)
    avg_p95 = sum(r["p95_latency_s"] for r in all_results) / len(all_results)

    print("\n" + "=" * 60)
    print("FINAL AVERAGED RESULTS")
    print("=" * 60)
    print(f"  Recall@{top_k}: {avg_recall:.4f} (avg over {len(seeds)} seeds)")
    print(f"  Avg latency: {avg_lat:.3f}s  p95: {avg_p95:.3f}s")
    print(f"  Backend: {backend}")
    for r in all_results:
        print(f"    seed: recall={r['recall_at_k']:.4f} lat={r['avg_latency_s']:.3f}s")

    # Save results
    out_dir = Path(__file__).parent
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = (
        out_dir
        / f"longmemeval_improved_{backend}_{timestamp}.json"
    )
    output = {
        "benchmark": "LongMemEval (oracle) — Improved Engine v2.0",
        "timestamp": timestamp,
        "total_questions": len(dataset),
        "total_evidence_sessions": total_unique_sessions,
        "top_k": top_k,
        "recall_at_k": round(avg_recall, 4),
        "avg_latency_s": round(avg_lat, 3),
        "p95_latency_s": round(avg_p95, 3),
        "backend": backend,
        "seeds": seeds,
        "individual_results": [
            {
                "seed": seeds[i],
                "recall_at_k": r["recall_at_k"],
                "avg_latency_s": r["avg_latency_s"],
                "p95_latency_s": r["p95_latency_s"],
            }
            for i, r in enumerate(all_results)
        ],
        "configuration": {
            "embedding_backend": backend,
            "use_lightweight_reranker": engine.lightweight_reranker is not None,
            "use_query_expansion": engine.use_query_expansion,
            "use_two_stage_reranker": engine.use_two_stage_reranker,
            "fusion_method": os.environ.get("FUSION_METHOD", "weighted"),
            "weights": {
                "semantic": float(os.environ.get("WEIGHT_SEM", "1.5")),
                "bm25": float(os.environ.get("WEIGHT_BM25", "1.0")),
                "entity": float(os.environ.get("WEIGHT_ENT", "5.0")),
                "temporal": float(os.environ.get("WEIGHT_TEMP", "1.0")),
            },
            "query_expansion_max_synonyms": int(
                os.environ.get("QUERY_EXPANSION_MAX_SYNONYMS", "2")
            ),
            "two_stage_reranker_candidates_multiplier": int(
                os.environ.get(
                    "TWO_STAGE_RERANKER_CANDIDATES_MULTIPLIER", "5"
                )
            ),
        },
    }
    with open(result_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {result_file}")

    return output


if __name__ == "__main__":
    main()