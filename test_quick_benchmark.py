#!/usr/bin/env python3
import os
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, '.')

# Configuration
os.environ['EMBEDDING_BACKEND'] = 'local'  # or 'nim'
os.environ['USE_LIGHTWEIGHT_RERANKER'] = 'true'
os.environ['LIGHTWEIGHT_RERANKER_MODEL'] = 'mixedbread-ai/mxbai-rerank-xsmall-v1'
os.environ['USE_TWO_STAGE_RERANKER'] = 'true'
os.environ['TWO_STAGE_RERANKER_CANDIDATES_MULTIPLIER'] = '10'
os.environ['FUSION_METHOD'] = 'rrf'
os.environ['WEIGHT_SEM'] = '1.5'
os.environ['WEIGHT_BM25'] = '1.0'
os.environ['WEIGHT_ENT'] = '5.0'
os.environ['WEIGHT_TEMP'] = '1.0'
os.environ['TOP_K'] = '5'

from memster.hybrid_retrieval import HybridRetrievalEngine
import psycopg2
import psycopg2.extras

DB_URL = "postgresql://house:@/memster?host=/run/postgresql&port=5433"

def get_conn():
    conn = psycopg2.connect(DB_URL)
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    return conn

def load_dataset(dataset_type="oracle"):
    data_dir = Path(__file__).parent / "benchmarks" / "LongMemEval_dataset" / "data"
    with open(data_dir / f"longmemeval_{dataset_type}.json") as f:
        return json.load(f)

def build_evidence(dataset):
    evidence = {}
    session_evidence = {}
    for q in dataset:
        qid = q["question_id"]
        qtype = q["question_type"]
        for si, session in enumerate(q["haystack_sessions"]):
            has_ev = any(turn.get("has_answer", False) for turn in session)
            session_key = f"{qid}|{qtype}|{si}"
            session_evidence[session_key] = has_ev
            evidence[session_key] = has_ev
    return evidence, session_evidence

def compute_recall_at_k(engine, question, evidence, qid, qtype, k=5):
    results = engine.retrieve(
        query=question["question"],
        top_k=k,
        semantic_weight=float(os.environ.get("WEIGHT_SEM", "1.5")),
        bm25_weight=float(os.environ.get("WEIGHT_BM25", "1.0")),
        entity_weight=float(os.environ.get("WEIGHT_ENT", "5.0")),
        temporal_weight=float(os.environ.get("WEIGHT_TEMP", "1.0")),
        fusion_method=os.environ.get("FUSION_METHOD", "weighted"),
    )
    retrieved_cats = {r["category"] for r in results if "category" in r}
    for cat in retrieved_cats:
        if cat in evidence and evidence[cat] and cat.startswith(f"{qid}|{qtype}|"):
            return True
    return False

def main():
    print("=" * 60)
    print(f"QUICK BENCHMARK - {os.environ['EMBEDDING_BACKEND'].upper()} backend")
    print("=" * 60)
    
    # Load dataset
    print("[1/3] Loading dataset...")
    dataset = load_dataset("oracle")
    print(f"  Loaded {len(dataset)} questions")
    
    # Build evidence
    print("[2/3] Building evidence...")
    evidence, session_evidence = build_evidence(dataset)
    total_evidence_sessions = sum(1 for v in session_evidence.values() if v)
    print(f"  Total evidence sessions: {total_evidence_sessions}")
    
    # Initialize engine
    print("[3/3] Initializing engine...")
    engine = HybridRetrievalEngine(get_conn)
    print(f"  Backend: {engine.embedding_backend}")
    print(f"  Embeddings available: {engine.embeddings_available}")
    print(f"  Lightweight reranker: {engine.lightweight_reranker is not None}")
    print(f"  Two-stage reranking: {engine.use_two_stage_reranker}")
    print(f"  Fusion: {os.environ.get('FUSION_METHOD', 'weighted')}")
    print(f"  Weights: sem={os.environ.get('WEIGHT_SEM', '1.5')} bm25={os.environ.get('WEIGHT_BM25', '1.0')} ent={os.environ.get('WEIGHT_ENT', '5.0')} temp={os.environ.get('WEIGHT_TEMP', '1.0')}")
    
    if not engine.embeddings_available:
        print("ERROR: No embedding backend available")
        return
    
    # Run evaluation on first 100 questions
    print("\nRunning evaluation on first 100 questions...")
    latencies = []
    total_found = 0
    
    for i, question in enumerate(dataset[:100]):
        qid = question["question_id"]
        qtype = question["question_type"]
        
        start = time.time()
        found = compute_recall_at_k(engine, question, evidence, qid, qtype, k=5)
        elapsed = time.time() - start
        
        latencies.append(elapsed)
        if found:
            total_found += 1
        
        if (i + 1) % 20 == 0:
            current_recall = total_found / (i + 1)
            avg_lat = sum(latencies) / len(latencies)
            print(f"  [{i+1}/100] recall={current_recall:.3f} ({total_found}/{i+1}) avg_lat={avg_lat:.3f}s")
    
    # Final results
    recall_at_5 = total_found / 100
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Recall@5: {recall_at_5:.4f} ({total_found}/100)")
    print(f"Avg latency: {avg_latency:.3f}s")
    print(f"P95 latency: {p95_latency:.3f}s")
    print(f"Target: Recall@5 >= 0.950, Latency < 1.0s")
    
    if recall_at_5 >= 0.950 and avg_latency < 1.0:
        print("✅ PASS: Meets targets")
    else:
        print("❌ FAIL: Does not meet targets")
        if recall_at_5 < 0.950:
            print(f"   Recall deficit: {0.950 - recall_at_5:.4f}")
        if avg_latency >= 1.0:
            print(f"   Latency excess: {avg_latency - 1.0:.3f}s")

if __name__ == "__main__":
    main()