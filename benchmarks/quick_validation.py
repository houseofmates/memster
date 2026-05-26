#!/usr/bin/env python3
"""Quick validation: run 50 LongMemEval questions to estimate recall before full run."""
import json, os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import psycopg2, psycopg2.extras

# Set NIM env
if os.path.exists(os.path.expanduser('~/.hermes/.env')):
    with open(os.path.expanduser('~/.hermes/.env')) as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                os.environ[k] = v
os.environ['EMBEDDING_BACKEND'] = 'nim'

DB_URL = os.environ.get("DATABASE_URL", "postgresql://house:@/memster?host=/run/postgresql&port=5433")

def get_conn():
    conn = psycopg2.connect(DB_URL)
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    return conn

# Load dataset
data_dir = Path(__file__).parent / "LongMemEval_dataset" / "data"
with open(data_dir / "longmemeval_oracle.json") as f:
    dataset = json.load(f)

# Build chunked evidence map
def chunk_session(session, chunk_size=4, overlap=2):
    chunks = []
    i = 0
    while i < len(session):
        chunks.append(session[i:i + chunk_size])
        i += (chunk_size - overlap)
    return chunks

evidence = {}
for q in dataset:
    qid = q["question_id"]
    qtype = q["question_type"]
    for si, session in enumerate(q["haystack_sessions"]):
        has_ev = any(t.get("has_answer", False) for t in session)
        chunks = chunk_session(session, 4, 2)
        for ci in range(len(chunks)):
            evidence[f"{qid}|{qtype}|{si}|chunk{ci}"] = has_ev

import importlib
import memster.hybrid_retrieval as hr
importlib.reload(hr)
from memster.hybrid_retrieval import HybridRetrievalEngine

engine = HybridRetrievalEngine(get_conn)
print(f"Backend: {engine.embedding_backend}, embeddings={engine.embeddings_available}")

# Test 50 questions
selected = random.Random(42).sample(range(len(dataset)), 50)
total_found = 0
total_ev = 0
latencies = []

for idx in selected:
    q = dataset[idx]
    qid = q["question_id"]
    qtype = q["question_type"]
    query = q["question"]

    # Count unique evidence sessions
    seen = set()
    for tag, has in evidence.items():
        if has and tag.startswith(f"{qid}|{qtype}|"):
            parts = tag.split("|")
            seen.add(f"{parts[0]}|{parts[1]}|{parts[2]}")
    q_ev = len(seen)
    total_ev += q_ev

    t0 = time.time()
    results = engine.retrieve(query, top_k=5, semantic_weight=1.5, bm25_weight=1.0, entity_weight=5.0, temporal_weight=1.0)
    latencies.append(time.time() - t0)

    cats = {r.get("category","") for r in results}
    found_sessions = set()
    for cat in cats:
        if cat in evidence and evidence[cat] and cat.startswith(f"{qid}|{qtype}|"):
            parts = cat.split("|")
            found_sessions.add(f"{parts[0]}|{parts[1]}|{parts[2]}")
    total_found += len(found_sessions)

recall = total_found / total_ev if total_ev > 0 else 0
avg_lat = sum(latencies) / len(latencies)
p95 = sorted(latencies)[int(len(latencies)*0.95)]

print(f"\n=== Quick 50-Question Validation ===")
print(f"Recall@5: {recall:.4f} ({total_found}/{total_ev})")
print(f"Avg latency: {avg_lat:.3f}s  p95: {p95:.3f}s")