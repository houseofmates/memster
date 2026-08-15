#!/usr/bin/env python3
import os
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, '.')

# Configuration - use local for fast testing, no API key needed
os.environ['EMBEDDING_BACKEND'] = 'local'
os.environ['USE_LIGHTWEIGHT_RERANKER'] = 'true'
os.environ['LIGHTWEIGHT_RERANKER_MODEL'] = 'mixedbread-ai/mxbai-rerank-xsmall-v1'
os.environ['USE_TWO_STAGE_RERANKER'] = 'true'
os.environ['TWO_STAGE_RERANKER_CANDIDATES_MULTIPLIER'] = '10'
os.environ['FUSION_METHOD'] = 'rrf'
os.environ['WEIGHT_SEM'] = '1.5'
os.environ['WEIGHT_BM25'] = '1.0'
os.environ['WEIGHT_ENT'] = '5.0'
os.environ['WEIGHT_TEMP'] = '1.0'

from memster.hybrid_retrieval import HybridRetrievalEngine
import psycopg2
import psycopg2.extras

DB_URL = "postgresql://house:@/memster?host=/run/postgresql&port=5433"

def get_conn():
    conn = psycopg2.connect(DB_URL)
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    return conn

def main():
    print("Initializing engine (this may take a moment to load models)...")
    start_init = time.time()
    engine = HybridRetrievalEngine(get_conn)
    init_time = time.time() - start_init
    print(f"Engine initialized in {init_time:.2f}s")
    print(f"Backend: {engine.embedding_backend}")
    print(f"Embeddings available: {engine.embeddings_available}")
    print(f"Lightweight reranker loaded: {engine.lightweight_reranker is not None}")
    if engine.lightweight_reranker is not None:
        print(f"Reranker model: {os.environ['LIGHTWEIGHT_RERANKER_MODEL']}")
    
    # Test a few queries
    test_queries = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet in our solar system?",
        "When did World War II end?",
        "What is the chemical symbol for gold?"
    ]
    
    print("\nTesting retrieval on 5 sample queries...")
    latencies = []
    for i, query in enumerate(test_queries):
        start = time.time()
        results = engine.retrieve(
            query=query,
            top_k=5,
            semantic_weight=1.5,
            bm25_weight=1.0,
            entity_weight=5.0,
            temporal_weight=1.0,
            fusion_method='rrf',
        )
        elapsed = time.time() - start
        latencies.append(elapsed)
        print(f"  Query {i+1}: '{query[:30]}...' -> {len(results)} results in {elapsed:.3f}s")
        if results:
            print(f"    Top result ID: {results[0]['id']}, score: {results[0].get('hybrid_score', 0):.4f}")
    
    avg_lat = sum(latencies) / len(latencies)
    print(f"\nAverage latency over {len(test_queries)} queries: {avg_lat:.3f}s")
    print(f"Target latency: < 1.0s")
    if avg_lat < 1.0:
        print("✅ Latency target met")
    else:
        print("❌ Latency target NOT met")
    
    # Also test without reranker to see the baseline
    print("\n" + "="*50)
    print("Testing WITHOUT lightweight reranker (fusion only)...")
    os.environ['USE_LIGHTWEIGHT_RERANKER'] = 'false'
    # Re-initialize engine to pick up the new setting
    engine_no_rerank = HybridRetrievalEngine(get_conn)
    latencies_no_rerank = []
    for i, query in enumerate(test_queries):
        start = time.time()
        results = engine_no_rerank.retrieve(
            query=query,
            top_k=5,
            semantic_weight=1.5,
            bm25_weight=1.0,
            entity_weight=5.0,
            temporal_weight=1.0,
            fusion_method='rrf',
        )
        elapsed = time.time() - start
        latencies_no_rerank.append(elapsed)
        print(f"  Query {i+1}: {len(results)} results in {elapsed:.3f}s")
    avg_lat_no_rerank = sum(latencies_no_rerank) / len(latencies_no_rerank)
    print(f"\nAverage latency without reranker: {avg_lat_no_rerank:.3f}s")

if __name__ == "__main__":
    main()