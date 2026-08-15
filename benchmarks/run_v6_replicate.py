#!/usr/bin/env python3
"""
Replicate V6 LongMemEval benchmark: high candidate count + RRF + cross-encoder.
Runs with sensible timeouts per query so a single slow call can't hang.
"""
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("EMBEDDING_BACKEND", "nim")
os.environ.setdefault("SIGNAL_CANDIDATE_MULTIPLIER", "100")  # 500 candidates at top_k=5
os.environ.setdefault("RRF_K", "300")
os.environ.setdefault("FUSION_METHOD", "rrf")
os.environ.setdefault("WEIGHT_SEM", "1.5")
os.environ.setdefault("WEIGHT_BM25", "1.0")
os.environ.setdefault("WEIGHT_ENT", "5.0")
os.environ.setdefault("WEIGHT_TEMP", "1.0")

import signal, psycopg2, psycopg2.extras

DB_URL = "postgresql://house:@/memster?host=/run/postgresql&port=5433"

def get_conn():
    conn = psycopg2.connect(DB_URL)
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    return conn

def load_dataset():
    data_dir = Path(__file__).parent / "LongMemEval_dataset" / "data"
    with open(data_dir / "longmemeval_oracle.json") as f:
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

from memster.hybrid_retrieval import HybridRetrievalEngine

def main():
    timeout = int(os.environ.get("QUERY_TIMEOUT", "120"))

    print("=" * 60)
    print("V6 REPLICATE — NIM backend, RRF k=300, 500 candidates, reranker")
    print("=" * 60)

    print("\n[1/4] Loading dataset...")
    dataset = load_dataset()
    print(f"  {len(dataset)} questions")

    print("\n[2/4] Building evidence map...")
    evidence, session_evidence = build_evidence(dataset)
    total_ev = sum(1 for v in session_evidence.values() if v)
    print(f"  {total_ev} evidence sessions")

    print("\n[3/4] Initializing engine...")
    engine = HybridRetrievalEngine(get_conn)
    print(f"  Backend: {engine.embedding_backend}")
    print(f"  Reranker: {engine.lightweight_reranker is not None}")
    print(f"  Fusion: {os.environ.get('FUSION_METHOD', 'rrf')}")
    print(f"  Weights: sem={os.environ.get('WEIGHT_SEM')} bm25={os.environ.get('WEIGHT_BM25')} "
          f"ent={os.environ.get('WEIGHT_ENT')} temp={os.environ.get('WEIGHT_TEMP')}")
    signal_k = int(os.environ.get("SIGNAL_CANDIDATE_MULTIPLIER", "100"))
    print(f"  Candidates/signal: 5 * {signal_k} = {5 * signal_k}")
    print(f"  Query timeout: {timeout}s")

    if not engine.embeddings_available:
        print("ERROR: no embeddings")
        return

    print("\n[4/4] Running benchmark...")
    latencies = []
    total_found = 0
    qtype_results = {}
    question_results = []
    timeouts = 0
    errors = 0

    # Use SIGALRM for per-query timeout
    def timeout_handler(signum, frame):
        raise TimeoutError("Query timed out")

    t0 = time.time()

    for qi, question in enumerate(dataset):
        qid = question["question_id"]
        qtype = question["question_type"]

        # Count evidence sessions for this question
        seen = set()
        for tag, has in evidence.items():
            if has and tag.startswith(f"{qid}|{qtype}|"):
                parts = tag.split("|")
                seen.add(f"{parts[0]}|{parts[1]}|{parts[2]}")
        q_ev_count = len(seen)

        # Run retrieval with timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        elapsed = timeout
        results = []
        try:
            start = time.time()
            results = engine.retrieve(
                query=question["question"],
                top_k=5,
                semantic_weight=1.5,
                bm25_weight=1.0,
                entity_weight=5.0,
                temporal_weight=1.0,
                rerank_with_llm=False,
                fusion_method="rrf",
            )
            elapsed = time.time() - start
            signal.alarm(0)
        except TimeoutError:
            timeouts += 1
            print(f"  TIMEOUT q{qi} ({qid}) after {timeout}s")
        except Exception as e:
            errors += 1
            print(f"  ERROR q{qi} ({qid}): {e}")
        finally:
            signal.alarm(0)

        latencies.append(elapsed)

        # Check found sessions
        retrieved_cats = {r["category"] for r in results if "category" in r}
        found = set()
        for cat in retrieved_cats:
            if cat in evidence and evidence[cat] and cat.startswith(f"{qid}|{qtype}|"):
                parts = cat.split("|")
                found.add(f"{parts[0]}|{parts[1]}|{parts[2]}")

        total_found += len(found)

        if qtype not in qtype_results:
            qtype_results[qtype] = {"found": 0, "total": 0}
        qtype_results[qtype]["found"] += len(found)
        qtype_results[qtype]["total"] += q_ev_count

        question_results.append({
            "qid": qid,
            "qtype": qtype,
            "found": len(found),
            "total": q_ev_count,
            "latency": round(elapsed, 3),
        })

        if (qi + 1) % 50 == 0:
            print(f"  [{qi+1}/{len(dataset)}] recall={total_found}/{total_ev} "
                  f"({total_found/total_ev*100:.1f}%) avg_lat={sum(latencies)/len(latencies):.3f}s")

    elapsed_total = time.time() - t0

    # Results
    recall = total_found / total_ev if total_ev > 0 else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    p95 = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    p99 = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  Recall: {recall:.4f} ({total_found}/{total_ev})")
    print(f"  Avg latency: {avg_lat:.3f}s  p95: {p95:.3f}s  p99: {p99:.3f}s")
    print(f"  Total time: {elapsed_total:.0f}s")
    print(f"  Timeouts: {timeouts}  Errors: {errors}")
    print(f"  Backend: {os.environ.get('EMBEDDING_BACKEND', 'nim')}")

    print("\n  By Question Type:")
    for qt in sorted(qtype_results.keys()):
        r = qtype_results[qt]
        rec = r["found"] / r["total"] if r["total"] > 0 else 0
        print(f"    {qt:30s}: {rec:.4f} ({r['found']}/{r['total']})")

    # Save
    out_dir = Path(__file__).parent
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = out_dir / f"v6_replicate_nim_{timestamp}.json"
    output = {
        "benchmark": "V6 Replicate — NIM backend",
        "timestamp": timestamp,
        "total_questions": len(dataset),
        "total_evidence_sessions": total_ev,
        "recall": round(recall, 4),
        "found": total_found,
        "total": total_ev,
        "avg_latency_s": round(avg_lat, 3),
        "p95_latency_s": round(p95, 3),
        "p99_latency_s": round(p99, 3),
        "total_time_s": round(elapsed_total, 1),
        "timeouts": timeouts,
        "errors": errors,
        "backend": "nim",
        "seeds": [42],
        "configuration": {
            "embedding_backend": "nim",
            "reranker": engine.lightweight_reranker is not None,
            "fusion_method": "rrf",
            "rrf_k": 300,
            "signal_candidate_multiplier": 100,
            "weights": {"semantic": 1.5, "bm25": 1.0, "entity": 5.0, "temporal": 1.0},
            "query_timeout_s": timeout,
        },
    }
    with open(result_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {result_file}")

if __name__ == "__main__":
    main()