#!/usr/bin/env bash
# Run LongMemEval benchmark with fixed NIM backend, output to file
set -o pipefail
cd /home/house/projects/memster || exit 1

export PYTHONUNBUFFERED=1
export EMBEDDING_BACKEND=nim
export FUSION_METHOD=weighted
export WEIGHT_SEM=1.5
export WEIGHT_BM25=1.0
export WEIGHT_ENT=5.0
export WEIGHT_TEMP=1.0
export USE_QUERY_EXPANSION=true
export USE_TWO_STAGE_RERANKER=false
export USE_LIGHTWEIGHT_RERANKER=false
export SIGNAL_CANDIDATE_MULTIPLIER=100

LOGFILE="/home/house/projects/memster/benchmarks/bench_run_$(date +%Y%m%d-%H%M%S).log"
echo "Starting benchmark at $(date)" | tee "$LOGFILE"
echo "Log: $LOGFILE" | tee -a "$LOGFILE"

# No reranker, no two-stage — just hybrid fusion
echo "Using: nim backend, query_expansion, no reranker" | tee -a "$LOGFILE"

python3 -u benchmarks/run_improved_longmemeval.py 2>&1 | stdbuf -oL tee -a "$LOGFILE"

EXIT_CODE=$?
echo "Exit code: $EXIT_CODE at $(date)" | tee -a "$LOGFILE"
exit $EXIT_CODE