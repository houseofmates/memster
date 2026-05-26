#!/usr/bin/env python3
"""Backfill NIM embeddings using direct API calls."""
import sys, os, json, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load env FIRST, before any memster imports
env_path = os.path.expanduser("~/.hermes/.env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k] = v

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BATCH = 20
DB_URL = os.environ.get("DATABASE_URL", "postgresql://house:@/memster?host=/run/postgresql&port=5433")

import psycopg2, psycopg2.extras

conn = psycopg2.connect(DB_URL)
conn.cursor_factory = psycopg2.extras.RealDictCursor
cur = conn.cursor()

cur.execute("SELECT id, content FROM memories WHERE source='longmemeval' AND local_embedding IS NULL")
rows = cur.fetchall()
print(f"Backfilling {len(rows)} NIM embeddings...", flush=True)

t0 = time.time()
for i in range(0, len(rows), BATCH):
    batch = rows[i:i+BATCH]
    ids = [r["id"] for r in batch]
    texts = [r["content"][:6000] for r in batch]

    payload = json.dumps({
        "model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        "input": texts
    }).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/embeddings",
        data=payload,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
        for j, d in enumerate(data["data"]):
            emb = d["embedding"]
            cur.execute(
                "UPDATE memories SET local_embedding = %s WHERE id = %s",
                (json.dumps(emb), ids[j]),
            )

    if (i + 1) % 200 == 0 or (i + 1) >= len(rows):
        conn.commit()
        elapsed = time.time() - t0
        print(f"  {min(i+BATCH, len(rows))}/{len(rows)} in {elapsed:.0f}s ({min(i+BATCH, len(rows))/elapsed:.1f}/s)", flush=True)

elapsed = time.time() - t0
print(f"Done! {len(rows)} embeddings in {elapsed:.0f}s ({len(rows)/elapsed:.1f}/s)", flush=True)
conn.close()