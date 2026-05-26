#!/usr/bin/env python3
"""Force re-backfill all longmemeval memories with NIM 2048-dim embeddings."""
import os, json, time, urllib.request
from pathlib import Path

env_path = Path.home() / ".hermes" / ".env"
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v

API_KEY = os.environ["OPENROUTER_API_KEY"]
BATCH = 20
DB_URL = "postgresql://house:@/memster?host=/run/postgresql&port=5433"

import psycopg2, psycopg2.extras
conn = psycopg2.connect(DB_URL)
conn.cursor_factory = psycopg2.extras.RealDictCursor
cur = conn.cursor()

cur.execute("SELECT id, content FROM memories WHERE source='longmemeval'")
rows = list(cur.fetchall())
print(f"Backfilling {len(rows)} memories with NIM (2048-dim)...", flush=True)

t0 = time.time()
done = 0
for i in range(0, len(rows), BATCH):
    batch = rows[i:i+BATCH]
    ids = [r["id"] for r in batch]
    texts = [r["content"][:6000] for r in batch]

    payload = json.dumps({"model":"nvidia/llama-nemotron-embed-vl-1b-v2:free","input":texts}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/embeddings", data=payload,
        headers={"Authorization":f"Bearer {API_KEY}","Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
        for j, d in enumerate(data["data"]):
            emb = d["embedding"]
            cur.execute("UPDATE memories SET local_embedding = %s WHERE id = %s", (json.dumps(emb), ids[j]))

    done += len(batch)
    if done % 200 == 0 or done >= len(rows):
        conn.commit()
        el = time.time() - t0
        print(f"  {done}/{len(rows)} in {el:.0f}s ({done/el:.1f}/s)", flush=True)

print(f"Done! {done} in {time.time()-t0:.0f}s", flush=True)
conn.close()