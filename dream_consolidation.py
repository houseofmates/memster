#!/usr/bin/env python3
"""dream_consolidation.py — autonomous memster memory consolidation runner

Invoked by memster-dream.service systemd timer. Performs offline memory
consolidation: decay-based cleanup, pattern discovery, cross-session linking.
"""

import math
import sys
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# Default memster DB
DB_PATH = Path.home() / "memster" / "memster_unified.db"

def sleep_consolidate(db_path=None, batch_size=100):
    """Consolidate old memories with access-weighted decay."""
    path = db_path or DB_PATH
    if not path.exists():
        print(f"DB not found: {path}")
        return {"error": "db missing"}
    
    conn = sqlite3.connect(str(path))
    c = conn.cursor()
    now = datetime.now()
    results = {"decayed": 0, "promoted": 0, "summarized": 0}
    
    # Get memories eligible for decay (L2 tier, older than 7 days, decay_score > 0.1)
    cutoff = (now - timedelta(days=7)).isoformat()
    c.execute("""
        SELECT id, decay_score, access_count
        FROM memories
        WHERE tier = 'L2'
          AND (t_recorded IS NULL OR t_recorded < ?)
          AND decay_score > 0.1
    """, (cutoff,))
    
    eligible = c.fetchall()
    decayed = 0
    for mem_id, decay_score, access_count in eligible:
        # Decay rate = base / (1 + log(access_count))
        decay_factor = 1.0 / (1 + math.log(access_count + 1))
        new_decay = decay_score * (0.5 + 0.5 * decay_factor)
        
        if new_decay < 0.05:
            # Archive/delete tier ? For now just mark as decayed
            c.execute("UPDATE memories SET decay_score = ? WHERE id = ?", (new_decay, mem_id))
            decayed += 1
        else:
            c.execute("UPDATE memories SET decay_score = ? WHERE id = ?", (new_decay, mem_id))
    
    results["decayed"] = decayed
    
    # Promote hot memories (accessed > 10 times) up to L0 (highest tier)
    c.execute("""
        SELECT id FROM memories
        WHERE access_count > 10 AND tier NOT IN ('L0')
    """)
    hot = c.fetchall()
    promoted = 0
    for (mem_id,) in hot:
        c.execute("UPDATE memories SET tier = 'L0' WHERE id = ?", (mem_id,))
        promoted += 1
    results["promoted"] = promoted
    
    conn.commit()
    conn.close()
    
    print(f"Consolidation: {decayed} decayed, {promoted} promoted")
    return results

if __name__ == "__main__":
    try:
        result = sleep_consolidate()
        print(f"Memster dream cycle complete: {result}")
        sys.exit(0 if result.get("error") is None else 1)
    except Exception as e:
        print(f"Dream cycle failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
