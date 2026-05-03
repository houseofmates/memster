#!/usr/bin/env python3
"""
Memster Spaced Repetition Module (SM-2++ Variant)

Based on:
  - Mnemosyne's SM-2 algorithm
  - Research from mem0 and autocontext on importance weighting
  - Adaptations for fused event horizons (beads-style temporal validity)
"""

import json
import sqlite3
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any

logger = None
DEFAULT_DB_PATH = os.path.expanduser("~/memster/memster_unified.db")

# SM-2++ parameters
INITIAL_EASINESS = 2.5
MIN_EASINESS = 1.3
MAX_EASINESS = 2.5
MAX_INTERVAL_DAYS = 36500

QUALITY_EASINESS_DELTA = {
    0: -1.0, 1: -0.6, 2: -0.3, 3: 0.0, 4: 0.15, 5: 0.3,
    0.5: -0.9, 1.5: -0.45, 2.5: -0.15, 3.5: 0.075, 4.5: 0.225,
}

def get_db(db_path: str = None):
    conn = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _column_exists(cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row["name"] == column for row in cursor.fetchall())

def ensure_schema(db_path: str = None) -> None:
    conn = get_db(db_path)
    c = conn.cursor()
    required = [
        ("repetition_strength", "REAL DEFAULT 2.5"),
        ("repetition_interval", "INTEGER DEFAULT 0"),
        ("next_review_date", "TEXT"),
        ("review_count", "INTEGER DEFAULT 0"),
        ("lapses", "INTEGER DEFAULT 0"),
        ("last_review_quality", "REAL"),
        ("review_history", "JSON DEFAULT '[]'"),
    ]
    for col, typedef in required:
        if not _column_exists(c, "memories", col):
            try:
                c.execute(f"ALTER TABLE memories ADD COLUMN {col} {typedef}")
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower():
                    pass
                else:
                    raise
    conn.commit()
    conn.close()

def schedule_review(memory_id: int, quality: float, db_path: str = None, importance_modifier: float = 1.0) -> Dict:
    conn = get_db(db_path)
    c = conn.cursor()
    c.execute("SELECT repetition_strength, repetition_interval, review_count, lapses, importance FROM memories WHERE id = ?", (memory_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return {"error": f"memory {memory_id} not found"}
    ef = row["repetition_strength"] or INITIAL_EASINESS
    interval = row["repetition_interval"] or 0
    review_count = row["review_count"] or 0
    lapses = row["lapses"] or 0
    importance = row["importance"] if row["importance"] is not None else 0.5
    # Importance modifier: low importance = more frequent (higher occurence)
    imp_mod = max(0.5, min(2.0, 1.0 + (0.5 - importance)))
    if importance_modifier != 1.0:
        imp_mod *= importance_modifier
    delta = QUALITY_EASINESS_DELTA.get(quality, -0.3 if quality < 3 else 0.15)
    ef = max(MIN_EASINESS, min(MAX_EASINESS, ef + delta))
    if quality < 3:
        interval = 1
        lapses += 1
    else:
        if review_count == 0:
            interval = 1
        elif review_count == 1:
            interval = 3
        elif review_count == 2:
            interval = 7
        else:
            interval = max(1, min(MAX_INTERVAL_DAYS, round(interval * ef * imp_mod)))
    review_count += 1
    next_review = date.today() + timedelta(days=interval)
    c.execute("SELECT review_history FROM memories WHERE id = ?", (memory_id,))
    hist_row = c.fetchone()
    history = json.loads(hist_row["review_history"] or "[]")
    history.append({"date": datetime.now().isoformat(), "quality": quality, "interval": interval, "easiness": round(ef, 3)})
    if len(history) > 50:
        history = history[-50:]
    c.execute("""UPDATE memories SET repetition_strength=?, repetition_interval=?, next_review_date=?, review_count=?, lapses=?, last_review_quality=?, review_history=? WHERE id=?""",
              (ef, interval, next_review.isoformat(), review_count, lapses, quality, json.dumps(history), memory_id))
    conn.commit()
    conn.close()
    return {"memory_id": memory_id, "new_interval": interval, "next_review": next_review.isoformat(), "easiness": round(ef, 3), "review_count": review_count, "lapses": lapses}

def get_due_reviews(db_path: str = None, date_str: str = None, limit: int = 50) -> List[Dict]:
    target = date_str or date.today().isoformat()
    conn = get_db(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT id, content, repetition_interval, next_review_date, review_count, lapses, importance
        FROM memories 
        WHERE next_review_date <= ?
          AND (is_ephemeral IS NULL OR is_ephemeral = 0)
        ORDER BY CASE WHEN lapses > 0 THEN 0 ELSE 1 END, next_review_date ASC, importance DESC
        LIMIT ?
    """, (target, limit))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def batch_review(memory_ids: List[int], qualities: List[float], db_path: str = None) -> Dict:
    if len(memory_ids) != len(qualities):
        return {"error": "length mismatch"}
    results = []
    conn = get_db(db_path)
    conn.execute("BEGIN")
    try:
        for mid, q in zip(memory_ids, qualities):
            res = schedule_review(mid, q, db_path=db_path)
            if "error" in res:
                raise Exception(res["error"])
            results.append(res)
        conn.commit()
    except Exception as e:
        conn.rollback()
        return {"error": str(e), "partial": results}
    finally:
        conn.close()
    return {"processed": len(results), "results": results}

def predict_retention(memory_id: int, future_date: date, db_path: str = None) -> float:
    conn = get_db(db_path)
    c = conn.cursor()
    c.execute("SELECT repetition_interval, repetition_strength FROM memories WHERE id = ?", (memory_id,))
    row = c.fetchone()
    conn.close()
    if not row or not row["repetition_interval"]:
        return 1.0
    interval = row["repetition_interval"]
    ef = row["repetition_strength"] or INITIAL_EASINESS
    days_ahead = max(0, (future_date - date.today()).days)
    if days_ahead == 0:
        return 1.0
    factor = ef * 0.5
    retention = max(0.0, min(1.0, 1.0 - (days_ahead / max(1, interval * factor * 10))))
    return round(retention, 3)

def auto_schedule_important(db_path: str = None, threshold: float = 0.8, max_count: int = 100) -> Dict:
    conn = get_db(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT id FROM memories 
        WHERE (next_review_date IS NULL OR repetition_interval = 0)
          AND importance >= ?
          AND (is_ephemeral IS NULL OR is_ephemeral = 0)
          AND (pinned IS NULL OR pinned = 0)
        ORDER BY importance DESC, t_recorded DESC
        LIMIT ?
    """, (threshold, max_count))
    ids = [r["id"] for r in c.fetchall()]
    conn.close()
    scheduled = []
    for mid in ids:
        result = schedule_review(mid, 3.0, db_path=db_path)
        if "error" not in result:
            scheduled.append(result)
    return {"candidates": len(ids), "scheduled": len(scheduled), "details": scheduled[:10]}

def init_spaced_repetition() -> Dict:
    try:
        ensure_schema()
        return {"initialized": True, "sm2": "ready"}
    except Exception as e:
        return {"initialized": False, "error": str(e)}
