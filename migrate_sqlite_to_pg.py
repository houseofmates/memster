#!/usr/bin/env python3
"""
Migrate memster data from SQLite (memster_core.db) to PostgreSQL.
Copies memories, memory_tags, and memory_edges.
"""

import sqlite3
import os

# postgres connection params
PG_HOST = "localhost"
PG_PORT = 5433
PG_DB = "memster"
PG_USER = "house"

SQLITE_DB = os.path.expanduser("~/.memster/memster_core.db")


def migrate():
    import psycopg2
    from psycopg2.extras import RealDictCursor

    # connect to sqlite
    sq = sqlite3.connect(SQLITE_DB)
    sq.row_factory = sqlite3.Row

    # connect to postgres
    pg = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, database=PG_DB, user=PG_USER
    )
    pg.autocommit = False
    cur = pg.cursor()

    print("=== Memster SQLite → PostgreSQL Migration ===")

    # columns that exist in both sqlite and postgres
    pg_cols = ['id', 'content', 'network_type', 't_event', 't_recorded', 'source',
               'conversation_id', 'embedding', 'category', 'tier', 'memory_type',
               'importance', 'decay_score', 'access_count', 'fronter_uid', 'fronter_name',
               'valid_from', 'valid_to', 'graph_x', 'graph_y', 'event_time',
               'local_embedding']

    # ── Verify columns ───────────────────────────────────────
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='memories'")
    pg_existing = {r[0] for r in cur.fetchall()}
    sq_cols = [d[0] for d in sq.execute("SELECT * FROM memories LIMIT 0").description]
    
    common_cols = [c for c in pg_cols if c in sq_cols and c in pg_existing]
    print(f"SQLite cols: {len(sq_cols)}, PG cols: {len(pg_existing)}")
    print(f"Migrating columns: {common_cols}")

    # ── 1. Memories ──────────────────────────────────────────
    rows = sq.execute(f"SELECT {','.join(common_cols)} FROM memories WHERE is_archived = 0").fetchall()
    print(f"\nFound {len(rows)} non-archived memories in SQLite")

    inserted = 0
    skipped = 0
    placeholders = ','.join(['%s'] * len(common_cols))
    col_list = ','.join(common_cols)

    for row in rows:
        vals = list(row)
        try:
            cur.execute(
                f"INSERT INTO memories ({col_list}) VALUES ({placeholders}) "
                f"ON CONFLICT (id) DO NOTHING",
                vals
            )
            if cur.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  WARN: memory {row[0]}: {e}")
            skipped += 1
            try:
                pg.rollback()
            except:
                pass
            # reconnect if needed
            if pg.closed:
                pg = psycopg2.connect(
                    host=PG_HOST, port=PG_PORT, database=PG_DB, user=PG_USER
                )
                pg.autocommit = False
                cur = pg.cursor()

    pg.commit()

    # fix sequence
    cur.execute("SELECT setval('memories_id_seq', (SELECT MAX(id) FROM memories))")
    pg.commit()
    print(f"Inserted {inserted} memories, skipped {skipped}")

    # ── 2. Memory Tags ───────────────────────────────────────
    tag_inserted = 0
    try:
        tag_rows = sq.execute("SELECT memory_id, tag FROM memory_tags").fetchall()
        print(f"\nFound {len(tag_rows)} memory tags in SQLite")
        for tr in tag_rows:
            try:
                cur.execute(
                    "INSERT INTO memory_tags (memory_id, tag) VALUES (%s, %s) "
                    "ON CONFLICT (memory_id, tag) DO NOTHING",
                    (tr['memory_id'], tr['tag'])
                )
                if cur.rowcount > 0:
                    tag_inserted += 1
            except Exception as e:
                pg.rollback()
        pg.commit()
        print(f"Inserted {tag_inserted} tags")
    except Exception as e:
        print(f"Tags table error: {e}")

    # ── 3. Memory Edges ──────────────────────────────────────
    edge_inserted = 0
    try:
        edge_rows = sq.execute(
            "SELECT source_memory_id, target_memory_id, relation_type, weight, created_at "
            "FROM memory_edges"
        ).fetchall()
        print(f"\nFound {len(edge_rows)} memory edges in SQLite")
        for er in edge_rows:
            try:
                cur.execute(
                    "INSERT INTO memory_edges (source_memory_id, target_memory_id, relation_type, weight, created_at) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT DO NOTHING",
                    (er['source_memory_id'], er['target_memory_id'],
                     er['relation_type'], er['weight'], er['created_at'])
                )
                if cur.rowcount > 0:
                    edge_inserted += 1
            except Exception:
                try:
                    pg.rollback()
                except:
                    pass
        pg.commit()
        print(f"Inserted {edge_inserted} edges")
    except Exception as e:
        print(f"Edges table error: {e}")

    # ── 4. search_vector is auto-generated ────────────────────
    print("\nsearch_vector is auto-generated in PostgreSQL — skipping rebuild.")

    # ── Verify ───────────────────────────────────────────────
    cur.execute("SELECT COUNT(*) FROM memories")
    pg_count = cur.fetchone()[0]
    print(f"\nPostgreSQL now has {pg_count} memories")

    cur.close()
    pg.close()
    sq.close()

    print("\n=== Migration complete! ===")


if __name__ == "__main__":
    migrate()
