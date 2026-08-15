#!/usr/bin/env python3
"""Backfill entity data for existing LongMemEval memories."""
import sys, json, os, psycopg2, psycopg2.extras
sys.path.insert(0, '/home/house/projects/memster')

from memster.entity_extraction import extract_entities_list

conn = psycopg2.connect('postgresql://house:@/memster?host=/run/postgresql&port=5433')
conn.cursor_factory = psycopg2.extras.RealDictCursor
cur = conn.cursor()

cur.execute("""
    SELECT m.id, m.content FROM memories m
    LEFT JOIN memory_entity_data e ON m.id = e.memory_id
    WHERE m.source = 'longmemeval' AND e.memory_id IS NULL
""")
rows = cur.fetchall()
print(f'Backfilling {len(rows)} memories...')

done = 0
for row in rows:
    try:
        entity_list = extract_entities_list(row['content'])
        entity_data = {}
        for ent in entity_list:
            t = ent.get('type', 'other')
            if t not in entity_data:
                entity_data[t] = []
            entity_data[t].append(ent['name'])
        if entity_data:
            cur.execute(
                "INSERT INTO memory_entity_data (memory_id, entities) VALUES (%s, %s) ON CONFLICT (memory_id) DO UPDATE SET entities = EXCLUDED.entities",
                (row['id'], json.dumps(entity_data))
            )
        done += 1
        if done % 100 == 0:
            conn.commit()
            print(f'  {done}/{len(rows)}')
    except Exception as e:
        print(f'  error on id={row["id"]}: {e}')

conn.commit()
print(f'Done: {done}/{len(rows)}')
conn.close()