#!/usr/bin/env python3
"""
Update wiki pages for people using SimplyPlural API data.
"""

import json
import urllib.request
import os

PG_HOST = "localhost"
PG_PORT = 5433
PG_DB = "memster"
PG_USER = "house"

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''


def update_page(slug, title, content, tags, category="people"):
    import psycopg2
    pg = psycopg2.connect(host=PG_HOST, port=PG_PORT, database=PG_DB, user=PG_USER)
    pg.autocommit = False
    cur = pg.cursor()
    cur.execute("""
        INSERT INTO wiki_pages (slug, title, content, tags, category, word_count, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (slug) DO UPDATE SET
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            tags = EXCLUDED.tags,
            category = EXCLUDED.category,
            word_count = EXCLUDED.word_count,
            updated_at = CURRENT_TIMESTAMP
    """, (slug, title, content, tags, category, len(content.split())))
    pg.commit()
    cur.close()
    pg.close()


def make_page(m):
    c = m.get('content', {})
    name = c.get('name', '?')
    desc = c.get('desc', '')
    color = c.get('color', '')
    pronouns = c.get('pronouns', '')
    avatar_url = c.get('avatarUrl', '')
    info = c.get('info', {})

    # extract structured fields from info
    birthday = info.get('6Z8nyzbXGUgs4EXW6bakSw0', '')
    fav_color = info.get('jcyeAYLaD9BPCdmzB3UAvB1', '')
    system_role = info.get('g7DMkqWpgKwdy7RFzTACnR3', '')
    likes = info.get('jBYR7LoFGpodFGx7ixvoKc4', '')
    dislikes = info.get('fUVdqJL6dNGAtaT85AYJP15', '')
    age = info.get('eHJMt52bo6fiMo4MGGTeLU6', '')

    # clean up markdown bullets in desc
    clean_desc = desc.replace('• **', '## ').replace('**\n', '\n').replace('**', '')

    slug = name.lower().replace(' ', '-').replace('/', '-').replace('.', '')
    if slug == 'john-h-':
        slug = 'john-h'

    # build tags
    tags = ['headmate', 'plural-system']
    if system_role:
        tags.append(system_role.lower().replace(' ', '-'))

    content = f"""# {name}

"""
    if avatar_url:
        content += f"![avatar]({avatar_url})\n\n"

    if pronouns:
        content += f"**pronouns:** {pronouns}\n\n"

    if age:
        content += f"**age:** {age}\n"

    if fav_color:
        content += f"**color:** {fav_color}\n"

    if system_role:
        content += f"**system role:** {system_role}\n"

    if birthday:
        content += f"**birthday:** {birthday}\n"

    content += "\n"

    if likes:
        content += f"**likes:** {likes}\n"

    if dislikes:
        content += f"**dislikes:** {dislikes}\n"

    content += "\n"

    # add the full desc
    if clean_desc:
        content += f"## description\n\n{clean_desc}\n"

    return slug, name, content, ' '.join(tags)


with open('/tmp/sp_members.json') as f:
    members = json.load(f)

print(f"Updating {len(members)} people pages from SimplyPlural data...")

for m in members:
    c = m.get('content', {})
    name = c.get('name', '?')
    if name == 'System' or not name:
        continue
    slug, title, content, tags = make_page(m)
    update_page(slug, title, content, tags)
    print(f"  updated: {name} ({slug})")

print(f"\nDone! Updated {len(members)} pages.")
