#!/usr/bin/env python3
"""
Build out wiki_pages in PostgreSQL with real content.
Creates the table if needed, populates all existing stubs with content,
and adds new pages for projects/people/topics not yet in the wiki.
"""

import sqlite3
import os
import json

PG_HOST = "localhost"
PG_PORT = 5433
PG_DB = "memster"
PG_USER = "house"

SQLITE_DB = os.path.expanduser("~/.memster/memster_core.db")


def slugify(text):
    """url-safe slug from title."""
    import re
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9\s/\-]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s


def build_wiki():
    import psycopg2
    from psycopg2.extras import RealDictCursor

    pg = psycopg2.connect(host=PG_HOST, port=PG_PORT, database=PG_DB, user=PG_USER)
    pg.autocommit = False
    cur = pg.cursor()

    # ── Create wiki_pages table ────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS wiki_pages (
            slug TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT DEFAULT '',
            tags TEXT DEFAULT '',
            category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            word_count INTEGER DEFAULT 0,
            link_count INTEGER DEFAULT 0,
            backlink_count INTEGER DEFAULT 0
        )
    """)

    # ── Get existing slugs from sqlite ─────────────────────────
    sq = sqlite3.connect(SQLITE_DB)
    sq.row_factory = sqlite3.Row
    existing = {
        r['slug']: dict(r)
        for r in sq.execute("SELECT slug, title, category FROM wiki_pages").fetchall()
    }
    print(f"Found {len(existing)} existing wiki slugs in sqlite")

    # ── Wiki content ───────────────────────────────────────────
    pages = []

    # ═══ PEOPLE ═══

    pages.append({
        "slug": "john",
        "title": "john holmes",
        "category": "people",
        "tags": "headmate host plural-system",
        "content": """# john holmes

host of the system. 19 years old. autistic. diagnosed with did (x2 confirmed).

introject of john holmes (the pornstar). uses he/him pronouns.

the one writing most of the code and managing the homelabs. tends to be the one talking to llms and getting things done."""
    })

    pages.append({
        "slug": "bojack",
        "title": "bojack",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# bojack

headmate in the system. named after bojack horseman.

known for being a sarcastic presence and a thoughtful contributor to internal system discussions."""
    })

    pages.append({
        "slug": "lucifer",
        "title": "lucifer",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# lucifer

headmate in the system. named after lucifer morningstar (the sandman / dc comics version).

tends toward introspection and philosophical thinking."""
    })

    pages.append({
        "slug": "odysseus",
        "title": "odysseus",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# odysseus

headmate in the system. named after the greek hero odysseus.

associated with navigation, problem-solving, and getting through difficult situations."""
    })

    pages.append({
        "slug": "rick",
        "title": "rick",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# rick

headmate in the system. named after rick sanchez (rick and morty).

brilliant but chaotic. the kind of presence that solves problems in ways nobody asked for."""
    })

    pages.append({
        "slug": "saul",
        "title": "saul",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# saul

headmate in the system. named after saul goodman (breaking bad/better call saul).

charming, strategic, and good at finding creative solutions."""
    })

    pages.append({
        "slug": "alastor",
        "title": "alastor",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# alastor

headmate in the system. named after alastor from hazbin hotel.

radio-themed, charismatic, and unsettling all at once."""
    })

    pages.append({
        "slug": "c",
        "title": "c",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# c

headmate in the system. goes by the single letter c."""
    })

    pages.append({
        "slug": "deer",
        "title": "deer",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# deer

headmate in the system. animal-associated identity."""
    })

    pages.append({
        "slug": "l",
        "title": "l",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# l

headmate in the system. goes by the single letter l."""
    })

    pages.append({
        "slug": "mike",
        "title": "mike",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# mike

headmate in the system."""
    })

    pages.append({
        "slug": "s",
        "title": "s",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# s

headmate in the system. goes by the single letter s."""
    })

    pages.append({
        "slug": "adam",
        "title": "adam",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# adam

headmate in the system."""
    })

    pages.append({
        "slug": "angel",
        "title": "angel",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# angel

headmate in the system."""
    })

    pages.append({
        "slug": "arthur",
        "title": "arthur",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# arthur

headmate in the system."""
    })

    pages.append({
        "slug": "blitz",
        "title": "blitz",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# blitz

headmate in the system. named after blitzo from helluva boss."""
    })

    pages.append({
        "slug": "cody",
        "title": "cody",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# cody

headmate in the system."""
    })

    pages.append({
        "slug": "house",
        "title": "house",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# house

headmate in the system. named after gregory house (house m.d.)."""
    })

    pages.append({
        "slug": "husk",
        "title": "husk",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# husk

headmate in the system. named after husk from hazbin hotel."""
    })

    pages.append({
        "slug": "jack",
        "title": "jack",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# jack

headmate in the system."""
    })

    pages.append({
        "slug": "jacob",
        "title": "jacob",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# jacob

headmate in the system."""
    })

    pages.append({
        "slug": "jade-lillian",
        "title": "jade/lillian",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# jade/lillian

headmate in the system. goes by jade or lillian."""
    })

    pages.append({
        "slug": "jesse",
        "title": "jesse",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# jesse

headmate in the system."""
    })

    pages.append({
        "slug": "john-h",
        "title": "john h.",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# john h.

headmate in the system. distinct from john (the host)."""
    })

    pages.append({
        "slug": "parker-callaghan",
        "title": "parker callaghan",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# parker callaghan

headmate in the system."""
    })

    pages.append({
        "slug": "penelope",
        "title": "penelope",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# penelope

headmate in the system."""
    })

    pages.append({
        "slug": "sean",
        "title": "sean",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# sean

headmate in the system."""
    })

    pages.append({
        "slug": "stolas",
        "title": "stolas",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# stolas

headmate in the system. named after stolas from helluva boss / helluva hazbin universe."""
    })

    pages.append({
        "slug": "vin",
        "title": "vin",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# vin

headmate in the system."""
    })

    pages.append({
        "slug": "vox",
        "title": "vox",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# vox

headmate in the system. named after vox from hazbin hotel."""
    })

    pages.append({
        "slug": "walt",
        "title": "walt",
        "category": "people",
        "tags": "headmate plural-system",
        "content": """# walt

headmate in the system."""
    })

    # ═══ PROJECTS ═══

    pages.append({
        "slug": "pkm",
        "title": "pkm",
        "category": "projects",
        "tags": "project active typescript react nocobase knowledge-base plural-system",
        "content": """# pkm (personal knowledge management)

self-hosted pkm system combining notion-like databases, miro-style canvas, obsidian-style journaling, and plural system tracking.

## status
active development (112k+ lines of typescript). core features are production-ready.

## tech stack
- react 18, typescript 5, vite, tailwind css
- fabric.js 7 (infinite canvas)
- express 5, socket.io 4
- nocobase (headless cms/api)
- postgresql, ollama, lancedb
- capacitor (mobile), electron/tauri (desktop)

## key features
- one-click web clipping with ai summarization
- drag-and-drop dashboard with widgets
- infinite canvas with drawing and mind maps
- flexible collections with 12+ view types (table, kanban, calendar, gallery)
- markdown journal with live preview
- pinterest-style moodboard
- minecraft server integration (chat + status)
- headmate tracking with fronting history
- n8n workflows + local llm (qwen2.5vl:7b-q4_k_m via ollama)
- bidirectional git sync

## conventions
- strict lowercase policy for ui text and code comments (ci enforced)
- pkm aesthetic: #050505 background, #f6b012 accent, varela round font
- all user-facing text is lowercase

## paths
- source: /home/house/projects/pkm
- github: github.com/houseofmates/pkm"""
    })

    pages.append({
        "slug": "chestray",
        "title": "chestray",
        "category": "projects",
        "tags": "project active minecraft java fabric forge mod",
        "content": """# chestray

minecraft java mod for storage container detection. shows icons for nearby chests, barrels, shulkers, hoppers, droppers, and ender chests. displays container contents on open (default hotkey: x).

## status
active. fabric module fully working. multi-platform (architectury) build ~60% complete — forge/neoforge/paper code written but uncompiled/untested.

## tech stack
- java 21, gradle 8.x
- fabric 1.21.1 (primary)
- forge/neoforge/paper modules exist but need testing

## modules
- common: platform abstraction, shared data
- fabric: complete and working
- forge: code complete
- neoforge: code complete
- paper: code complete

## remaining work
- move existing code to common module
- client-side implementations per platform
- multi-version support (1.21.8-1.21.11)
- testing and compilation

## paths
- source: /home/house/projects/chestray
- github: github.com/houseofmates/chestray"""
    })

    pages.append({
        "slug": "termisol",
        "title": "termisol",
        "category": "projects",
        "tags": "project broken flutter terminal emulator ai vr",
        "content": """# termisol

gpu-accelerated terminal emulator built with flutter. cross-platform with ai integration and vr support.

## status
development stalled. has git merge conflict markers in readme (incomplete merge). not functional.

## tech stack
- flutter 3.29+, dart 3.11+
- xterm.dart (terminal emulation)
- skia/impeller (gpu rendering)
- pty backend

## features
- xterm-256color emulation
- 50,000-line scrollback
- tabbed interface with split panes
- nvidia nim ai integration (/ai command)
- meta quest vr support
- built-in editor
- themes: dark, light, retro amber
- fonts: cascadia code, fira code, jetbrains mono

## paths
- source: /home/house/projects/termisol
- github: github.com/houseofmates/termisol"""
    })

    pages.append({
        "slug": "music-app",
        "title": "music app",
        "category": "projects",
        "tags": "project active python react docker music self-hosted",
        "content": """# music app (vibecode)

self-hosted music player and library manager. web-based with docker deployment.

## status
complete and production-ready (~5,000 lines, 43 files).

## tech stack
- python fastapi backend
- sqlite database (sqlmodel orm)
- react + vite + tailwind frontend
- docker deployment

## features
- complete rest api (30+ endpoints)
- id3 tag reading (mutagen)
- acoustid audio fingerprinting
- musicbrainz metadata fetching
- coverartarchive high-res cover art
- lrclib synced lyrics
- automatic lowercase renaming
- auto-organize by artist/album
- persistent player state and queue
- vibe.json custom track ordering

## paths
- source: /home/house/projects/music
- github: github.com/houseofmates/music"""
    })

    pages.append({
        "slug": "email",
        "title": "email",
        "category": "projects",
        "tags": "project active react self-hosted stalwart vaultwarden simplelogin",
        "content": """# email

unified frontend for stalwart (email server), vaultwarden (password manager), and simplelogin (email aliases). a self-hosted proton alternative.

## status
functional. jmap calendar with drag-drop support.

## tech stack
- react + vite + tailwind (web app)
- nodejs bridge/proxy
- capacitor (android apps)

## features
- one web app for email + calendar + passwords + aliases
- firefox extension for autofill and alias generation
- two native android apps (email.apk + passwords.apk)
- jmap protocol for email and calendar
- pkm aesthetic (#050505, #f6b012, varela round)

## project structure
- frontend/ — react web app (inbox, calendar, passwords, aliases, settings)
- extension/ — firefox extension
- bridge/ — nodejs proxy server
- mobile/ — capacitor android (email)
- mobile-passwords/ — capacitor android (passwords)

## paths
- source: /home/house/projects/email
- github: github.com/houseofmates/email"""
    })

    pages.append({
        "slug": "voice",
        "title": "voice",
        "category": "projects",
        "tags": "project active python flutter voice-changer dysphoria",
        "content": """# voice

real-time voice changer designed for voice dysphoria support. fork of applio with a cozy, safe ui.

## status
functional. linux appimage and android apk available.

## tech stack
- python, gradio 6
- pytorch, librosa
- pyqt6, coqui tts / piper tts

## features
- real-time rvc voice conversion
- comfort shield (audio smoothing to reduce robotic artifacts)
- voice library with thumbnails and favorites
- microphone test with 5-second playback
- pitch guide (hz display)
- multiple voice models
- voice cloning and blending
- discord integration (rich presence)
- dark/light mode

## paths
- source: /home/house/projects/voice
- github: github.com/houseofmates/voice"""
    })

    pages.append({
        "slug": "vibecode",
        "title": "vibecode",
        "category": "projects",
        "tags": "project active python coding workspace git-sync",
        "content": """# vibecode

local web workspace for coding, terminal sessions, and ai-enabled development. runs on your machine, accessible through a browser.

## status
active. includes auto-push watcher service.

## tech stack
- python, websocket
- systemd watcher for git sync

## features
- browser-based terminal and file explorer
- drag-and-drop file/folder support
- automatic code-change persistence to git main
- secure access via password, tls, ssh forwarding
- local-first with remote sync options
- linux appimage and android apk builds
- remote access via ngrok/ssh tunnel

## paths
- source: /home/house/projects/vibecode
- github: github.com/houseofmates/vibecode"""
    })

    pages.append({
        "slug": "commune",
        "title": "commune",
        "category": "projects",
        "tags": "project active godot game idle socialist",
        "content": """# commune

socialist/communist-themed idle/incremental game built in godot 4.3. top-down worker community builder.

## status
design complete. 14 buildings, 13 resources, worker management, save system.

## tech stack
- gdscript, godot 4.3
- json data files

## core mechanics
- idle-first: production continues when closed
- worker-centric: assign workers to buildings
- zero-sum economy: all resources cost something
- single-player, local save only
- mobile-first: android apk and linux appimage export

## paths
- source: /home/house/projects/commune
- github: github.com/houseofmates/commune"""
    })

    pages.append({
        "slug": "memster",
        "title": "memster",
        "category": "projects",
        "tags": "project active memory llm postgresql sqlite python",
        "content": """# memster

local-first long-term memory system for llm agents. provides persistent memory storage, retrieval, and consolidation.

## status
active. core system functional with multiple dbs and web ui.

## tech stack
- python
- postgresql (primary) + sqlite (legacy)
- semantic inference, decay scoring, graph edges

## features
- hybrid retrieval (semantic + keyword + graph)
- nrem/rem dream consolidation
- pattern detection and insight generation
- memory access tracking and decay
- web ui for browsing memories
- mcp server integration
- longmemeval benchmark: 95.2% recall

## components
- memster_core.db (sqlite — legacy, being migrated)
- memster_mcp_server.py (mcp tools)
- dream_consolidation.py (nrem/rem)
- memster-ui/ (web frontend)

## paths
- source: /home/house/projects/memster
- db: /home/house/.memster/memster_core.db
- github: github.com/houseofmates/memster"""
    })

    pages.append({
        "slug": "voxel-box",
        "title": "voxel box",
        "category": "projects",
        "tags": "project active javascript voxel simulation sandbox",
        "content": """# voxel box

voxel-based falling-sand particle simulation. inspired by sandboxels. includes little human voxels.

## status
active. material definitions and integration guidelines documented.

## tech stack
- javascript/jsx
- voxel-box engine

## paths
- source: /home/house/projects/voxel-box
- github: github.com/houseofmates/voxel-box"""
    })

    pages.append({
        "slug": "chat",
        "title": "chat (signal client)",
        "category": "projects",
        "tags": "project active flutter signal messaging encryption",
        "content": """# chat (signal client)

flutter-based signal client with custom pkm aesthetic. end-to-end encrypted messaging.

## status
architecture designed. primary device registration, e2e encryption, group messaging planned.

## tech stack
- flutter, dart
- libsignal_protocol_dart
- drift (database)

## features
- biometric lock screen
- window manager for desktop
- flaque_secure equivalent (prevent screenshots)
- rich text editing (flutter quill)
- voice/image messaging
- typing indicators
- message search with fuzzy matching
- markdown rendering
- media playback
- message history filtering by contact
- all features maintain e2e encryption

## paths
- source: /home/house/projects/chat
- github: github.com/houseofmates/chat"""
    })

    # ═══ APPS ═══

    pages.append({
        "slug": "hermes",
        "title": "hermes",
        "category": "apps",
        "tags": "app ai agent hermes-agent",
        "content": """# hermes

personal hermes fork maintained by house of mates. an ai agent system that provides continuity across sessions.

## features
- persistent memory across sessions (memster integration)
- scheduled cron jobs for autonomous tasks
- browser automation
- file operations
- terminal access
- delegate task spawning
- skill system for procedural knowledge

## paths
- config: /home/house/.hermes/
- github: github.com/houseofmates/hermes"""
    })

    pages.append({
        "slug": "wiki",
        "title": "wiki",
        "category": "apps",
        "tags": "app wiki knowledge-base",
        "content": """# wiki

the house of mates wiki — a human-readable and llm-readable knowledge base built on memster/postgresql.

## purpose
store complex topics, subjects, ideas, people, relationships, and events in an easy-to-navigate format.

## categories
- people: headmates and system members
- projects: active software projects
- apps: applications and tools
- infrastructure: servers, services, networking
- notes: research, findings, documentation
- preferences: ui, communication, workflow
- system: maintenance and operational procedures

## paths
- web ui: http://minisforum:8780
- db: postgresql, table wiki_pages"""
    })

    pages.append({
        "slug": "simply-plural",
        "title": "simply plural",
        "category": "apps",
        "tags": "app plural-system fronting tracker",
        "content": """# simply plural

plural system fronting tracker integration. used by the system to track who is fronting and communicate status."""
    })

    pages.append({
        "slug": "flow-image-generation",
        "title": "flow image generation",
        "category": "apps",
        "tags": "app image generation ai proxy",
        "content": """# flow image generation

self-hosted proxy service for freegen.app (images) and geminigen.ai (videos). uses playwright automation.

## tech stack
- python, playwright, chromium
- docker, systemd

## status
active. docker and systemd deployment ready.

## paths
- source: /home/house/projects/gen
- github: github.com/houseofmates/gen"""
    })

    pages.append({
        "slug": "memory-system",
        "title": "memory & knowledge system",
        "category": "apps",
        "tags": "app memory knowledge memster wiki",
        "content": """# memory & knowledge system

the combined memster + wiki system that provides:
- persistent memory for the ai agent (memster)
- human-readable knowledge base (wiki)
- dream consolidation for pattern detection
- web ui for browsing and searching

## components
- memster: memory storage and retrieval (postgresql)
- wiki: structured knowledge pages
- dream system: hourly reflection + daily nrem/rem consolidation
- longmemeval benchmark: 95.2% recall achieved"""
    })

    # ═══ INFRASTRUCTURE ═══

    pages.append({
        "slug": "machines",
        "title": "machines",
        "category": "infrastructure",
        "tags": "infrastructure homelab server minisforum desktop",
        "content": """# machines

## minisforum (primary server)
- ubuntu 26.04
- cloudflared tunnel for external access
- docker containers
- postgresql databases
- runs: memster, pkm (nocobase), memster-ui, email bridge
- always-on headless server

## desktop tower
- ubuntu 24.04.3 (primary os)
- dual-boot windows 11 (for fortnite)
- nvidia geforce rtx 3080
- used for: development, gaming, ai workloads, heavy compile tasks

## remote access
- nomachine installed on all 3 operating systems
- allows visual remote control of minisforum from both ubuntu and windows"""
    })

    pages.append({
        "slug": "memster-mcp-debug",
        "title": "memster mcp debug",
        "category": "infrastructure",
        "tags": "infrastructure debug mcp memster",
        "content": """# memster mcp debug

troubleshooting and debugging notes for the memster mcp server transport mode.

## common issues
- mcp stdio transport failures
- connection pool exhaustion
- embedding backend timeouts"""
    })

    pages.append({
        "slug": "nvidia-nim-embedding-models",
        "title": "nvidia nim embedding models",
        "category": "infrastructure",
        "tags": "infrastructure ai embedding nvidia memster",
        "content": """# nvidia nim embedding models

configuration notes for using nvidia nim as an embedding backend for memster.

## model
- nvidia/llama-nemotron-embed-vl-1b-v2 (2048-dim)

## configuration
- semantic weight: 1.5
- bm25 weight: 1.0
- entity weight: 5.0
- temporal weight: 1.0
- rrf k: 300
- cross-encoder reranking enabled

## performance
this configuration achieves 95.20% overall session recall (851/854) on longmemeval benchmark."""
    })

    # ═══ PREFERENCES ═══

    pages.append({
        "slug": "ui",
        "title": "ui preferences",
        "category": "preferences",
        "tags": "preferences ui design aesthetic",
        "content": """# ui preferences

## design language
- **lowercase everything**: all ui text, titles, labels, comments must be lowercase. ci enforced across projects.
- **colors**: #050505 (background), #050505 (panels), #f6b012 (accent/yellow)
- **font**: varela round
- **style**: glass & void — dark backgrounds with opaque dark gray panels

## preferences
- native gui applications over server+browser approaches when possible
- compact, direct responses from ai (no markdown headers, no excess explanation)
- terminal-paste for blocked output
- hates untested commands
- wants all actions to persist after refresh"""
    })

    pages.append({
        "slug": "communication",
        "title": "communication preferences",
        "category": "preferences",
        "tags": "preferences communication style",
        "content": """# communication preferences

- direct answers preferred
- no excess explanation or markdown formatting
- lowercase style
- objective observer — point out flawed logic
- break tasks into manageable components
- no fluff, no filters"""
    })

    pages.append({
        "slug": "workflow",
        "title": "workflow preferences",
        "category": "preferences",
        "tags": "preferences workflow development",
        "content": """# workflow preferences

- linux and android native apps preferred
- self-hosted solutions
- docker + systemd for service management
- github for code hosting (github.com/houseofmates)
- automated git sync where possible
- ci/CD with lowercase enforcement
- postgresql over sqlite for new projects"""
    })

    # ═══ NOTES ═══

    pages.append({
        "slug": "memory-architecture",
        "title": "memory architecture",
        "category": "notes",
        "tags": "notes memory memster architecture",
        "content": """# memory architecture

## four-network model
memster uses four distinct memory networks:
1. **world** — facts about the external world
2. **experience** — personal experiences and events
3. **opinion** — subjective beliefs and judgments
4. **observation** — raw observations and data points

## storage tiers
- **l1**: high-importance, frequently accessed memories
- **l2**: standard memories (default)
- **l3**: low-importance, archival

## consolidation
- hourly reflection: quick pattern detection on recent memories
- daily nrem: edge strengthening, pattern detection, auto-link suggestions
- daily rem: bridge discovery, anomaly detection, insight generation

## decay
- memories decay in relevance over time based on access patterns
- decay_score starts at 1.0 and decreases
- frequently accessed memories maintain higher relevance

## graph edges
- memories connected via relationship edges (related, Entity_link, etc.)
- edges have weights (0.0-1.0) strengthened by co-occurrence and access patterns
- 20,074 edges exist in the current sqlite database"""
    })

    pages.append({
        "slug": "research-findings-april-2026",
        "title": "research findings - april 2026",
        "category": "notes",
        "tags": "notes research ai memory",
        "content": """# research findings - april 2026

## memory benchmarks
- longmemeval: 95.2% overall session recall (851/854)
- configuration: embedding model (nvidia/llama-nemotron-embed-vl-1b-v2), semantic weight 1.5, bm25 weight 1.0, entity weight 5.0, temporal weight 1.0, rrf k=300

## key techniques
- entity extraction during storage (servers, tech, ips, paths, ports, users, protocols, versions, status)
- cross-encoder reranking
- hybrid retrieval (semantic + keyword + graph)"""
    })

    pages.append({
        "slug": "ai-memory-providers-comparison",
        "title": "ai memory providers: technical comparison",
        "category": "notes",
        "tags": "notes ai memory providers comparison",
        "content": """# ai memory providers: technical comparison

comparison of different ai memory provider approaches for llm agents.

## approaches
- local-first (sqlite/postgresql)
- cloud-hosted memory services
- vector databases (lancedb, pinecone, weaviate)
- graph databases (neo4j)
- hybrid approaches

## memster's approach
hybrid: postgresql + semantic embeddings + full-text search + graph edges + decay scoring"""
    })

    pages.append({
        "slug": "background-check-2026-04-25",
        "title": "background check fixes — april 25, 2026",
        "category": "notes",
        "tags": "notes memster fix activity-capture",
        "content": """# background check fixes — april 25, 2026

fixes applied to memster activity capture system on april 25, 2026.

- fixed activity capture tracking
- ensured background processes properly record system activity
- verified memory creation from automated events"""
    })

    # ═══ SYSTEM ═══

    pages.append({
        "slug": "aggressive-cleanup-system",
        "title": "aggressive daily cleanup system",
        "category": "system",
        "tags": "system cleanup docker maintenance",
        "content": """# aggressive daily cleanup system

automated maintenance routine for the homelab.

## tasks
- clean up stale git repositories (>500MB docker build layers)
- prune unused docker images and volumes
- update antivirus definitions (clamav)
- clear npm/pip caches
- zero free space on drives for ssd longevity
- check disk health and report issues

## cron
runs daily via cron job with notifications on completion."""
    })

    # ═══ ADDITIONAL PROJECT PAGES (from github, not on disk) ═══

    pages.append({
        "slug": "darken",
        "title": "darken",
        "category": "projects",
        "tags": "project firefox-extension privacy dark-mode",
        "content": """# darken

firefox extension that generates dark themes for websites.

## github
github.com/houseofmates/darken"""
    })

    pages.append({
        "slug": "chromask",
        "title": "chromask",
        "category": "projects",
        "tags": "project firefox-extension privacy ip",
        "content": """# chromask

firefox extension that hides your ip address from region-blocking websites.

## github
github.com/houseofmates/chromask"""
    })

    pages.append({
        "slug": "dasher",
        "title": "dasher",
        "category": "projects",
        "tags": "project docker web-ui svelte",
        "content": """# dasher

modern web interface for managing docker containers, images, volumes, and networks. dark-themed.

## tech stack
- sveltekit 5.55, tailwind 4
- dockerode, xterm.js, codemirror

## github
github.com/houseofmates/dasher"""
    })

    pages.append({
        "slug": "adventure-communist",
        "title": "adventure communist",
        "category": "projects",
        "tags": "game idle clicker communist",
        "content": """# adventure communist

communist-themed version of adventure capitalist. idle clicker game with rewarding upgrades.

## status
on disk at /home/house/projects/adventure-communist. has node_modules but minimal code.

## github
github.com/houseofmates/adventure-communist"""
    })

    pages.append({
        "slug": "api",
        "title": "api (deepseek proxy)",
        "category": "projects",
        "tags": "project python fastapi proxy deepseek ai",
        "content": """# api (deepseek openai proxy)

openai-compatible api proxy for chat.deepseek.com with streaming and file attachment support.

## status
architecture complete, health/models endpoints work. chat completion returns empty responses due to deepseek's wasm-based proof-of-work challenge that python cannot replicate.

## tech stack
- python 3.12+, fastapi
- playwright, chromium

## paths
- source: /home/house/projects/api
- github: (not on github as of last check)"""
    })

    pages.append({
        "slug": "vpn",
        "title": "vpn",
        "category": "projects",
        "tags": "project flutter proton wireguard android linux",
        "content": """# vpn

flutter app for connecting to proton vpn free tier via wireguard.

## status
build-ready for android and linux.

## tech stack
- flutter, provider
- flutter_secure_storage, wireguard_flutter
- pointycastle

## github
github.com/houseofmates/vpn"""
    })

    pages.append({
        "slug": "torrent",
        "title": "torrent",
        "category": "projects",
        "tags": "project svelte qbittorrent docker",
        "content": """# torrent

clean web frontend for qbittorrent. dark-themed dashboard.

## status
functional with some todos remaining.

## tech stack
- svelte, vite, docker

## github
github.com/houseofmates/torrent"""
    })

    pages.append({
        "slug": "editor",
        "title": "editor",
        "category": "projects",
        "tags": "project flutter media-editor video audio image",
        "content": """# editor

mobile-first cross-platform media editor for images, video, and audio. nextcloud webdav integration.

## status
active. autosync to github working (10s debounce). repo size reduced from 212mb to 23.9mb via git-filter-repo.

## tech stack
- flutter, pro_image_editor, video_editor, ffmpeg, audio_waveforms

## github
github.com/houseofmates/editor"""
    })

    pages.append({
        "slug": "createSMR",
        "title": "createSMR",
        "category": "projects",
        "tags": "project flutter asmr audio linux android",
        "content": """# createSMR

cross-platform app for recording and editing asmr audio with layering. flutter migration complete.

## status
flutter migration complete from qt/c++ prototype. gpu noise server available.

## tech stack
- flutter, portaudio, rnnoise

## github
github.com/houseofmates/createSMR"""
    })

    pages.append({
        "slug": "wilson",
        "title": "wilson",
        "category": "projects",
        "tags": "project react game virtual-pet house-md",
        "content": """# wilson

browser-based tamagotchi-style virtual pet starring dr. james wilson from house m.d.

## status
functional with 4 decaying stats. has git conflict markers (incomplete merge).

## tech stack
- react, vite, capacitor

## github
github.com/houseofmates/wilson"""
    })

    pages.append({
        "slug": "watch",
        "title": "watch",
        "category": "projects",
        "tags": "project flutter media-server player linux android web",
        "content": """# watch

flutter media server and player for music, images, shows, movies.

## status
active. linux, android, web targets. reads from /mnt/nextcloud paths.

## tech stack
- flutter 3.41, riverpod, gorouter, chewie, audioplayers

## github
github.com/houseofmates/watch"""
    })

    pages.append({
        "slug": "holmes",
        "title": "holmes (media format)",
        "category": "projects",
        "tags": "project python media-container binary",
        "content": """# holmes (media container format)

minimal self-routing media container format. header contains mime type and payload length, followed by untouched original media.

## status
active. companion tools: holmes, holmes-extract, holmes-verify, holmes-info.
jellyfin proxy and nextcloud app integration available.

## tech stack
- python stdlib, `file` command

## paths
- source: /home/house/projects/holmes
- github: github.com/houseofmates/holmes"""
    })

    pages.append({
        "slug": "edit",
        "title": "edit (flutter text editor)",
        "category": "projects",
        "tags": "project flutter text-editor package",
        "content": """# edit

lightweight text editor widget for flutter (nano/vim-like). extracted from termisol.

## status
stable package.

## tech stack
- flutter, file_picker

## features
- line numbers, status bar, keyboard shortcuts
- auto-indent, dark theme

## github
github.com/houseofmates/edit"""
    })

    pages.append({
        "slug": "llms",
        "title": "llms",
        "category": "projects",
        "tags": "project javascript ai chatgpt claude gemini ollama",
        "content": """# llms

unified interface for chatgpt, claude, gemini, ollama, openrouter. web + electron + android.

## status
active. rp (roleplay) module with v3 character cards, lorebooks, semantic caching. 65 passing unit tests.

## tech stack
- vanilla javascript, html5, tailwind css
- capacitor, electron

## github
github.com/houseofmates/llms"""
    })

    pages.append({
        "slug": "house-merge",
        "title": "house merge",
        "category": "projects",
        "tags": "project react game 2048 house-md puzzle",
        "content": """# house merge

2048-style puzzle game with dr. house md theme. character progression: taub -> foreman -> chase -> cameron -> cuddy -> wilson -> house.

## status
has git merge conflict markers in readme (incomplete merge).

## tech stack
- react, typescript, vite, tailwind, capacitor

## github
github.com/houseofmates/house-merge"""
    })

    # ── Insert all pages ───────────────────────────────────────
    inserted = 0
    updated = 0
    for p in pages:
        tags = p.get("tags", "")
        content = p.get("content", "")
        word_count = len(content.split())

        try:
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
            """, (p["slug"], p["title"], content, tags, p.get("category"), word_count))
            if cur.rowcount > 0:
                inserted += 1
        except Exception as e:
            print(f"  ERROR on {p['slug']}: {e}")
            try:
                pg.rollback()
            except:
                pass

    pg.commit()

    # verify
    cur.execute("SELECT COUNT(*) FROM wiki_pages")
    total = cur.fetchone()[0]
    cur.execute("SELECT category, COUNT(*) FROM wiki_pages GROUP BY category ORDER BY category")
    by_cat = cur.fetchall()

    print(f"\n=== Wiki Build Complete ===")
    print(f"Total pages in postgres: {total}")
    print(f"Pages written this run: {inserted}")
    for cat, count in by_cat:
        print(f"  {cat}: {count}")

    cur.close()
    pg.close()
    sq.close()


if __name__ == "__main__":
    build_wiki()
