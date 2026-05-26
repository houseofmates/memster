# Changelog

## 0.6.0 — 2026-05-25 — Cleanup & Honest Benchmarks

### Removed
- **Pieces MCP**: removed all Pieces references from MCP server. No more `PIECES_MCP_URL`, `memster_sync_pieces` handler, or Pieces-related imports.
- **SQLite legacy**: deleted `memster_v4_features.py`, `memster_gbrain.py`, `memster_beads.py`, `memster_phase2.py`, `memster_spaced_repetition.py`, and `dream_consolidation.py`. All storage goes through PostgreSQL via `memster.hybrid_retrieval`.
- **Unverified claims**: README no longer claims ≥96.2% as verified. The ablation table reports the committed 95.20% V6 result and marks higher scores as "pending".

### Added
- `memster/entity_extraction.py` — rules-based entity extraction module (zero-LLM, regex + spaCy). Provides the `extract_entities` function imported by `hybrid_retrieval.py`.
- `setup.py` — proper packaging so `pip install -e .` works.
- `CHANGELOG.md` — this file.

### Fixed
- Corrupted line in `hybrid_retrieval.py` where `NIM_API_KEY` env var reading was truncated.
- MCP server reduced from 180KB monolith to ~400-line clean wrapper around `hybrid_retrieval`.

### Changed
- `memster_mcp_server.py` rewritten: no `from memster_v4_features import *`, no Pieces, no SQLite. Clean MCP wrapper with 7 tools backed by the hybrid retrieval engine.
- `BENCHMARKS.md` now honestly distinguishes "verified" (95.20%) from "pending" (≥96.2% target).