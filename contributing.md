# contributing to memster

thanks for your interest in improving memster. this document covers development setup, coding standards, and pull request procedures.

## development setup

```bash
# clone and install
git clone https://github.com/houseofmates/memster.git
cd memster

# install core dependencies
pip install -e .

# install mcp extra
pip install -e ".[mcp]"

# install nim extra (for nvidia embedding backend)
pip install -e ".[nim]"

# install benchmark dependencies
pip install -r benchmarks/requirements.txt 2>/dev/null || true
```

### prerequisites
- python 3.12+
- postgresql 15+ (with pgvector extension for vector search)
- nvidia api key (optional, for the nim embedding backend)

### database setup
```bash
# start postgres with pgvector
docker run -d --name memster-db \
  -e POSTGRES_PASSWORD=house \
  -e POSTGRES_DB=memster \
  -p 5433:5432 \
  ankane/pgvector:latest

# apply schema
psql -h localhost -p 5433 -U house -d memster -f memster/schema.sql
```

## project structure
```
memster/
├── memster/                    # core package
│   ├── hybrid_retrieval.py     # multi-signal retrieval engine
│   ├── local_embeddings.py     # sentence-transformer local embeddings
│   ├── entity_extraction.py    # entity extraction from memories
│   └── __init__.py
├── benchmarks/                 # evaluation scripts (not unit tests)
│   ├── run_v6.py
│   └── run_improved_longmemeval.py
├── memster_mcp_server.py       # mcp server for tool access
├── setup.py                    # python packaging
├── tox.ini                     # test configuration
└── requirements.txt            # (see setup.py for actual deps)
```

## coding standards

- type hints throughout (mypy compatible)
- comprehensive docstrings on all public functions
- lazy-loaded model caching with global singleton pattern
- graceful fallback chains (nim → local → fallback)
- exponential backoff with jitter for rate limiting
- numpy batch operations for vector similarity
- parameterized sql queries (`cursor.execute` with `%s` placeholders) — never use f-strings for sql
- snake_case for all function and variable names
- lowercase module-level constants

## testing

memster uses pytest for test infrastructure. the `benchmarks/` directory contains evaluation scripts that benchmark retrieval performance (recall@5, mrr, etc.) — these are not regression tests.

to add a unit test:
1. create `tests/test_<module>.py`
2. use `pytest` with `AsyncMock` or `Mock` for database connections
3. mock psycopg2 connections to avoid needing a live database
4. run `python -m pytest tests/ -v`

### recommended test areas (currently missing, needs contribution)
- `hybrid_retrieval.py` — mock psycopg2, test signal fusion (weighted vs rrf), test temporal decay
- `entity_extraction.py` — test entity extraction accuracy
- `local_embeddings.py` — test model caching and embedding output dimensions

## pull request procedure

1. fork the repository
2. create a feature branch (`git checkout -b feature/your-thing`)
3. make your changes
4. run `python -m pytest tests/ -v` (if tests exist)
5. run `python -m mypy memster/ --ignore-missing-imports` for type checking
6. ensure no merge conflict markers exist
7. if you changed `hybrid_retrieval.py` or retrieval logic, run benchmarks to verify no regression: `python benchmarks/run_v6.py`
8. push and submit a pull request

## ci checks

all prs are checked against:
- python syntax check (`py_compile`)
- pytest test suite (if `tests/` directory exists)
- mypy type checking
- conflict marker detection

## backend configuration

memster supports two embedding backends:
- `local` (default) — sentence-transformers on CPU, works out of the box
- `nim` — nvidia nim / openrouter api, requires `NVIDIA_API_KEY` or `OPENROUTER_API_KEY`

switch via the `EMBEDDING_BACKEND` environment variable.

the database connection string is configured via `DATABASE_URL` or `MEMSTER_PG_URL` environment variables.
