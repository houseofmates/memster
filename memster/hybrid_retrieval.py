"""
Memster Hybrid Retrieval Engine v2.0

Multi-signal retrieval pipeline fusing:
  - Semantic vector similarity (NVIDIA NIM or local embeddings)
  - BM25 keyword scoring (PostgreSQL tsvector/tsquery)
  - Entity-based boosting
  - Temporal-proximity boosting
  - Configurable fusion (weighted linear combination or RRF)
  - Lightweight CrossEncoder reranker (fast, CPU-based)
  - Two-stage reranking (hybrid fusion -> top-N -> reranker -> top-K)
  - Optional query expansion (WordNet synonyms)

Backends:
  'local' — SentenceTransformer on CPU (384-dim, BAAI/bge-small-en-v1.5 default)
  'nim'   — NVIDIA NIM / OpenRouter API (2048-dim, nvidia/llama-nemotron-embed-vl-1b-v2)

Default: local (works out of the box, zero API keys).
Set EMBEDDING_BACKEND=nim and NVIDIA_API_KEY for NIM.
"""

import json
import logging
import math
import os
import re
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("memster.hybrid")

# ── try CrossEncoder (lightweight reranker) ──────────────────────
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

import psycopg2
import psycopg2.extras


# ── defaults ──────────────────────────────────────────────────────
DEFAULT_TOP_K = 20
TEMPORAL_HALF_LIFE_DAYS = 30.0
RRF_K = 60

# Embedding backend config
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "local").lower()
LOCAL_EMBEDDING_MODEL = os.environ.get(
    "LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
)

# NIM API config
NIM_API_KEY = os.environ.get("NVIDIA_API_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")
NIM_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
NIM_BASE_URL = "https://openrouter.ai/api/v1/embeddings"


# ── NIM embedding helpers (direct HTTP, no external package) ──────

def _nim_embed(text: str, retries: int = 3) -> Optional[List[float]]:
    """Embed via NIM/OpenRouter API. Returns 2048-dim vector or None."""
    if not NIM_API_KEY:
        return None
    text = text[:6000]
    for attempt in range(retries):
        try:
            payload = json.dumps({
                "model": f"{NIM_MODEL}:free",
                "input": [text],
            }).encode()
            req = urllib.request.Request(
                NIM_BASE_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {NIM_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
                if data.get("data") and len(data["data"]) > 0:
                    return data["data"][0]["embedding"]
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.warning(f"NIM embed failed: {exc}")
    return None


def _nim_embed_batch(texts: List[str]) -> Optional[List[List[float]]]:
    """Batch embed via NIM."""
    if not NIM_API_KEY:
        return None
    texts = [t[:6000] for t in texts]
    try:
        payload = json.dumps({
            "model": f"{NIM_MODEL}:free",
            "input": texts,
        }).encode()
        req = urllib.request.Request(
            NIM_BASE_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {NIM_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
            if data.get("data"):
                result = [d["embedding"] for d in data["data"]]
                return result
    except Exception as exc:
        logger.warning(f"NIM batch embed failed: {exc}")
    return None


# ── Local embedding helpers ───────────────────────────────────────

_LOCAL_MODEL_INSTANCE = None

def _get_local_model():
    """Lazy-load and cache the SentenceTransformer model."""
    global _LOCAL_MODEL_INSTANCE
    if _LOCAL_MODEL_INSTANCE is not None:
        return _LOCAL_MODEL_INSTANCE

    try:
        from sentence_transformers import SentenceTransformer
        models_dir = os.path.expanduser("~/memster/models")
        os.makedirs(models_dir, exist_ok=True)
        logger.info(f"Loading local embedding model: {LOCAL_EMBEDDING_MODEL}")
        t0 = time.time()
        _LOCAL_MODEL_INSTANCE = SentenceTransformer(
            LOCAL_EMBEDDING_MODEL, cache_folder=models_dir
        )
        logger.info(f"Local model loaded in {time.time()-t0:.1f}s")
        return _LOCAL_MODEL_INSTANCE
    except ImportError:
        logger.warning("sentence_transformers not available for local embeddings")
        return None
    except Exception as e:
        logger.warning(f"Failed to load local embedding model: {e}")
        return None


def _local_embed(text: str) -> Optional[List[float]]:
    """Embed via local SentenceTransformer. Returns 384-dim vector or None."""
    model = _get_local_model()
    if model is None:
        return None
    try:
        return model.encode([text[:8192]], convert_to_numpy=True)[0].tolist()
    except Exception as e:
        logger.warning(f"Local embed failed: {e}")
        return None


def _local_embed_batch(texts: List[str]) -> Optional[List[List[float]]]:
    """Batch embed via local model."""
    model = _get_local_model()
    if model is None:
        return None
    try:
        texts = [t[:8192] for t in texts]
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]
    except Exception as e:
        logger.warning(f"Local batch embed failed: {e}")
        return None


def _get_local_dim() -> int:
    """Get embedding dimension from local model (default 384)."""
    try:
        model = _get_local_model()
        if model is not None:
            probe = model.encode(["probe"], convert_to_numpy=True)
            return probe.shape[1]
    except Exception:
        pass
    return 384


# ── backend dispatch ──────────────────────────────────────────────

def embed_text(text: str) -> Optional[List[float]]:
    """Embed text using the configured backend."""
    if EMBEDDING_BACKEND == "nim":
        result = _nim_embed(text)
        if result is not None:
            return result
        logger.debug("NIM embed failed, falling back to local")
        return _local_embed(text)
    # default: local
    result = _local_embed(text)
    if result is not None:
        return result
    logger.debug("Local embed failed, trying NIM fallback")
    return _nim_embed(text)


def embed_batch(texts: List[str]) -> Optional[List[List[float]]]:
    """Batch embed using the configured backend."""
    if EMBEDDING_BACKEND == "nim":
        result = _nim_embed_batch(texts)
        if result is not None:
            return result
        return _local_embed_batch(texts)
    result = _local_embed_batch(texts)
    if result is not None:
        return result
    return _nim_embed_batch(texts)


def is_embedding_available() -> bool:
    """Check if any embedding backend is available."""
    if EMBEDDING_BACKEND == "nim" and NIM_API_KEY:
        return True
    if EMBEDDING_BACKEND == "local":
        return _get_local_model() is not None
    return _get_local_model() is not None or bool(NIM_API_KEY)


def get_embedding_dim() -> int:
    """Return embedding dimension for the configured backend."""
    if EMBEDDING_BACKEND == "nim":
        return 2048
    return _get_local_dim()


def get_backend_info() -> Dict[str, Any]:
    """Return info about the current embedding backend."""
    return {
        "backend": EMBEDDING_BACKEND,
        "available": is_embedding_available(),
        "dimension": get_embedding_dim(),
        "nim_key_set": bool(NIM_API_KEY),
        "nim_model": NIM_MODEL if EMBEDDING_BACKEND == "nim" else None,
        "local_model": LOCAL_EMBEDDING_MODEL if EMBEDDING_BACKEND == "local" else None,
    }


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── vector similarity search helpers ──────────────────────────────

def _get_conn_internal():
    """Get a database connection for internal use."""
    db_url = os.environ.get(
        "DATABASE_URL",
        os.environ.get(
            "MEMSTER_PG_URL",
            "postgresql://house:@/memster?host=/run/postgresql&port=5433",
        ),
    )
    conn = psycopg2.connect(db_url)
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    return conn


def _vector_search_sql(query_embedding: List[float], limit: int = 100, threshold: float = 0.1) -> Dict[int, float]:
    """Brute-force cosine similarity search via local_embedding jsonb column.
    Uses numpy for batch computation.
    """
    conn = _get_conn_internal()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id, local_embedding
            FROM memories
            WHERE local_embedding IS NOT NULL
            ORDER BY t_event DESC
            LIMIT 5000
        """)
        rows = cursor.fetchall()

        if not rows:
            return {}

        import numpy as np

        query_np = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_np)
        if query_norm == 0:
            return {}

        query_np = query_np / query_norm

        ids = []
        vectors = []
        for row in rows:
            mem_id = row["id"]
            emb_json = row["local_embedding"]
            if isinstance(emb_json, str):
                emb = json.loads(emb_json)
            else:
                emb = emb_json
            if emb is None:
                continue
            ids.append(mem_id)
            vectors.append(emb)

        if not vectors:
            return {}

        vectors_np = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors_np = vectors_np / norms

        # Batch dot product
        similarities = np.dot(vectors_np, query_np)

        # Filter and sort
        indices = np.where(similarities >= threshold)[0]
        if len(indices) == 0:
            return {}

        sorted_indices = indices[np.argsort(-similarities[indices])]
        keep = min(limit, len(sorted_indices))

        return {ids[i]: round(float(similarities[i]), 4) for i in sorted_indices[:keep]}
    except Exception as e:
        logger.debug(f"Vector search failed: {e}")
        return {}
    finally:
        conn.close()


def _store_local_embedding(memory_id: int, embedding: List[float]) -> bool:
    """Store a local embedding in the local_embedding jsonb column."""
    conn = _get_conn_internal()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE memories SET local_embedding = %s WHERE id = %s",
            (json.dumps(embedding), memory_id),
        )
        conn.commit()
        return True
    except Exception as e:
        logger.debug(f"Failed to store local embedding: {e}")
        return False
    finally:
        conn.close()


# ════════════════════════════════════════════════════════════════════
# HybridRetrievalEngine class
# ════════════════════════════════════════════════════════════════════

class HybridRetrievalEngine:
    """Multi-signal retrieval engine for Memster."""

    def __init__(self, db_connection_factory):
        """Initialize with a callable that returns a psycopg2 connection."""
        self.get_conn = db_connection_factory
        self.embeddings_available = False

        # Determine embedding backend
        self.embedding_backend = os.environ.get("EMBEDDING_BACKEND", "local").lower()

        # Try local embeddings first (default)
        try:
            from memster.local_embeddings import get_shared_local_embedding_model, is_local_embedding_available
            if is_local_embedding_available():
                self.local_embedding_model = get_shared_local_embedding_model(LOCAL_EMBEDDING_MODEL)
                self.local_embeddings_available = True
            else:
                self.local_embedding_model = None
                self.local_embeddings_available = False
        except ImportError:
            self.local_embedding_model = None
            self.local_embeddings_available = False
        except Exception:
            self.local_embedding_model = None
            self.local_embeddings_available = False

        # Check NIM key availability
        self.nim_available = bool(NIM_API_KEY)

        # Determine final availability
        if self.embedding_backend == "nim" and self.nim_available:
            self.embeddings_available = True
            logger.info("Using NVIDIA NIM embeddings backend (2048-dim)")
        elif self.embedding_backend == "local" and self.local_embeddings_available:
            self.embeddings_available = True
            logger.info("Using local embeddings backend (384-dim)")
        else:
            if self.local_embeddings_available:
                self.embeddings_available = True
                self.embedding_backend = "local"
                logger.info("Defaulting to local embeddings backend")
            elif self.nim_available:
                self.embeddings_available = True
                self.embedding_backend = "nim"
                logger.info("Defaulting to NVIDIA NIM embeddings backend")
            else:
                self.embeddings_available = False
                logger.error("No embedding backend available")

        # Query expansion settings
        self.use_query_expansion = os.environ.get("USE_QUERY_EXPANSION", "false").lower() == "true"
        self.query_expansion_max_synonyms = int(os.environ.get("QUERY_EXPANSION_MAX_SYNONYMS", "2"))

        # NLTK WordNet for query expansion
        self.wordnet_available = False
        self.stop_words = set()
        if self.use_query_expansion:
            try:
                from nltk.corpus import wordnet as wn
                from nltk.corpus import stopwords
                try:
                    self.stop_words = set(stopwords.words("english"))
                except Exception:
                    import nltk
                    nltk.download("stopwords", quiet=True)
                    from nltk.corpus import stopwords
                    self.stop_words = set(stopwords.words("english"))
                self.wordnet = wn
                self.wordnet_available = True
                logger.info("WordNet available for query expansion")
            except ImportError:
                logger.warning("NLTK WordNet not available - query expansion disabled")
                self.wordnet_available = False
            except Exception as e:
                logger.warning(f"Query expansion init failed: {e}")

        # Two-stage reranking settings
        self.use_two_stage_reranker = os.environ.get("USE_TWO_STAGE_RERANKER", "false").lower() == "true"
        self.two_stage_reranker_candidates_multiplier = int(
            os.environ.get("TWO_STAGE_RERANKER_CANDIDATES_MULTIPLIER", "5")
        )

        # Lightweight reranker
        self.lightweight_reranker = None
        self.use_lightweight_reranker = os.environ.get("USE_LIGHTWEIGHT_RERANKER", "true").lower() == "true"
        if self.use_lightweight_reranker and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                model_name = os.environ.get("LIGHTWEIGHT_RERANKER_MODEL", "mixedbread-ai/mxbai-rerank-xsmall-v1")
                self.lightweight_reranker = CrossEncoder(model_name, max_length=512)
                logger.info(f"Loaded lightweight reranker: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load lightweight reranker: {e}")
                self.lightweight_reranker = None
        elif self.use_lightweight_reranker:
            logger.warning("Lightweight reranker requested but sentence_transformers not available")

    def embed_and_store(self, memory_id: int, content: str) -> bool:
        """Embed content and store in the local_embedding column."""
        emb = embed_text(content)
        if emb is None:
            return False
        return _store_local_embedding(memory_id, emb)

    # ── public API ────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        semantic_weight: float = 0.40,
        bm25_weight: float = 0.30,
        entity_weight: float = 0.20,
        temporal_weight: float = 0.10,
        rerank_with_llm: bool = False,
        fusion_method: str = "weighted",
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval with multi-signal fusion.

        Returns list of MemoryBead-like dicts with scores.
        """
        # 1. Optional query expansion
        expanded_query = query
        if self.use_query_expansion and self.wordnet_available:
            expanded_query = self._expand_query(query)
            if expanded_query != query:
                logger.debug(f"Query expanded from '{query}' to '{expanded_query}'")

        # 2. Parallel signal extraction (using expanded query)
        signals = {}

        if semantic_weight > 0 and self.embeddings_available:
            signals["semantic"] = self._semantic_search(expanded_query, top_k * 2)

        if bm25_weight > 0:
            signals["bm25"] = self._bm25_search(expanded_query, top_k * 2)

        if entity_weight > 0:
            signals["entity"] = self._entity_search(expanded_query, top_k * 2)

        if temporal_weight > 0:
            signals["temporal"] = self._temporal_boost(top_k * 2)

        # If no signals at all, fall back to simple LIKE search
        if not signals:
            return self._fallback_search(query, top_k)

        # 3. Fusion
        if fusion_method == "rrf":
            final = self._reciprocal_rank_fusion(signals, top_k)
        else:
            final = self._weighted_fusion(
                signals,
                top_k,
                {
                    "semantic": semantic_weight,
                    "bm25": bm25_weight,
                    "entity": entity_weight,
                    "temporal": temporal_weight,
                },
            )

        # 4. Fetch full memory objects
        memories = self._fetch_memories(list(final.keys()))

        # 5. Attach scores
        for mem in memories:
            mem["hybrid_score"] = round(final.get(mem["id"], 0), 4)

        # Sort by final score
        memories.sort(key=lambda m: m["hybrid_score"], reverse=True)

        # 6. Two-stage reranking (uses lightweight reranker on top-N)
        if self.use_two_stage_reranker and self.lightweight_reranker is not None and len(memories) > 3:
            memories = self._two_stage_rerank(query, memories, top_k)
        # 7. Lightweight rerank (fast, CPU-based) - fallback if two-stage is off
        elif self.use_lightweight_reranker and self.lightweight_reranker is not None and len(memories) > 3:
            memories = self._lightweight_rerank(query, memories, top_k)

        # 8. Optional LLM rerank
        if rerank_with_llm and len(memories) > 3:
            memories = self._llm_rerank(query, memories, top_k)

        # 9. Bump access counts
        self._bump_access_counts([m["id"] for m in memories[:top_k]])

        return memories[:top_k]

    # ── query expansion ───────────────────────────────────────────

    def _expand_query(self, query: str) -> str:
        """Expand query with WordNet synonyms for content words."""
        if not self.wordnet_available:
            return query

        tokens = re.findall(r"\w+", query.lower())
        expanded = set(tokens)

        for token in tokens:
            if token in self.stop_words or len(token) < 2:
                continue
            synsets = self.wordnet.synsets(token)
            added = 0
            for syn in synsets:
                if added >= self.query_expansion_max_synonyms:
                    break
                for lemma in syn.lemmas():
                    term = lemma.name().replace("_", " ")
                    if term.isalpha() and len(term) > 1 and term not in self.stop_words:
                        expanded.add(term)
                        added += 1
                        if added >= self.query_expansion_max_synonyms:
                            break

        return " ".join(expanded)

    # ── signal extraction ─────────────────────────────────────────

    def _semantic_search(self, query: str, limit: int) -> Dict[int, float]:
        """Vector similarity via configured embedding backend."""
        try:
            query_emb = embed_text(query)
            if query_emb is None:
                return {}
            results = _vector_search_sql(query_emb, limit=limit, threshold=0.1)
            return results
        except Exception as e:
            logger.debug(f"Semantic search failed: {e}")
            return {}

    def _bm25_search(self, query: str, limit: int) -> Dict[int, float]:
        """BM25-style full-text search via PostgreSQL tsquery."""
        conn = self.get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT id, ts_rank(search_vector, plainto_tsquery('english', %s)) as rank
                FROM memories
                WHERE search_vector @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
            """,
                (query, query, limit),
            )
            results = {}
            for row in cursor.fetchall():
                results[row["id"]] = float(row["rank"]) if row["rank"] else 0.0
            return results
        except Exception as e:
            logger.debug(f"BM25 search failed: {e}")
            return {}
        finally:
            conn.close()

    def _entity_search(self, query: str, limit: int) -> Dict[int, float]:
        """Entity-based boosting: boost memories sharing entities with query."""
        conn = self.get_conn()
        cursor = conn.cursor()
        try:
            from memster.entity_extraction import extract_entities

            query_entities = extract_entities(query)
            if not query_entities:
                return {}

            query_set = set()
            for key, values in query_entities.items():
                if isinstance(values, list):
                    query_set.update(v.lower() for v in values)
                else:
                    query_set.add(str(values).lower())

            if not query_set:
                return {}

            cursor.execute("SELECT memory_id, entities FROM memory_entity_data")
            rows = cursor.fetchall()

            scored = []
            for row in rows:
                mem_id = row["memory_id"]
                try:
                    mem_entities = json.loads(row["entities"]) if row["entities"] else {}
                except (json.JSONDecodeError, TypeError):
                    mem_entities = {}

                mem_set = set()
                for key, values in mem_entities.items():
                    if isinstance(values, list):
                        mem_set.update(v.lower() for v in values)
                    else:
                        mem_set.add(str(values).lower())

                overlap = query_set & mem_set
                if overlap:
                    score = len(overlap)
                    scored.append((mem_id, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            return {mem_id: float(score) for mem_id, score in scored[:limit]}
        except Exception as e:
            logger.debug(f"Entity search failed: {e}")
            return {}
        finally:
            conn.close()

    def _temporal_boost(self, limit: int) -> Dict[int, float]:
        """Temporal-proximity boosting: more recent memories score higher."""
        conn = self.get_conn()
        cursor = conn.cursor()
        try:
            now = datetime.now()
            cursor.execute(
                """
                SELECT id, t_event
                FROM memories
                ORDER BY t_event DESC
                LIMIT %s
            """,
                (limit,),
            )
            results = {}
            for row in cursor.fetchall():
                try:
                    t = datetime.fromisoformat(str(row["t_event"]))
                    days = (now - t).total_seconds() / 86400
                    boost = math.exp(-days / TEMPORAL_HALF_LIFE_DAYS)
                except (ValueError, TypeError):
                    boost = 0.1
                results[row["id"]] = round(boost, 4)
            return results
        except Exception as e:
            logger.debug(f"Temporal boost failed: {e}")
            return {}
        finally:
            conn.close()

    def _fallback_search(self, query: str, limit: int) -> List[Dict]:
        """Simple LIKE search when no signals are available."""
        conn = self.get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT id, content, category, tier, importance, network_type, t_event
                FROM memories
                WHERE content ILIKE %s
                ORDER BY importance DESC, t_event DESC
                LIMIT %s
            """,
                (f"%{query}%", limit),
            )
            results = [dict(r) for r in cursor.fetchall()]
            for r in results:
                r["hybrid_score"] = r.get("importance", 0.5) or 0.5
            return results
        finally:
            conn.close()

    # ── fusion algorithms ────────────────────────────────────

    def _weighted_fusion(
        self, signals: Dict[str, Dict[int, float]], top_k: int, weights: Dict[str, float],
    ) -> Dict[int, float]:
        """Weighted linear combination of signal scores (normalized per signal)."""
        all_ids = set()
        for scores in signals.values():
            all_ids.update(scores.keys())

        if not all_ids:
            return {}

        normalized = {}
        for signal_name, scores in signals.items():
            if not scores:
                continue
            max_s = max(scores.values())
            min_s = min(scores.values())
            rng = max_s - min_s if max_s > min_s else 1.0
            normalized[signal_name] = {mid: (s - min_s) / rng for mid, s in scores.items()}

        final = {}
        for mid in all_ids:
            total = 0.0
            total_weight = 0.0
            for signal_name, norm_scores in normalized.items():
                w = weights.get(signal_name, 0)
                s = norm_scores.get(mid, 0)
                total += w * s
                total_weight += w
            if total_weight > 0:
                final[mid] = round(total / total_weight, 4)

        sorted_ids = sorted(final.keys(), key=lambda i: final[i], reverse=True)
        return {mid: final[mid] for mid in sorted_ids[:top_k]}

    def _reciprocal_rank_fusion(self, signals: Dict[str, Dict[int, float]], top_k: int) -> Dict[int, float]:
        """Reciprocal Rank Fusion (RRF) - no normalization needed."""
        k = int(os.environ.get("RRF_K", str(RRF_K)))

        rankings = {}
        for signal_name, scores in signals.items():
            sorted_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
            rankings[signal_name] = {mid: rank + 1 for rank, mid in enumerate(sorted_ids)}

        all_ids = set()
        for ranks in rankings.values():
            all_ids.update(ranks.keys())

        rrf_scores = {}
        for mid in all_ids:
            score = 0.0
            for ranks in rankings.values():
                rank = ranks.get(mid, len(ranks) + 1)
                score += 1.0 / (k + rank)
            rrf_scores[mid] = round(score, 4)

        sorted_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        return {mid: rrf_scores[mid] for mid in sorted_ids[:top_k]}

    # ── helpers ──────────────────────────────────────────────

    def _fetch_memories(self, memory_ids: List[int]) -> List[Dict[str, Any]]:
        """Fetch full memory objects for given IDs."""
        if not memory_ids:
            return []
        conn = self.get_conn()
        cursor = conn.cursor()
        try:
            placeholders = ",".join(["%s"] * len(memory_ids))
            cursor.execute(
                f"""
                SELECT id, content, network_type, category, t_event, t_recorded,
                       importance, decay_score, access_count, tier, source,
                       fronter_uid, fronter_name, valid_from, valid_to
                FROM memories
                WHERE id IN ({placeholders})
            """,
                memory_ids,
            )
            return [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()

    def _bump_access_counts(self, memory_ids: List[int]) -> None:
        """Increment access_count for retrieved memories."""
        if not memory_ids:
            return
        conn = self.get_conn()
        cursor = conn.cursor()
        try:
            for mid in memory_ids:
                cursor.execute(
                    "UPDATE memories SET access_count = access_count + 1 WHERE id = %s",
                    (mid,),
                )
            conn.commit()
        finally:
            conn.close()

    def _lightweight_rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Lightweight reranking pass on top candidates using CrossEncoder."""
        if not self.lightweight_reranker:
            return candidates
        try:
            pairs = [[query, mem.get("content", "")] for mem in candidates]
            scores = self.lightweight_reranker.predict(pairs)
            scored_indices = list(enumerate(scores))
            scored_indices.sort(key=lambda x: x[1], reverse=True)
            reranked = [candidates[i] for i, _ in scored_indices[:top_k]]
            logger.debug(f"Lightweight reranker reordered {len(candidates)} candidates")
            return reranked
        except Exception as e:
            logger.warning(f"Lightweight reranker failed: {e}, returning original order")
            return candidates

    def _two_stage_rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Two-stage reranking: hybrid fusion -> top-N -> lightweight reranker -> top-K."""
        if not self.lightweight_reranker:
            return candidates
        try:
            rerank_candidates = min(len(candidates), top_k * self.two_stage_reranker_candidates_multiplier)
            if rerank_candidates <= top_k:
                return candidates[:top_k]
            top_candidates = candidates[:rerank_candidates]
            pairs = [[query, mem.get("content", "")] for mem in top_candidates]
            scores = self.lightweight_reranker.predict(pairs)
            scored_indices = list(enumerate(scores))
            scored_indices.sort(key=lambda x: x[1], reverse=True)
            reranked = [top_candidates[i] for i, _ in scored_indices[:top_k]]
            logger.debug(f"Two-stage reranker: reranked {len(top_candidates)} candidates to top {top_k}")
            return reranked
        except Exception as e:
            logger.warning(f"Two-stage reranker failed: {e}, returning original order")
            return candidates[:top_k]

    def _llm_rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Optional LLM reranking pass on top candidates."""
        return candidates


# ── MCP tool handler ─────────────────────────────────────────────

async def handle_hybrid_retrieve(args: Dict[str, Any], engine: HybridRetrievalEngine) -> str:
    """MCP tool handler: memster_hybrid_retrieve."""
    query = args.get("query", "")
    top_k = int(args.get("top_k", DEFAULT_TOP_K))
    semantic_weight = float(args.get("semantic_weight", 0.40))
    bm25_weight = float(args.get("bm25_weight", 0.30))
    entity_weight = float(args.get("entity_weight", 0.20))
    temporal_weight = float(args.get("temporal_weight", 0.10))
    rerank = args.get("rerank_with_llm", False)
    fusion = args.get("fusion_method", "weighted")

    results = engine.retrieve(
        query=query,
        top_k=top_k,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
        entity_weight=entity_weight,
        temporal_weight=temporal_weight,
        rerank_with_llm=rerank,
        fusion_method=fusion,
    )

    return json.dumps(
        {
            "query": query,
            "count": len(results),
            "fusion_method": fusion,
            "weights": {
                "semantic": semantic_weight,
                "bm25": bm25_weight,
                "entity": entity_weight,
                "temporal": temporal_weight,
            },
            "embeddings_available": engine.embeddings_available,
            "results": results,
        },
        indent=2,
        default=str,
    )


HYBRID_RETRIEVAL_TOOL_DEF = {
    "name": "memster_hybrid_retrieve",
    "description": (
        "Multi-signal hybrid memory retrieval fusing semantic vectors, BM25 keyword "
        "search, entity boosting, and temporal-proximity scoring. Configurable weights "
        "and optional lightweight or LLM reranking."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "default": 20},
            "semantic_weight": {"type": "number", "default": 0.40},
            "bm25_weight": {"type": "number", "default": 0.30},
            "entity_weight": {"type": "number", "default": 0.20},
            "temporal_weight": {"type": "number", "default": 0.10},
            "rerank_with_llm": {"type": "boolean", "default": False},
            "fusion_method": {
                "type": "string",
                "enum": ["weighted", "rrf"],
                "default": "weighted",
            },
        },
        "required": ["query"],
    },
}