"""
Memster Entity Extraction Module

Rules-based (zero-LLM) entity extraction for hybrid retrieval boosting.
Uses regex patterns and spaCy NER when available. Works on CPU, zero API calls.

Extracts: servers, technologies, statuses, IPs, ports, paths, persons,
protocols, dates, locations, organizations, and other named entities.
"""

import re
from typing import Dict, List, Optional

try:
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
except ImportError:
    nlp = None

# ── Regex patterns ────────────────────────────────────────────────

SERVER_PAT = re.compile(r"\b(server|host|node|instance|container)[\- ]?([\w-]+)\b", re.I)
TECH_PAT = re.compile(
    r"\b(PostgreSQL|Redis|Docker|Kubernetes|Nginx|MongoDB|MySQL|"
    r"Elasticsearch|RabbitMQ|Kafka|Prometheus|Grafana|Terraform|"
    r"Ansible|AWS|GCP|Azure|Flask|FastAPI|Django|React|Vue|Svelte|"
    r"PyTorch|TensorFlow|Jupyter|Git|Linux|Ubuntu|NVIDIA|CUDA)\b",
    re.I,
)
STATUS_PAT = re.compile(
    r"\b(running|failed|crashed|deployed|started|stopped|restarted|"
    r"healthy|unhealthy|down|up|error|success|completed|pending|"
    r"degraded|scaling|migrated|replicated|back(ed)? ?up|recovered|"
    r"disabled|enabled)\b",
    re.I,
)
IP_PAT = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
PORT_PAT = re.compile(r"\bport\s+(\d+)\b", re.I)
PATH_PAT = re.compile(r"\b(?:/[a-zA-Z0-9_.-]+)+")
PERSON_PAT = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b")
TITLE_PAT = re.compile(
    r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
)
PROTOCOL_PAT = re.compile(
    r"\b(http|https|ssh|ftp|tcp|udp|ws|wss|grpc|tls|ssl)\b", re.I
)
VERSION_PAT = re.compile(r"\bv?(\d+)\.(\d+)(?:\.(\d+))?\b")
URL_PAT = re.compile(r"https?://[^\s]+")


def _extract_regex(text: str) -> Dict[str, List[str]]:
    """Extract entities using regex patterns."""
    return {
        "servers": list(set(m.group() for m in SERVER_PAT.finditer(text))),
        "technologies": list(set(m.group() for m in TECH_PAT.finditer(text))),
        "statuses": list(
            set(m.group().lower() for m in STATUS_PAT.finditer(text))
        ),
        "ips": [m.group() for m in IP_PAT.finditer(text)],
        "ports": [m.group(1) for m in PORT_PAT.finditer(text)],
        "paths": [
            m.group()
            for m in PATH_PAT.finditer(text)
            if len(m.group()) > 5
        ],
        "persons": list(
            set(m.group() for m in PERSON_PAT.finditer(text)) |
            set(m.group() for m in TITLE_PAT.finditer(text))
        ),
        "protocols": [
            m.group().lower() for m in PROTOCOL_PAT.finditer(text)
        ],
    }


def _extract_spacy(text: str) -> Dict[str, List[str]]:
    """Extract entities using spaCy NER."""
    if nlp is None:
        return {}
    doc = nlp(text)
    entities: Dict[str, List[str]] = {}
    for ent in doc.ents:
        label = ent.label_.lower()
        entities.setdefault(label, []).append(ent.text)
    # Deduplicate
    for key in entities:
        entities[key] = list(set(entities[key]))
    return entities


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract entities from text using regex patterns and spaCy NER.

    Returns a dict mapping entity type labels to lists of entity strings,
    e.g. {'technologies': ['Docker', 'PostgreSQL'], 'statuses': ['running']}.

    Works entirely on CPU with zero API calls.
    """
    entities = _extract_regex(text)
    if nlp is not None:
        spacy_ents = _extract_spacy(text)
        for key, values in spacy_ents.items():
            entities.setdefault(key, []).extend(values)
            entities[key] = list(set(entities[key]))
    return entities


def extract_entities_list(text: str) -> List[Dict[str, str]]:
    """Return extracted entities as a list of {name, type} dicts.

    Convenience wrapper for downstream consumers that prefer the
    list-of-dicts format.
    """
    raw = extract_entities(text)
    result: List[Dict[str, str]] = []
    for ent_type, names in raw.items():
        for name in names:
            # Find relationships from context
            relationship = _infer_relationship(text, name)
            result.append({
                "name": name,
                "type": ent_type,
                "relationship": relationship,
            })
    return result


def _infer_relationship(text: str, entity: str) -> str:
    """Simple heuristic to infer entity relationship from nearby text."""
    idx = text.lower().find(entity.lower())
    if idx < 0:
        return "mentioned"
    window = text[max(0, idx - 40) : idx + len(entity) + 40].lower()
    if any(w in window for w in ["running", "using", "deployed", "started"]):
        return "active"
    if any(w in window for w in ["failed", "crashed", "down", "error"]):
        return "failed"
    if any(w in window for w in ["configured", "set up", "installed"]):
        return "configured"
    if any(w in window for w in ["migrated", "moved", "transferred"]):
        return "migrated"
    return "mentioned"


# ── Conflict detection ────────────────────────────────────────────

OPPOSITE_PAIRS = [
    ("is up", "is down"),
    ("is running", "is stopped"),
    ("is running", "is down"),
    ("is running", "is broken"),
    ("is running", "is failing"),
    ("deployed successfully", "deployment failed"),
    ("healthy", "unhealthy"),
    ("running", "crashed"),
    ("running", "failed"),
    ("active", "inactive"),
    ("enabled", "disabled"),
    ("started", "stopped"),
    ("increased", "decreased"),
    ("connected", "disconnected"),
    ("online", "offline"),
    ("available", "unavailable"),
    ("passed", "failed"),
    ("completed", "pending"),
]


def detect_conflicts(new_text: str, existing_texts: List[str]) -> List[dict]:
    """Detect semantic conflicts between new and existing memories."""
    new_lower = new_text.lower()
    conflicts = []
    for existing in existing_texts:
        existing_lower = existing.lower()
        for a, b in OPPOSITE_PAIRS:
            has_a = a in new_lower or a in existing_lower
            has_b = b in new_lower or b in existing_lower
            if has_a and has_b:
                new_ents = set(
                    extract_entities(new_text).get("technologies", [])
                )
                existing_ents = set(
                    extract_entities(existing).get("technologies", [])
                )
                if new_ents & existing_ents:
                    conflicts.append(
                        {
                            "pair": (a, b),
                            "shared_entities": list(new_ents & existing_ents),
                        }
                    )
    return conflicts