"""
Memster Core Package

Local-first long-term memory system for AI agents.
PostgreSQL backend, hybrid retrieval, entity extraction,
dual embedding backends (local/NIM), and MCP integration.
"""

from memster.hybrid_retrieval import HybridRetrievalEngine

__all__ = ["HybridRetrievalEngine"]
__version__ = "0.6.0"