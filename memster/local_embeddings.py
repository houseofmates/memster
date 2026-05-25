#!/usr/bin/env python3
"""
Local embeddings module for Memster.
Supports CPU-efficient embedding models that work on machines with 8GB RAM and no GPU.
"""

import os
import json
import numpy as np
from typing import List, Optional, Union
import logging

logger = logging.getLogger("memster.local_embeddings")

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence_transformers not available - local embeddings disabled")

# Try to import ONNX Runtime for even faster inference
try:
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.debug("onnxruntime not available - using standard transformers")


class LocalEmbeddingModel:
    """Local embedding model wrapper."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: str = "./models"):
        """
        Initialize local embedding model.
        
        Args:
            model_name: HuggingFace model identifier
            cache_dir: Directory to cache models
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence_transformers required for local embeddings")
            
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.dimension = None
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading local embedding model: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name, 
                cache_folder=self.cache_dir
            )
            # Get embedding dimension by testing with a dummy string
            test_embedding = self.model.encode(["test"], convert_to_numpy=True)
            self.dimension = test_embedding.shape[1]
            logger.info(f"Loaded model {self.model_name} with dimension {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Embed a single text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding, or None on failure
        """
        if not self.model:
            logger.error("Model not loaded")
            return None
            
        try:
            # Truncate text to reasonable length
            text = text[:8192]  # Most models have token limits
            embedding = self.model.encode([text], convert_to_numpy=True)[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None
    
    def embed_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Embed a batch of text strings.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding lists, or None on failure
        """
        if not self.model:
            logger.error("Model not loaded")
            return None
            
        try:
            # Truncate texts
            texts = [t[:8192] for t in texts]
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return None
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension or 384  # Default for bge-small-en-v1.5


# Predefined model configurations
LOCAL_MODELS = {
    "nomic-embed-text-v2": {
        "name": "nomic-embed-text-v2",
        "dimension": 768,
        "description": "Nomic's general purpose text embedding model"
    },
    "BAAI/bge-small-en-v1.5": {
        "name": "BAAI/bge-small-en-v1.5", 
        "dimension": 384,
        "description": "BAAI's small English embedding model - efficient and effective"
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "description": "Popular general purpose sentence transformer"
    }
}


def get_local_embedding_model(model_key: str = "BAAI/bge-small-en-v1.5") -> LocalEmbeddingModel:
    """
    Get a local embedding model instance.
    
    Args:
        model_key: Key from LOCAL_MODELS or full model name
        
    Returns:
        LocalEmbeddingModel instance
    """
    if model_key in LOCAL_MODELS:
        model_name = LOCAL_MODELS[model_key]["name"]
    else:
        model_name = model_key
        
    return LocalEmbeddingModel(model_name)


# Global model instance for reuse (singleton pattern)
_LOCAL_MODEL_INSTANCE = None
_MODEL_KEY = None


def get_shared_local_embedding_model(model_key: str = "BAAI/bge-small-en-v1.5") -> LocalEmbeddingModel:
    """
    Get or create a shared local embedding model instance.
    
    Args:
        model_key: Key from LOCAL_MODELS or full model name
        
    Returns:
        Shared LocalEmbeddingModel instance
    """
    global _LOCAL_MODEL_INSTANCE, _MODEL_KEY
    
    if _LOCAL_MODEL_INSTANCE is None or _MODEL_KEY != model_key:
        _LOCAL_MODEL_INSTANCE = get_local_embedding_model(model_key)
        _MODEL_KEY = model_key
        
    return _LOCAL_MODEL_INSTANCE


def is_local_embedding_available() -> bool:
    """Check if local embeddings are available."""
    return SENTENCE_TRANSFORMERS_AVAILABLE


if __name__ == "__main__":
    # Test the local embedding module
    logging.basicConfig(level=logging.INFO)
    
    try:
        model = get_shared_local_embedding_model("BAAI/bge-small-en-v1.5")
        print(f"Model dimension: {model.get_dimension()}")
        
        # Test embedding
        test_text = "This is a test sentence for embedding."
        embedding = model.embed_text(test_text)
        if embedding:
            print(f"Embedding length: {len(embedding)}")
            print(f"First 5 values: {embedding[:5]}")
        else:
            print("Embedding failed")
    except Exception as e:
        print(f"Error: {e}")
def get_embedding(text: str) -> Optional[List[float]]:
    """
    Get embedding for a single text string using the shared local embedding model.
    """
    model = get_shared_local_embedding_model()
    return model.embed_text(text)

