# app/embedding/rag_embedder.py
"""
RAG-optimized CV embedder with caching.
Handles chunk-based embedding and storage for efficient retrieval.
"""

import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import torch
from app.utils.rag import chunk_cv


class RAGEmbedder:
    """
    Manages chunk-based embeddings for CVs with caching support.
    Uses SentenceTransformers for fast GPU-accelerated embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG embedder with in-memory cache.
        
        Args:
            model_name: Transformer model for embeddings
                - "all-MiniLM-L6-v2" (90MB, FAST, recommended)
                - "all-mpnet-base-v2" (420MB, best quality)
        """
        self._cache: Dict[str, Dict] = {}
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"🚀 RAG Embedder loaded: {model_name} on {self.device.upper()}")
    
    def get_cv_hash(self, cv_text: str) -> str:
        """
        Generate hash for CV text.
        
        Args:
            cv_text: Clean CV text
            
        Returns:
            SHA256 hash of the CV
        """
        return hashlib.sha256(cv_text.encode()).hexdigest()
    
    def embed_cv(
        self,
        cv_text: str,
        force_recompute: bool = False
    ) -> Dict:
        """
        Chunk and embed a CV using FAST SentenceTransformers, with caching.
        
        Args:
            cv_text: Clean CV text
            force_recompute: If True, bypass cache and recompute
            
        Returns:
            Dictionary with chunks, embeddings, and metadata
        """
        cv_hash = self.get_cv_hash(cv_text)
        
        # Check cache
        if not force_recompute and cv_hash in self._cache:
            return self._cache[cv_hash]
        
        # Chunk the CV
        chunks = chunk_cv(cv_text)
        
        # Embed chunks using SentenceTransformers (GPU-accelerated, batch processing)
        embeddings = self.model.encode(
            chunks, 
            convert_to_numpy=True, 
            show_progress_bar=False,
            batch_size=32  # Process in batches for speed
        )
        embeddings = [emb for emb in embeddings]  # Convert to list
        
        # Store in cache
        result = {
            "cv_hash": cv_hash,
            "chunks": chunks,
            "embeddings": embeddings,
            "num_chunks": len(chunks),
            "cv_text": cv_text
        }
        
        self._cache[cv_hash] = result
        
        return result
    
    def get_cached_cv(self, cv_text: str) -> Optional[Dict]:
        """
        Get cached CV embeddings if available.
        
        Args:
            cv_text: Clean CV text
            
        Returns:
            Cached CV data or None
        """
        cv_hash = self.get_cv_hash(cv_text)
        return self._cache.get(cv_hash)
    
    def retrieve_for_jd(
        self,
        cv_data: Dict,
        jd_text: str,
        top_k: int = 5
    ) -> Tuple[List[str], List[float], np.ndarray]:
        """
        Retrieve top K CV chunks relevant to a job description.
        
        Args:
            cv_data: CV data from embed_cv()
            jd_text: Job description text
            top_k: Number of chunks to retrieve
            
        Returns:
            Tuple of (relevant chunks, scores, jd_embedding)
        """
        from app.utils.rag import retrieve_relevant_chunks
        
        # Embed JD using SentenceTransformers
        jd_embedding = self.model.encode(
            jd_text, 
            convert_to_numpy=True, 
            show_progress_bar=False
        )
        
        # Retrieve relevant chunks
        relevant_chunks, scores = retrieve_relevant_chunks(
            cv_data["chunks"],
            cv_data["embeddings"],
            jd_embedding,
            top_k=top_k
        )
        
        return relevant_chunks, scores, jd_embedding
    
    def retrieve_for_jd_with_embedding(
        self,
        cv_data: Dict,
        jd_embedding: np.ndarray,
        top_k: int = 5
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieve top K CV chunks using pre-computed JD embedding.
        Faster when processing multiple CVs with the same JD.
        
        Args:
            cv_data: CV data from embed_cv()
            jd_embedding: Pre-computed JD embedding
            top_k: Number of chunks to retrieve
            
        Returns:
            Tuple of (relevant chunks, scores)
        """
        from app.utils.rag import retrieve_relevant_chunks
        
        # Retrieve relevant chunks using cached JD embedding
        relevant_chunks, scores = retrieve_relevant_chunks(
            cv_data["chunks"],
            cv_data["embeddings"],
            jd_embedding,
            top_k=top_k
        )
        
        return relevant_chunks, scores
    
    def get_similarity_score(
        self,
        cv_embeddings: List[np.ndarray],
        jd_embedding: np.ndarray
    ) -> float:
        """
        Calculate overall similarity between CV and JD.
        Uses max pooling over chunk similarities.
        
        Args:
            cv_embeddings: List of CV chunk embeddings
            jd_embedding: JD embedding
            
        Returns:
            Similarity score (0-100)
        """
        from app.utils.rag import cosine_similarity
        
        # Calculate similarity for each chunk
        similarities = [
            cosine_similarity(cv_emb, jd_embedding)
            for cv_emb in cv_embeddings
        ]
        
        # Use max similarity (best match)
        # Alternative: could use mean or weighted average
        max_similarity = max(similarities) if similarities else 0.0
        
        return round(max_similarity * 100, 1)
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
    
    def cache_size(self) -> int:
        """Get number of cached CVs."""
        return len(self._cache)
    
    def remove_from_cache(self, cv_text: str) -> bool:
        """
        Remove a CV from cache.
        
        Args:
            cv_text: Clean CV text
            
        Returns:
            True if removed, False if not in cache
        """
        cv_hash = self.get_cv_hash(cv_text)
        if cv_hash in self._cache:
            del self._cache[cv_hash]
            return True
        return False


# Global instance for use across the application
_rag_embedder = RAGEmbedder()


def get_rag_embedder() -> RAGEmbedder:
    """
    Get the global RAG embedder instance.
    
    Returns:
        Global RAGEmbedder instance
    """
    return _rag_embedder
