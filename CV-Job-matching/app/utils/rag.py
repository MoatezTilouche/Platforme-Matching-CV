# app/utils/rag.py
"""
RAG (Retrieval Augmented Generation) utilities for CV-Job matching.
Implements chunking, embedding, and retrieval to reduce LLM context size.
"""

import ollama
import numpy as np
import re
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
import torch

# Global GPU-accelerated embedding model
_embedding_model = None

def get_embedding_model():
    """Get or create GPU-accelerated embedding model."""
    global _embedding_model
    if _embedding_model is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print(f"🚀 Embedding model loaded on: {device.upper()}")
    return _embedding_model


def chunk_cv(text: str, max_chars: int = 500) -> List[str]:
    """
    Chunk CV into logical sections using section headers and character limits.
    
    Prioritizes:
    - Skills sections
    - Each job experience
    - Education
    - Projects
    - Certifications
    
    Args:
        text: Clean CV text
        max_chars: Maximum characters per chunk
        
    Returns:
        List of CV chunks
    """
    chunks = []
    
    # First, try to split by common section headers
    section_patterns = [
        r'\n\s*(?:SKILLS?|TECHNICAL SKILLS?|CORE COMPETENCIES)\s*\n',
        r'\n\s*(?:EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE)\s*\n',
        r'\n\s*(?:EDUCATION|ACADEMIC BACKGROUND)\s*\n',
        r'\n\s*(?:PROJECTS?|KEY PROJECTS?)\s*\n',
        r'\n\s*(?:CERTIFICATIONS?|CERTIFICATES?)\s*\n',
    ]
    
    # Split by sections first
    sections = [text]
    for pattern in section_patterns:
        new_sections = []
        for section in sections:
            parts = re.split(pattern, section, flags=re.IGNORECASE)
            new_sections.extend([p.strip() for p in parts if p.strip()])
        sections = new_sections
    
    # Now chunk each section by character limit
    for section in sections:
        if len(section) <= max_chars:
            if section.strip():
                chunks.append(section.strip())
        else:
            # Split long sections by paragraphs/lines
            current = ""
            for line in section.split("\n"):
                if len(current) + len(line) + 1 > max_chars:
                    if current.strip():
                        chunks.append(current.strip())
                    current = line
                else:
                    current += "\n" + line if current else line
            
            if current.strip():
                chunks.append(current.strip())
    
    return chunks


def chunk_jd(text: str, max_chars: int = 500) -> List[str]:
    """
    Chunk Job Description into logical sections.
    
    Args:
        text: Clean JD text
        max_chars: Maximum characters per chunk
        
    Returns:
        List of JD chunks
    """
    chunks = []
    
    # Split by common JD sections
    section_patterns = [
        r'\n\s*(?:REQUIREMENTS?|QUALIFICATIONS?|REQUIRED SKILLS?)\s*\n',
        r'\n\s*(?:RESPONSIBILITIES|JOB DUTIES|ROLE)\s*\n',
        r'\n\s*(?:NICE TO HAVE|PREFERRED|BONUS)\s*\n',
        r'\n\s*(?:ABOUT|COMPANY|OVERVIEW)\s*\n',
    ]
    
    sections = [text]
    for pattern in section_patterns:
        new_sections = []
        for section in sections:
            parts = re.split(pattern, section, flags=re.IGNORECASE)
            new_sections.extend([p.strip() for p in parts if p.strip()])
        sections = new_sections
    
    # Chunk by size
    for section in sections:
        if len(section) <= max_chars:
            if section.strip():
                chunks.append(section.strip())
        else:
            current = ""
            for line in section.split("\n"):
                if len(current) + len(line) + 1 > max_chars:
                    if current.strip():
                        chunks.append(current.strip())
                    current = line
                else:
                    current += "\n" + line if current else line
            
            if current.strip():
                chunks.append(current.strip())
    
    return chunks


def embed_chunks(chunks: List[str], model: str = "nomic-embed-text") -> List[np.ndarray]:
    """
    Embed multiple chunks using GPU-accelerated sentence-transformers.
    10x faster than Ollama sequential calls.
    
    Args:
        chunks: List of text chunks
        model: Embedding model to use (ignored, using sentence-transformers)
        
    Returns:
        List of embedding vectors
    """
    embedding_model = get_embedding_model()
    # Batch encode all chunks at once (GPU accelerated)
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    return [emb for emb in embeddings]


def embed_text(text: str, model: str = "nomic-embed-text") -> np.ndarray:
    """
    Embed a single text using GPU-accelerated sentence-transformers.
    
    Args:
        text: Text to embed
        model: Embedding model to use (ignored, using sentence-transformers)
        
    Returns:
        Embedding vector
    """
    embedding_model = get_embedding_model()
    return embedding_model.encode(text, convert_to_numpy=True, show_progress_bar=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_relevant_chunks(
    chunks: List[str],
    chunk_embeddings: List[np.ndarray],
    query_embedding: np.ndarray,
    top_k: int = 5
) -> Tuple[List[str], List[float]]:
    """
    Retrieve top K most relevant chunks using cosine similarity.
    
    Args:
        chunks: Original text chunks
        chunk_embeddings: Embeddings for each chunk
        query_embedding: Query embedding (e.g., JD embedding)
        top_k: Number of top chunks to retrieve
        
    Returns:
        Tuple of (top chunks, similarity scores)
    """
    # Calculate similarity scores
    scores = [
        cosine_similarity(chunk_emb, query_embedding)
        for chunk_emb in chunk_embeddings
    ]
    
    # Get top K indices
    top_k = min(top_k, len(chunks))
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    # Return top chunks and their scores
    top_chunks = [chunks[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]
    
    return top_chunks, top_scores


def format_retrieved_context(chunks: List[str], scores: List[float]) -> str:
    """
    Format retrieved chunks into a readable context string.
    
    Args:
        chunks: Retrieved chunks
        scores: Similarity scores for each chunk
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
        context_parts.append(f"[Relevant Section {i}] (relevance: {score:.2f})")
        context_parts.append(chunk)
        context_parts.append("")  # Empty line
    
    return "\n".join(context_parts)


def rag_cv_to_jd(
    cv_text: str,
    jd_text: str,
    cv_chunks: List[str] = None,
    cv_embeddings: List[np.ndarray] = None,
    top_k: int = 5
) -> Dict:
    """
    RAG pipeline: retrieve relevant CV chunks for a job description.
    
    Args:
        cv_text: Full CV text
        jd_text: Full JD text
        cv_chunks: Pre-computed CV chunks (optional)
        cv_embeddings: Pre-computed CV embeddings (optional)
        top_k: Number of chunks to retrieve
        
    Returns:
        Dictionary with retrieved chunks, scores, and formatted context
    """
    # Chunk CV if not provided
    if cv_chunks is None:
        cv_chunks = chunk_cv(cv_text)
    
    # Embed CV chunks if not provided
    if cv_embeddings is None:
        cv_embeddings = embed_chunks(cv_chunks)
    
    # Embed JD
    jd_embedding = embed_text(jd_text)
    
    # Retrieve relevant chunks
    relevant_chunks, scores = retrieve_relevant_chunks(
        cv_chunks,
        cv_embeddings,
        jd_embedding,
        top_k=top_k
    )
    
    # Format context
    formatted_context = format_retrieved_context(relevant_chunks, scores)
    
    return {
        "chunks": relevant_chunks,
        "scores": scores,
        "formatted_context": formatted_context,
        "total_chunks": len(cv_chunks),
        "retrieved_chunks": len(relevant_chunks),
        "avg_score": float(np.mean(scores))
    }
