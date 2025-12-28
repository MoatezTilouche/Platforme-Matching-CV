# app/pipeline.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.extract.pdf_extractor import extract_pdf_text
from app.extract.cleaner import clean_text
from app.agents.cv_parser import parse_cv
from app.agents.jd_parser import parse_jd
from app.embedding.similarity import embedding_score, embedding_score_rag
from app.agents.scorer import score_cv, score_cv_rag
from app.utils.cache import CVCache
from app.utils.text_trimmer import trim_cv, trim_jd
from app.embedding.rag_embedder import get_rag_embedder
from app.utils.rag import chunk_jd

# Global cache instance
_cache = CVCache()

# Global RAG embedder
_rag_embedder = get_rag_embedder()

# Global JD embedding cache (reused across CVs)
_jd_embedding_cache = {}

def run_pipeline(cv_path, jd_text):
    """
    Optimized pipeline with:
    - CV caching
    - Text trimming
    - Parallel processing
    - Token limits
    
    LEGACY: Uses full documents (not recommended for large CVs)
    """
    # 1. Extract and clean
    cv_raw = extract_pdf_text(cv_path)
    cv_clean = clean_text(cv_raw)
    jd_clean = clean_text(jd_text)
    
    # 2. Check cache
    cv_hash = _cache.get_hash(cv_clean)
    cached_cv = _cache.get(cv_hash)
    
    # 3. Trim text for faster LLM processing
    cv_trimmed = trim_cv(cv_clean)
    jd_trimmed = trim_jd(jd_clean)
    
    # 4. Parallel processing using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Parse CV (or use cache)
        if cached_cv and "parsed" in cached_cv:
            cv_json = cached_cv["parsed"]
            future_cv = None
        else:
            future_cv = executor.submit(parse_cv, cv_trimmed)
        
        # Parse JD and calculate embedding in parallel
        future_jd = executor.submit(parse_jd, jd_trimmed)
        future_emb = executor.submit(embedding_score, cv_clean, jd_clean)
        
        # Wait for results
        cv_json = cv_json if cached_cv and "parsed" in cached_cv else future_cv.result()
        jd_json = future_jd.result()
        sim_score = future_emb.result()
    
    # 5. Cache CV results
    if not cached_cv:
        _cache.set(cv_hash, {
            "parsed": cv_json,
            "clean_text": cv_clean
        })
    
    # 6. Score
    return score_cv(cv_json, jd_json, sim_score)


def run_pipeline_async(cv_path, jd_text):
    """
    Async version for even better performance with many CVs
    """
    async def process():
        # Extract and clean
        cv_raw = extract_pdf_text(cv_path)
        cv_clean = clean_text(cv_raw)
        jd_clean = clean_text(jd_text)
        
        # Check cache
        cv_hash = _cache.get_hash(cv_clean)
        cached_cv = _cache.get(cv_hash)
        
        # Trim text
        cv_trimmed = trim_cv(cv_clean)
        jd_trimmed = trim_jd(jd_clean)
        
        # Run in parallel
        loop = asyncio.get_event_loop()
        
        if cached_cv and "parsed" in cached_cv:
            cv_task = asyncio.create_task(asyncio.sleep(0))
            cv_json = cached_cv["parsed"]
        else:
            cv_task = loop.run_in_executor(None, parse_cv, cv_trimmed)
        
        jd_task = loop.run_in_executor(None, parse_jd, jd_trimmed)
        emb_task = loop.run_in_executor(None, embedding_score, cv_clean, jd_clean)
        
        if not (cached_cv and "parsed" in cached_cv):
            cv_json = await cv_task
        await jd_task
        await emb_task
        
        jd_json = jd_task.result() if hasattr(jd_task, 'result') else await jd_task
        sim_score = emb_task.result() if hasattr(emb_task, 'result') else await emb_task
        
        # Cache
        if not cached_cv:
            _cache.set(cv_hash, {
                "parsed": cv_json,
                "clean_text": cv_clean
            })
        
        return score_cv(cv_json, jd_json, sim_score)
    
    return asyncio.run(process())


def clear_cache():
    """Clear all cached CV data"""
    _cache.clear()


def run_pipeline_rag(cv_path, jd_text, top_k=5):
    """
    RAG-optimized pipeline with:
    - CV chunking & embedding (cached)
    - Retrieval of top K relevant chunks
    - Reduced LLM context (70-90% smaller)
    - Better performance & accuracy
    
    RECOMMENDED: Use this for production
    
    Args:
        cv_path: Path to CV PDF
        jd_text: Job description text
        top_k: Number of CV chunks to retrieve (default: 5)
        
    Returns:
        Scoring result as JSON string
    """
    # 1. Extract and clean
    cv_raw = extract_pdf_text(cv_path)
    cv_clean = clean_text(cv_raw)
    jd_clean = clean_text(jd_text)
    
    # 2. Check if CV chunks are cached
    cached_cv_data = _rag_embedder.get_cached_cv(cv_clean)
    
    if cached_cv_data:
        # Use cached chunks and embeddings
        cv_data = cached_cv_data
    else:
        # Chunk and embed CV (will be cached)
        cv_data = _rag_embedder.embed_cv(cv_clean)
    
    # 3. Get or compute JD embedding (cached globally across all CVs)
    import hashlib
    jd_hash = hashlib.sha256(jd_clean.encode()).hexdigest()
    
    if jd_hash not in _jd_embedding_cache:
        from app.utils.rag import embed_text
        _jd_embedding_cache[jd_hash] = embed_text(jd_clean)
    
    jd_embedding = _jd_embedding_cache[jd_hash]
    
    # Retrieve relevant CV chunks using cached JD embedding
    relevant_chunks, chunk_scores = _rag_embedder.retrieve_for_jd_with_embedding(
        cv_data,
        jd_embedding,
        top_k=top_k
    )
    
    # 4. Calculate overall similarity score (RAG-based)
    sim_score = embedding_score_rag(
        cv_data["embeddings"],
        jd_embedding,
        pooling="max"
    )
    
    # 5. Parse CV and JD (use trimmed versions for parsing)
    cv_trimmed = trim_cv(cv_clean)
    jd_trimmed = trim_jd(jd_clean)
    
    # Optional: chunk JD for even more optimization
    jd_chunks = chunk_jd(jd_clean, max_chars=400)
    top_jd_chunks = jd_chunks[:3] if len(jd_chunks) > 3 else jd_chunks
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Parse CV and JD in parallel
        future_cv = executor.submit(parse_cv, cv_trimmed)
        future_jd = executor.submit(parse_jd, jd_trimmed)
        
        cv_json = future_cv.result()
        jd_json = future_jd.result()
    
    # 6. Score using RAG (only relevant chunks sent to LLM)
    return score_cv_rag(
        cv_json=cv_json,
        jd_json=jd_json,
        relevant_cv_chunks=relevant_chunks,
        chunk_scores=chunk_scores,
        similarity_score=sim_score,
        jd_chunks=top_jd_chunks
    )
