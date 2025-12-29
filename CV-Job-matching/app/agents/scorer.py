# app/agents/scorer.py
from ollama import AsyncClient
from typing import List, Optional

async def score_cv(cv_json, jd_json, similarity_score):
    """
    Legacy scorer using full CV/JD (not recommended).
    Use score_cv_rag() for better performance.
    """
    prompt = f"""
You are an ATS scoring agent.

Use:
- Skills match (40%)
- Experience (30%)
- Domain relevance (20%)
- Penalties (10%)

Embedding similarity score: {similarity_score}/100

CV:
{cv_json}

JOB:
{jd_json}

Return JSON:
{{
  "final_score": number,
  "strengths": [],
  "gaps": [],
  "recommendation": "reject | maybe | shortlist"
}}
"""

    client = AsyncClient()
    res = await client.chat(
        model="mistral:7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"num_predict": 500}
    )

    return res["message"]["content"]


async def score_cv_rag(
    cv_json: str,
    jd_json: str,
    relevant_cv_chunks: List[str],
    chunk_scores: List[float],
    similarity_score: float,
    jd_chunks: Optional[List[str]] = None
):
    """
    RAG-optimized scorer: uses only retrieved CV chunks instead of full CV.
    
    Args:
        cv_json: Parsed CV (structured info only - skills, experience, etc.)
        jd_json: Parsed JD (structured requirements)
        relevant_cv_chunks: Top K relevant CV chunks
        chunk_scores: Similarity scores for each chunk
        similarity_score: Overall similarity score
        jd_chunks: Optional top JD requirement chunks
        
    Returns:
        Scoring result as JSON string
    """
    # Format relevant CV information (limit context)
    cv_context = "\n\n".join([
        f"[CV Section {i+1}] (score: {score:.2f})\n{chunk[:300]}"  # Limit chunk size
        for i, (chunk, score) in enumerate(zip(relevant_cv_chunks[:3], chunk_scores[:3]))  # Top 3 only
    ])
    
    # Format JD context (use chunks if available, otherwise truncate)
    if jd_chunks:
        jd_context = "\n".join([f"- {chunk[:200]}" for chunk in jd_chunks[:3]])
    else:
        jd_context = jd_json[:800]
    
    prompt = f"""Score candidate (0-100). Weights: Skills 40%, Experience 30%, Domain 20%, Other 10%.

Embedding: {similarity_score}/100

CV: {cv_json[:400]}

Top CV Sections:
{cv_context}

Job: {jd_context}

JSON:
{{
  "final_score": 75,
  "strengths": ["item1", "item2"],
  "gaps": ["item1"],
  "recommendation": "shortlist"
}}
"""

    client = AsyncClient()
    res = await client.chat(
        model="mistral:7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={
            "num_predict": 250,
            "temperature": 0.2
        }
    )

    return res["message"]["content"]
