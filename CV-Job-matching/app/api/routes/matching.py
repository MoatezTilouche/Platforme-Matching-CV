# app/api/routes/matching.py
"""CV matching endpoints"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from typing import List
import json
import os
import tempfile
import time

from app.pipeline import run_pipeline, run_pipeline_rag
from app.api.models import MatchResult, RankingResult, BulkRankingResponse

router = APIRouter(prefix="", tags=["Matching"])


@router.post("/match", response_model=MatchResult)
async def match_single_cv(
    file: UploadFile = File(..., description="CV file in PDF format"),
    job_description: str = Form(..., description="Job description text"),
    use_rag: bool = Form(True, description="Use RAG for better performance (recommended)")
):
    """
    Match a single CV against a job description
    
    RAG mode (use_rag=True, default):
    - Chunks CV into sections
    - Retrieves only relevant parts
    - 70-90% faster with better accuracy
    
    Legacy mode (use_rag=False):
    - Uses full CV text
    - Slower but comprehensive
    
    Returns score, recommendation, strengths, and gaps
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Process CV
        start_time = time.time()
        
        if use_rag:
            # RAG pipeline (recommended)
            result = await run_pipeline_rag(tmp_path, job_description, top_k=5)
        else:
            # Legacy pipeline
            result = await run_pipeline(tmp_path, job_description)
        
        processing_time = time.time() - start_time
        
        # Parse result
        result_data = json.loads(result)
        
        return MatchResult(
            score=result_data.get('final_score', 0),
            recommendation=result_data.get('recommendation', 'unknown'),
            strengths=result_data.get('strengths', []),
            gaps=result_data.get('gaps', []),
            processing_time=round(processing_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CV: {str(e)}")
    
    finally:
        # Cleanup temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/rank", response_model=BulkRankingResponse)
async def rank_multiple_cvs(
    files: List[UploadFile] = File(..., description="Multiple CV files in PDF format"),
    job_description: str = Form(..., description="Job description text"),
    use_rag: bool = Form(True, description="Use RAG for better performance (recommended)")
):
    """
    Rank multiple CVs against a job description
    
    RAG mode (use_rag=True, default):
    - Much faster for bulk processing
    - CV embeddings cached across requests
    - 2-4x speedup
    
    Returns sorted list of candidates with scores and recommendations
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate all files are PDFs
    for file in files:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail=f"File {file.filename} is not a PDF"
            )
    
    results = []
    total_start = time.time()
    
    for file in files:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Process CV
            start_time = time.time()
            
            if use_rag:
                # RAG pipeline (recommended for bulk)
                result = await run_pipeline_rag(tmp_path, job_description, top_k=5)
            else:
                # Legacy pipeline
                result = await run_pipeline(tmp_path, job_description)
            
            processing_time = time.time() - start_time
            
            # Parse result
            result_data = json.loads(result)
            
            results.append(RankingResult(
                filename=file.filename,
                score=result_data.get('final_score', 0),
                recommendation=result_data.get('recommendation', 'unknown'),
                strengths=result_data.get('strengths', []),
                gaps=result_data.get('gaps', []),
                processing_time=round(processing_time, 2)
            ))
            
        except json.JSONDecodeError as e:
            # JSON parsing error
            results.append(RankingResult(
                filename=file.filename,
                score=0,
                recommendation='error',
                strengths=[],
                gaps=[f"JSON parse error: {str(e)[:100]}"],
                processing_time=round(time.time() - start_time, 2) if 'start_time' in locals() else 0
            ))
        except Exception as e:
            # Add error result
            results.append(RankingResult(
                filename=file.filename,
                score=0,
                recommendation='error',
                strengths=[],
                gaps=[f"{type(e).__name__}: {str(e)[:100]}"],
                processing_time=0
            ))
        
        finally:
            # Cleanup temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    total_time = time.time() - total_start
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x.score, reverse=True)
    
    return BulkRankingResponse(
        total_cvs=len(results),
        total_time=round(total_time, 2),
        average_time=round(total_time / len(results), 2) if results else 0,
        results=results
    )
