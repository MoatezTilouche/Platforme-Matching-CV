# API_USAGE.md

`- LLM: mistral (Ollama)- Embeddings: all-MiniLM-L6-v2 (SentenceTransformers)Models Used:└─────────────────────────────────────────────────────┘│                                                      ││     └─> RAG-based scoring with retrieved chunks     ││  4. Scoring (Mistral)                               ││                                                      ││     └─> Calculate similarity                        ││     ├─> Parse JD (Mistral)                          ││     ├─> Parse CV (Mistral)                          ││  3. Async Processing (single-threaded)              ││                                                      ││     └─> Batch encode all chunks at once             ││  2. Embeddings (SentenceTransformers + GPU)         ││                                                      ││     └─> PDF → Text → Cleaned Text                   ││  1. Extract & Clean (synchronous)                   ││                                                      │├─────────────────────────────────────────────────────┤│                   Pipeline Flow                      │┌─────────────────────────────────────────────────────┐`## Architecture Summary`python run_api.py`bashStart the API server:`python test_installation.py`bashRun the installation test:## Testing4. **Batch processing**: More efficient than parallel chunk processing3. **Single model (Mistral)**: Simpler setup, less memory usage, unified behavior2. **SentenceTransformers**: 10x faster embeddings with GPU acceleration1. **Single-threaded async execution**: More predictable, easier to debug## Benefits`pip install sentence-transformers torch# Already in requirements.txt`bash### Dependencies`ollama pull mistral# Only one model needed now`bash### Required Model## Setup Instructions- Updated documentation to reflect SentenceTransformers usage (no Ollama needed for embeddings)- Added `mistral` as the only required Ollama model- Removed `qwen2.5:7b`, `llama3.1:8b`, and `nomic-embed-text` from model checks**Changes:**- [API_USAGE.md](API_USAGE.md)- [test_installation.py](test_installation.py)- [run_api.py](run_api.py)**Files Modified:**### 4. ✅ Updated Model Checks and Documentation- Scorer: `mistral`- JD Parser: `mistral`- CV Parser: `mistral`**After:**- Scorer: `llama3.1:8b`- JD Parser: `qwen2.5:7b`- CV Parser: `qwen2.5:7b`**Before:**- Now using single model for all agent operations- Replaced `model="llama3.1:8b"` → `model="mistral"`- Replaced `model="qwen2.5:7b"` → `model="mistral"`**Changes:**- [app/agents/scorer.py](app/agents/scorer.py)- [app/agents/jd_parser.py](app/agents/jd_parser.py)- [app/agents/cv_parser.py](app/agents/cv_parser.py)**Files Modified:**### 3. ✅ Replaced Qwen and Llama with Mistral`    return [emb for emb in embeddings]    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)    # Batch encode all chunks at once (GPU accelerated)    embedding_model = get_embedding_model()def embed_chunks(chunks: List[str]) -> List[np.ndarray]:    return _embedding_model        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)        device = 'cuda' if torch.cuda.is_available() else 'cpu'    if _embedding_model is None:    global _embedding_modeldef get_embedding_model():_embedding_model = None# Global GPU-accelerated embedding model`python**Implementation:**- ✅ No parallel chunking needed - batch encoding is more efficient- ✅ GPU acceleration when available- ✅ Batch processing with `embedding_model.encode(chunks, convert_to_numpy=True)`- ✅ Already using `SentenceTransformer` from `sentence_transformers`**Status:**- [app/utils/rag.py](app/utils/rag.py)**Files Verified:**### 2. ✅ SentenceTransformers for Embeddings (Already Implemented)`cv_json, jd_json = asyncio.run(process_async())    return cv_json, jd_json    jd_json = await jd_task    cv_json = await cv_task    jd_task = asyncio.create_task(asyncio.to_thread(parse_jd, jd_trimmed))    cv_task = asyncio.create_task(asyncio.to_thread(parse_cv, cv_trimmed))async def process_async():`python**After:**`    jd_json = future_jd.result()    cv_json = future_cv.result()    future_jd = executor.submit(parse_jd, jd_trimmed)    future_cv = executor.submit(parse_cv, cv_trimmed)with ThreadPoolExecutor(max_workers=3) as executor:`python**Before:**- Single-threaded async execution using `asyncio.run()`- Used `asyncio.to_thread()` and `asyncio.create_task()` for non-blocking execution- Replaced all `ThreadPoolExecutor(max_workers=3)` and `ThreadPoolExecutor(max_workers=2)` with asyncio patterns- Removed `from concurrent.futures import ThreadPoolExecutor`**Changes:**- [app/pipeline.py](app/pipeline.py)**Files Modified:**### 1. ✅ Replaced ThreadPoolExecutor with Asyncio (Single Thread)## Changes Made# 🚀 CV-Job Matching API Documentation

## Quick Start

### 1. Start the API Server

```bash
python run_api.py
```

The server will start at:

- **Local**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (Interactive Swagger UI)
- **ReDoc**: http://localhost:8000/redoc (Alternative docs)

---

## 📡 API Endpoints

### 1. **Root Endpoint**

```http
GET /
```

**Response:**

```json
{
  "message": "CV-Job Matching API",
  "version": "2.0.0",
  "endpoints": { ... }
}
```

---

### 2. **Health Check**

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "optimizations": {
    "caching": true,
    "text_trimming": true,
    "parallelization": true,
    "token_limits": true
  },
  "timestamp": "2025-12-26T10:30:00"
}
```

---

### 3. **Match Single CV** ⭐ Most Common

```http
POST /match
```

**Request:**

- `file`: PDF file (multipart/form-data)
- `job_description`: Job description text (form field)

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/match" \
  -F "file=@cv_ahmed.pdf" \
  -F "job_description=Looking for Python developer with 5 years experience in FastAPI"
```

**Python Example:**

```python
import requests

url = "http://localhost:8000/match"

with open("cv_ahmed.pdf", "rb") as f:
    files = {"file": f}
    data = {"job_description": "Python developer with FastAPI experience"}
    response = requests.post(url, files=files, data=data)

print(response.json())
```

**Response:**

```json
{
  "score": 85.5,
  "recommendation": "shortlist",
  "strengths": [
    {
      "type": "Skills match",
      "score": 34
    },
    {
      "type": "Experience",
      "score": 25.5
    }
  ],
  "gaps": [
    {
      "area": "AWS experience",
      "severity": "minor"
    }
  ],
  "processing_time": 8.45
}
```

---

### 4. **Rank Multiple CVs** 🏆

```http
POST /rank
```

**Request:**

- `files`: Multiple PDF files (multipart/form-data)
- `job_description`: Job description text (form field)

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/rank" \
  -F "files=@cv_ahmed.pdf" \
  -F "files=@cv_moatez.pdf" \
  -F "files=@cv_oussama.pdf" \
  -F "job_description=Backend developer with Node.js and TypeScript"
```

**Python Example:**

```python
import requests

url = "http://localhost:8000/rank"

files = [
    ("files", open("cv_ahmed.pdf", "rb")),
    ("files", open("cv_moatez.pdf", "rb")),
    ("files", open("cv_oussama.pdf", "rb"))
]

data = {"job_description": "Backend developer with Node.js"}
response = requests.post(url, files=files, data=data)

result = response.json()
print(f"Total CVs: {result['total_cvs']}")
print(f"Total time: {result['total_time']}s")

for rank, cv in enumerate(result['results'], 1):
    print(f"{rank}. {cv['filename']}: {cv['score']}")
```

**Response:**

```json
{
  "total_cvs": 3,
  "total_time": 18.45,
  "average_time": 6.15,
  "results": [
    {
      "filename": "cv_ahmed.pdf",
      "score": 85.5,
      "recommendation": "shortlist",
      "strengths": [...],
      "gaps": [...],
      "processing_time": 8.23
    },
    {
      "filename": "cv_moatez.pdf",
      "score": 74.0,
      "recommendation": "maybe",
      "strengths": [...],
      "gaps": [...],
      "processing_time": 5.12
    },
    {
      "filename": "cv_oussama.pdf",
      "score": 62.5,
      "recommendation": "reject",
      "strengths": [...],
      "gaps": [...],
      "processing_time": 5.10
    }
  ]
}
```

---

### 5. **Cache Statistics**

```http
GET /cache/stats
```

**Response:**

```json
{
  "cached_cvs": 15,
  "cache_size_mb": 2.34,
  "status": "active"
}
```

---

### 6. **Clear Cache**

```http
DELETE /cache
```

**Response:**

```json
{
  "message": "Cache cleared successfully",
  "status": "success"
}
```

---

## 🔥 Real-World Examples

### Example 1: Screen Candidates for a Job

```python
import requests
import os

API_URL = "http://localhost:8000"

# Your job description
JOB_DESCRIPTION = """
We're looking for a Senior Backend Developer with:
- 5+ years Python experience
- FastAPI expertise
- Microservices architecture
- AWS/Docker knowledge
"""

# Get all CVs from a folder
cv_folder = "candidates/"
cv_files = [f for f in os.listdir(cv_folder) if f.endswith('.pdf')]

# Upload and rank all CVs
files = [("files", open(f"{cv_folder}/{cv}", "rb")) for cv in cv_files]
data = {"job_description": JOB_DESCRIPTION}

response = requests.post(f"{API_URL}/rank", files=files, data=data)
results = response.json()

# Show top 3 candidates
print("\n🏆 TOP 3 CANDIDATES:")
for i, candidate in enumerate(results['results'][:3], 1):
    print(f"\n{i}. {candidate['filename']}")
    print(f"   Score: {candidate['score']}/100")
    print(f"   Recommendation: {candidate['recommendation'].upper()}")
    print(f"   Processing time: {candidate['processing_time']}s")
```

---

### Example 2: Integration with Frontend

```javascript
// JavaScript/React example
async function rankCandidates(cvFiles, jobDescription) {
  const formData = new FormData();

  // Add all CV files
  cvFiles.forEach((file) => {
    formData.append("files", file);
  });

  // Add job description
  formData.append("job_description", jobDescription);

  const response = await fetch("http://localhost:8000/rank", {
    method: "POST",
    body: formData,
  });

  const results = await response.json();

  // Display results
  results.results.forEach((candidate, index) => {
    console.log(`${index + 1}. ${candidate.filename}: ${candidate.score}`);
  });

  return results;
}
```

---

### Example 3: Batch Processing with Progress

```python
import requests
from tqdm import tqdm

API_URL = "http://localhost:8000"

def process_cv_batch(cv_paths, job_description):
    """Process CVs one by one with progress bar"""
    results = []

    for cv_path in tqdm(cv_paths, desc="Processing CVs"):
        with open(cv_path, "rb") as f:
            files = {"file": f}
            data = {"job_description": job_description}

            response = requests.post(f"{API_URL}/match", files=files, data=data)

            if response.status_code == 200:
                result = response.json()
                results.append({
                    "filename": cv_path,
                    "score": result['score'],
                    "recommendation": result['recommendation']
                })

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

# Usage
cvs = ["cv1.pdf", "cv2.pdf", "cv3.pdf"]
jd = "Python developer with FastAPI"

results = process_cv_batch(cvs, jd)
print(f"\nTop candidate: {results[0]['filename']} ({results[0]['score']})")
```

---

## 🎯 Response Codes

| Code | Meaning                                             |
| ---- | --------------------------------------------------- |
| 200  | Success                                             |
| 400  | Bad request (invalid file type, missing parameters) |
| 500  | Server error (LLM failure, processing error)        |

---

## ⚡ Performance Tips

1. **Use `/rank` for multiple CVs** - Processes them in one request (faster)
2. **Cache is automatic** - Second time processing same CV is 3-5× faster
3. **Clear cache periodically** - Use `DELETE /cache` if needed
4. **File size limits** - Keep PDFs under 10MB for best performance

---

## 🔧 Testing the API

### Using the Interactive Docs

1. Go to http://localhost:8000/docs
2. Click "Try it out" on any endpoint
3. Upload files and test directly in browser

### Using Postman

1. Import the collection (see `postman_collection.json`)
2. Set base URL to `http://localhost:8000`
3. Test all endpoints

### Using Python

```python
import requests

# Test health
health = requests.get("http://localhost:8000/health")
print(health.json())

# Test match
with open("cv.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/match",
        files={"file": f},
        data={"job_description": "Python developer"}
    )
print(response.json())
```

---

## 🛡️ Error Handling

All endpoints return structured errors:

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "timestamp": "2025-12-26T10:30:00"
}
```

Common errors:

- **Invalid file type**: Only PDF files are supported
- **Missing Ollama models**: Install Mistral model (`ollama pull mistral:7b-instruct`)
- **Processing timeout**: CV might be too large or complex

---

## 📊 Monitoring

Check API status:

```bash
curl http://localhost:8000/health
```

Check cache usage:

```bash
curl http://localhost:8000/cache/stats
```

---

**API is ready! Start with:** `python run_api.py` 🚀
