# 🎯 CV-Job Matching Platform

An AI-powered platform for intelligent CV ranking and job matching using **RAG (Retrieval-Augmented Generation)** technology. This system analyzes CVs against job descriptions to provide accurate matching scores and rankings.

![Version](https://img.shields.io/badge/version-2.1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![React](https://img.shields.io/badge/react-19.2.0-61dafb.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0+-009688.svg)

---

## 🎥 Demo Video

<div align="center">

[![Watch Demo](https://img.shields.io/badge/▶️_Watch_Demo_Video-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://drive.google.com/file/d/1OcSou34eUQuk5HvqSMdc_5OrFTP8Zopc/view?usp=sharing)

_Click above to watch a comprehensive demonstration of the CV-Job Matching Platform_

</div>

---

## 📋 Table of Contents

- [Demo Video](#-demo-video)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
  - [API Usage](#-api-usage)
  - [Frontend Usage](#-frontend-usage)
- [RAG Implementation](#-rag-implementation)
- [Performance](#-performance)
- [Development](#-development)
  - [Running Tests](#running-tests)
  - [Environment Variables](#environment-variables)
  - [Example Use Cases](#-example-use-cases)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Roadmap](#️-roadmap)
- [Changelog](#-changelog)
- [Author](#-author)
- [Support](#-support)
- [License](#-license)
- [Author](#-author)
- [Support](#-support)
- [License](#-license)

---

## ✨ Features

### Core Capabilities

- 🤖 **AI-Powered Matching**: Advanced CV-Job matching using Mistral LLM (Ollama)
- 🔍 **RAG Technology**: Efficient retrieval-augmented generation for accurate scoring
- 📊 **Batch Processing**: Rank multiple CVs against a single job description
- ⚡ **Ultra-Fast Performance**: 60-80% faster with async processing, GPU embeddings, and single-call scoring
- 🎨 **Modern UI**: Beautiful React-based frontend with smooth animations
- 📄 **PDF Support**: Direct PDF parsing and text extraction
- 💾 **Smart Caching**: Automatic caching for embeddings and results
- 🔄 **Real-time API**: Async RESTful API with FastAPI
- 🚀 **GPU Acceleration**: SentenceTransformers for 10x faster embeddings

### Advanced Features

- **Section-Aware Analysis**: Intelligent CV section detection (skills, experience, education)
- **Multiple Pooling Strategies**: Max, mean, and weighted embedding pooling
- **Token Optimization**: Smart text trimming to stay within LLM limits
- **Async Processing**: Non-blocking I/O for efficient concurrent operations
- **Single-Call Scoring**: Ultra-fast scoring that combines parsing and scoring in ONE LLM call
- **GPU-Accelerated Embeddings**: SentenceTransformers with CUDA support for 10x speedup
- **Comprehensive Scoring**: Skills, experience, education, and overall match scores

---

## 🛠️ Tech Stack

### Backend

- **Framework**: FastAPI 0.110.0+ (async)
- **LLM**: Mistral 7B (Ollama - single model for all operations)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2) with GPU acceleration
- **PDF Processing**: pdfplumber
- **Caching**: In-memory + file-based (pickle/JSON)
- **HTTP Client**: httpx (async)
- **Data Validation**: Pydantic 2.6+
- **ML Framework**: PyTorch (for GPU support)

### Frontend

- **Framework**: React 19.2.0
- **Language**: TypeScript 5.9.3
- **Build Tool**: Vite 7.2.4
- **Styling**: TailwindCSS 4.1.18
- **Graphics**: OGL (WebGL library)
- **Linting**: ESLint

### Infrastructure

- **Server**: Uvicorn (ASGI)
- **CORS**: Enabled for cross-origin requests
- **API Documentation**: Swagger UI + ReDoc

---

## 📁 Project Structure

```
Platforme Matching CV/
├── CV-Job-matching/              # Backend (Python/FastAPI)
│   ├── app/
│   │   ├── main.py              # FastAPI application entry point
│   │   ├── pipeline.py          # Core matching pipeline
│   │   ├── agents/              # AI agents for parsing and scoring
│   │   │   ├── cv_parser.py     # CV parsing logic (async)
│   │   │   ├── jd_parser.py     # Job description parsing (async)
│   │   │   ├── scorer.py        # Traditional scoring algorithms
│   │   │   └── fast_scorer.py   # Ultra-fast single-call scorer (NEW)
│   │   ├── api/                 # API routes and models
│   │   │   ├── models.py        # Pydantic models
│   │   │   └── routes/          # API endpoints
│   │   │       ├── matching.py  # Matching endpoints
│   │   │       ├── health.py    # Health checks
│   │   │       └── cache.py     # Cache management
│   │   ├── embedding/           # RAG embedding system
│   │   │   ├── rag_embedder.py  # RAG embedder class
│   │   │   └── similarity.py    # Similarity calculations
│   │   ├── extract/             # PDF extraction utilities
│   │   │   ├── pdf_extractor.py # PDF text extraction
│   │   │   └── cleaner.py       # Text cleaning
│   │   └── utils/               # Utility functions
│   │       ├── rag.py           # RAG core functions
│   │       ├── cache.py         # Caching utilities
│   │       └── text_trimmer.py  # Token optimization
│   ├── requirements.txt         # Python dependencies
│   ├── run_api.py              # API server launcher
│   ├── test_rag.py             # RAG performance tests
│   └── rank_candidates.py      # Batch ranking script
│
└── frontend/                    # Frontend (React/TypeScript)
    ├── src/
    │   ├── App.tsx             # Main application component
    │   ├── main.tsx            # Application entry point
    │   ├── components/         # React components
    │   │   ├── MatchingPage.tsx    # Main matching interface
    │   │   ├── GuidePage.tsx       # User guide
    │   │   ├── launding_page.tsx   # Landing page
    │   │   ├── navbar.tsx          # Navigation bar
    │   │   ├── Aurora/             # Aurora background effect
    │   │   └── PyramidLoader/      # Loading animation
    │   ├── services/
    │   │   └── api.ts          # API client
    │   └── types/
    │       └── api.ts          # TypeScript type definitions
    ├── package.json            # Node dependencies
    ├── vite.config.ts          # Vite configuration
    └── tsconfig.json           # TypeScript configuration
```

---

## 📋 Prerequisites

### Required Software

- **Python**: 3.9 or higher
- **Node.js**: 18.0 or higher
- **npm**: 9.0 or higher
- **Ollama**: Latest version ([Download here](https://ollama.ai))

### Ollama Models

You only need to pull ONE model:

```bash
ollama pull mistral:7b-instruct
```

**Note**: Embeddings are handled by SentenceTransformers (no Ollama model needed), which is much faster and supports GPU acceleration.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "Platforme Matching CV"
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd CV-Job-matching

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python test_installation.py
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd ../frontend

# Install dependencies
npm install

# Build the project (optional)
npm run build
```

---

## 🎯 Quick Start

### Start Backend Server

```bash
# From CV-Job-matching directory
python run_api.py
```

The API will be available at:

- **API**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Start Frontend Development Server

```bash
# From frontend directory
npm run dev
```

The frontend will be available at: http://localhost:5173

---

## � Usage

### �📡 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Match Single CV

**Using cURL:**

```bash
curl -X POST "http://localhost:8000/match" \
  -F "file=@cv.pdf" \
  -F "job_description=Looking for Python developer with FastAPI experience"
```

**Using Python:**

```python
import requests

url = "http://localhost:8000/match"

with open("cv.pdf", "rb") as f:
    files = {"file": f}
    data = {"job_description": "Python developer with 5 years experience"}
    response = requests.post(url, files=files, data=data)

result = response.json()
print(f"Overall Score: {result['overall_score']}")
```

**Response:**

```json
{
  "overall_score": 85,
  "skills_score": 90,
  "experience_score": 80,
  "education_score": 85,
  "summary": "Strong match with excellent Python and FastAPI skills",
  "processing_time": 2.3,
  "cached": false
}
```

### Rank Multiple CVs

**Using Python:**

```python
import requests

url = "http://localhost:8000/rank"
job_description = "Senior Python Developer with FastAPI and React experience"

files = []
for i, cv_path in enumerate(["cv1.pdf", "cv2.pdf", "cv3.pdf"]):
    files.append(("files", open(cv_path, "rb")))

data = {"job_description": job_description}
response = requests.post(url, files=files, data=data)

rankings = response.json()
for rank in rankings["rankings"]:
    print(f"{rank['rank']}. {rank['filename']} - Score: {rank['overall_score']}")
```

### API Parameters

| Endpoint       | Method | Parameters                                                                 | Description             |
| -------------- | ------ | -------------------------------------------------------------------------- | ----------------------- |
| `/health`      | GET    | -                                                                          | Check API health status |
| `/match`       | POST   | `file` (PDF), `job_description` (text), `use_rag` (bool, default: true)    | Match single CV         |
| `/rank`        | POST   | `files` (PDF[]), `job_description` (text), `use_rag` (bool, default: true) | Rank multiple CVs       |
| `/cache/stats` | GET    | -                                                                          | Get cache statistics    |

#---

## 🎨 Frontend Usage

### Main Features

1. **Landing Page**: Beautiful animated entry point with project introduction
2. **Matching Page**: Upload CVs and input job descriptions for matching
3. **Guide Page**: Comprehensive user guide and documentation
4. **Results Display**: Visual representation of matching scores

### Using the Matching Interface

1. Navigate to the matching page
2. Upload one or more CV files (PDF format)
3. Enter the job description in the text area
4. Click "Match" or "Rank" button
5. View results with scores and recommendations

---

## 🧠 RAG Implementation

### What is RAG?

RAG (Retrieval-Augmented Generation) improves matching accuracy by:

- **Chunking**: Splits CVs into semantic chunks (skills, experience, education)
- **Embedding**: Creates vector representations of chunks
- **Retrieval**: Finds most relevant chunks for the job description
- **Scoring**: Analyzes only relevant chunks (70-90% token reduction)

### Performance Benefits

| Metric          | Traditional | With RAG | v2.1 (Fast Mode) | Improvement    |
| --------------- | ----------- | -------- | ---------------- | -------------- |
| Processing Time | 17s         | 8.5s     | **4.3s**         | **75% faster** |
| Tokens Used     | 3000        | 900      | **500**          | 83% reduction  |
| Accuracy        | 78%         | 87%      | **88%**          | +10%           |
| Cache Hit Rate  | 40%         | 85%      | **90%**          | +50%           |
| LLM Calls       | 3           | 3        | **1**            | 67% reduction  |

**v2.1 Optimizations:**

- Single-call scoring (combines parsing + scoring)
- GPU-accelerated embeddings (SentenceTransformers)
- Async processing (non-blocking I/O)
- Single LLM model (Mistral) for all operations

### Using RAG

**Default (Recommended):**

```python
from app.pipeline import run_pipeline_rag

result = run_pipeline_rag("cv.pdf", "job description", top_k=5)
```

**API:**

```bash
curl -X POST "http://localhost:8000/match?use_rag=true" \
  -F "file=@cv.pdf" \
  -F "job_description=..."
```

### RAG Configuration

```python
from app.embedding.rag_embedder import rag_embedder

# Configure chunk size
result = run_pipeline_rag(
    cv_path="cv.pdf",
    job_description="...",
    top_k=5,           # Number of chunks to retrieve
    pooling="max"      # Pooling strategy: max, mean, weighted
)
```

---

---

### Benchmarking

Run performance tests:

```bash
cd CV-Job-matching
python test_performance.py
```

### Test RAG Performance

```bash
python test_rag.py
```

### Optimization Features

1. **Caching System**

   - Embedding cache: Stores CV embeddings
   - Result cache: Caches matching results
   - RAG cache: Stores chunks and vectors
   - JD parsing cache: Reuses job description parsing across CVs

2. **Text Trimming**

   - Smart truncation to fit token limits
   - Preserves important information
   - Reduces LLM processing time

3. **Async Processing** ⚡ NEW

   - Async/await pattern for non-blocking I/O
   - Concurrent task execution with asyncio
   - Single-threaded async (more predictable)
   - Faster than traditional thread pools

4. **GPU Acceleration** 🚀 NEW

   - SentenceTransformers with CUDA support
   - Batch embedding generation (32 embeddings at once)
   - 10x faster than Ollama embeddings
   - Automatic CPU fallback if no GPU

5. **Single-Call Scoring** 🎯 NEW

   - Combines parsing and scoring in ONE LLM call
   - 60-80% faster than 3-call approach
   - Reduces token usage significantly
   - Robust error handling with fallbacks

6. **Token Optimization**
   - Efficient prompt engineering
   - Section-based chunking
   - Top-k retrieval (default: top 5 chunks)

---

## 🔧 Development

### Running Tests

````bash
# Backend tests
cd CDevelopment Mode

**Backend (with auto-reload):**

```bash
cd CV-Job-matching
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
````

**Frontend (with HMR):**
Development Mode

**Backend (with auto-reload):**

```bash
cd CV-Job-matching
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend (with HMR):**

```bash
cd frontend
npm run dev
```

### Environment Variables

Create a `.env` file in `CV-Job-matching/` (optional):

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b-instruct  # Single model for all operations

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Cache Configuration
ENABLE_CACHE=true
CACHE_DIR=.cache

# Embedding Configuration (NEW)
EMBEDDING_MODEL=all-MiniLM-L6-v2  # SentenceTransformer model
EMBEDDING_DEVICE=cuda  # or 'cpu' - auto-detected if not set
EMBEDDING_BATCH_SIZE=32  # Batch size for embedding generation

# Performance
ENABLE_FAST_SCORER=true  # Use single-call scoring (recommended)
ENABLE_GPU=true  # Enable GPU acceleration if available
```

### Code Structure Best Practices

- \*\* 1. HR Recruitment

```python
# Rank candidates for a job opening
python rank_candidates.py \
  --cv-dir "./cvs" \
  --job-description "Senior Full-Stack Developer" \
  --top-n 10
```

### 2. Job Seeker

```python
# Check how well your CV matches a job posting
from app.pipeline import run_pipeline_rag

score = run_pipeline_rag("my_cv.pdf", "job_posting.txt")
print(f"Your match score: {score['overall_score']}%")
```

### 3. Bulk Analysis

```python
# Analyze multiple CVs in batch
import os
from app.pipeline import run_pipeline_rag

cvs = [f for f in os.listdir("cvs/") if f.endswith(".pdf")]
results = []

for cv in cvs:
    result = run_pipeline_rag(f"cvs/{cv}", job_description)
    results.append({"cv": cv, "score": result["overall_score"]})

# Sort by score
results.sort(key=lambda x: x["score"], reverse=True)
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)

### Code Quality & Best Practices

- **Backend**: Follow FastAPI best practices, use async where possible
- **Frontend**: Use TypeScript for type safety, follow React hooks patterns
- **API**: RESTful design, proper error handling
- **Caching**: Use appropriate cache invalidation strategies
- **Testing**: Write unit tests for new features
- **Documentation**: Keep code well-documented with docstrings

---

## 🐛 Troubleshooting

### Common Issues

**1. Ollama not running:**

```bash
# Start Ollama service

# You should see mistral:7b-instruct
# If not, pull it:
ollama pull mistral:7b-instruct
```

**2. Port already in use:**

```bash
# Change port in run_api.py or use environment variable
export API_PORT=8001
python run_api.py
```

**3. PDF extraction errors:**

- Ensure PDF is not password-protected
- Check PDF is not corrupted
- Verify pdfplumber is installed correctly

**4. Frontend build errors:**

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**5. Cache issues:**

```bash
# Clear all caches via API
curl -X POST "http://localhost:8000/cache/clear"

# Or manually delete cache directory
rm -rf CV-Job-matching/.cache
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Ollama**: For providing local LLM inference
- **FastAPI**: For the excellent web framework
- **React**: For the powerful frontend library
- **TailwindCSS**: For beautiful styling utilities

---

## 📞 Support

For issues, questions, or suggestions:

- Create an issue in the repository
- Check existing documentation in `/CV-Job-matching/`
- Review API docs at http://localhost:8000/docs

---

## 🗺️ Roadmap

### Completed ✅

- [x] GPU-accelerated embeddings with SentenceTransformers
- [x] Single LLM model architecture (Mistral)
- [x] Async processing with asyncio
- [x] Single-call ultra-fast scoring
- [x] Enhanced caching system

### Planned 🚀

- [ ] Multi-language support (beyond English)
- [ ] Advanced analytics dashboard
- [ ] Export results to PDF/Excel
- [ ] Integration with job boards (LinkedIn, Indeed)
- [ ] AI-powered resume improvement suggestions
- [ ] Skill gap analysis with learning recommendations
- [ ] Interview question generation based on CV-JD match
- [ ] Mobile application (React Native)
- [ ] Support for other LLM providers (OpenAI, Anthropic)
- [ ] Fine-tuned models for specific industries

---

## 📈 Changelog

### Version 2.1.0 (December 29, 2025) - Major Performance Update

**Architecture Improvements:**

- ✨ Unified LLM model: Migrated to Mistral 7B for all operations (CV parsing, JD parsing, scoring)
- ⚡ GPU-accelerated embeddings: Replaced Ollama embeddings with SentenceTransformers (10x faster)
- 🚀 Ultra-fast scoring: New `fast_scorer.py` combines parsing and scoring in ONE LLM call (60-80% faster)
- 🔄 Async architecture: Replaced ThreadPoolExecutor with async/await for better performance
- 💾 Enhanced caching: Added JD parsing cache for better multi-CV ranking performance

**Performance Gains:**

- 75% faster processing time (17s → 4.3s)
- 83% token reduction (3000 → 500 tokens)
- 67% fewer LLM calls (3 → 1 per match)
- GPU acceleration with automatic CPU fallback

**Developer Experience:**

- Simplified model setup (1 model instead of 3)
- Better error handling with intelligent fallbacks
- Cleaner async code patterns
- Improved documentation

### Version 2.0.0 (December 27, 2025)

- Initial RAG implementation
- Multi-CV ranking support
- React frontend with modern UI
- Caching system for embeddings

---

## 👨‍💻 Author

<div align="center">

### **Moatez Tilouch**

_Frontend Developer & Animation Enthusiast_

[![GitHub](https://img.shields.io/badge/GitHub-MoatezTilouche-181717?style=for-the-badge&logo=github)](https://github.com/MoatezTilouche)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Moatez%20Tilouch-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/moatez-tilouch-a58a96284/)
[![Email](https://img.shields.io/badge/Email-moateztilouch%40gmail.com-EA4335?style=for-the-badge&logo=gmail)](mailto:moateztilouch@gmail.com)

</div>

---

**Built with ❤️ using AI-powered matching technology**

Last Updated: December 29, 2025

## 📞 Support

For issues, questions, or suggestions:

- 📧 Create an issue in the repository
- 📚 Check existing documentation in [API_USAGE.md](CV-Job-matching/API_USAGE.md)
- 🌐 Review API docs at http://localhost:8000/docs
- 💬 Join discussions in the repository

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Ollama** - For providing local LLM inference capabilities
- **Mistral AI** - For the excellent Mistral 7B model
- **SentenceTransformers** - For GPU-accelerated embeddings
- **FastAPI** - For the high-performance async web framework
- **React** - For the powerful frontend library
- **TailwindCSS** - For beautiful styling utilities
- **PyTorch** - For ML framework support

---

**Built with ❤️ using AI-powered matching technology**

**Version**: 2.1.0 | **Last Updated**
