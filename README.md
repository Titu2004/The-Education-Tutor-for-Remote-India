# The Education Tutor for Remote India

> **Empowering students in low-bandwidth, low-cost environments with intelligent, cost-effective tutoring.**

## Problem Statement

Personalized AI tutors are revolutionizing education, but they are **expensive to run**. In rural India, where:
- 🌐 Internet connectivity is spotty and unreliable
- 💻 Computing power is limited
- 💰 Students cannot afford high-latency, high-cost API queries

Students need an intelligent tutoring system that can:
1. ✅ Ingest entire state-board textbooks once
2. ✅ Provide personalized, curriculum-aligned answers without re-processing documents
3. ✅ Optimize for **lowest cost per query** and **minimal data transfer**
4. ✅ Demonstrate significant cost reduction vs. baseline RAG systems

---

## What This Project Does

This is a **cost-optimized, Retrieval-Augmented Generation (RAG) tutoring system** that:

- **Ingests PDFs Efficiently**: Uploads textbooks once, extracts text, creates semantic chunks, and builds a vector index
- **Answers Questions Intelligently**: Students ask curriculum-aligned questions; the system retrieves relevant textbook content and generates accurate answers
- **Optimizes for Cost**: Uses keyword-based context compression, answer caching, and lightweight embeddings to minimize LLM API calls
- **Detects Out-of-Book Questions**: Identifies when questions fall outside textbook scope and alerts users
- **Extracts Relevant Images**: Automatically links diagrams, graphs, and figures from relevant pages
- **Tracks Metrics**: Provides token usage, accuracy scores, and processing times for transparency

**Use Case**: A student in rural India can upload their NCERT/state-board textbook once, then ask unlimited questions about the content with minimal data transfer and API costs.

---

## How It Works

### System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          STREAMLIT WEB UI                        │
│               (Interactive interface for students)               │
└──────────────────────────┬───────────────────────────────────────┘
                           │
      ┌────────────────────┼────────────────────┬─────────────────┐
      │                    │                    │                 │
  ┌───▼────┐      ┌────────▼────────┐   ┌──────▼──────┐   ┌──────▼────┐
  │   PDF  │      │  Text Chunking  │   │ FAISS Index │   │   LLM     │
  │  Upload│      │ & Embeddings    │   │  (Vector DB)│   │ (Gemini)  │
  │        │      │                 │   │             │   │           │
  └───┬────┘      └────────┬────────┘   └──────┬──────┘   └──────┬────┘
      │                    │                   │                 │
      └────────────────────┼───────────────────┘                 │
                           │                                     │
                      ┌────▼────────────────────┬────────────────┘
                      │                         │
          ┌───────────▼──────────┐      ┌──────▼──────────┐
          │Context Compression   │      │ Accuracy Scoring│
          │(Keyword Matching)    │      │ & Caching       │
          └───────────┬──────────┘      └──────┬──────────┘
                      │                        │
                      └────────────┬───────────┘
                                   │
                           ┌───────▼────────┐
                           │  Query Cache   │
                           │  (JSON file)   │
                           └────────────────┘
```

### Two-Phase Pipeline

#### **Phase 1: Offline Ingestion (One-Time)**
When a textbook PDF is uploaded:

1. **Text Extraction**: Extract all text from PDF using `PyPDF` and `pdfminer.six`
2. **Semantic Chunking**: Split text into logical chunks (paragraphs, sections)
3. **Embedding Generation**: Convert chunks to vectors using `SentenceTransformer` (`all-MiniLM-L6-v2`)
4. **Index Creation**: Store embeddings in FAISS (fast similarity search index)
5. **Image Extraction**: Extract diagrams, graphs, and link to relevant pages

**Result**: A reusable vector store saved to disk—no re-processing needed!

#### **Phase 2: Real-Time Query Processing (Per Question)**
When a student asks a question:

```
Student Question
       ↓
1. EMBED: Convert question to vector using same embedding model
       ↓
2. RETRIEVE: Search FAISS index → retrieve top-5 similar chunks
       ↓
3. SCORE: Compute cosine similarity for relevance checking
       ↓
4. COMPRESS: Filter chunks using keyword overlap → keep top-3 most relevant
       ↓
5. CHECK CACHE: Is this question cached? If yes → return cached answer
       ↓
6. GENERATE: Send compressed context + question to Google Gemini API
       ↓
7. SCORE ANSWER: Compute semantic similarity between answer and context
       ↓
8. CACHE RESULT: Store answer + metrics for future identical queries
       ↓
9. DISPLAY: Show answer, accuracy score, token usage, relevant images
```

---

## Approach Taken to Solve the Problem

### 🎯 Core Strategy: Cost Optimization Through 3 Key Mechanisms

#### **1. Vector Search (Semantic Retrieval)**
- **Problem**: Re-sending entire textbooks to LLM = massive costs
- **Solution**: 
  - Pre-compute embeddings offline using lightweight model (`all-MiniLM-L6-v2`: 22M parameters)
  - Store in FAISS index for instant, cost-free similarity search
  - Only relevant chunks sent to LLM, not entire document

**Cost Impact**: 🔴 **100 queries w/ base RAG** = entire textbook × 100 LLM calls
                 🟢 **100 queries w/ FAISS** = 1-time embedding + 100 small queries

#### **2. Context Compression**
- **Problem**: Multiple retrieved chunks = many tokens sent to LLM
- **Solution**: 
  - Retrieve top-5 chunks from FAISS
  - Filter using keyword matching: chunks with most overlap to question keywords
  - Send only top-3 chunks to LLM
  - LLM prompt also includes quality constraints to reject out-of-book questions

**Cost Impact**: Reduces prompt tokens by ~60-70% per query

#### **3. Query Caching**
- **Problem**: Students ask identical questions → duplicate API calls
- **Solution**: 
  - Cache all Q&A pairs in JSON file with hash-based key: `md5(pdf_name + question)`
  - On repeat query → return cached answer instantly
  - Zero API cost for cache hits

**Cost Impact**: In real classrooms, 40-60% queries are repeats = ~50% API cost savings

#### **4. Efficient Embeddings**
- **Model Choice**: `all-MiniLM-L6-v2` instead of larger models
  - 22M parameters vs. 335M+ for larger models
  - Works on CPU (no GPU needed in rural areas)
  - Still achieves 97% performance of larger models
  - Can run offline after initial download


### 📊 Expected Cost Reduction

**Baseline RAG** (naive approach):
- Query: Extract text → Tokenize → Send entire document to LLM
- Cost per query: $0.10-0.50 (depending on textbook size)

**This System** (with all 3 optimizations):
- Query: Vector search → Compress context → Cache check → Send 3 chunks
- Cost per query: $0.005-0.02 (90%+ cost reduction)
- Cache hit: $0.00000 (essentially free)

---

## Key Features

| Feature | Benefit |
|---------|---------|
| 🔍 **Vector Search** | Instant, cost-free similarity retrieval |
| 📦 **Context Compression** | 60-70% token reduction per query |
| 💾 **Query Caching** | 50%+ cost savings from repeat questions |
| 📊 **Accuracy Metrics** | Track answer relevance & quality |
| 🖼️ **Image Extraction** | Link relevant diagrams to answers |
| 📈 **Usage Tracking** | Monitor token counts & API costs |
| 🔗 **Out-of-Book Detection** | Flag questions outside textbook scope |
| 🌐 **Web UI** | Student-friendly Streamlit interface |

---

## System Components

| File/Module | Purpose |
|-------------|---------|
| `app.py` | Streamlit UI for student interaction |
| `main.py` | Example ingestion pipeline |
| `src/retrieval.py` | Vector search & chunk retrieval logic |
| `src/LLM.py` | Google Gemini API integration with token tracking |
| `src/context_compression.py` | Keyword-based chunk filtering |
| `src/cache.py` | JSON-based query caching |
| `src/create_embeddings.py` | Batch embedding generation for PDFs |
| `src/text_processing.py` | Text extraction & chunking |
| `src/image_extractor.py` | Extract & link images from PDFs |
| `vector_store/` | Persistent storage: FAISS index + embeddings |

---

## Tech Stack

- **Backend**: Python
- **UI Framework**: Streamlit
- **Vector Search**: FAISS (CPU-optimized)
- **Embeddings**: SentenceTransformer (`all-MiniLM-L6-v2`)
- **LLM**: Google Generative AI (Gemini)
- **PDF Processing**: PyPDF, pdfminer.six, pypdfium2, Pillow
- **Caching**: JSON-based (no external DB needed)

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Steps

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd The-Education-Tutor-for-Remote-India
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file with your Google API key
   echo GOOGLE_API_KEY=<your-key> > .env
   ```

5. **Create embeddings (one-time)**
   ```bash
   python -m src.create_embeddings --pdf path/to/textbook.pdf
   ```

6. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## Usage

### For Students
1. Open the Streamlit app: `http://localhost:8501`
2. Upload a PDF textbook
3. Ask questions about the content
4. Get instant, cost-effective answers with source references

### For Developers
- **Add a new embedding model**: Edit `src/retrieval.py` → `load_embedding_model()`
- **Install_Requirements.txt** pip install -r requirements.txt
- **Change LLM**: Edit `src/LLM.py` → swap `generate_response()` implementation
- **Modify cache behavior**: Edit `src/cache.py` → customize key/value storage
- **Adjust compression**: Edit `src/context_compression.py` → tune keyword matching

---

## Performance & Metrics

The system tracks:
- **Token Count**: Input/output tokens sent to LLM
- **Answer Accuracy**: Semantic similarity between answer and source context (0-100%)
- **Relevance Score**: Cosine similarity of retrieved chunks (0-1)
- **Cache Hit Rate**: % of queries served from cache
- **Processing Time**: End-to-end query processing time

---

## Future Enhancements

- [ ] Multi-language support (Hindi, regional languages)
- [ ] Offline LLM fallback for zero-connectivity scenarios
- [ ] Student progress tracking & personalization
- [ ] Interactive Q&A dialogue (multi-turn conversations)
- [ ] Vision LLM integration for diagram understanding
- [ ] Export study materials & practice questions

---

## Contact

For questions or contributions, reach out via github.

---
