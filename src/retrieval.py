from __future__ import annotations

import os, pickle, time
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

try:
    from src.context_compression import compress_chunks
    from src.LLM import generate_response
except ModuleNotFoundError:
    from context_compression import compress_chunks
    from LLM import generate_response

STORE_DIR   = "vector_store"
INDEX_PATH  = os.path.join(STORE_DIR, "textbook_vectors.index")
CHUNKS_PATH = os.path.join(STORE_DIR, "chunks.pkl")

NOT_IN_BOOK_MARKER  = "CONTENT_NOT_IN_BOOK"

# Lowered threshold — MiniLM L2 distances mean cosine sims for relevant
# content typically fall in the 0.15–0.40 range. 0.10 lets real matches
# through; the LLM prompt is the true out-of-book guard.
RELEVANCE_THRESHOLD = 0.10


@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_vector_store() -> tuple:
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"Vector store not found at '{STORE_DIR}/'. "
            "Run `python -m src.create_embeddings` first."
        )
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def create_vector_store(chunks: list) -> faiss.IndexFlatL2:
    model  = load_embedding_model()
    texts  = [c["content"] if isinstance(c, dict) else c for c in chunks]
    embeds = model.encode(texts, show_progress_bar=False).astype("float32")
    index  = faiss.IndexFlatL2(embeds.shape[1])
    index.add(embeds)
    return index


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b  = a.flatten(), b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def _compute_accuracy(answer: str, context: str, model: SentenceTransformer) -> float:
    vecs = model.encode([answer, context])
    sim  = _cosine_similarity(vecs[0], vecs[1])
    return round(max(0.0, min(1.0, sim)) * 100, 1)


def retrieve_chunks(question: str, index, chunks: list, k: int = 5) -> tuple:
    model     = load_embedding_model()
    query_vec = model.encode([question]).astype("float32")
    D, I      = index.search(query_vec, k)
    retrieved = [chunks[i] for i in I[0] if i < len(chunks)]

    best_sim = 0.0
    if retrieved:
        top_text = retrieved[0]["content"] if isinstance(retrieved[0], dict) else retrieved[0]
        top_vec  = model.encode([top_text]).astype("float32")
        best_sim = _cosine_similarity(query_vec[0], top_vec[0])

    return retrieved, best_sim


def generate_answer(question: str, index, chunks: list, k: int = 5) -> tuple:
    model        = load_embedding_model()
    total_chunks = index.ntotal
    t0           = time.perf_counter()

    # 1. Retrieve
    t1 = time.perf_counter()
    retrieved, best_sim = retrieve_chunks(question, index, chunks, k=k)
    retrieval_ms = round((time.perf_counter() - t1) * 1000, 1)

    # 2. Hard out-of-book guard — only fire when similarity is extremely low
    if best_sim < RELEVANCE_THRESHOLD:
        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        return NOT_IN_BOOK_MARKER, {
            "retrieval_time_ms": retrieval_ms, "compression_time_ms": 0,
            "llm_time_ms": 0, "total_time_ms": total_ms,
            "chunks_scanned": total_chunks, "chunks_retrieved": len(retrieved),
            "chunks_after_compress": 0, "accuracy_pct": 0.0,
            "epochs": 1, "from_cache": False,
            "best_similarity": round(best_sim * 100, 1),
        }

    # 3. Compress
    t2 = time.perf_counter()
    context, used_chunks = compress_chunks(question, retrieved, return_count=True)
    compression_ms = round((time.perf_counter() - t2) * 1000, 1)

    # 4. LLM — prompt is the primary out-of-book gate
    prompt = f"""You are a strict textbook tutor for Indian school students.
Answer ONLY using the context provided below. Do not use any outside knowledge.
If the context does not contain enough information to answer, respond with exactly the word: {NOT_IN_BOOK_MARKER}

Context:
{context}

Question:
{question}

Answer:"""

    t3         = time.perf_counter()
    raw_answer = generate_response(prompt)
    llm_ms     = round((time.perf_counter() - t3) * 1000, 1)
    total_ms   = round((time.perf_counter() - t0) * 1000, 1)

    # 5. Check if LLM said out-of-book
    if raw_answer.strip() == NOT_IN_BOOK_MARKER or NOT_IN_BOOK_MARKER in raw_answer:
        return NOT_IN_BOOK_MARKER, {
            "retrieval_time_ms": retrieval_ms, "compression_time_ms": compression_ms,
            "llm_time_ms": llm_ms, "total_time_ms": total_ms,
            "chunks_scanned": total_chunks, "chunks_retrieved": len(retrieved),
            "chunks_after_compress": used_chunks, "accuracy_pct": 0.0,
            "epochs": 1, "from_cache": False,
            "best_similarity": round(best_sim * 100, 1),
        }

    accuracy = _compute_accuracy(raw_answer, context, model)

    return raw_answer.strip(), {
        "retrieval_time_ms": retrieval_ms, "compression_time_ms": compression_ms,
        "llm_time_ms": llm_ms, "total_time_ms": total_ms,
        "chunks_scanned": total_chunks, "chunks_retrieved": len(retrieved),
        "chunks_after_compress": used_chunks, "accuracy_pct": accuracy,
        "epochs": 1, "from_cache": False,
        "best_similarity": round(best_sim * 100, 1),
    }