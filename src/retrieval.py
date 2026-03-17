import faiss
import numpy as np
import streamlit as st
import time
from sentence_transformers import SentenceTransformer
from context_compression import compress_chunks
from LLM import generate_response


@st.cache_resource
def load_embedding_model():
    """Load embedding model once and cache it."""
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()


def create_vector_store(chunks):

    embeddings = model.encode(chunks).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def retrieve_chunks(question, index, chunks, k=5):

    query_vector = model.encode([question]).astype("float32")

    D, I = index.search(query_vector, k)

    retrieved = [chunks[i] for i in I[0]]

    return retrieved


def generate_answer(question, index, chunks):
    timing = {}
    total_start = time.time()
    
    # Step 1: Retrieve chunks
    retrieval_start = time.time()
    retrieved_chunks = retrieve_chunks(question, index, chunks)
    timing["retrieval"] = time.time() - retrieval_start
    
    # Step 2: Compress/filter chunks
    compression_start = time.time()
    context = compress_chunks(question, retrieved_chunks)
    timing["compression"] = time.time() - compression_start

    prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

    # Step 3: Generate response from LLM
    llm_start = time.time()
    answer = generate_response(prompt)
    timing["llm"] = time.time() - llm_start
    
    timing["total"] = time.time() - total_start
    
    return answer, timing

    