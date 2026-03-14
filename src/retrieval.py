import faiss
import numpy as np
import streamlit as st
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

    retrieved_chunks = retrieve_chunks(question, index, chunks)

    context = compress_chunks(question, retrieved_chunks)

    prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

    answer = generate_response(prompt)

    return answer

    