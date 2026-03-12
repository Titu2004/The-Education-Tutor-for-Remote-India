import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from compression import compress_chunks

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# load vector database
index = faiss.read_index("vector_store/textbook_vectors.index")

# load chunks
with open("vector_store/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


def retrieve_chunks(question, k=5):

    query_vector = model.encode([question]).astype("float32")

    D, I = index.search(query_vector, k)

    retrieved = [chunks[i] for i in I[0]]

    return retrieved


def generate_answer(question):

    print("Retrieving chunks...")

    retrieved_chunks = retrieve_chunks(question)

    print("Compressing chunks...")

    context = compress_chunks(question, retrieved_chunks)

    print("Generating answer...")

    answer = f"""
Context:
{context}

Question:
{question}

Answer:
"""

    return answer

    