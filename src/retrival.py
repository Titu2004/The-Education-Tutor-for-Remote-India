from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("vector_store/textbook_vectors.index")

# Load stored chunks
with open("vector_store/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


def retrieve_chunks(question, k=3):

    # Convert question to embedding
    query_vector = model.encode([question])

    # Search vector database
    distances, indices = index.search(np.array(query_vector), k)

    retrieved = []

    for i in indices[0]:
        retrieved.append(chunks[i])

    return retrieved