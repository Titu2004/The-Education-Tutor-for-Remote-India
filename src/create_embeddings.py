from sentence_transformers import SentenceTransformer
from load_pdfs import load_all_pdfs

import faiss
import numpy as np
import pickle

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(documents):

    all_chunks = []
    all_vectors = []

    for name, text in documents.items():

        print(f"Creating embeddings for {name}...")

        chunks = text.split("\n")

        vectors = model.encode(chunks)

        for chunk, vector in zip(chunks, vectors):
            all_chunks.append(chunk)
            all_vectors.append(vector)

    return all_chunks, np.array(all_vectors)


def create_vector_db(vectors):

    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(vectors)

    return index


if __name__ == "__main__":

    docs = load_all_pdfs()

    chunks, vectors = create_embeddings(docs)

    index = create_vector_db(vectors)

    import os
    os.makedirs("vector_store", exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, "vector_store/textbook_vectors.index")

    # Save chunks
    with open("vector_store/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("\nVector database created successfully")
    print("Total chunks stored:", len(chunks))

    