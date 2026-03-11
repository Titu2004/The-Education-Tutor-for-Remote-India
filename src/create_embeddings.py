from sentence_transformers import SentenceTransformer
from load_pdfs import load_all_pdfs

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(documents):
    embeddings = {}

    for name, text in documents.items():

        print(f"Creating embeddings for {name}...")

        # simple chunking
        chunks = text.split("\n")

        vectors = model.encode(chunks)

        embeddings[name] = list(zip(chunks, vectors))

    return embeddings


if __name__ == "__main__":

    docs = load_all_pdfs()

    emb = create_embeddings(docs)

    for doc, data in emb.items():
        print(f"\n{doc}")
        print("Total chunks:", len(data))