import re


def clean_text(text: str) -> str:
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove standalone page numbers (e.g. "\n42\n")
    text = re.sub(r'(?<!\w)\d{1,4}(?!\w)', '', text)

    # Keep Unicode so Hindi/regional scripts survive (removed only junk chars)
    text = re.sub(r'[^\w\s.,!?;:\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F]', '', text)

    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into overlapping word-level chunks for better retrieval."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def process_documents(documents: dict) -> list:
    """Clean and chunk all documents, returning a list of chunk dicts."""
    all_chunks = []

    for doc_name, text in documents.items():
        cleaned = clean_text(text)
        chunks = chunk_text(cleaned)

        for chunk in chunks:
            all_chunks.append({
                "source": doc_name,
                "content": chunk,
            })

    return all_chunks
