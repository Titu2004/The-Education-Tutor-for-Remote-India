import re


def clean_text(text):

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    # remove page numbers
    text = re.sub(r'\n\d+\n', '\n', text)

    # remove special characters
    text = re.sub(r'[^a-zA-Z0-9.,!?;:\s]', '', text)

    return text


def chunk_text(text, chunk_size=500):

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def process_documents(documents):

    all_chunks = []

    for doc_name, text in documents.items():

        cleaned = clean_text(text)

        chunks = chunk_text(cleaned)

        for chunk in chunks:

            all_chunks.append({
                "source": doc_name,
                "content": chunk
            })

    return all_chunks