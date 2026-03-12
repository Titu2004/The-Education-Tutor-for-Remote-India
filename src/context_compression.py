def compress_chunks(question, chunks, max_chunks=3):
    """
    Selects the most relevant chunks based on keywords in the question
    and returns a compressed context.
    """

    keywords = question.lower().split()

    relevant_chunks = []

    for chunk in chunks:
        chunk_lower = chunk.lower()

        for word in keywords:
            if word in chunk_lower:
                relevant_chunks.append(chunk)
                break

    # limit context size
    compressed_context = " ".join(relevant_chunks[:max_chunks])

    return compressed_context