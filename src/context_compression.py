from __future__ import annotations


def compress_chunks(question: str, chunks: list, max_chunks: int = 3, return_count: bool = False):
    """
    Select most relevant chunks by keyword overlap with the question.
    If return_count=True, returns (context_str, n_chunks_used).
    """
    def get_text(chunk) -> str:
        return chunk["content"] if isinstance(chunk, dict) else chunk

    keywords = set(w for w in question.lower().split() if len(w) > 2)
    scored   = []

    for chunk in chunks:
        text  = get_text(chunk)
        score = sum(1 for kw in keywords if kw in text.lower())
        if score > 0:
            scored.append((score, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_texts = [t for _, t in scored[:max_chunks]]

    if not top_texts:
        top_texts = [get_text(c) for c in chunks[:max_chunks]]

    context = "\n\n".join(top_texts)
    return (context, len(top_texts)) if return_count else context