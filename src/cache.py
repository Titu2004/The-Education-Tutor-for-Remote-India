from __future__ import annotations
import json, os, hashlib, time

CACHE_PATH = "vector_store/answer_cache.json"


def _load() -> dict:
    """Load cache from file, handling missing or corrupted files gracefully."""
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache file: {e}")
            return {}
    return {}


def _save(cache: dict) -> None:
    """Save cache to file, creating directory if needed."""
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save cache file: {e}")


def _key(question: str, pdf_name: str) -> str:
    return hashlib.md5(f"{pdf_name}::{question.strip().lower()}".encode()).hexdigest()


def get(question: str, pdf_name: str):
    return _load().get(_key(question, pdf_name))


def set(question: str, pdf_name: str, answer: str, metrics: dict) -> None:
    cache = _load()
    cache[_key(question, pdf_name)] = {
        "answer": answer, "metrics": metrics,
        "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save(cache)


def stats() -> dict:
    return {"total_entries": len(_load())}


def clear() -> None:
    """Clear all cached answers."""
    try:
        _save({})
    except Exception as e:
        print(f"Error clearing cache: {e}")
        raise