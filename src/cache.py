from __future__ import annotations
import json, os, hashlib, time

CACHE_PATH = "vector_store/answer_cache.json"


def _load() -> dict:
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


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