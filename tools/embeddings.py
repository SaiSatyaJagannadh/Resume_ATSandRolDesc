"""Embeddings with an on-disk cache.

The ATS scorer runs twice per analysis (pre- and post-tailoring) over largely the
same keyword set, so the cache is what keeps the second run nearly free.
"""

import hashlib
import json

import numpy as np

import config

# Loaded lazily on first use; None means "not loaded yet".
_cache: dict[str, list[float]] | None = None


def _openai_embed(texts: list[str]) -> list[list[float]]:
    from langchain_openai import OpenAIEmbeddings

    config.require_embedding_key()
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL).embed_documents(texts)


# Module-level hook so tests can monkeypatch the network call away.
_embed_fn = _openai_embed


def _key(text: str) -> str:
    return hashlib.sha256(f"{config.EMBEDDING_MODEL}:{text}".encode()).hexdigest()


def _load_cache() -> dict[str, list[float]]:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(config.EMBEDDING_CACHE.read_text())
        except Exception:
            # Missing or corrupt cache is never fatal: start empty and rewrite.
            _cache = {}
    return _cache


def embed(texts: list[str]) -> list[list[float]]:
    cache = _load_cache()
    misses = list(dict.fromkeys(t for t in texts if _key(t) not in cache))
    if misses:
        for text, vec in zip(misses, _embed_fn(misses)):  # one API call for all misses
            cache[_key(text)] = list(vec)
        config.EMBEDDING_CACHE.parent.mkdir(parents=True, exist_ok=True)
        config.EMBEDDING_CACHE.write_text(json.dumps(cache))
    return [cache[_key(t)] for t in texts]


def cosine_sim(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def best_match(query: str, candidates: list[str]) -> tuple[str, float]:
    if not candidates:
        return "", 0.0
    vecs = embed([query] + candidates)
    sims = [cosine_sim(vecs[0], v) for v in vecs[1:]]
    i = int(np.argmax(sims))
    return candidates[i], sims[i]
