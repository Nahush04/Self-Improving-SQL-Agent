"""Local, free embeddings for memory text. CPU only, no paid vendor."""

from __future__ import annotations

from functools import lru_cache

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL_NAME, device="cpu")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings. Loads the model once per process, lazily."""
    if not texts:
        return []
    vectors = _model().encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def embed_one(text: str) -> list[float]:
    return embed_texts([text])[0]
