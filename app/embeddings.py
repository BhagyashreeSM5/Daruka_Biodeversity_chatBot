"""
Embedding module using fastembed (ONNX-based, ultra-lightweight).

Uses sentence-transformers/all-MiniLM-L6-v2 (384-dimensional).
Runs via ONNX Runtime without PyTorch, using only ~80-100MB RAM.
This prevents Out-Of-Memory crashes on Render Free Tier (512MB RAM cap)
and requires zero external API keys.
"""

from functools import lru_cache
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_model():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=MODEL_NAME)


def embed_text(text: str) -> list[float]:
    model = get_model()
    # model.embed returns a generator of numpy ndarrays
    vector = next(model.embed([text]))
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_model()
    vectors = list(model.embed(texts))
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    return embed_text(text)


def get_embedding_dim() -> int:
    return EMBEDDING_DIM
