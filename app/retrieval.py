import json
from pathlib import Path
import numpy as np
from sqlalchemy import select

from app.db import SessionLocal, KnowledgeChunk
from app.embeddings import embed_text, embed_batch
from app.config import settings
from app.models import EvidenceChunk

_LOCAL_KB_CACHE: list[EvidenceChunk] | None = None
_LOCAL_KB_EMBEDDINGS: np.ndarray | None = None


def _load_local_kb() -> tuple[list[EvidenceChunk], np.ndarray]:
    global _LOCAL_KB_CACHE, _LOCAL_KB_EMBEDDINGS
    if _LOCAL_KB_CACHE is None or _LOCAL_KB_EMBEDDINGS is None:
        kb_path = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"
        with open(kb_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        chunks = [
            EvidenceChunk(
                id=item["id"],
                text=item["text"],
                source=item["source"],
                metric_tags=item["metric_tags"],
                year=item.get("year"),
            )
            for item in raw_data
        ]
        embeddings = np.array(embed_batch([c.text for c in chunks]))
        _LOCAL_KB_CACHE = chunks
        _LOCAL_KB_EMBEDDINGS = embeddings
    return _LOCAL_KB_CACHE, _LOCAL_KB_EMBEDDINGS


def _local_retrieve_fallback(query_vector: list[float], top_k: int) -> list[EvidenceChunk]:
    chunks, embeddings = _load_local_kb()
    q_vec = np.array(query_vector)
    sims = np.dot(embeddings, q_vec)
    ranked_indices = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in ranked_indices]


def retrieve_evidence(query: str, top_k: int | None = None) -> list[EvidenceChunk]:
    """Cosine-similarity retrieval over the pgvector knowledge_chunks table.
    This is the explicit, inspectable retrieval step the brief asks for —
    every recommendation traces back to one of these rows via `source`."""
    top_k = top_k or settings.top_k_evidence
    query_vector = embed_text(query)

    try:
        session = SessionLocal()
        try:
            stmt = (
                select(KnowledgeChunk)
                .order_by(KnowledgeChunk.embedding.cosine_distance(query_vector))
                .limit(top_k)
            )
            rows = session.execute(stmt).scalars().all()
            if rows:
                return [
                    EvidenceChunk(
                        id=r.id,
                        text=r.text,
                        source=r.source,
                        metric_tags=r.metric_tags or [],
                        year=r.year,
                    )
                    for r in rows
                ]
        finally:
            session.close()
    except Exception:
        pass

    return _local_retrieve_fallback(query_vector, top_k)

