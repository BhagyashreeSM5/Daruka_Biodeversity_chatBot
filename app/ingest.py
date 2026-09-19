"""
Run with: python -m app.ingest

Reads data/knowledge_base.json, embeds each chunk locally (sentence-transformers),
and upserts into the pgvector-backed knowledge_chunks table.
"""

import json
from pathlib import Path

from app.db import init_db, SessionLocal, KnowledgeChunk
from app.embeddings import embed_batch

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"


def run_ingest() -> None:
    init_db()
    with open(DATA_PATH) as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    vectors = embed_batch(texts)

    session = SessionLocal()
    try:
        for chunk, vector in zip(chunks, vectors):
            existing = session.get(KnowledgeChunk, chunk["id"])
            if existing:
                existing.text = chunk["text"]
                existing.source = chunk["source"]
                existing.metric_tags = chunk["metric_tags"]
                existing.year = chunk.get("year")
                existing.embedding = vector
            else:
                session.add(
                    KnowledgeChunk(
                        id=chunk["id"],
                        text=chunk["text"],
                        source=chunk["source"],
                        metric_tags=chunk["metric_tags"],
                        year=chunk.get("year"),
                        embedding=vector,
                    )
                )
        session.commit()
        print(f"Ingested {len(chunks)} knowledge chunks into pgvector.")
    finally:
        session.close()


if __name__ == "__main__":
    run_ingest()
