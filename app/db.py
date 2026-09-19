from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, JSON
from sqlalchemy.sql import expression

from app.config import settings

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output size

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_recycle=300,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    text: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    metric_tags: Mapped[list] = mapped_column(JSON, default=list)
    year: Mapped[int] = mapped_column(Integer, nullable=True)
    embedding = mapped_column(Vector(EMBEDDING_DIM))


class ConversationState(Base):
    __tablename__ = "conversation_state"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    state_json: Mapped[dict] = mapped_column(JSON, default=dict)
    turn_count: Mapped[int] = mapped_column(Integer, default=0, server_default=expression.text("0"))


def init_db() -> None:
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(
            text("ALTER TABLE IF EXISTS conversation_state ADD COLUMN IF NOT EXISTS turn_count INTEGER DEFAULT 0")
        )
        conn.commit()
    Base.metadata.create_all(engine)
