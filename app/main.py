import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.models import ChatTurn, ChatResponse, SiteInput
from app.memory import load_state, save_state, clear_state, increment_turn_count, load_and_increment_state
from app.agent import get_graph
from app.db import init_db
from app.embeddings import get_model

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

app = FastAPI(
    title="Darukaa.Earth Biodiversity Intelligence Chatbot",
    version="1.1.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def on_startup():
    init_db()
    # Pre-warm embedding model so first request doesn't experience ~5s cold-start
    get_model()


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/knowledge")
def knowledge_page():
    return FileResponse("static/knowledge.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/knowledge")
def api_knowledge():
    """Powers the Knowledge Base browser page — the curated, source-attributed
    evidence entries the retrieval/reasoning pipeline draws from. Served
    straight from the source file so it's inspectable even before ingestion
    into pgvector has run."""
    with open(DATA_DIR / "knowledge_base.json") as f:
        return json.load(f)


@app.get("/api/examples")
def api_examples():
    """Preloaded example scenarios for one-click demoing."""
    with open(DATA_DIR / "example_scenarios.json") as f:
        return json.load(f)


@app.post("/chat", response_model=ChatResponse)
def chat(turn: ChatTurn):
    """Multi-turn endpoint. Accepts free text and/or structured `data`.
    State persists per session_id across calls (Postgres-backed)."""
    if not turn.message and not turn.data:
        raise HTTPException(400, "Provide `message` and/or `data`.")

    existing, turn_count = load_and_increment_state(turn.session_id)

    graph = get_graph()
    result = graph.invoke(
        {
            "session_id": turn.session_id,
            "turn_count": turn_count,
            "message": turn.message,
            "incoming_data": turn.data.model_dump() if turn.data else None,
            "site_input": existing,
        }
    )

    save_state(turn.session_id, result["site_input"])

    return ChatResponse(
        session_id=turn.session_id,
        reply=result["reply"],
        status=result["status"],
        missing_fields=result.get("missing_fields", []),
        recommendations=result.get("recommendations", []),
    )


@app.post("/chat/structured", response_model=ChatResponse)
def chat_structured(session_id: str, data: SiteInput):
    """Pure structured-JSON entry point — also what the guided form and the
    raw-JSON-paste UI both submit to under the hood."""
    return chat(ChatTurn(session_id=session_id, message=None, data=data))


@app.post("/session/{session_id}/reset")
def reset_session(session_id: str):
    clear_state(session_id)
    return {"status": "reset", "session_id": session_id}
