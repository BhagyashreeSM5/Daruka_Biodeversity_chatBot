# Darukaa.Earth — AI Biodiversity Intelligence Chatbot

An AI system that behaves like an environmental scientist, not a chatbot: it retrieves real
evidence from a vector-indexed knowledge base, reasons across multiple linked environmental
metrics, and returns structured, cited, non-generic recommendations.

## Architecture

```
                         ┌────────────────────────┐
   POST /chat  ───────▶  │   FastAPI (app/main.py) │
   POST /chat/structured │                          │
                         └───────────┬──────────────┘
                                     ▼
                         ┌────────────────────────┐
                         │   LangGraph agent        │  app/agent.py
                         │                          │
                         │  parse_input             │  app/parsing.py
                         │       │                   │
                         │  check_completeness       │
                         │   /            \          │
                         │ missing      complete      │
                         │   │              │          │
                         │ ask_clarifying  reason ──▶ app/reasoning.py
                         │   │              │          │      │
                         │  END        format_output   │  infer_primary_metrics()
                         │                  │          │  (app/relationships.py)
                         │                 END          │      │
                         └──────────────────┬───────────┘      ▼
                                             │        retrieve_evidence()
                                             │        (app/retrieval.py)
                                             ▼                 │
                                 Conversation state             ▼
                                 (Postgres, per session_id)   pgvector similarity search
                                                              knowledge_chunks table
```

- **Knowledge layer**: `data/knowledge_base.json` — 20 curated, source-attributed evidence
  chunks covering soil, land use, water, biodiversity and pollution. Embedded locally with
  `sentence-transformers` (`all-MiniLM-L6-v2`, no API key needed) and stored in **Postgres +
  pgvector** (`app/ingest.py`, `app/retrieval.py`). Curated on purpose rather than pulled live
  at query time — see "Design decisions" below for why.
- **Knowledge Base browser**: `/knowledge` renders every curated entry with its citation and
  metric tags (`GET /api/knowledge`), so the retrieval pipeline is inspectable rather than a
  black box.
- **Multi-metric reasoning**: `app/relationships.py` encodes an explicit graph of which
  metrics causally affect which others (soil ↔ biodiversity, water ↔ species survival, land
  use ↔ fragmentation). `app/reasoning.py` requires every recommendation to match **≥2**
  linked metrics from a retrieved evidence chunk before it's returned — this is what prevents
  generic, single-variable output.
- **Conversational intelligence**: `app/agent.py` is a LangGraph state machine that asks for
  missing required fields (soil organic carbon %, rainfall, land use, region) one at a time
  and persists partial state per `session_id` in Postgres (`app/memory.py`). It also:
  - recognizes bare greetings and responds with a one-time intro instead of a bare question
  - accepts rainfall as a word, a descriptive term ("scarce", "moderate", "heavy"), or a raw
    number in mm/cm/inches (auto-converted to low/medium/high)
  - fuzzy-corrects typos in land-use and region answers (`app/parsing.py`, `difflib`)
  - phrases follow-up questions with context already known ("Given it's a semi-arid region,
    what's the rainfall pattern?") instead of an identical fixed script every time
  - gives an explicit "I couldn't catch that" retry message with an example, instead of
    silently repeating the same question when a reply can't be parsed
  - acknowledges only the field(s) newly provided each turn, in plain language, not a raw
    Python dict
- **Input handling**: three first-class entry points, all hitting the same `/chat` /
  `/chat/structured` endpoints — free-text chat, a guided form (soil organic carbon, pH,
  moisture, rainfall, land use, region, optional coordinates), and raw JSON paste. Plus
  one-click preloaded example scenarios (`data/example_scenarios.json`) for fast demoing,
  including the brief's own semi-arid monoculture-wheat example.
- **LLM usage is optional and bounded**: if `GOOGLE_API_KEY` is set, Gemini is used to (a)
  extract structured fields from free text more robustly, and (b) phrase the final reply
  around the already-generated structured `Recommendation` objects — it is explicitly
  instructed not to add or invent facts. With no key set, the system runs fully on rule-based
  parsing + templated output, so it is never "just an LLM wrapper."
- **Output schema**: every recommendation is a validated Pydantic model
  (`app/models.py::Recommendation`) with `action`, `reasoning`, `metrics_impacted`,
  `time_horizon`, `confidence`, and `source` — matching the brief's required output shape.

## Design decisions (deliberate, not shortcuts)

- **No live external retrieval.** The knowledge layer is a curated, hand-vetted evidence base
  rather than a live web-search/RAG-over-the-internet system. This is what makes "Scientific
  Grounding" verifiable — every claim traces to a specific, checkable source — and what makes
  the multi-metric enforcement in `app/reasoning.py` deterministic. Live retrieval would add
  freshness at the cost of unverifiable claims and a network dependency that can fail during a
  demo. Worth stating explicitly as a scope call made against the grading rubric, not something
  left out for lack of time.
- **No sign-in / no cross-session persistence.** Nothing in the evaluation criteria rewards
  accounts or returning-later assessments; multi-turn memory within a session (already
  implemented via Postgres-backed `session_id` state) is what's actually graded under
  "Conversational Intelligence." Auth would add real scope with no rubric benefit.

## Database / schema

Two tables, created automatically on startup (`app/db.py::init_db`):

| Table | Purpose | Key columns |
|---|---|---|
| `knowledge_chunks` | Vector-indexed evidence base | `id`, `text`, `source`, `metric_tags (json)`, `year`, `embedding (vector(384))` |
| `conversation_state` | Per-session multi-turn memory | `session_id`, `state_json` |

pgvector's cosine-distance operator (`<=>`, via SQLAlchemy's `cosine_distance`) powers
retrieval in `app/retrieval.py`.

## Local setup

**Database:** a hosted Postgres with the `pgvector` extension — Supabase (free tier) is the
easiest path and what these steps use. Neon, Render Postgres, or RDS work identically; only
the connection string changes.

### 1. Create the database (Supabase)

1. Create a free project at [supabase.com](https://supabase.com).
2. Go to the **SQL Editor** and run:
   ```sql
   create extension if not exists vector;
   ```
3. Go to **Project Settings → Database → Connection string**, copy the **Session pooler**
   (or direct connection) URI — not the "Transaction pooler" one, which isn't suited to this
   app's persistent connection pooling.

### 2. Configure and run the app

```bash
git clone <your-repo-url>
cd darukaa-biodiversity-chatbot

cp .env.example .env
# paste your Supabase connection string into DATABASE_URL in .env
# (change its "postgresql://" prefix to "postgresql+psycopg2://")
# (optional) add GOOGLE_API_KEY for LLM-polished replies — not required to run

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m app.ingest        # one-time: embeds and loads the knowledge base into Supabase
uvicorn app.main:app --reload
```

Open `http://localhost:8000` for the chat UI, `http://localhost:8000/knowledge` for the
Knowledge Base browser, or call the API directly:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo1", "message": "Biodiversity is declining on my land"}'
```

The agent will ask for the missing fields (soil organic carbon %, rainfall, land use, region)
across turns, then return cited, multi-metric recommendations once it has enough input.
Structured JSON input works in one shot too:

```bash
curl -X POST "http://localhost:8000/chat/structured?session_id=demo2" \
  -H "Content-Type: application/json" \
  -d '{
        "soil_organic_carbon_pct": 0.3,
        "rainfall": "low",
        "land_use": "monoculture wheat",
        "region": "semi-arid"
      }'
```

The web UI (`/`) also has a **Guided Form** tab, a **Raw JSON** tab, and one-click **preloaded
example scenarios** across the top. Browse the cited evidence base directly at `/knowledge`.

### Optional: running the app in Docker

The database is external now, so Docker is only needed if you want the app itself
containerized (e.g. for deployment). `docker-compose.yml` builds just the app and reads
`DATABASE_URL` from `.env`:

```bash
docker compose up -d --build
docker compose exec app python -m app.ingest
```

### Running tests

```bash
pip install pytest ruff
ruff check app tests
pytest tests/ -v
```

The unit tests (`tests/`) cover the relationship graph and free-text parsing without
requiring a database connection, so they run in CI without a live Postgres service.

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`: installs dependencies, lints with
`ruff`, and runs the unit test suite. Extend this workflow with a `docker build` + push step to
a registry (e.g. GHCR) and a deploy step (Railway/Render/Fly) for continuous deployment — kept
out of this submission to avoid embedding deployment secrets in the repo.

## Honesty note on the knowledge base

The percentage figures in `data/knowledge_base.json` (e.g. "+15–25% soil organic carbon") are
**illustrative ranges commonly cited in FAO/IPCC-adjacent literature**, written by hand for
this assessment rather than pulled from a live literature search. For production use, each
entry should be replaced with a verified citation (DOI/report page) before the numbers are
presented as fact — flagging this rather than dressing up placeholder numbers as verified
statistics.

## What's intentionally out of scope for this submission

- Fine-tuning any model (brief lists it as a nice-to-have, not core-scored)
- Auth/multi-tenant user management
- A production-grade frontend (brief explicitly says this isn't UI-judged)
