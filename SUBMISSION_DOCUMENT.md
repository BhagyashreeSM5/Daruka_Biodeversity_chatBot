# Darukaa.Earth AI Biodiversity Intelligence System — Project Submission Document

---

# SECTION 1: PROJECT REPOSITORY & LIVE DEPLOYMENT LINKS

### Primary Project Links
* **GitHub Repository (Source Code):**  
  [https://github.com/BhagyashreeSM5/Daruka_Biodeversity_chatBot](https://github.com/BhagyashreeSM5/Daruka_Biodeversity_chatBot)  
  *Branch:* `main` (Fully synchronized with all scientific reasoning, memory isolation, and lightweight embedding updates).

* **Live Interactive Application (Web UI):**  
  [https://daruka-biodeversity-chatbot.onrender.com](https://daruka-biodeversity-chatbot.onrender.com)  
  *Access:* Open publicly with no login required. Connects directly to the cloud vector database.

* **Live Knowledge Base Explorer:**  
  [https://daruka-biodeversity-chatbot.onrender.com/knowledge](https://daruka-biodeversity-chatbot.onrender.com/knowledge)  
  *Function:* Enables evaluators to inspect the underlying curated scientific evidence chunks, attribution sources, and metric tags.

* **Interactive API Documentation (Swagger / OpenAPI):**  
  [https://daruka-biodeversity-chatbot.onrender.com/docs](https://daruka-biodeversity-chatbot.onrender.com/docs)  
  *Endpoints:* Full documentation for `/chat`, `/chat/structured`, `/health`, `/api/knowledge`, and `/session/{id}/reset`.

* **Production Health Check Endpoint:**  
  [https://daruka-biodeversity-chatbot.onrender.com/health](https://daruka-biodeversity-chatbot.onrender.com/health)  
  *Payload:* `{"status": "ok"}`

---

# SECTION 2: SYSTEM ARCHITECTURE & DESIGN OVERVIEW

### 2.1 Executive Summary
The **Darukaa.Earth AI Biodiversity Intelligence System** is an evidence-driven environmental intelligence assistant designed to deliver rigorous, parcel-specific ecological recommendations. Standard conversational models frequently suffer from "generic hallucination," single-variable myopia (e.g., prescribing irrigation purely for low rainfall without checking soil drainage or salinity), and cross-conversation memory contamination. 

To overcome these failure modes, this system is engineered with:
1. **Multi-Metric Ecological Reasoning:** Generates non-obvious recommendations derived from causal chains connecting $\ge 3$ interacting environmental variables.
2. **Strict Ecosystem Applicability:** Validates interventions against land-use and climate boundaries (e.g., pasture $\ne$ cropland; orchard $\ne$ annual rotation; runoff $\ne$ pesticide drift).
3. **Deterministic Scientific Grounding:** Enforces that every recommendation cites retrieved, source-attributed peer-reviewed evidence (FAO, IPCC, IPBES, Ramsar).
4. **Isolated Conversational Memory:** Prevents state contamination across independent site queries while maintaining provenance tracking (`KNOWN`, `INFERRED`, `UNKNOWN`).

```
                              ┌────────────────────────────────────────┐
                              │            Client Layer                │
                              │  - Web Browser UI (static/index.html)  │
                              │  - REST API / External Systems         │
                              └───────────────────┬────────────────────┘
                                                  │ HTTP POST /chat
                                                  ▼
                              ┌────────────────────────────────────────┐
                              │        FastAPI Application Layer       │
                              │             (app/main.py)              │
                              └───────────────────┬────────────────────┘
                                                  │
                                                  ▼
                              ┌────────────────────────────────────────┐
                              │    LangGraph State Orchestrator        │
                              │             (app/agent.py)             │
                              ├────────────────────────────────────────┤
                              │ 1. Parse & Normalize Input             │
                              │ 2. Detect Site Shift / Continuation    │
                              │ 3. Validate Environmental Variables    │
                              │ 4. Route: Missing Info vs. Reasoning   │
                              └───────┬────────────────────────┬───────┘
                                      │                        │
               Needs Missing Fields   │                        │ Has Minimum Viable Metrics
                                      ▼                        ▼
                      ┌──────────────────────┐  ┌────────────────────────────────────────┐
                      │ Clarification Engine │  │      Ecological Reasoning Engine       │
                      │  - Targeted queries  │  │         (app/reasoning.py)             │
                      │  - Scientific why    │  ├────────────────────────────────────────┤
                      └──────────────────────┘  │ • Multi-metric causal chains (>=3 var) │
                                                │ • Ecosystem applicability filters      │
                                                │ • Trade-off & risk synthesis           │
                                                └──────────────────┬─────────────────────┘
                                                                   │
                                                                   ▼
                                                ┌────────────────────────────────────────┐
                                                │    pgvector Semantic Retrieval Engine  │
                                                │    (app/retrieval.py + embeddings.py)  │
                                                ├────────────────────────────────────────┤
                                                │ • fastembed ONNX (all-MiniLM-L6-v2)    │
                                                │ • Cosine similarity search (top-k=15)  │
                                                │ • Neon Cloud PostgreSQL Vector DB      │
                                                └────────────────────────────────────────┘
```

### 2.2 Core Modules Breakdown

1. **State Isolation & Input Parsing (`app/parsing.py`, `app/models.py`):**
   * Normalizes both conversational natural language and structured JSON (flat and nested) into a unified `SiteInput` model.
   * Employs heuristic and structural shift detection: when a user introduces a new parcel with distinct ecosystem markers, the system immediately resets the prior site context unless the user explicitly indicates continuation.
   * Tracks origin provenance for every environmental metric via `field_origin`:
     - `KNOWN`: Explicitly provided by the user.
     - `INFERRED`: Derived scientifically (e.g., rainfall < 500mm $\rightarrow$ `water_availability: low`).
     - `UNKNOWN`: Not yet provided; never hallucinated as a fact.

2. **Multi-Metric Ecological Reasoning Engine (`app/reasoning.py`, `app/relationships.py`):**
   * Implements `_build_ecological_mechanism()` to link interacting metrics into an explicit causal mechanism:
     $$\text{Initial Variables} \longrightarrow \text{Biogeochemical Mechanism} \longrightarrow \text{Target Ecological Outcomes}$$
   * **Rule Constraint:** Input variables (e.g., rainfall, land use) are strictly excluded from `metrics_impacted`, focusing exclusively on true ecosystem endpoints (e.g., `microbial_diversity`, `soil_organic_carbon_retention`, `canopy_structural_diversity`).
   * Evaluates mutual environmental constraints (e.g., high erosion risk on slopes under low organic carbon and intense rainfall).

3. **Vector Retrieval & Lightweight Embeddings (`app/embeddings.py`, `app/retrieval.py`):**
   * Evaluates semantic distance between synthesized ecological queries and peer-reviewed literature chunks stored in `pgvector`.
   * Built on **`fastembed`** (ONNX Runtime) using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). This delivers high semantic precision while reducing RAM overhead to **<100MB**, allowing seamless deployment on free-tier containers without PyTorch or external API billing.

---

# SECTION 3: DATABASE ARCHITECTURE & SCHEMA SPECIFICATION

### 3.1 Database Technology
* **Database Engine:** PostgreSQL 16 (Hosted on Neon Serverless).
* **Extensions:** `pgvector` enabled for vector similarity operations.
* **Driver & ORM:** `psycopg2-binary` with SQLAlchemy 2.0.

### 3.2 Relational & Vector Schema

```sql
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Table 1: Curated Scientific Knowledge Chunks
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id VARCHAR PRIMARY KEY,             -- Unique ID, e.g. "kb001", "kb002"
    text TEXT NOT NULL,                 -- Scientific guidance text from peer-reviewed source
    source VARCHAR NOT NULL,             -- Authoritative body (e.g. FAO, IPCC, IPBES)
    metric_tags JSONB DEFAULT '[]',      -- Impacted domains (e.g. ["soil_organic_carbon"])
    year INTEGER NULL,                   -- Publication year (e.g. 2021)
    embedding VECTOR(384)               -- 384-dimensional dense semantic vector
);

-- Table 2: Conversational Session State
CREATE TABLE IF NOT EXISTS conversation_state (
    session_id VARCHAR PRIMARY KEY,     -- Client session token
    state_json JSONB DEFAULT '{}',       -- Stored parcel attributes & field_origin tracking
    turn_count INTEGER DEFAULT 0         -- Turn counter for multi-turn dialogue management
);

-- Cosine Distance Retrieval Index (Auto-managed)
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding 
ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 10);
```

### 3.3 Schema Field Definitions

| Table | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| `knowledge_chunks` | `id` | `VARCHAR` | Primary key representing the unique chunk identifier (e.g., `kb001` to `kb020`). |
| | `text` | `TEXT` | Verbatim text describing quantitative, peer-reviewed conservation interventions. |
| | `source` | `VARCHAR` | Attribution reference (e.g., *FAO Conservation Agriculture*, *IPCC AR6 WG3*). |
| | `metric_tags` | `JSONB` | Array of ecological indicators impacted by the intervention. |
| | `year` | `INTEGER` | Year of publication. |
| | `embedding` | `VECTOR(384)` | Dense vector embedding generated by `all-MiniLM-L6-v2`. |
| `conversation_state` | `session_id` | `VARCHAR` | Client identifier for multi-turn conversational isolation. |
| | `state_json` | `JSONB` | Current site data: `soil_ph`, `organic_carbon`, `land_use`, plus `field_origin`. |
| | `turn_count` | `INTEGER` | Monotonic counter tracking turns in the current dialogue session. |

---

# SECTION 4: LOCAL SETUP, INSTALLATION & EXECUTION GUIDE

### 4.1 Prerequisites
* Python 3.11 or higher
* Git
* Access to PostgreSQL with `pgvector` (or use the pre-configured live Neon connection)

### 4.2 Step-by-Step Installation Commands

#### Step 1: Clone Repository
```bash
git clone https://github.com/BhagyashreeSM5/Daruka_Biodeversity_chatBot.git
cd Daruka_Biodeversity_chatBot
```

#### Step 2: Set Up Virtual Environment
```bash
# Using Python 3.11 launcher (Windows):
py -3.11 -m venv .venv
.venv\Scripts\activate

# On macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate
```

#### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 4: Configure Environment Variables
Create a `.env` file in the project root (a sample `.env.example` is provided):
```ini
# Neon Cloud Hosted PostgreSQL with pgvector (Active & Accessible)
DATABASE_URL=postgresql+psycopg2://neondb_owner:npg_pi8VqlMd2yAx@ep-crimson-sunset-az8b29dx.c-3.ap-southeast-1.aws.neon.tech/neondb?sslmode=require

# Lightweight embedding model (no API key needed)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Optional Google API key (Leave blank for deterministic scientific mode)
GOOGLE_API_KEY=

# Retrieval threshold
TOP_K_EVIDENCE=4
```

#### Step 5: Ingest Knowledge Base into Vector DB
```bash
python -m app.ingest
```
*Expected Output:* `Ingested 20 knowledge chunks into pgvector.`

#### Step 6: Start Local Development Server
```bash
python -m uvicorn app.main:app --reload --port 8000
```
Open your browser to: **`http://127.0.0.1:8000`**

#### Step 7: Run Automated Verification Test Suite
```bash
pytest tests/
```
*Verification Result:* **36 passed in ~34s (100% pass rate)**. Tests validate memory isolation, multi-metric reasoning, ecological mechanisms, applicability filters, and output formatting.

---

# SECTION 5: CI/CD PIPELINE & CLOUD HOSTING INFRASTRUCTURE

### 5.1 Infrastructure as Code (`render.yaml`)
Deployment is automated using a declarative Render Blueprint:
```yaml
services:
  - type: web
    name: daruka-biodeversity-chatbot
    runtime: docker
    plan: free
    region: singapore
    healthCheckPath: /health
    envVars:
      - key: DATABASE_URL
        sync: false
      - key: GOOGLE_API_KEY
        sync: false
      - key: EMBEDDING_MODEL
        value: all-MiniLM-L6-v2
      - key: TOP_K_EVIDENCE
        value: "4"
```

### 5.2 Containerization Strategy (`Dockerfile`)
* **Base Image:** `python:3.11-slim` for minimal image footprint.
* **Build-Time Model Pre-Caching:**  
  The Dockerfile executes an ONNX pre-caching step during build:
  ```dockerfile
  RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding('sentence-transformers/all-MiniLM-L6-v2').embed(['warmup']))"
  ```
  *Benefit:* Avoids downloading model weights at runtime, guaranteeing that the container starts in under 2 seconds.
* **Memory Optimization:**  
  Total runtime consumption is **~177 MB RSS**, eliminating Out-of-Memory (OOM) risks on free-tier containers with a 512 MB ceiling.

### 5.3 Continuous Deployment Workflow
1. Developer pushes commits to the `main` branch of `BhagyashreeSM5/Daruka_Biodeversity_chatBot`.
2. Render detects the push via GitHub Webhook.
3. Render pulls the latest commit, builds the multi-stage Docker image, runs the health check at `/health`, and switches traffic with zero downtime.

---

# SECTION 6: EVALUATOR REVIEW NOTES & TEST SCENARIOS

### 6.1 Reviewer Operational Notes
1. **Free-Tier Sleep Wake-Up:** The live demo is hosted on Render’s free tier. If the service has been idle for $>15$ minutes, the container spins down. On your first visit, **please allow 30–45 seconds** for the container to wake up. All subsequent queries will respond instantaneously.
2. **Deterministic & Scientific Mode:** The system intentionally runs in deterministic mode without relying on external generative LLM APIs to ensure consistent, auditable, and reproducible scientific outputs.

---

### 6.2 Recommended Evaluation Test Cases

#### Test Case 1: Multi-Metric Land Degradation
* **User Input:**  
  `"Our site has a soil pH of 5.2, organic carbon at 0.7%, annual rainfall around 420mm, and severe erosion on sloping pasture."`
* **Expected System Behavior:**
  * **Site Assessment:** Correctly tags soil pH as acidic (5.2), organic carbon as low (0.7%), rainfall as low (420mm), and land use as pasture.
  * **Ecological Mechanism:** Demonstrates how low organic carbon reduces soil aggregate stability, compounding erosion on sloping land during sparse but intense precipitation events.
  * **Recommendations:** Proposes rotational grazing management and contour buffer strips of native drought-tolerant grasses citing FAO and IUCN guidelines.

#### Test Case 2: Strict Ecosystem Applicability (Orchard vs. Cropland)
* **User Input:**  
  `"We manage a Mediterranean olive and fruit orchard with high pesticide exposure from neighboring plots."`
* **Expected System Behavior:**
  * **Filtering:** Explicitly rejects recommendations for annual crop rotation, monoculture tillage reduction, or flooded pasture buffers.
  * **Targeted Interventions:** Prescribes flowering understory cover and native shrub buffer strips to mitigate pesticide drift and protect native pollinators (grounded in IPBES Pollinator Assessment).

#### Test Case 3: Memory Contamination Isolation
* **Step 1 Input:**  
  `"Region: Tropical, Land use: Pasture, Pollution: High, Deforestation: Increasing"`
* **Step 2 Input (New Site):**  
  `"Region: Mediterranean, Land use: Orchard, Human impact: Moderate pesticide use"`
* **Expected System Behavior:**
  * The system identifies that a new independent site profile has been provided.
  * Tropical climate, pasture land use, and deforestation trends from Step 1 are **completely discarded**.
  * The new assessment addresses solely the Mediterranean orchard without carrying over the prior site's parameters.
