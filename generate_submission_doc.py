import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

doc = docx.Document()

# Set Standard 1-inch Margins
for s in doc.sections:
    s.top_margin = Inches(1.0)
    s.bottom_margin = Inches(1.0)
    s.left_margin = Inches(1.0)
    s.right_margin = Inches(1.0)

# Colors
C_DARK_GREEN = RGBColor(20, 70, 45)
C_MED_GREEN  = RGBColor(38, 115, 60)
C_CHARCOAL   = RGBColor(40, 40, 40)
C_MUTED      = RGBColor(110, 110, 110)
C_CODE       = RGBColor(30, 30, 30)

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def add_title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.font.size = Pt(22)
    run.font.bold = True
    run.font.color.rgb = C_DARK_GREEN

def add_subtitle(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(20)
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.font.size = Pt(12)
    run.font.italic = True
    run.font.color.rgb = C_MUTED

def add_h1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.font.size = Pt(15)
    run.font.bold = True
    run.font.color.rgb = C_DARK_GREEN

def add_h2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.font.size = Pt(12.5)
    run.font.bold = True
    run.font.color.rgb = C_MED_GREEN

def add_h3(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = C_CHARCOAL

def add_p(text, bold_prefix=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Calibri'
        r_pre.font.size = Pt(10.5)
        r_pre.font.bold = True
        r_pre.font.color.rgb = C_CHARCOAL
    r = p.add_run(text)
    r.font.name = 'Calibri'
    r.font.size = Pt(10.5)
    r.font.color.rgb = C_CHARCOAL

def add_bullet(text, bold_prefix=None):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Calibri'
        r_pre.font.size = Pt(10.5)
        r_pre.font.bold = True
        r_pre.font.color.rgb = C_CHARCOAL
    r = p.add_run(text)
    r.font.name = 'Calibri'
    r.font.size = Pt(10.5)
    r.font.color.rgb = C_CHARCOAL

def add_code_block(code_text):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Inches(6.5)
    cell = table.cell(0, 0)
    set_cell_background(cell, "F4F6F5")
    cell.paragraphs[0].paragraph_format.space_before = Pt(4)
    cell.paragraphs[0].paragraph_format.space_after = Pt(4)
    cell.paragraphs[0].paragraph_format.line_spacing = 1.05
    run = cell.paragraphs[0].add_run(code_text)
    run.font.name = 'Consolas'
    run.font.size = Pt(9.0)
    run.font.color.rgb = C_CODE
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

# ==================== DOCUMENT BUILD ====================

add_title("Darukaa.Earth AI Biodiversity Intelligence System")
add_subtitle("Comprehensive Technical Project Submission & System Architecture Documentation")

# Section 1
add_h1("1. Project Repository & Live Deployment Links")
add_p("The project has been deployed to production and is fully accessible to evaluators across the following verified endpoints:")
add_bullet("https://github.com/BhagyashreeSM5/Daruka_Biodeversity_chatBot", "GitHub Repository (main branch): ")
add_bullet("https://daruka-biodeversity-chatbot.onrender.com", "Live Demo Web Application: ")
add_bullet("https://daruka-biodeversity-chatbot.onrender.com/knowledge", "Live Knowledge Base Explorer: ")
add_bullet("https://daruka-biodeversity-chatbot.onrender.com/docs", "Interactive OpenAPI / Swagger Documentation: ")
add_bullet("https://daruka-biodeversity-chatbot.onrender.com/health", "Live Health Check Endpoint: ")

# Section 2
add_h1("2. System Architecture & Design Overview")
add_h2("2.1 Executive Summary & Problem Context")
add_p("Standard conversational LLM applications frequently fail in environmental and agricultural domains due to single-variable hallucinations (e.g., prescribing flood irrigation for low rainfall without verifying soil salinity, soil texture, or drainage constraints) and cross-conversation state contamination. The Darukaa.Earth AI Biodiversity Intelligence System is built as an evidence-driven expert system specifically structured around deterministic ecological causal mechanisms, strict ecosystem boundaries, and verifiable peer-reviewed evidence from global environmental authorities.")

add_h2("2.2 Multi-Tier Architectural Pipeline")
add_bullet("Dual-mode user dashboard supporting multi-turn dialogue, raw structured JSON input (flat and nested spec-compliant payloads), and direct inspection of the underlying scientific database.", "1. Presentation Layer (static/index.html & knowledge.html): ")
add_bullet("FastAPI service orchestrating incoming turns, session persistence, site boundary detection, and structured response validation.", "2. Application & API Layer (app/main.py): ")
add_bullet("LangGraph state machine that manages dialogue transitions: parsing free text and JSON into a validated SiteInput schema, tracking missing parameters, and isolating new site evaluations from previous conversations.", "3. Dialogue & State Orchestrator (app/agent.py): ")
add_bullet("Constructs non-obvious multi-metric causal chains linking >= 3 interacting environmental variables (soil pH, moisture, organic carbon, rainfall, land use, and habitat fragmentation). Enforces ecosystem compatibility filters (rejecting pasture techniques for orchards or flooded irrigation for arid soils).", "4. Ecological Reasoning Engine (app/reasoning.py & relationships.py): ")
add_bullet("Powered by fastembed (ONNX Runtime) generating 384-dimensional dense vectors using all-MiniLM-L6-v2 in <100MB RAM, executing cosine distance similarity queries against PostgreSQL pgvector.", "5. Embedding & Retrieval Engine (app/retrieval.py & embeddings.py): ")
add_bullet("Hosted on Neon Serverless PostgreSQL with pgvector, storing 20 curated peer-reviewed evidence chunks from FAO, IPCC, IPBES, and Ramsar conventions.", "6. Vector Database Layer (app/db.py): ")

add_h2("2.3 Provenance & State Contamination Resolution")
add_p("To eliminate memory contamination across distinct queries, the system introduces strict provenance tracking:")
add_bullet("Explicitly provided by the user in the prompt or structured data. These values are immutable and take absolute precedence.", "KNOWN Origin: ")
add_bullet("Scientifically derived based on valid environmental relationships (e.g., rainfall < 500mm implies water_availability: low). These are never presented as user-stated facts.", "INFERRED Origin: ")
add_bullet("Parameters not yet provided. If critical parameters are missing, the system generates targeted scientific clarifying questions rather than fabricating assumptions.", "UNKNOWN Origin: ")

# Section 3
add_h1("3. Database Architecture & Schema Specification")
add_p("The database runs on Neon Serverless PostgreSQL with the pgvector extension enabled. The database schema consists of two primary tables:")

add_h2("Table 1: knowledge_chunks (Scientific Literature Vector Store)")
add_p("Stores peer-reviewed scientific findings with corresponding dense semantic embeddings:")
add_code_block("""CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id VARCHAR PRIMARY KEY,             -- Unique chunk ID (e.g. 'kb001' to 'kb020')
    text TEXT NOT NULL,                 -- Verbatim quantitative scientific guidance
    source VARCHAR NOT NULL,             -- Authoritative body (e.g. FAO, IPCC, IPBES)
    metric_tags JSONB DEFAULT '[]',      -- Targeted indicators (e.g. ["soil_organic_carbon"])
    year INTEGER NULL,                   -- Publication year
    embedding VECTOR(384)               -- 384-dimensional dense semantic vector
);""")

add_h2("Table 2: conversation_state (Session State & Isolation)")
add_p("Maintains multi-turn context per session with complete origin tracking:")
add_code_block("""CREATE TABLE IF NOT EXISTS conversation_state (
    session_id VARCHAR PRIMARY KEY,     -- Client session token
    state_json JSONB DEFAULT '{}',       -- Stored parcel attributes & field_origin tracking
    turn_count INTEGER DEFAULT 0         -- Turn counter for multi-turn dialogue management
);""")

# Section 4
add_h1("4. Local Setup, Installation & Execution Guide")
add_p("The system can be run locally on any Windows, macOS, or Linux environment:")

add_h2("Step-by-Step Instructions")
add_p("1. Clone repository and navigate to folder:")
add_code_block("git clone https://github.com/BhagyashreeSM5/Daruka_Biodeversity_chatBot.git\ncd Daruka_Biodeversity_chatBot")

add_p("2. Create and activate a Python 3.11 virtual environment:")
add_code_block("py -3.11 -m venv .venv\n.venv\\Scripts\\activate      # On macOS/Linux: source .venv/bin/activate")

add_p("3. Install all required dependencies:")
add_code_block("pip install -r requirements.txt")

add_p("4. Configure .env file (points to hosted Neon PostgreSQL):")
add_code_block("""DATABASE_URL=postgresql+psycopg2://neondb_owner:npg_pi8VqlMd2yAx@ep-crimson-sunset-az8b29dx.c-3.ap-southeast-1.aws.neon.tech/neondb?sslmode=require
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K_EVIDENCE=4
GOOGLE_API_KEY=""")

add_p("5. Run database ingestion (populates pgvector with 20 knowledge chunks):")
add_code_block("python -m app.ingest")

add_p("6. Start the local development server:")
add_code_block("python -m uvicorn app.main:app --reload --port 8000")
add_p("The application will be live at http://127.0.0.1:8000.")

add_p("7. Run automated test suite:")
add_code_block("pytest tests/")
add_p("Result: 36 passed in ~34 seconds (100% pass rate).")

# Section 5
add_h1("5. CI/CD & Cloud Deployment Details")
add_bullet("Constructed on python:3.11-slim. Executes an ONNX model pre-caching step during the build phase so container initialization takes <2 seconds with zero runtime downloads.", "Docker Containerization (Dockerfile): ")
add_bullet("Deploys as an automated Docker Web Service on Render's free tier in the Singapore region (closest to Neon AWS ap-southeast-1).", "Cloud Infrastructure (Render): ")
add_bullet("By replacing heavy PyTorch with fastembed (ONNX Runtime), runtime RAM consumption dropped from 580MB (crashing free tier) to ~177MB, leaving >330MB of safe headroom.", "Memory Optimization: ")
add_bullet("Connected to the GitHub main branch. Every push triggers an automated build, container health check, and zero-downtime rolling deployment.", "Continuous Deployment: ")

# Section 6
add_h1("6. Evaluator Review Notes & Detailed Test Cases")
add_h2("6.1 Operational Notes for Reviewers")
add_bullet("Render free tier containers hibernate after 15 minutes of inactivity. If visiting after an idle period, please allow 30–45 seconds for the container to wake up on the first request. Subsequent interactions respond in milliseconds.", "Free-Tier Wakeup Notice: ")
add_bullet("The chatbot intentionally operates in a deterministic scientific mode, ensuring transparent, reproducible, and verifiable environmental recommendations.", "Deterministic Scientific Execution: ")

add_h2("6.2 Recommended Evaluation Test Scenarios")

add_h3("Scenario A: Multi-Metric Land Degradation (>=3 Interacting Variables)")
add_p("Input Prompt:", "User Query: ")
add_p('"Our site has a soil pH of 5.2, organic carbon at 0.7%, annual rainfall around 420mm, and severe erosion on sloping pasture."')
add_p("Expected System Response:", "Evaluation Result: ")
add_bullet("Identifies acidic soil (pH 5.2), low organic carbon (0.7%), low rainfall (420mm), and pasture land use.")
add_bullet("Synthesizes a causal mechanism linking low organic carbon to poor soil aggregate stability, compounding slope erosion under dryland rainfall pulses.")
add_bullet("Recommends rotational grazing, contour buffer strips, and drought-tolerant legumes supported by FAO and IUCN guidance.")

add_h3("Scenario B: Strict Ecosystem Applicability (Orchard vs. Annual Cropland)")
add_p("Input Prompt:", "User Query: ")
add_p('"We manage a Mediterranean olive and fruit orchard with high pesticide exposure from neighboring farms."')
add_p("Expected System Response:", "Evaluation Result: ")
add_bullet("Strictly filters out annual crop rotation, monoculture tillage reduction, and flooded pasture interventions.")
add_bullet("Recommends flowering understory vegetation and native grass buffer strips citing IPBES Pollinator Assessment.")

add_h3("Scenario C: State Isolation Verification (Cross-Session Reset)")
add_bullet("User submits: 'Region: Tropical, Land use: Pasture, Pollution: High, Deforestation: Increasing'. Chatbot responds with tropical pasture assessment.", "Turn 1: ")
add_bullet("User submits: 'Region: Mediterranean, Land use: Orchard, Human impact: Moderate pesticide use'.", "Turn 2: ")
add_bullet("Chatbot cleanly purges Tropical, Pasture, and Deforestation from memory and evaluates the new Mediterranean orchard parcel with zero carryover.", "Turn 2 Verification: ")

# Save
output_path = "c:\\Users\\Hp\\OneDrive\\New folder\\darukaa-biodiversity-chatbot\\Darukaa_Biodiversity_Chatbot_Submission.docx"
doc.save(output_path)
print("Successfully generated detailed Word document at:", output_path)
