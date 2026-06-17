# InspectAI — Backend Agent & API

This is the core intelligence layer of the InspectAI platform. It hosts the agentic NL2SQL pipeline, the semantic validation engine, and the FastAPI server that powers the frontend.

## 🧠 Agent Architecture: Orchestrator-Generator

The system uses a split-brain architecture to maximize accuracy and reasoning:

1.  **The Orchestrator (Llama 3.1 8B)**: 
    - Decodes user intent and identifies required data.
    - Decides whether to query the **Inspection Workflow** (PostgreSQL) or **Form Answers** (AI Semantic Layer).
    - Retrieves relevant schema fragments (Table definitions, Column types).
    - Prepares a "Schema Hint" for the generator.

2.  **The Generator (DeepSeek Coder V2)**:
    - Receives the specialized schema hint.
    - Generates high-precision, optimized SQL.
    - Follows strictly defined domain rules (e.g., date-truncation, draft exclusion).

## 🛡️ Semantic Safety & Validation

To prevent hallucinations and "silent metric corruption," the backend uses a multi-layered validation system:

- **Semantics Registry (`core/semantics_registry.py`)**: The single source of truth for all domain rules (e.g., "completed" vs "conducted" inspections).
- **Business Glossary (`core/business_glossary.py`)**: Enforces specific SQL patterns. For example, it forces `SUM(po_amount)` for CAPEX conversion and blocks incorrect join paths.
- **SQL Validator (`core/validator.py`)**: A final gatekeeper that parses the generated SQL and checks it against security and business constraints before execution.

## 📊 Chart Advisor

The `chart_advisor.py` module analyzes the shape of the SQL result set to recommend the best Apache ECharts visualization. It uses a combination of heuristics (Cardinality, Data Types, Semantics) and optional LLM advice to ensure users see their data in the most meaningful format.

## 🚀 API Endpoints

- `POST /api/chat`: The main entry point. Accepts a message and returns an answer, SQL, reasoning steps, and a suggested chart configuration.
- `POST /api/chart`: Force-regenerate a chart with specific column overrides.
- `GET /api/columns/{session_id}`: Returns available columns and their semantic types (Category, Numeric, Date) for the last query.
- `POST /api/page`: Handles high-performance server-side pagination for large result sets.

## 🛠️ Development & Setup

### Requirements
- Python 3.13
- PostgreSQL
- Ollama (running `llama3.1` and `deepseek-coder-v2`)

### Setup
1. Create a virtual environment: `python -m venv venv`
2. Install dependencies: `pip install -r requirements.txt`
3. Configure `.env`:
   ```env
   DATABASE_URL=postgresql://user:pass@localhost/dbname
   OLLAMA_URL=http://localhost:11434
   USE_LLM_CHART_ADVISOR=true
   ```
4. Run the server: `uvicorn fastapi_app:app --port 8001`
