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

## 🗺️ Request flow

The package layout under `backend/src/` maps directly onto the stages a
question passes through:

```
                         ┌──────────────┐
   question ───────────► │    agent/    │  plan (Llama reasoning loop)
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │ retrieval/   │  find relevant schema + few-shot
                         │              │  examples (embeddings, reranking)
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │    sql/      │  generate ─► validate ─► execute
                         │              │  (DeepSeek, business rules, RBAC)
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │    agent/    │  synthesize natural-language answer
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │   charts/    │  recommend a visualization
                         └──────────────┘

   llm/       → shared provider abstraction, used by agent/ (charts/ is rules-based only)
   knowledge/ → domain facts (schema, glossary, status rules) consumed
                throughout the pipeline
   security/  → auth + RBAC scoping, enforced in agent/ and sql/validation/
```

Packages are named for **what they do**, not for who calls them or where the
code happened to be written first — there is no `core/` grab-bag. See
`docs/Backend_Directory_Structure.docx` for a full file-by-file breakdown of
every package.

## 📁 Directory structure

```
backend/src/
├── llm/                LLM provider abstraction (base, ollama, factory)
├── sql/                SQL generation, execution, validation, business rules,
│                        few-shot example corpus — SQL as a first-class capability
│   ├── validation/       SQLValidator (cleaner/shape/semantic/scope mixins)
│   ├── rules/             BLOCK/WARN/AUTOCORRECT/INJECT registry
│   └── examples/          NL->SQL seed corpus + SQL-generation prompt templates
├── retrieval/           Schema/example retrieval — embeddings, reranking
├── knowledge/           Domain knowledge: schema introspection, semantics
│                        registry, business glossary, question patterns
├── security/            Authentication + RBAC data-visibility scoping
├── charts/               Chart-type recommendation (peer capability to agent/)
│   └── advisor/
├── agent/                Orchestration only — plan/act/observe loop
│   ├── orchestrator/       AgentOrchestrator, session state, synthesis
│   ├── tools.py             simple query tools + dispatch registry
│   ├── tool_dispatcher.py   parses/validates LLM tool-call JSON
│   └── prompts/             reasoning-LLM ("Llama-side") system prompt
├── api/                  FastAPI app, routes, request/response glue
└── config/               Environment settings + logging setup
```

## 🛡️ Semantic Safety & Validation

To prevent hallucinations and "silent metric corruption," the backend uses a multi-layered validation system:

- **Semantics Registry (`backend/src/knowledge/semantics_registry.py`)**: The single source of truth for all domain rules (e.g., "completed" vs "conducted" inspections).
- **Business Glossary (`backend/src/knowledge/business_glossary.py`)**: Enforces specific SQL patterns. For example, it forces `SUM(po_amount)` for CAPEX conversion and blocks incorrect join paths.
- **SQL Validator (`backend/src/sql/validation/`)**: A final gatekeeper that parses the generated SQL and checks it against security and business constraints before execution.

## 📊 Chart Advisor

The `backend/src/charts/advisor/` package analyzes the shape of the SQL result set to recommend the best Apache ECharts visualization, using rules-based heuristics (cardinality, data types, semantics) — no LLM in the loop.

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

### Layout
The Python backend lives under `backend/src/` (a proper package, imported
as `src.*`). `run.py` at the repo root is a thin wrapper that forwards to
`backend/scripts/run.py`, kept so existing commands don't change. Tests live
in `backend/tests/`, the eval harness in `backend/eval/`. The Angular
frontend lives in `frontend/`.

### Setup
1. Create a virtual environment: `python -m venv venv`
2. Install dependencies: `pip install -r backend/requirements.txt`
3. Configure `.env` at the repo root:
   ```env
   DATABASE_URL=postgresql://user:pass@localhost/dbname
   OLLAMA_URL=http://localhost:11434
   ```
4. Run the server (from the repo root, so `.env` and the Angular build resolve correctly):
   `uvicorn src.api.app:app --app-dir backend --port 8000`
   (port 8000 matches `frontend/proxy.conf.json`, which `ng serve` uses to proxy `/api` calls to the backend)
5. Run the CLI: `python run.py`
6. Run tests: `python -m pytest backend/tests/` (or run a test file directly, e.g. `python backend/tests/core/test_business_rules_consumption.py`)
7. Or start both backend and frontend together: `backend\scripts\start_dev.bat` — starts the backend, waits for `/api/ready`, then starts `ng serve`.
