"""
NL2SQL Engine — LlamaIndex Version
====================================
Connects to your PostgreSQL database, generates SQL using sqlcoder:7b,
executes queries, and returns results.

Components:
  - SQLDatabase: connects to PostgreSQL, introspects schema
  - ObjectIndex: embedding-based table selection (picks 10-15 from 100)
  - Ollama: runs sqlcoder:7b locally on your GPU
  - Custom prompt: injects JSONB rules, examples, schema knowledge
  - Validator: catches hallucinated columns, fixes ->> arrows
"""

import os
import yaml
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import create_engine, text as sql_text

from llama_index.core import VectorStoreIndex, Settings

# LlamaIndex moves things between versions — try multiple paths
try:
    from llama_index.core import SQLDatabase
except ImportError:
    try:
        from llama_index.core.utilities.sql_database import SQLDatabase
    except ImportError:
        from llama_index.core.utilities import SQLDatabase

try:
    from llama_index.core.query_engine import NLSQLTableQueryEngine
except ImportError:
    from llama_index.core.indices.struct_store.sql_query import NLSQLTableQueryEngine

try:
    from llama_index.core.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
except ImportError:
    from llama_index.core.query_engine import SQLTableRetrieverQueryEngine

try:
    from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
except ImportError:
    from llama_index.core.objects.base import SQLTableNodeMapping, ObjectIndex, SQLTableSchema

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config.custom_prompts import TEXT_TO_SQL_PROMPT, RESPONSE_SYNTHESIS_PROMPT
from core.validator import SQLValidator
from core.templates import SQLTemplateBypass
from core.semantic import SemanticQuestionIndex
from core.router import IntentRouter


@dataclass
class QueryResult:
    """Result from the NL2SQL pipeline."""
    question: str
    sql: str
    raw_result: str
    answer: str
    tables_used: list[str]
    validation_passed: bool
    validation_errors: list[str]


class NL2SQLEngine:
    """
    The main engine. Connects LlamaIndex to your PostgreSQL via Ollama.
    
    Usage:
        engine = NL2SQLEngine.from_env()     # reads .env file
        result = engine.query("How many forms are there?")
        print(result.answer)
        print(result.sql)
    """

    def __init__(
        self,
        db_connection_string: str,
        db_schema: str = "public",
        ollama_url: str = "http://localhost:11434",
        sql_model: str = "sqlcoder:7b",
        chat_model: str = "mistral:latest",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        table_descriptions_path: str = "config/table_descriptions.yaml",
        top_k_tables: int = 15,
    ):
        self.validator = SQLValidator()
        self.top_k_tables = top_k_tables
        self.chat_model_name = chat_model
        self.ollama_url = ollama_url

        print("Setting up NL2SQL engine...")

        # --- Step 1: Connect to PostgreSQL ---
        print(f"  Connecting to database...")
        self.db_engine = create_engine(db_connection_string)
        # Test the connection
        with self.db_engine.connect() as conn:
            result = conn.execute(sql_text("SELECT 1"))
            result.fetchone()
        print(f"  ✓ Database connected")

        # --- Step 2: Create LlamaIndex SQLDatabase ---
        # This introspects all tables in the schema
        print(f"  Introspecting schema '{db_schema}'...")
        self.sql_database = SQLDatabase(
            self.db_engine,
            schema=db_schema,
        )
        all_tables = list(self.sql_database.get_usable_table_names())
        print(f"  ✓ Found {len(all_tables)} tables")

        # --- Step 3: Set up embedding model (CPU, ~130MB) ---
        print(f"  Loading embedding model: {embedding_model}...")
        embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        Settings.embed_model = embed_model
        print(f"  ✓ Embedding model loaded")

        # --- Step 4: Build semantic question index ---
        self.semantic_index = SemanticQuestionIndex(
            db_engine=self.db_engine,
            embed_model=embed_model,
        )
        self.templates = SQLTemplateBypass(semantic_index=self.semantic_index)

        # --- Step 5: Build intent router ---
        self.router = IntentRouter(embed_model=embed_model)

        # --- Step 6: Set up Ollama LLM for SQL generation ---
        print(f"  Configuring LLM: {sql_model} via Ollama...")
        self.sql_llm = Ollama(
            model=sql_model,
            base_url=ollama_url,
            temperature=0.0,
            request_timeout=120.0,
            additional_kwargs={
                "num_predict": 512,
                "top_p": 0.9,
            },
        )
        Settings.llm = self.sql_llm
        print(f"  ✓ LLM configured")

        # --- Step 5: Load table descriptions and build ObjectIndex ---
        print(f"  Building table retrieval index...")
        table_schemas = self._load_table_descriptions(
            table_descriptions_path, all_tables
        )

        table_node_mapping = SQLTableNodeMapping(self.sql_database)
        self.object_index = ObjectIndex.from_objects(
            table_schemas,
            table_node_mapping,
            VectorStoreIndex,
        )
        print(f"  ✓ Table index built ({len(table_schemas)} tables with descriptions)")

        # --- Step 6: Create the query engine ---
        print(f"  Creating query engine...")
        self.query_engine = SQLTableRetrieverQueryEngine(
            sql_database=self.sql_database,
            llm=self.sql_llm,
            table_retriever=self.object_index.as_retriever(
                similarity_top_k=self.top_k_tables,
            ),
            text_to_sql_prompt=TEXT_TO_SQL_PROMPT,
            sql_only=True,  # We handle response synthesis separately
        )
        print(f"  ✓ Query engine ready")
        print()
        print("=" * 50)
        print("  NL2SQL Engine is ready!")
        print(f"  Tables: {len(all_tables)}")
        print(f"  SQL model: {sql_model}")
        print(f"  Table retrieval: top-{self.top_k_tables}")
        print("=" * 50)

    @classmethod
    def from_env(cls, env_path: str = ".env") -> "NL2SQLEngine":
        """Create engine from .env file."""
        from dotenv import load_dotenv
        load_dotenv(env_path)

        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "postgres")
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD", "")
        db_schema = os.getenv("DB_SCHEMA", "public")

        connection_string = (
            f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

        return cls(
            db_connection_string=connection_string,
            db_schema=db_schema,
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            sql_model=os.getenv("OLLAMA_SQL_MODEL", "sqlcoder:7b"),
            chat_model=os.getenv("OLLAMA_CHAT_MODEL", "mistral:latest"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        )

    def _load_table_descriptions(
        self, path: str, available_tables: list[str]
    ) -> list[SQLTableSchema]:
        """Load table descriptions from YAML and create SQLTableSchema objects."""
        with open(path, "r") as f:
            descriptions = yaml.safe_load(f)

        schemas = []
        for table_name in available_tables:
            desc_entry = descriptions.get(table_name, {})
            if isinstance(desc_entry, dict):
                desc = desc_entry.get("description", f"Table: {table_name}")
            elif isinstance(desc_entry, str):
                desc = desc_entry
            else:
                desc = f"Table: {table_name}"

            schemas.append(
                SQLTableSchema(
                    table_name=table_name,
                    context_str=desc.strip(),
                )
            )

        return schemas

    def query(self, question: str) -> QueryResult:
        """
        Two-path query engine:
          Path 1 (SEMANTIC): content searches → embedding similarity → direct results
          Path 2 (STRUCTURAL): counts, listings, config → templates/LLM → SQL → execute
        """
        # --- Step 1: Route the question ---
        decision = self.router.route(question)

        # --- Path 1: SEMANTIC — no SQL needed ---
        if decision.path == "semantic" and self.semantic_index:
            return self._handle_semantic(question, decision)

        # --- Path 2: STRUCTURAL — needs SQL ---
        return self._handle_structural(question)

    def _handle_semantic(self, question: str, decision) -> QueryResult:
        """
        Content search: embed the question, find similar labels, return directly.
        No SQL, no database query, no LLM for generation. Instant.
        """
        matches = self.semantic_index.search(
            query=decision.search_term or question,
            entity_type=decision.entity_type if decision.entity_type != "QUESTION" else None,
            form_name=decision.form_name,
            top_k=10,
        )

        # Filter by a reasonable threshold
        good_matches = [m for m in matches if m.score >= 0.4]

        if not good_matches:
            return QueryResult(
                question=question,
                sql="[semantic] No SQL — direct embedding search",
                raw_result="No matching questions found.",
                answer=f"No questions found matching '{decision.search_term}'" + 
                       (f" in form '{decision.form_name}'" if decision.form_name else "") + ".",
                tables_used=[],
                validation_passed=True,
                validation_errors=[],
            )

        # Format results
        raw_lines = []
        for m in good_matches:
            raw_lines.append(f"Form: {m.form_name} | Type: {m.entity_type} | "
                           f"Label: {m.text} | Score: {m.score:.2f} | ID: {m.element_id}")
        raw_result = "\n".join(raw_lines)

        # Synthesize a natural answer
        try:
            answer = self._synthesize_answer(question, "[semantic search]", raw_result)
        except Exception:
            # Fallback: format manually
            answer_parts = []
            forms_seen = set()
            for m in good_matches:
                if m.score >= 0.5:  # Only include strong matches in the answer
                    answer_parts.append(f"- \"{m.text}\" (in form: {m.form_name}, score: {m.score:.0%})")
                    forms_seen.add(m.form_name)
            if answer_parts:
                answer = f"Found {len(answer_parts)} matching questions:\n" + "\n".join(answer_parts)
            else:
                answer = "No strong matches found."

        return QueryResult(
            question=question,
            sql=f"[semantic] search_term='{decision.search_term}' | form='{decision.form_name or 'all'}' | "
                f"confidence={decision.confidence:.2f} | matches={len(good_matches)}",
            raw_result=raw_result,
            answer=answer,
            tables_used=[],
            validation_passed=True,
            validation_errors=[],
        )

    def _handle_structural(self, question: str) -> QueryResult:
        """
        Structural query: try templates first, fall through to LLM.
        Generates SQL, executes, synthesizes answer.
        """
        # --- Try template bypass (instant, deterministic) ---
        template_sql = self.templates.try_match(question)
        if template_sql:
            sql = template_sql
            source = "template"
        else:
            # --- Fall through to LLM ---
            source = "llm"
            try:
                response = self.query_engine.query(question)
                raw_sql = str(response.metadata.get("sql_query", str(response)))
                sql = self.validator.clean_sql(raw_sql)
            except Exception as e:
                return QueryResult(
                    question=question,
                    sql=f"-- ERROR generating SQL: {e}",
                    raw_result="",
                    answer=f"Error generating SQL: {e}",
                    tables_used=[],
                    validation_passed=False,
                    validation_errors=[str(e)],
                )

        # Validate
        is_valid, errors = self.validator.validate(sql)

        # Execute if valid
        raw_result = ""
        if is_valid:
            try:
                with self.db_engine.connect() as conn:
                    result = conn.execute(sql_text(sql))
                    rows = result.fetchall()
                    columns = list(result.keys())
                    if rows:
                        raw_result = f"Columns: {columns}\n"
                        for row in rows[:50]:
                            raw_result += str(dict(zip(columns, row))) + "\n"
                    else:
                        raw_result = "No results found."
            except Exception as e:
                raw_result = f"Execution error: {e}"
                errors.append(f"EXECUTION ERROR: {e}")
                is_valid = False

        # Synthesize answer
        answer = raw_result
        if raw_result and not raw_result.startswith("Execution error"):
            try:
                answer = self._synthesize_answer(question, sql, raw_result)
            except Exception:
                answer = raw_result

        return QueryResult(
            question=question,
            sql=f"[{source}] {sql}",
            raw_result=raw_result,
            answer=answer,
            tables_used=[],
            validation_passed=is_valid,
            validation_errors=errors,
        )

    def query_sql_only(self, question: str) -> QueryResult:
        """Generate SQL or semantic results without executing. Safe mode."""

        # --- Route the question ---
        decision = self.router.route(question)

        if decision.path == "semantic" and self.semantic_index:
            # Show what semantic search would find
            matches = self.semantic_index.search(
                query=decision.search_term or question,
                form_name=decision.form_name,
                top_k=10,
            )
            good = [m for m in matches if m.score >= 0.4]
            result_lines = [f"  {m.score:.0%} | {m.form_name} | {m.text}" for m in good]
            return QueryResult(
                question=question,
                sql=f"[semantic] search='{decision.search_term}' form='{decision.form_name or 'all'}'",
                raw_result="\n".join(result_lines) if result_lines else "No matches",
                answer="",
                tables_used=[],
                validation_passed=True,
                validation_errors=[],
            )

        # --- Structural: try template ---
        template_sql = self.templates.try_match(question)
        if template_sql:
            is_valid, errors = self.validator.validate(template_sql)
            return QueryResult(
                question=question,
                sql=f"[template] {template_sql}",
                raw_result="(not executed — sql_only mode)",
                answer="",
                tables_used=[],
                validation_passed=is_valid,
                validation_errors=errors,
            )

        # --- Fall through to LLM ---
        try:
            response = self.query_engine.query(question)
            raw_sql = str(response.metadata.get("sql_query", str(response)))
        except Exception as e:
            return QueryResult(
                question=question,
                sql=f"-- ERROR: {e}",
                raw_result="",
                answer="",
                tables_used=[],
                validation_passed=False,
                validation_errors=[str(e)],
            )

        sql = self.validator.clean_sql(raw_sql)
        is_valid, errors = self.validator.validate(sql)

        return QueryResult(
            question=question,
            sql=f"[llm] {sql}",
            raw_result="(not executed — sql_only mode)",
            answer="",
            tables_used=[],
            validation_passed=is_valid,
            validation_errors=errors,
        )

    def _retry_with_feedback(self, question: str, bad_sql: str, errors: list[str]) -> Optional[str]:
        """
        When the first attempt fails validation, call the LLM directly
        with the error feedback and a very explicit correct pattern.
        """
        error_text = "\n".join(f"- {e}" for e in errors)

        retry_prompt = f"""### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Your previous SQL was WRONG:
{bad_sql}

### Errors found:
{error_text}

### MANDATORY FIX — use this exact pattern for any query needing labels/titles/names:
SELECT elem->>'translatedText' AS label
FROM fb_forms f
JOIN fb_translation_json tj ON f.translations_id = tj.id,
     jsonb_array_elements(tj.translations) AS elem
WHERE f.name ILIKE '%TEST%'
  AND elem->>'language' = 'eng'
  AND elem->>'attribute' = 'NAME'
  AND elem->>'entityType' = 'QUESTION'
LIMIT 100;

Replace '%TEST%' with the actual form name from the user's question.

RULES:
- NEVER use fb_question.label — it does not exist
- NEVER use fb_section.title or fb_page.title — they do not exist
- NEVER use translations->>'key' — must use jsonb_array_elements() first
- Always use ->> not ->
- "show all forms" → SELECT f.name, f.status, f.active FROM fb_forms f LIMIT 100;
- For simple counts (no labels needed): SELECT COUNT(*) FROM fb_forms;

### Answer
Given the errors above, here is the corrected SQL query that [QUESTION]{question}[/QUESTION]
[SQL]
"""
        try:
            response = self.sql_llm.complete(retry_prompt)
            return str(response).strip()
        except Exception:
            return None

    def _synthesize_answer(self, question: str, sql: str, raw_result: str) -> str:
        """Use mistral to convert raw SQL results to natural language."""
        try:
            chat_llm = Ollama(
                model=self.chat_model_name,
                base_url=self.ollama_url,
                temperature=0.3,
                request_timeout=60.0,
            )
            prompt = RESPONSE_SYNTHESIS_PROMPT.format(
                query_str=question,
                sql_query=sql,
                sql_result=raw_result[:2000],  # Truncate long results
            )
            response = chat_llm.complete(prompt)
            return str(response).strip()
        except Exception as e:
            return raw_result  # Fall back to raw results

    def test_connection(self) -> bool:
        """Test if the database connection works."""
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(sql_text("SELECT COUNT(*) FROM fb_forms"))
                count = result.fetchone()[0]
                print(f"  ✓ Connection OK — {count} forms in database")
                return True
        except Exception as e:
            print(f"  ✗ Connection failed: {e}")
            return False
