"""
Custom Prompt Templates for LlamaIndex NL2SQL
==============================================
These inject our hard-won JSONB knowledge into LlamaIndex's query engine.
Without these, the engine would hallucinate fb_question.label on every query.
"""

from llama_index.core.prompts import PromptTemplate


# ============================================================
# The main text-to-SQL prompt
# This replaces LlamaIndex's default prompt with our schema rules
# ============================================================

TEXT_TO_SQL_PROMPT = PromptTemplate(
    """### Task
Generate a SQL query to answer [QUESTION]{query_str}[/QUESTION]

### Instructions
- Generate a PostgreSQL SELECT query only.
- IMPORTANT: fb_question has NO label/title/text column. fb_section has NO title/name column. fb_page has NO title/name column.
- The ONLY tables with a name column are: fb_forms and fb_modules.
- All question labels, section titles, and page names are inside fb_translation_json.translations (a JSONB ARRAY).
- You MUST use jsonb_array_elements() to unpack the JSONB array. Never use ->> directly on the translations column.
- Always use ->> (double arrow) not -> (single arrow) for JSONB text extraction.
- Use ILIKE for text matching. Include LIMIT 100 unless aggregating.

### Database Schema
The query will run on a database with the following schema:
{schema}

### MANDATORY PATTERNS — follow these exactly:

SIMPLE QUERY (counts, listing forms/modules — no JSONB needed):
SELECT COUNT(*) FROM fb_forms;
SELECT f.name, f.status, f.active FROM fb_forms f LIMIT 100;
SELECT m.name FROM fb_modules m LIMIT 100;

QUERY NEEDING LABELS/TITLES (questions, sections, pages of a specific form):
SELECT elem->>'translatedText' AS label
FROM fb_forms f
JOIN fb_translation_json tj ON f.translations_id = tj.id,
     jsonb_array_elements(tj.translations) AS elem
WHERE f.name ILIKE '%TEST%'
  AND elem->>'language' = 'eng'
  AND elem->>'attribute' = 'NAME'
  AND elem->>'entityType' = 'QUESTION'
LIMIT 100;

Replace '%TEST%' with the form name from the user's question.
For pages use: elem->>'entityType' = 'PAGE'
For sections use: elem->>'entityType' = 'SECTION'
For all elements use: elem->>'attribute' = 'NAME' (without entityType filter)

"show all forms" → SELECT f.name, f.status, f.active FROM fb_forms f LIMIT 100;
"questions in TEST form" → use JSONB pattern above with '%TEST%'
"how many questions in TEST form" → SELECT COUNT(*) FROM fb_forms f JOIN fb_translation_json tj ON f.translations_id = tj.id, jsonb_array_elements(tj.translations) AS elem WHERE f.name ILIKE '%TEST%' AND elem->>'language' = 'eng' AND elem->>'attribute' = 'NAME' AND elem->>'entityType' = 'QUESTION';

### Answer
Given the database schema, here is the SQL query that [QUESTION]{query_str}[/QUESTION]
[SQL]
"""
)


# ============================================================
# Response synthesis prompt (converts SQL results to English)
# Used by mistral:latest, not sqlcoder
# ============================================================

RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(
    """Given a user question, the SQL query that was run, and the SQL result,
answer the user's question in a clear, concise way.

User question: {query_str}
SQL Query: {sql_query}
SQL Result: {sql_result}

If the result is empty, say "No results found" and suggest checking the form/module name.
If the result has data, summarize it naturally. For lists, include all items.
Do not show raw SQL or technical details unless the user asks.

Answer: """
)
