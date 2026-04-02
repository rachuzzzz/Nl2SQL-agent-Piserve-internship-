"""
SQL Template Bypass
====================
For known query patterns, generates SQL deterministically without the LLM.
100% accurate, instant, zero GPU.

The LLM only handles queries that don't match any template.
"""

import re
from typing import Optional


class SQLTemplateBypass:
    """
    Detects known query patterns and emits correct SQL directly.
    Returns None if the query doesn't match any template (falls through to LLM).
    
    If a SemanticQuestionIndex is provided, content-based searches
    ("about age", "about safety") use embeddings instead of ILIKE.
    """

    def __init__(self, semantic_index=None):
        self.semantic_index = semantic_index

    def try_match(self, question: str) -> Optional[str]:
        """
        Try to match the question to a known SQL template.
        Returns SQL string if matched, None if no match (let LLM handle it).
        ORDER MATTERS — specific patterns must come before general ones.
        """
        q_lower = question.lower().strip()

        # Try each pattern in order of specificity (most specific first)
        result = (
            self._help_text_in_form(q_lower, question) or
            self._placeholder_in_form(q_lower, question) or
            self._search_by_label_content(q_lower, question) or
            self._list_all_forms(q_lower, question) or
            self._list_all_modules(q_lower, question) or
            self._count_entities_per_form(q_lower, question) or
            self._count_all_entities(q_lower, question) or
            self._count_entities_in_form(q_lower, question) or
            self._list_entity_labels_in_form(q_lower, question) or
            self._forms_by_status(q_lower, question) or
            self._forms_in_module(q_lower, question) or
            self._all_elements_in_form(q_lower, question)
        )

        return result

    # ==============================================================
    # Helper: extract form name from question
    # ==============================================================

    def _extract_form_name(self, question: str) -> Optional[str]:
        """Extract form name from quotes or 'in/of the X form' pattern."""
        # Try quoted strings first
        quoted = re.findall(r'["\']([^"\']+)["\']', question)
        if quoted:
            return quoted[0]

        # Try "in the X form" / "of the X form" / "in form X"
        patterns = [
            r'(?:in|of|from)\s+(?:the\s+)?["\']?(\w[\w\s]*?)["\']?\s+form',
            r'(?:in|of|from)\s+form\s+["\']?(\w[\w\s]*?)["\']?(?:\s|$|\?)',
            r'form\s+["\']?(\w[\w\s]*?)["\']?(?:\s+have|\s+has|\?|$)',
        ]
        for p in patterns:
            m = re.search(p, question, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    def _extract_module_name(self, question: str) -> Optional[str]:
        """Extract module name from the question."""
        quoted = re.findall(r'["\']([^"\']+)["\']', question)
        if quoted:
            return quoted[0]

        patterns = [
            r'(?:in|of|from|under)\s+(?:the\s+)?["\']?(\w[\w\s]*?)["\']?\s+module',
            r'module\s+["\']?(\w[\w\s]*?)["\']?(?:\s|$|\?)',
        ]
        for p in patterns:
            m = re.search(p, question, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    # ==============================================================
    # JSONB base query builder
    # ==============================================================

    def _jsonb_query(
        self,
        select: str = "elem->>'translatedText' AS label, elem->>'elementId' AS element_id",
        form_name: Optional[str] = None,
        entity_type: Optional[str] = None,
        attribute: str = "NAME",
        text_filter: Optional[str] = None,
        is_count: bool = False,
        group_by: Optional[str] = None,
    ) -> str:
        """Build a JSONB query from parameters."""

        if is_count:
            select_clause = "SELECT COUNT(*)"
        else:
            select_clause = f"SELECT {select}"

        parts = [
            select_clause,
            "FROM fb_forms f",
            "JOIN fb_translation_json tj ON f.translations_id = tj.id,",
            "     jsonb_array_elements(tj.translations) AS elem",
        ]

        where_parts = [
            f"elem->>'language' = 'eng'",
            f"elem->>'attribute' = '{attribute}'",
        ]

        if form_name:
            where_parts.insert(0, f"f.name ILIKE '%{form_name}%'")

        if entity_type:
            where_parts.append(f"elem->>'entityType' = '{entity_type}'")

        if text_filter:
            where_parts.append(f"elem->>'translatedText' ILIKE '%{text_filter}%'")

        parts.append("WHERE " + "\n  AND ".join(where_parts))

        if group_by:
            parts.append(f"GROUP BY {group_by}")

        if not is_count and not group_by:
            parts.append("LIMIT 100")

        return ";\n".join(["\n".join(parts)]) + ";"

    # ==============================================================
    # Pattern matchers — each returns SQL or None
    # ==============================================================

    def _count_all_entities(self, q_lower: str, question: str) -> Optional[str]:
        """'how many forms/modules/pages are there'"""

        if not re.search(r'how\s+many|count', q_lower):
            return None

        # "how many forms" (no specific form mentioned)
        if re.search(r'how\s+many\s+(?:total\s+)?forms', q_lower) and not self._extract_form_name(question):
            return "SELECT COUNT(*) AS form_count FROM fb_forms;"

        if re.search(r'how\s+many\s+(?:total\s+)?modules', q_lower):
            return "SELECT COUNT(*) AS module_count FROM fb_modules;"

        return None

    def _count_entities_in_form(self, q_lower: str, question: str) -> Optional[str]:
        """'how many questions/pages/sections in the TEST form'"""

        if not re.search(r'how\s+many|count', q_lower):
            return None

        form_name = self._extract_form_name(question)
        if not form_name:
            return None

        if re.search(r'question', q_lower):
            return self._jsonb_query(form_name=form_name, entity_type="QUESTION", is_count=True)

        if re.search(r'page', q_lower):
            return self._jsonb_query(form_name=form_name, entity_type="PAGE", is_count=True)

        if re.search(r'section', q_lower):
            return self._jsonb_query(form_name=form_name, entity_type="SECTION", is_count=True)

        return None

    def _list_entity_labels_in_form(self, q_lower: str, question: str) -> Optional[str]:
        """
        'what questions are in the TEST form'
        'show page titles in the TEST form'
        'show section titles in the TEST form'
        """

        form_name = self._extract_form_name(question)
        if not form_name:
            return None

        # Determine entity type
        entity_type = None
        if re.search(r'question|field|asked', q_lower):
            entity_type = "QUESTION"
        elif re.search(r'page', q_lower):
            entity_type = "PAGE"
        elif re.search(r'section', q_lower):
            entity_type = "SECTION"
        elif re.search(r'sub.?form|subform|repeatable', q_lower):
            entity_type = "SUB_FORM"

        if not entity_type:
            return None

        # Must have an action word suggesting listing
        if re.search(r'what|show|list|display|get|give|tell|which|all|titles?|labels?|names?', q_lower):
            return self._jsonb_query(form_name=form_name, entity_type=entity_type)

        return None

    def _search_by_label_content(self, q_lower: str, question: str) -> Optional[str]:
        """
        'which form has a question about grade'
        'is there a question about age in form TEST'
        'find the question about employee name'
        'which form mentions "what is your age"'
        
        Uses semantic search if available, falls back to ILIKE.
        """

        # First check for quoted search terms — highest priority
        quoted = re.findall(r'["\']([^"\']+)["\']', question)
        search_term = None
        form_name = None

        if quoted:
            # If there are multiple quoted strings, the longer one is probably the search term
            # and the shorter one might be a form name
            if len(quoted) >= 2:
                sorted_q = sorted(quoted, key=len, reverse=True)
                search_term = sorted_q[0]
                form_name = sorted_q[1]
            else:
                # Single quoted string — is it a form name or search term?
                # If question also mentions "form", check context
                if re.search(r'(?:about|mention|contain|ask|related|find|search|question)', q_lower):
                    search_term = quoted[0]
                    # Try to find form name without quotes
                    fm = re.search(r'(?:in|of|from)\s+(?:the\s+)?(\w+)\s+form', q_lower)
                    if fm:
                        form_name = fm.group(1)
                else:
                    form_name = quoted[0]

        # If no quoted search term, look for "about X", "mentions X", etc.
        if not search_term:
            search_match = re.search(
                r'(?:about|containing|contains|related\s+to|mentioning|mentions?|with.*word|labeled|called|asks?\s+about|asks?\s+for)\s+["\']?(\w[\w\s]*?)(?:["\']|\s+in\s+|\s+form|\?|$)',
                q_lower
            )
            if not search_match:
                return None
            search_term = search_match.group(1).strip()

        # Clean up trailing common words
        search_term = re.sub(r'\s+(in|form|the|and|or)$', '', search_term).strip()
        if len(search_term) < 2:
            return None

        # Extract form name if not already found
        if not form_name:
            form_name = self._extract_form_name(question)

        # --- Try semantic search first ---
        if self.semantic_index:
            sql = self.semantic_index.search_and_build_sql(
                search_term=search_term,
                entity_type="QUESTION",
                form_name=form_name,
                threshold=0.45,
            )
            if sql:
                return sql

        # --- Fall back to ILIKE ---
        if re.search(r'which\s+form|what\s+form|find.*form|is\s+there', q_lower):
            return (
                f"SELECT DISTINCT f.name AS form_name,\n"
                f"       elem->>'translatedText' AS matching_label\n"
                f"FROM fb_forms f\n"
                f"JOIN fb_translation_json tj ON f.translations_id = tj.id,\n"
                f"     jsonb_array_elements(tj.translations) AS elem\n"
                f"WHERE elem->>'language' = 'eng'\n"
                f"  AND elem->>'attribute' = 'NAME'\n"
                f"  AND elem->>'translatedText' ILIKE '%{search_term}%'\n"
                f"LIMIT 100;"
            )

        if form_name:
            return self._jsonb_query(
                form_name=form_name,
                entity_type="QUESTION",
                text_filter=search_term,
            )
        else:
            return (
                f"SELECT f.name AS form_name,\n"
                f"       elem->>'translatedText' AS label,\n"
                f"       elem->>'elementId' AS element_id\n"
                f"FROM fb_forms f\n"
                f"JOIN fb_translation_json tj ON f.translations_id = tj.id,\n"
                f"     jsonb_array_elements(tj.translations) AS elem\n"
                f"WHERE elem->>'language' = 'eng'\n"
                f"  AND elem->>'attribute' = 'NAME'\n"
                f"  AND elem->>'entityType' = 'QUESTION'\n"
                f"  AND elem->>'translatedText' ILIKE '%{search_term}%'\n"
                f"LIMIT 100;"
            )

    def _list_all_forms(self, q_lower: str, question: str) -> Optional[str]:
        """'show all forms', 'list forms'"""
        if re.search(r'(?:show|list|display|get|all)\s+(?:the\s+)?(?:all\s+)?forms', q_lower):
            if not self._extract_form_name(question):  # Don't match "show questions in TEST form"
                return "SELECT f.name, f.status, f.active, m.name AS module_name FROM fb_forms f LEFT JOIN fb_modules m ON f.module_id = m.id ORDER BY f.name LIMIT 100;"
        return None

    def _list_all_modules(self, q_lower: str, question: str) -> Optional[str]:
        """'list all modules', 'show modules'"""
        if re.search(r'(?:show|list|display|get|all)\s+(?:the\s+)?(?:all\s+)?modules', q_lower):
            return "SELECT m.name, m.active, pm.name AS parent_module FROM fb_modules m LEFT JOIN fb_modules pm ON m.parent_module_id = pm.id ORDER BY m.name LIMIT 100;"
        return None

    def _forms_by_status(self, q_lower: str, question: str) -> Optional[str]:
        """'which forms are in DRAFT status', 'show published forms'"""
        status_match = re.search(r'(DRAFT|PUBLISHED|DELETED|ACTIVE)', question, re.IGNORECASE)
        if status_match and re.search(r'form', q_lower):
            status = status_match.group(1).upper()
            if status == 'ACTIVE':
                return "SELECT f.name, f.status FROM fb_forms f WHERE f.active = true LIMIT 100;"
            return f"SELECT f.name, f.status, f.active FROM fb_forms f WHERE f.status = '{status}' LIMIT 100;"
        return None

    def _forms_in_module(self, q_lower: str, question: str) -> Optional[str]:
        """'how many forms in the Inspection module', 'show forms in HR module'"""
        module_name = self._extract_module_name(question)
        if not module_name:
            return None

        if re.search(r'how\s+many', q_lower) and re.search(r'form', q_lower):
            return f"SELECT COUNT(*) AS form_count FROM fb_forms f JOIN fb_modules m ON f.module_id = m.id WHERE m.name ILIKE '%{module_name}%';"

        if re.search(r'(?:show|list|which|what)\s+.*form', q_lower):
            return f"SELECT f.name, f.status, f.active FROM fb_forms f JOIN fb_modules m ON f.module_id = m.id WHERE m.name ILIKE '%{module_name}%' LIMIT 100;"

        return None

    def _help_text_in_form(self, q_lower: str, question: str) -> Optional[str]:
        """'show help text for questions in TEST form'"""
        if not re.search(r'help.?text', q_lower):
            return None

        form_name = self._extract_form_name(question)
        if not form_name:
            return self._jsonb_query(
                select="elem->>'translatedText' AS help_text, elem->>'elementId' AS element_id",
                entity_type="QUESTION",
                attribute="HELP_TEXT",
            )

        return self._jsonb_query(
            select="elem->>'translatedText' AS help_text, elem->>'elementId' AS element_id",
            form_name=form_name,
            entity_type="QUESTION",
            attribute="HELP_TEXT",
        )

    def _placeholder_in_form(self, q_lower: str, question: str) -> Optional[str]:
        """'show placeholder text for questions in TEST form'"""
        if not re.search(r'placeholder', q_lower):
            return None

        form_name = self._extract_form_name(question)
        return self._jsonb_query(
            select="elem->>'translatedText' AS placeholder, elem->>'elementId' AS element_id",
            form_name=form_name,
            entity_type="QUESTION",
            attribute="PLACEHOLDER",
        )

    def _count_entities_per_form(self, q_lower: str, question: str) -> Optional[str]:
        """'how many questions in each form', 'questions per form'"""
        if not re.search(r'(?:each|every|per)\s+form|(?:in|across|for)\s+(?:all|every)\s+form', q_lower):
            return None
        # Must also mention a countable entity
        if not re.search(r'(?:how\s+many|count|number\s+of)\s+(?:question|page|section)', q_lower) and \
           not re.search(r'(?:question|page|section)s?\s+(?:per|in\s+each|in\s+every)', q_lower):
            return None

        entity_type = "QUESTION"
        if re.search(r'page', q_lower):
            entity_type = "PAGE"
        elif re.search(r'section', q_lower):
            entity_type = "SECTION"

        return (
            f"SELECT f.name AS form_name, COUNT(*) AS count\n"
            f"FROM fb_forms f\n"
            f"JOIN fb_translation_json tj ON f.translations_id = tj.id,\n"
            f"     jsonb_array_elements(tj.translations) AS elem\n"
            f"WHERE elem->>'language' = 'eng'\n"
            f"  AND elem->>'attribute' = 'NAME'\n"
            f"  AND elem->>'entityType' = '{entity_type}'\n"
            f"GROUP BY f.name\n"
            f"ORDER BY count DESC;"
        )

    def _all_elements_in_form(self, q_lower: str, question: str) -> Optional[str]:
        """'show all elements in the TEST form', 'what's in the TEST form'"""
        form_name = self._extract_form_name(question)
        if not form_name:
            return None

        if re.search(r'all\s+element|everything|what.*in\s+(?:the\s+)?["\']', q_lower):
            return self._jsonb_query(
                select="elem->>'translatedText' AS name, elem->>'entityType' AS type, elem->>'elementId' AS element_id",
                form_name=form_name,
            )

        return None
