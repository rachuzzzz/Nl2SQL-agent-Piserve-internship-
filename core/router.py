"""
Intent Router — Embedding-Based
=================================
Classifies user questions into two paths using embeddings, not regex:

  Path 1: SEMANTIC — "which forms ask about age?"
    → Cosine similarity against pre-loaded question labels
    → Returns matches directly, no SQL needed

  Path 2: STRUCTURAL — "how many forms?", "list pages in TEST"
    → Templates or LLM generate SQL
    → Executes against database

No regex for phrasing detection. Embeddings handle all variations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# Intent examples — the router embeds these once at startup
# and compares every user question against them
SEMANTIC_INTENTS = [
    "which form has a question about",
    "find questions related to",
    "is there a question about",
    "search for questions mentioning",
    "forms that ask about",
    "questions containing",
    "does any form mention",
    "find forms with questions on",
    "which form covers the topic of",
    "look for questions regarding",
    "is there a question on something in any of the forms",
    "which form asks about a topic",
    "does any form have a question on",
    "find questions on a specific topic",
    "forms with questions related to",
    "which forms mention a subject",
    "is there anything about a topic in the forms",
    "search questions about a concept",
    "forms covering a specific area",
    "questions dealing with a subject",
]

STRUCTURAL_INTENTS = [
    "how many forms are there",
    "list all forms",
    "show all modules",
    "count the number of questions in a specific form",
    "show page titles in a specific form",
    "list all section titles in a form",
    "which forms are in draft status",
    "show me the help text for a form",
    "how many pages does a specific form have",
    "how many questions in each form",
    "show all forms with their status",
    "what questions are in a specific named form",
    "list all questions in a named form",
    "show placeholder text for a form",
    "show all forms and modules",
    "count forms by status",
]


@dataclass
class RouterDecision:
    """Output of the intent router."""
    path: str                          # "semantic" or "structural"
    confidence: float                  # how confident the classification is
    search_term: Optional[str] = None  # extracted concept for semantic search
    form_name: Optional[str] = None    # form filter if specified
    entity_type: str = "QUESTION"      # what to search for


class IntentRouter:
    """
    Embeds the user question and compares against semantic vs structural
    intent examples. No regex needed.
    """

    def __init__(self, embed_model):
        self.embed_model = embed_model

        print("  Building intent router...")
        # Embed all intent examples once
        self.semantic_embeddings = np.array([
            embed_model.get_text_embedding(t) for t in SEMANTIC_INTENTS
        ])
        self.structural_embeddings = np.array([
            embed_model.get_text_embedding(t) for t in STRUCTURAL_INTENTS
        ])
        print(f"  ✓ Intent router ready ({len(SEMANTIC_INTENTS)} semantic, {len(STRUCTURAL_INTENTS)} structural intents)")

    def route(self, question: str) -> RouterDecision:
        """
        Classify the question as semantic or structural.
        Returns a RouterDecision with the path and extracted parameters.
        """
        q_lower = question.lower().strip()

        # Embed the question
        q_embedding = np.array(self.embed_model.get_text_embedding(question))

        # Compare against semantic intents
        semantic_scores = self._cosine_similarities(q_embedding, self.semantic_embeddings)
        max_semantic = float(np.max(semantic_scores))

        # Compare against structural intents
        structural_scores = self._cosine_similarities(q_embedding, self.structural_embeddings)
        max_structural = float(np.max(structural_scores))

        # Extract form name from question (simple pattern — forms are always quoted or after "in/of")
        form_name = self._extract_form_name(question)

        # Extract entity type
        entity_type = "QUESTION"
        if any(w in q_lower for w in ["page", "pages"]):
            entity_type = "PAGE"
        elif any(w in q_lower for w in ["section", "sections"]):
            entity_type = "SECTION"

        # Decision — semantic wins if it scores higher AND above minimum threshold
        # Debug: print scores for tuning
        print(f"  [router] semantic={max_semantic:.2f} structural={max_structural:.2f} → ", end="")

        if max_semantic > max_structural and max_semantic > 0.55:
            # It's a semantic/content search
            search_term = self._extract_search_concept(question, form_name)
            print(f"SEMANTIC (search='{search_term}')")
            return RouterDecision(
                path="semantic",
                confidence=max_semantic,
                search_term=search_term,
                form_name=form_name,
                entity_type=entity_type,
            )
        else:
            print(f"STRUCTURAL")
            return RouterDecision(
                path="structural",
                confidence=max_structural,
                form_name=form_name,
                entity_type=entity_type,
            )

    def _cosine_similarities(self, query: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all target embeddings."""
        norms = np.linalg.norm(targets, axis=1) * np.linalg.norm(query)
        norms = np.where(norms == 0, 1, norms)
        return np.dot(targets, query) / norms

    def _extract_form_name(self, question: str) -> Optional[str]:
        """Extract form name from quotes or 'in the X form' pattern."""
        import re
        # Quoted form names
        quoted = re.findall(r'["\']([^"\']+)["\']', question)
        if quoted:
            # If multiple quotes, shortest is likely the form name
            # But also check context — after "in/of/from"
            for q in quoted:
                q_lower = question.lower()
                q_pos = q_lower.find(q.lower())
                before = q_lower[:q_pos].rstrip().rstrip("\"'")
                if before.endswith(('in', 'of', 'from', 'in the', 'of the', 'from the', 'form')):
                    return q
            # If only one quote and it looks like a form name (short, title case)
            if len(quoted) == 1 and len(quoted[0].split()) <= 4:
                return quoted[0]

        # Unquoted: "in the TEST form" / "in form TEST"
        m = re.search(r'(?:in|of|from)\s+(?:the\s+)?(?:form\s+)?([A-Z][\w\s]*?)(?:\s+form|\?|$)', question)
        if m:
            name = m.group(1).strip()
            # Don't extract "any", "all", "each" as form names
            if name.lower() not in ('any', 'all', 'each', 'every', 'some'):
                return name

        return None

    def _extract_search_concept(self, question: str, form_name: Optional[str]) -> str:
        """
        Extract the search concept from the question.
        Instead of regex for "about X" / "on X", we strip known structural
        words and form names, leaving just the concept.
        """
        import re
        q = question.lower().strip().rstrip('?').rstrip('.')

        # Remove form name references
        if form_name:
            q = re.sub(re.escape(form_name.lower()), '', q, flags=re.IGNORECASE)

        # Remove structural words — what's left is the concept
        structural_words = [
            r'\b(which|what|is|are|there|does|do|any|all|the|a|an)\b',
            r'\b(form|forms|question|questions|page|pages|section|sections)\b',
            r'\b(have|has|ask|asks|about|on|regarding|related|to|mention|mentions)\b',
            r'\b(containing|contains|find|show|list|search|look|for|in|of|from|with)\b',
            r'\b(that|this|those|these|it|they|my|our|me|every|each|some)\b',
            r'\b(cover|covers|deal|deals|involve|involves|discuss|discusses)\b',
            r'["\']',
        ]

        for pattern in structural_words:
            q = re.sub(pattern, ' ', q)

        # Clean up whitespace
        q = re.sub(r'\s+', ' ', q).strip()

        # If we stripped everything, fall back to the original minus form name
        if not q or len(q) < 2:
            q = question.lower()
            if form_name:
                q = q.replace(form_name.lower(), '')
            q = re.sub(r'["\']', '', q).strip()

        return q
