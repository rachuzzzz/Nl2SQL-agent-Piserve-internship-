"""
DOMAIN_REGISTRY — maps a domain name ("inspection" | "audit") to the content
modules AgentOrchestrator needs to wire itself up for that domain: prompts,
seed examples, business-rules engine, schema introspection, join graph, and
the RBAC scope-injection primary table.

AgentOrchestrator.__init__(domain=...) looks up this registry instead of
importing inspection's content modules directly at the top of loop.py (see
the domain-router plan's §3). Both domains' content is otherwise fully
independent — this module only holds the lookup table, not any content
itself.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DomainConfig:
    name: str
    system_prompt: str
    force_summary_prompt: str
    sql_generation_prompt: str
    seed_examples: list
    rules_engine_getter: Callable[[], object]
    introspect_fn: Callable
    schema_placeholder: str
    builder_module: object            # exposes build_dynamic_prompt_additions(question, score_fn=None)
    join_graph_module: object         # exposes path_as_hint(source, target), validate_join_graph(engine)
    core_context: str
    build_core_context_additions_fn: Callable[[str], str]
    # RBAC scope-injection (sql/validation/scope.py's enforce_scope primary table)
    scope_primary_table: str
    scope_primary_alias: str
    # SeedExampleIndex domain-tag overrides (retrieval/examples.py) — None
    # keeps that class's built-in inspection-tuned keyword map, which is
    # correct for the inspection domain but actively wrong for any other
    # domain's index (see SeedExampleIndex.__init__'s docstring for the
    # confirmed-live bug this avoids: inspection vocabulary overlapping
    # with ordinary audit phrasing silently penalized valid audit seed
    # examples). Audit passes {} / ([], '') below — neutral, no domain-tag
    # boost/penalty — until audit has its own tuned keyword map.
    seed_query_domain_tags: dict | None = None
    seed_query_domain_fallback: tuple | None = None


def _build_registry() -> dict:
    from src.agent.prompts import SYSTEM_PROMPT, FORCE_SUMMARY_PROMPT
    import src.agent.prompts.builder as _builder_inspection
    from src.sql.prompts import SQL_GENERATION_PROMPT, CORE_CONTEXT, build_core_context_additions
    from src.sql.examples import SEED_EXAMPLES
    from src.sql.rules import get_engine
    from src.knowledge.schema_introspector import introspect_schema
    import src.sql.join_graph as _join_graph_inspection

    from src.agent.prompts.builder_audit import SYSTEM_PROMPT_AUDIT
    import src.agent.prompts.builder_audit as _builder_audit
    from src.sql.prompts_audit import (
        SQL_GENERATION_PROMPT_AUDIT, CORE_CONTEXT_AUDIT,
        build_core_context_additions as build_core_context_additions_audit,
    )
    from src.sql.examples.audit_seed_data import AUDIT_SEED_EXAMPLES
    from src.sql.rules import get_audit_engine
    from src.knowledge.schema_introspector import introspect_audit_schema
    import src.sql.audit_join_graph as _join_graph_audit

    return {
        "inspection": DomainConfig(
            name="inspection",
            system_prompt=SYSTEM_PROMPT,
            force_summary_prompt=FORCE_SUMMARY_PROMPT,
            sql_generation_prompt=SQL_GENERATION_PROMPT,
            seed_examples=SEED_EXAMPLES,
            rules_engine_getter=get_engine,
            introspect_fn=introspect_schema,
            schema_placeholder="{inspection_schema}",
            builder_module=_builder_inspection,
            join_graph_module=_join_graph_inspection,
            core_context=CORE_CONTEXT,
            build_core_context_additions_fn=build_core_context_additions,
            scope_primary_table="inspection_report",
            scope_primary_alias="ir",
        ),
        "audit": DomainConfig(
            name="audit",
            system_prompt=SYSTEM_PROMPT_AUDIT,
            force_summary_prompt=FORCE_SUMMARY_PROMPT,  # shared — content is domain-agnostic
            sql_generation_prompt=SQL_GENERATION_PROMPT_AUDIT,
            seed_examples=AUDIT_SEED_EXAMPLES,
            rules_engine_getter=get_audit_engine,
            introspect_fn=introspect_audit_schema,
            schema_placeholder="{audit_schema}",
            builder_module=_builder_audit,
            join_graph_module=_join_graph_audit,
            core_context=CORE_CONTEXT_AUDIT,
            build_core_context_additions_fn=build_core_context_additions_audit,
            scope_primary_table="audit_schedule",
            scope_primary_alias="asch",
            seed_query_domain_tags={},
            seed_query_domain_fallback=([], ''),
        ),
    }


_REGISTRY: dict | None = None


def get_domain_config(domain: str) -> DomainConfig:
    """Lazily build DOMAIN_REGISTRY on first call (avoids import-order issues
    between agent.prompts / sql.prompts / sql.rules at module-load time) and
    cache it — content modules are already singletons (module-level constants),
    so there's nothing per-call to recompute."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    if domain not in _REGISTRY:
        raise ValueError(f"Unknown domain {domain!r} — expected one of {sorted(_REGISTRY)}")
    return _REGISTRY[domain]


DOMAIN_NAMES = ("inspection", "audit")
