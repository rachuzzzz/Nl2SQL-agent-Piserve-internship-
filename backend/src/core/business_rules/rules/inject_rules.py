"""
INJECT_RULES: extracted verbatim from the original
core/business_rules.py REGISTRY list (all "type": INJECT entries,
in original relative order). See core.business_rules.engine for how
these are aggregated and consumed.
"""

from src.core.business_rules._types import INJECT
from src.core.business_rules.engine import _STATUS_HINT, _FINANCIAL_HINT, _DRAFT_EXCLUSION

INJECT_RULES: list[dict] = [

    # ══════════════════════════════════════════════════════════════════════════
    # INJECT — append hint to schema_hint when question matches
    # Higher priority fires first; hints are concatenated in priority order.
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id":             "inject_top_capex_converters",
        "type":           INJECT,
        "priority":       97,
        "description":    "Top N capex converters — inject exact CTE SQL, bypass embedding.",
        "required_tables": ["po_details"],
        "forbidden_patterns": [r"ORDER BY ica\.capex", r"capex_status\s*=\s*'ACCEPTED'", r"ica\.expenditure"],
        # Suppress the broader generic FM-manager routing hint — its routing
        # advice can conflict with this rule's "USE THIS EXACT SQL" template
        # when both fire on an ambiguous "top capex converter for FM manager" question.
        "suppresses":     ["inject_fm_manager_routing"],
        "question_pattern": (
            r"\b(top|best|highest|leading).{0,20}(capex.{0,15}converter|convert.{0,15}capex)s?\b"
            r"|\b(capex.{0,15}converter|convert.{0,15}capex)s?.{0,20}(top|best|rank|leaderboard)\b"
        ),
        "hint": (
            "\nUSE THIS EXACT SQL FOR TOP CAPEX CONVERTERS (do not deviate):\n"
            "WITH po_by_ica AS (\n"
            "    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po\n"
            "    FROM po_details WHERE active = true GROUP BY 1\n"
            ")\n"
            "SELECT\n"
            "    u.first_name || ' ' || u.last_name  AS fm_manager,\n"
            "    COUNT(DISTINCT ica.id)              AS ca_count,\n"
            "    COALESCE(SUM(ica.capex), 0)         AS capex_generated,\n"
            "    COALESCE(SUM(p.total_po), 0)        AS capex_converted,\n"
            "    ROUND(COALESCE(SUM(p.total_po), 0) * 100.0 / NULLIF(SUM(ica.capex), 0), 1) AS conversion_pct\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN inspection_report ir ON ica.inspection_id = ir.id\n"
            "JOIN users u ON ir.inspectee_user_id = u.id\n"
            "LEFT JOIN po_by_ica p ON p.ica_id = ica.id\n"
            "WHERE ir.active = true AND ica.active = true AND u.active = true AND ica.capex > 0\n"
            "GROUP BY u.id, u.first_name, u.last_name\n"
            "ORDER BY capex_converted DESC LIMIT 5;\n"
            "NEVER: ORDER BY ica.capex — that ranks by GENERATED not CONVERTED\n"
            "NEVER: ica.expenditure filter — that is a category flag, not conversion\n"
        ),
    },

    {
        "id":             "inject_top_capex_generators",
        "type":           INJECT,
        "priority":       96,
        "description":    "Top N capex generators — ranked by SUM(ica.capex), includes po join for context.",
        "required_tables": ["po_details"],
        "forbidden_patterns": [r"ORDER BY.*capex_converted", r"ica\.expenditure"],
        "suppresses":     ["inject_fm_manager_routing"],
        "question_pattern": (
            # Must explicitly mention fm manager / facility manager context
            r"\b(top|best|highest|leading|most).{0,25}capex.{0,20}generat\w*.{0,20}(fm|facility.{0,5}manager|inspectee)\b"
            r"|\b(top|best|highest|leading|most).{0,25}(fm|facility.{0,5}manager).{0,25}capex.{0,20}generat\w*\b"
            r"|\bcapex.{0,20}generat\w*.{0,20}(top|best|rank|leaderboard).{0,20}(fm|facility|manager)\b"
            r"|\b(fm|facility).{0,10}manager.{0,30}generat.{0,20}(most|highest|top).{0,20}capex\b"
            r"|\bwhich.{0,10}(fm|facility).{0,10}manager.{0,30}(most|highest).{0,20}capex\b"
        ),
        "hint": (
            "\nUSE THIS EXACT SQL FOR TOP CAPEX GENERATORS (ranked by generated, NOT converted):\n"
            "WITH po_by_ica AS (\n"
            "    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po\n"
            "    FROM po_details WHERE active = true GROUP BY 1\n"
            ")\n"
            "SELECT\n"
            "    u.first_name || ' ' || u.last_name  AS fm_manager,\n"
            "    COUNT(DISTINCT ica.id)              AS ca_count,\n"
            "    COALESCE(SUM(ica.capex), 0)         AS capex_generated,\n"
            "    COALESCE(SUM(p.total_po), 0)        AS capex_converted,\n"
            "    ROUND(COALESCE(SUM(p.total_po), 0) * 100.0 / NULLIF(SUM(ica.capex), 0), 1) AS conversion_pct\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN inspection_report ir ON ica.inspection_id = ir.id\n"
            "JOIN users u ON ir.inspectee_user_id = u.id\n"
            "LEFT JOIN po_by_ica p ON p.ica_id = ica.id\n"
            "WHERE ir.active = true AND ica.active = true AND u.active = true AND ica.capex > 0\n"
            "GROUP BY u.id, u.first_name, u.last_name\n"
            "ORDER BY capex_generated DESC LIMIT 10;\n"
            "NEVER: ORDER BY capex_converted — that ranks converters not generators\n"
        ),
    },

    {
        "id":             "inject_top_fm_performers",
        "type":           INJECT,
        "priority":       96,
        "description":    "Top performing FM managers — inject CTE SQL ranking by conversion rate.",
        "required_tables": ["po_details"],
        "forbidden_patterns": [r"AVG\(ir\.inspection_score\)", r"ORDER BY avg_score", r"GROUP BY ica\.responsible"],
        "suppresses":     ["inject_fm_manager_routing"],
        "question_pattern": (
            r"\b(top|best|highest).{0,20}(perform|fm.{0,10}manager|facility.{0,10}manager).{0,30}"
            r"(manager|fm|perform|rank|leaderboard)\b"
            r"|\b(who.{0,20}(best|top).{0,20}(fm|facility).{0,10}manager"
            r"|best.{0,20}fm.{0,10}manager)\b"
        ),
        "hint": (
            "\nUSE THIS EXACT SQL FOR TOP FM MANAGERS — rank by conversion_pct, filter by FM role, NOT avg_score:\n"
            "NEVER: AVG(ir.inspection_score) — that is inspector quality, not FM performance\n"
            "WITH po_by_ica AS (\n"
            "    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_po\n"
            "    FROM po_details WHERE active = true GROUP BY 1\n"
            "),\n"
            "pr_by_ica AS (\n"
            "    SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_pr\n"
            "    FROM pr_details WHERE active = true GROUP BY 1\n"
            ")\n"
            "SELECT\n"
            "    u.first_name || ' ' || u.last_name          AS fm_manager,\n"
            "    COALESCE(SUM(ica.capex), 0)                  AS capex_generated,\n"
            "    COALESCE(SUM(p.total_po), 0)                 AS capex_converted,\n"
            "    ROUND(COALESCE(SUM(p.total_po), 0) * 100.0 / NULLIF(SUM(ica.capex), 0), 1) AS conversion_pct,\n"
            "    COALESCE(SUM(pr.total_pr), 0)                AS opex_incurred\n"
            "FROM inspection_report ir\n"
            "JOIN users u ON ir.inspectee_user_id = u.id\n"
            "LEFT JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id AND ica.active = true AND ica.capex > 0\n"
            "LEFT JOIN po_by_ica p  ON p.ica_id  = ica.id\n"
            "LEFT JOIN pr_by_ica pr ON pr.ica_id = ica.id\n"
            "WHERE ir.active = true AND u.active = true\n"
            "  AND EXISTS (\n"
            "      SELECT 1 FROM user_roles ur JOIN role r ON ur.role_id = r.id\n"
            "      WHERE ur.user_id = u.id\n"
            "        AND r.name IN ('FM Manager','FM Engineer','AD FM',\n"
            "                       'Asst FM Manager','FM Admin','FM Supervisor')\n"
            "  )\n"
            "GROUP BY u.id, u.first_name, u.last_name\n"
            "HAVING SUM(ica.capex) > 0\n"
            "ORDER BY conversion_pct DESC NULLS LAST, opex_incurred ASC\n"
            "LIMIT 5;\n"
        ),
    },

    {
        "id":             "inject_avg_ma_close_time",
        "type":           INJECT,
        "priority":       85,
        "description":    "Average time to close mitigative action — use target_close_out_date not close_on.",
        "question_pattern": (
            r"\b(average|avg|mean).{0,20}(time|days?|hours?|duration).{0,20}"
            r"(close|clos|resolv|complet).{0,20}(mitigative|action|ma)\b"
            r"|\b(how.{0,10}long|time.{0,10}taken?).{0,30}(close|clos|resolv)\b"
        ),
        "hint": (
            "\nAVERAGE TIME TO CLOSE MITIGATIVE ACTION:\n"
            "NEVER use close_on — it is NULL on most rows (only set in special workflows).\n"
            "USE target_close_out_date - created_on for the planned close window:\n"
            "SELECT\n"
            "    AVG(EXTRACT(EPOCH FROM (ica.target_close_out_date - ica.created_on)) / 86400) AS avg_days,\n"
            "    COUNT(*) AS total_closed\n"
            "FROM inspection_corrective_action ica\n"
            "WHERE ica.status IN (\'CLOSED\', \'CLOSE_WITH_DEFERRED\')\n"
            "  AND ica.active = true\n"
            "  AND ica.target_close_out_date IS NOT NULL\n"
            "  AND ica.created_on IS NOT NULL;\n"
            "NEVER: EXTRACT(EPOCH FROM (close_on - created_on)) — close_on IS NULL on most rows\n"
        ),
    },

    {
        "id":             "inject_qa_recent_form",
        "type":           INJECT,
        "priority":       92,
        "description":    "Q&A from most recent form — provide complete 2-table SQL direct.",
        "suppresses":     [],
        "question_pattern": (
            r"\b(list|show|get|fetch|display|what).{0,20}"
            r"(question|answer|qa|form.{0,10}answer|answer.{0,10}form).{0,40}"
            r"(most.{0,10}recent(?:ly)?|latest|last)\b"
            r"|\b(most.{0,10}recent(?:ly)?|latest|last).{0,30}"
            r"(question|answer|qa|form.{0,10}answer)\b"
            # 'questions and answers for the most recently' — noun comes before recency qualifier
            r"|\b(question|answer)s?.{0,50}(most.{0,10}recent(?:ly)?|latest|last)\b"
        ),
        "hint": (
            "\nQ&A FROM MOST RECENT FORM — use this EXACT SQL. ONE query, NOT two separate queries:\n"
            "SELECT aq.label AS question, aa.answer_text AS answer, aa.module_name AS form_section\n"
            "FROM ai_answers aa\n"
            "JOIN ai_questions aq ON aa.element_id = aq.element_id\n"
            "WHERE aa.inspection_report_id = (\n"
            "    SELECT aa2.inspection_report_id FROM ai_answers aa2\n"
            "    ORDER BY aa2.submitted_on DESC LIMIT 1\n"
            ")\n"
            "\n"
            "ORDER BY aa.module_name, aq.label\n"
            "LIMIT 200;\n"
            "CRITICAL: This is a SINGLE SQL query — do NOT first lookup the inspection_id and then\n"
            "do a second query. The subquery handles the inspection lookup internally.\n"
            "NEVER: SELECT ir.inspection_id first, then ask for Q&A — use the complete SQL above.\n"
        ),
    },

    {
        "id":             "inject_pct_with_ma",
        "type":           INJECT,
        "priority":       100,
        "description":    "Percentage of inspections with mitigative action — inject exact SQL.",
        "question_pattern": (
            r"\b(percentage|percent|pct|proportion|ratio).{0,40}"
            r"(mitigative|finding)\b"
            r"|\b(mitigative|finding).{0,40}"
            r"(percentage|percent|pct|proportion|ratio)\b"
        ),
        "hint": (
            "\nUSE THIS EXACT SQL — do not generate your own:\n"
            "SELECT ROUND(\n"
            "  100.0 * COUNT(DISTINCT CASE WHEN ica.id IS NOT NULL THEN ir.id END)\n"
            "  / NULLIF(COUNT(DISTINCT ir.id), 0), 1\n"
            ") AS pct_with_mitigative_action\n"
            "FROM inspection_report ir\n"
            "LEFT JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id\n"
            f"WHERE ir.{_DRAFT_EXCLUSION};\n"
        ),
    },

    {
        "id":             "inject_inspector_performance",
        "type":           INJECT,
        "priority":       93,
        "description":    "Inspector performance summary — assigned/conducted/observations — inject CTE pattern.",
        "question_pattern": (
            r"\b(inspector).{0,50}(performance|summary|breakdown|stats?|report|overview)\b"
            r"|\b(performance|summary|breakdown|stats?).{0,50}(inspector)\b"
            r"|\b(assigned|conducted).{0,30}(inspection|inspector)\b"
            r"|\b(inspector).{0,30}(assigned|conducted|observations?|high.?risk)\b"
        ),
        "hint": (
            "\nINSPECTOR PERFORMANCE SUMMARY — CRITICAL DEFINITIONS:\n"
            "  assigned   = COUNT(ir.id)                               — ALL reports for inspector\n"
            "  conducted  = COUNT FILTER (WHERE ir.status IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION'))\n"
            "               NEVER status != 'DRAFT' — that includes CLOSED\n"
            "               NEVER status = 'SUBMITTED' alone — misses UNDER_REVIEW and RETURN_FOR_MOD\n"
            "  total_observations = SUM(COUNT DISTINCT aa.submission_id per inspection)\n"
            "               NEVER COUNT(*) on ai_answers — overcounts (many rows per observation)\n"
            "  high_risk_observations = COUNT DISTINCT submission_id WHERE Risk Level = High\n"
            "  NEVER join ai_answers twice in main query — causes cross-join fan-out\n"
            "  NEVER join inspection_schedule — irrelevant to performance counts\n"
            "  Pre-aggregate observations in a CTE:\n"
            "WITH obs AS (\n"
            "    SELECT aa.inspection_report_id,\n"
            "           COUNT(DISTINCT aa.submission_id)   AS total_obs,\n"
            "           COUNT(DISTINCT CASE\n"
            "               WHEN aq.label = 'Risk Level' AND aa.answer_text ILIKE '%High%'\n"
            "               THEN aa.submission_id END)     AS high_risk_obs\n"
            "    FROM ai_answers aa\n"
            "    JOIN ai_questions aq ON aa.element_id = aq.element_id\n"
            "    WHERE aa.module_name = 'Inspection Form'\n"
            "    GROUP BY aa.inspection_report_id\n"
            ")\n"
            "SELECT u.first_name || ' ' || u.last_name          AS inspector_name,\n"
            "       COUNT(ir.id)                                AS assigned,\n"
            "       COUNT(ir.id) FILTER (WHERE ir.status IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION')) AS conducted,\n"
            "       COUNT(ir.id) FILTER (WHERE ir.status = 'CLOSED') AS completed,\n"
            "       COALESCE(SUM(obs.total_obs), 0)             AS total_observations,\n"
            "       COALESCE(SUM(obs.high_risk_obs), 0)         AS high_risk_observations\n"
            "FROM users u\n"
            "JOIN inspection_report ir ON ir.inspector_user_id = u.id AND ir.active = true\n"
            "LEFT JOIN obs ON obs.inspection_report_id = ir.id\n"
            "WHERE u.active = true\n"
            "GROUP BY u.id, u.first_name, u.last_name\n"
            "ORDER BY conducted DESC NULLS LAST LIMIT 100;\n"
        ),
    },

    {
        "id":             "inject_inspector_count",
        "type":           INJECT,
        "priority":       90,
        "description":    "Inspector count query — prevents hardcoded years.",
        "question_pattern": (
            r"\b(which|who|top|most|count|how\s+many|list).{0,30}"
            r"(inspector|inspectors?).{0,30}"
            r"(most|conducted|done|most\s+inspections?|count|number|year)\b"
        ),
        "hint": (
            "\nUSE THIS EXACT PATTERN for inspector count queries:\n"
            "-- conducted = IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION') — in-flight\n"
            "-- completed = CLOSED only\n"
            "-- assigned  = all reports linked to inspector (any non-draft status)\n"
            "SELECT u.first_name || ' ' || u.last_name AS inspector_name,\n"
            "       COUNT(*) AS assigned,\n"
            "       COUNT(*) FILTER (WHERE ir.status IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION')) AS conducted,\n"
            "       COUNT(*) FILTER (WHERE ir.status = 'CLOSED') AS completed\n"
            "FROM inspection_report ir\n"
            "JOIN users u ON ir.inspector_user_id = u.id\n"
            "WHERE ir.active = true AND ir.status != 'DRAFT'\n"
            "  AND EXTRACT(YEAR FROM ir.submitted_on) = EXTRACT(YEAR FROM CURRENT_DATE)\n"
            "GROUP BY u.id, u.first_name, u.last_name\n"
            "ORDER BY conducted DESC LIMIT 1;\n"
            "CRITICAL: NEVER write '= 2023' or '= 2026' — always use EXTRACT(YEAR FROM CURRENT_DATE).\n"
            "CRITICAL: conducted = IN ('SUBMITTED','UNDER_REVIEW','RETURN_FOR_MODIFICATION'). NEVER status = 'SUBMITTED' only.\n"
        ),
    },

    {
        "id":             "inject_financial_routing",
        "required_tables": [],
        "forbidden_patterns": [r"SUM\\(ica\\.opex\\)", r"capex_status\\s*=\\s*'ACCEPTED'"],

        "type":           INJECT,
        "priority":       88,
        "description":    "CAPEX/OPEX financial queries — route to po_details/pr_details/ai_answers.",
        "question_pattern": (
            r"\b(po[_\s]detail|pr[_\s]detail|purchase\s+order|purchase\s+requisition|"
            r"purchase\s+order\s+amount|po\s+amount|total\s+po|"
            r"capex.{0,30}(generat|convert|lost|loss|incurr)|"
            r"opex.{0,30}(incurr|facilit|spend|higher|exceed)|"
            r"(generat|convert|lost|incurr).{0,30}(capex|opex)|"
            r"how\s+much.{0,40}(convert|capex|opex|po|purchase)|"
            r"opex.{0,30}(higher|more|greater|exceed).{0,30}capex|"
            r"conversion\s+rate|capex\s+conversion|"
            r"(convert|capex).{0,20}(inspectee|fm|manager|facility)|"
            # plural 'converters' / 'generators' explicitly
            r"(capex|opex)\s+(converter|generator)s?|"
            r"(converter|generator)s?\s+(capex|opex)|"
            r"top.{0,15}(capex|opex).{0,15}(converter|generator)s?|"
            # facility cost / expenditure queries
            r"(facility|site).{0,30}(costs?|spend|expenditure|expensive)|"
            r"(cost|spend|expenditure).{0,30}(facility|site)|"
            r"most\s+expensive\s+(facility|site)|"
            r"rival.{0,20}(quote|bid|amount)|financial\s+summary)\b"
        ),
        "hint": _FINANCIAL_HINT,
    },

    {
        "id":             "inject_fm_manager_routing",
        "forbidden_patterns": [r"GROUP\\s+BY\\s+ica\\.responsible", r"AVG\\(ir\\.inspection_score\\)"],

        "type":           INJECT,
        "priority":       85,
        "description":    "FM Manager / inspectee queries — route to inspectee_user_id not inspector_user_id.",
        "question_pattern": (
            r"\b(fm\s+manager|facility\s+manager|inspectee|fm\s+engineer|"
            r"fm\s+auditee|fm\s+leaderboard|capex\s+leaderboard|"
            # 'top N capex converters/generators' — plural handled by 's?' suffix
            r"top.{0,25}(fm|manager|capex|opex).{0,25}(generator|converter|performer|leaderboard)s?|"
            # standalone 'capex converters/generators'
            r"capex\s+(generator|converter|generat|convert)s?|"
            r"(generator|converter)s?.{0,20}capex|"
            # 'who are the top capex converters' type phrasing
            r"(best|highest|top|leading|rank).{0,30}(capex|opex).{0,30}(converter|generator)s?|"
            r"(converter|generator)s?.{0,30}(capex|opex))\b"
        ),
        "hint": (
            "\nFM MANAGER ROUTING — CRITICAL RULES:\n"
            "RULE 1: ica.responsible is a STRING ENUM (CLIENT/INTERNAL_OPERATIONS/SUB_CONTRACTOR)\n"
            "  It is NOT a person. NEVER GROUP BY ica.responsible for FM manager queries.\n"
            "  NEVER SELECT ica.responsible AS fm_manager — this gives 3 rows, not people.\n"
            "RULE 2: NEVER rank FM managers by avg_inspection_score — that is inspector quality.\n"
            "  FM performance = CAPEX conversion rate + low OPEX (pr_details), not inspection scores.\n"
            "RULE 3: FM Manager = inspectee_user_id (NOT inspector_user_id)\n"
            "  CORRECT JOIN: JOIN users u ON ir.inspectee_user_id = u.id\n"
            "  CORRECT GROUP: GROUP BY u.id, u.first_name, u.last_name\n"
            "\nFM MANAGER PERFORMANCE — correct ranking:\n"
            "  GOOD: High conversion = SUM(po.po_amount) / NULLIF(SUM(ica.capex),0) * 100\n"
            "  BAD:  High OPEX = SUM(pr.pr_amount) from pr_details (Khidmah pays internally)\n"
            "  NEVER SUM(ica.opex) — use pr_details for actual OPEX paid\n"
            "  Best FM manager = highest conversion_pct DESC, opex_incurred ASC\n"
            "\nCAPEX LEADERBOARD:\n"
            "  CAPEX Generated = SUM(ica.capex)\n"
            "  CAPEX Converted = SUM(po.po_amount) via po_details — NOT capex_status='ACCEPTED'\n"
            "  CAPEX Lost      = SUM(ica.capex) - COALESCE(SUM(po.po_amount), 0)\n"
            "  OPEX Incurred   = SUM(pr.pr_amount) via pr_details\n"
            "  capex_status IS a valid column but is a workflow flag — NOT a financial metric\n"
            "\nPRE-AGGREGATE to avoid row multiplication:\n"
            "  WITH po_by_ica AS (SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total FROM po_details WHERE active=true GROUP BY 1)\n"
            "  WITH pr_by_ica AS (SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total FROM pr_details WHERE active=true GROUP BY 1)\n"
        ),
    },

    {
        "id":             "inject_risk_observation_routing",
        "required_tables": ["risk_level"],
        "forbidden_patterns": [r"ai_answers.*label.*Risk Level", r"aa\.answer_text.*ILIKE.*High"],

        "type":           INJECT,
        "priority":       87,
        "description":    "ICA mitigative action risk queries — route to risk_level table via UUID FK. Does NOT apply to form observation queries which correctly use ai_answers.",
        "suppresses":     [],
        "question_pattern": (
            # Only fires for MITIGATIVE ACTION risk queries.
            # Form observation queries ("high risk observations") use ai_answers correctly.
            r"\b(high.{0,10}risk|critical.{0,10}risk).{0,30}"
            r"(mitigative.{0,10}action)|"
            r"\b(mitigative.{0,10}action).{0,30}"
            r"(high.{0,10}risk|critical.{0,10}risk)|"
            r"\b(risk.{0,10}breakdown|breakdown.{0,10}risk|"
            r"risk.{0,10}distribut|risk.{0,10}count|count.{0,10}risk|"
            r"risk.{0,10}by.{0,20}facilit|by.{0,10}risk.{0,10}level)\b"
        ),
        "hint": (
            "\nICA RISK ROUTING — mitigative action risk levels use inspection_corrective_action:\n"
            "  For risk level COUNTS / BREAKDOWNS on mitigative actions:\n"
            "  ALWAYS use the risk_level table via inspection_corrective_action:\n"
            "  NOTE: form 'high risk observations' uses ai_answers correctly — this rule\n"
            "  only applies to mitigative-action risk queries.\n"
            "  ALWAYS use the risk_level table via inspection_corrective_action:\n"
            "  JOIN risk_level rl ON ica.risk_level_id = rl.id\n"
            "  WHERE rl.name = \'High\'   -- or \'Medium\' / \'Low\'\n"
            "\nCorrect pattern for high-risk observations this month:\n"
            "SELECT COUNT(*) AS high_risk_count\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN risk_level rl ON ica.risk_level_id = rl.id\n"
            "JOIN inspection_report ir ON ica.inspection_id = ir.id\n"
            "WHERE rl.name = \'High\'\n"
            "  AND ir.active = true AND ica.active = true\n"
            "  AND date_trunc(\'month\', ir.submitted_on) = date_trunc(\'month\', CURRENT_DATE);\n"
            "\nNEVER: ai_answers WHERE label=\'Observation\' AND answer_text ILIKE \'%High%\'\n"
            "NEVER: ai_questions WHERE label=\'Risk Level\'\n"
        ),
    },

    {
        "id":             "inject_ica_join",
        "type":           INJECT,
        "priority":       80,
        "description":    "ICA + facility/grouping — prevent varchar=UUID join mismatch.",
        "question_pattern": (
            r"\b(mitigative.?action|capex|opex).{0,50}"
            r"(facility|client|inspector|type|count|by|group|breakdown)\b"
            r"|\b(facility|client|inspector|type|breakdown).{0,50}"
            r"(mitigative.?action|capex|opex)\b"
        ),
        "hint": (
            "\nCRITICAL JOIN — inspection_corrective_action to inspection_report:\n"
            "  ica.inspection_id → UUID FK to inspection_report.id (UUID PK)\n"
            "  ir.inspection_id  → VARCHAR human-readable ('2026/04/ST001/INS001') — DIFFERENT\n"
            "  ALWAYS: JOIN inspection_corrective_action ica ON ica.inspection_id = ir.id\n"
            "  NEVER:  ON ir.inspection_id = ica.inspection_id  ← varchar=UUID type mismatch\n"
        ),
    },

    {
        "id":             "inject_most_recent_inspection",
        "type":           INJECT,
        "priority":       70,
        "description":    "Most recent inspection — force direct inspection_report query.",
        "question_pattern": (
            r"\b(most\s+recent|latest|last)\s+(inspection|site|facility|form)\b"
            r"|\b(which|what)\s+(facility|site).{0,30}(most\s+recent|latest|last)\b"
        ),
        "hint": (
            "\nDIRECT QUERY — do NOT route through ai_answers for this:\n"
            "SELECT ir.inspection_id, fac.name AS facility_name, ir.submitted_on, ir.status\n"
            "FROM inspection_report ir\n"
            "JOIN facility fac ON ir.facility_id = fac.id\n"
            "WHERE ir.status != 'DRAFT'\n"
            "ORDER BY ir.submitted_on DESC LIMIT 1;\n"
            "CRITICAL: always SELECT ir.inspection_id — needed for multi-turn follow-up.\n"
        ),
    },

    {
        "id":             "inject_deferred_chain",
        "type":           INJECT,
        "priority":       80,
        "description":    "Deferred / recurring action chain — CLOSE_WITH_DEFERRED pattern.",
        "question_pattern": (
            r"\b(deferred|recurring|recur|carried.?forward|repeated.?finding|"
            r"same.?issue|chronic|persistent|repeat.?observation|"
            r"close.?with.?deferred|multiple.?cycles?)\b"
        ),
        "hint": (
            "\nCLOSE_WITH_DEFERRED — deferred/recurring action pattern:\n"
            "  Status value: 'CLOSE_WITH_DEFERRED' (exact uppercase)\n"
            "  ALWAYS join: ica → ir (ON ica.inspection_id = ir.id) → facility\n"
            "Example (recurring issues at a facility):\n"
            "SELECT ica.cause, COUNT(*) AS recurrence_count, fac.name AS facility_name\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN inspection_report ir ON ica.inspection_id = ir.id\n"
            "JOIN facility fac ON ir.facility_id = fac.id\n"
            "WHERE ica.status = 'CLOSE_WITH_DEFERRED'\n"
            "GROUP BY ica.cause, fac.name ORDER BY recurrence_count DESC LIMIT 20;\n"
        ),
    },

    {
        "id":             "inject_return_for_modification",
        "type":           INJECT,
        "priority":       70,
        "description":    "Return for modification — exact WHERE clause, no extra conditions.",
        "question_pattern": r"\breturn.{0,10}modif\b",
        "hint": (
            "\nUSE EXACTLY: WHERE ir.status = 'RETURN_FOR_MODIFICATION'"
            " — no date filter, no extra conditions.\n"
        ),
    },

    # Portfolio "sent back for modification" was getting hijacked by the
    # semantically-similar-but-wrong-table "inspections returned for modification"
    # seed example (2026-07-08: DeepSeek ignored Llama's correct ip.status='MODIFICATION'
    # schema_hint and generated ir.status='RETURN_FOR_MODIFICATION' SQL instead —
    # executed fine, silently answered the wrong question). required_tables forces
    # a guarded retry if inspector_portfolio doesn't end up in the generated SQL.
    {
        "id":             "inject_portfolio_modification",
        "type":           INJECT,
        "priority":       92,
        "description":    "Portfolio sent-back/modification questions — inspector_portfolio.status, NOT inspection_report.status.",
        "required_tables": ["inspector_portfolio"],
        "forbidden_patterns": [r"RETURN_FOR_MODIFICATION"],
        "suppresses":     ["inject_return_for_modification"],
        "question_pattern": (
            r"\bportfolio.{0,30}(modif|sent\s*back|returned)\b"
            r"|(modif|sent\s*back|returned).{0,30}\bportfolio\b"
        ),
        "hint": (
            "\nPORTFOLIO MODIFICATION — this is about inspector_portfolio, NOT inspection_report:\n"
            "  inspector_portfolio.status = 'MODIFICATION' means the portfolio was sent back for changes.\n"
            "  This is a DIFFERENT enum on a DIFFERENT table from inspection_report.status = "
            "'RETURN_FOR_MODIFICATION' — do not use ir.status here.\n"
            "USE: SELECT u.first_name || ' ' || u.last_name AS inspector_name, ip.submitted_date\n"
            "FROM inspector_portfolio ip\n"
            "JOIN users u ON ip.user_id = u.id\n"
            "WHERE ip.status = 'MODIFICATION' AND ip.active = true AND u.active = true\n"
            "ORDER BY ip.submitted_date DESC;\n"
        ),
    },

    {
        "id":             "inject_overdue_quarter",
        "type":           INJECT,
        "priority":       70,
        "description":    "Overdue inspections this quarter — route to inspection_schedule.",
        "question_pattern": (
            r"\boverdue.{0,20}(this\s+quarter|quarter)\b"
            r"|\b(this\s+quarter|quarter).{0,20}overdue\b"
        ),
        "hint": (
            "\nUSE: SELECT COUNT(*) AS overdue_count FROM inspection_schedule\n"
            "WHERE status = 'OVERDUE'\n"
            "  AND due_date >= date_trunc('quarter', CURRENT_DATE);\n"
        ),
    },

    {
        "id":             "inject_raised_vs_closed_quarter",
        "type":           INJECT,
        "priority":       85,
        "description":    "Raised vs closed mitigative actions — must use FILTER, not GROUP BY.",
        "question_pattern": (
            r"\b(raised|created|opened).{0,30}(closed|completed|resolved)\b"
            r"|\b(closed|completed|resolved).{0,30}(raised|created|opened)\b"
            r"|\bhow\s+many.{0,30}mitigative.{0,20}(raised|closed|quarter)\b"
        ),
        "hint": (
            "\nUSE THIS EXACT SINGLE-ROW FORM — NOT GROUP BY status:\n"
            "SELECT COUNT(*) AS total_raised,\n"
            "       COUNT(*) FILTER (WHERE status IN ('CLOSED','CLOSE_WITH_DEFERRED')) AS total_closed,\n"
            "       COUNT(*) FILTER (WHERE status IN ('OPEN','OVERDUE')) AS still_open\n"
            "FROM inspection_corrective_action\n"
            "WHERE created_on >= date_trunc('quarter', NOW());\n"
            "CRITICAL: NEVER use GROUP BY status — that gives multiple rows. Return ONE row.\n"
        ),
    },

    {
        "id":             "inject_avg_inspection_duration",
        "type":           INJECT,
        "priority":       80,
        "description":    "Average inspection duration — EPOCH extraction with IS NOT NULL guard.",
        "question_pattern": (
            r"\b(average|avg|mean).{0,20}(duration|time|hours?|length)\b"
            r"|\b(duration|time|hours?|length).{0,20}(average|avg|inspection)\b"
        ),
        "hint": (
            "\nUSE THIS EXACT PATTERN — start_date_time has NULLs, filter them out:\n"
            "SELECT ROUND(\n"
            "    AVG(EXTRACT(EPOCH FROM (ir.submitted_on - ir.start_date_time)) / 3600)::numeric,\n"
            "    2\n"
            ") AS avg_duration_hours\n"
            "FROM inspection_report ir\n"
            "WHERE ir.status != 'DRAFT'\n"
            "  AND ir.start_date_time IS NOT NULL;\n"
            "NEVER use: AVG(ir.submitted_on - ir.start_date_time) — returns interval not hours.\n"
        ),
    },

    {
        "id":             "inject_most_recent_form_answers",
        "type":           INJECT,
        "priority":       90,
        "description":    "Most recent inspection form Q&A — use aa subquery not ir subquery.",
        # Suppresses inject_most_recent_inspection when both patterns match, preventing
        # contradictory hints (one says 'use ir.submitted_on', the other says 'use aa subquery').
        "suppresses":     ["inject_most_recent_inspection"],
        "question_pattern": (
            # verb + answer/question/form + recency qualifier
            r"\b(list|show|get|fetch|display|what|give).{0,30}"
            r"(question|answer(?:s|ed)?|form).{0,30}"
            r"(most\s+recent(?:ly)?|latest|last)"
            # recency qualifier + answer/question/form (any order for the noun)
            r"|\b(most\s+recent(?:ly)?|latest|last).{0,30}"
            r"(question|answer(?:s|ed)?|form)"
            # answer/question/form + recency qualifier (handles "answers for the latest")
            r"|\b(answer(?:s|ed)?|question|form).{0,30}"
            r"(most\s+recent(?:ly)?|latest|last)\b"
        ),
        "hint": (
            "\nMOST RECENT FORM — always anchor on ai_answers.submitted_on, not ir.submitted_on.\n"
            "ai_answers is a complete flattened snapshot so every inspection has rows,\n"
            "but the submission timestamps in ai_answers may differ from ir.submitted_on.\n"
            "Use the aa subquery to guarantee you get the inspection with the latest FORM submission:\n"
            "\n-- Pattern A: user wants Q&A from the most recent form\n"
            "SELECT aq.label AS question, aa.answer_text AS answer, aa.module_name\n"
            "FROM ai_answers aa\n"
            "JOIN ai_questions aq ON aa.element_id = aq.element_id\n"
            "WHERE aa.inspection_report_id = (\n"
            "    SELECT aa2.inspection_report_id FROM ai_answers aa2\n"
            "    ORDER BY aa2.submitted_on DESC LIMIT 1\n"
            ")\n"
            "ORDER BY aa.module_name, aq.label LIMIT 200;\n"
            "\n-- Pattern B: user wants inspection metadata for the most recently answered inspection\n"
            "SELECT ir.inspection_id, fac.name AS facility_name,\n"
            "       it.name AS inspection_type_name,\n"
            "       u.first_name || ' ' || u.last_name AS inspector_name,\n"
            "       ir.submitted_on, ir.status, ir.inspection_score\n"
            "FROM inspection_report ir\n"
            "JOIN facility fac ON ir.facility_id = fac.id\n"
            "JOIN inspection_type it ON ir.inspection_type_id = it.id\n"
            "JOIN users u ON ir.inspector_user_id = u.id\n"
            "WHERE ir.id = (\n"
            "    SELECT aa2.inspection_report_id FROM ai_answers aa2\n"
            "    ORDER BY aa2.submitted_on DESC LIMIT 1\n"
            ");\n"
            "\nNEVER: ORDER BY ir.submitted_on DESC LIMIT 1 — use aa subquery instead.\n"
            "NEVER: WHERE aa2.module_name = 'Inspection Form' in the inner subquery — "
            "if the actual module_name differs the subquery returns NULL and the whole query returns empty.\n"
            "Only filter by module_name in the OUTER WHERE using ILIKE if the user names a specific form.\n"
        ),
    },

    # Fault 1 — "completed/conducted/total" semantic mapping
    # hint comes from src.core.semantics_registry — edit there, not here
    # SCOPE LIMIT (eval regression FREQ-05): excludes schedule-context questions —
    # "completed vs overdue in the SCHEDULE" is an inspection_schedule question with
    # different status enums (COMPLETED/OVERDUE/PENDING), handled by the rule below.
    # SCOPE LIMIT 2 (2026-06-12): also excludes overdue/pending/cancelled vocabulary
    # even when "schedule" isn't said literally — "how many inspections are overdue
    # vs pending vs completed" is schedule-status vocabulary (inspection_schedule's
    # own enum), not inspection_report's conducted/completed split.
    # inject_schedule_status (priority 76, below) still requires the literal word
    # "schedule" by design — it is NOT broadened here. If NEITHER rule fires for
    # such a question, the model falls back to the relevant worked example in
    # prompts.py — considered acceptable.
    {
        "id":             "inject_completed_inspections",
        "type":           INJECT,
        "priority":       75,
        "description":    "Total/completed/conducted inspections — canonical two-row category summary.",
        "question_pattern": (
            r"(?i)^(?!.*\bschedule)(?!.*\b(overdue|pending|cancelled)\b)"
            r"(?:.*(?:\b(completed|conducted|total|how\s+many|count)\s*(inspections?|inspection\s+report)\b"
            r"|\binspections?\s*(completed|conducted|total|count|in\s+\w+)\b))"
        ),
        "hint": _STATUS_HINT,
    },

    # "responsible party" phrasing — proactively prevent the users-join instinct
    {
        "id":             "inject_responsible_party_enum",
        "type":           INJECT,
        "priority":       79,
        "description":    "'responsible party' = ica.responsible enum, never a users join.",
        "question_pattern": (
            r"(?i)\bresponsible\s*(party|parties)?\b"
        ),
        "hint": (
            "\n'responsible' / 'responsible party' = ica.responsible directly.\n"
            "  It is a VARCHAR ENUM: CLIENT | INTERNAL_OPERATIONS | SUB_CONTRACTOR — a category, not a person.\n"
            "  SELECT ica.responsible AS responsible_party — NO JOIN to users table.\n"
            "  NEVER: LEFT JOIN users u ON ica.responsible = '...' AND ...u.id — this is not a valid join,\n"
            "  ica.responsible has no FK relationship to users at all.\n"
        ),
    },

    # MA approval-workflow questions — route to implementation_status, never capex_status
    {
        "id":             "inject_ma_approval_workflow",
        "type":           INJECT,
        "priority":       78,
        "description":    "MA approval/progress questions — implementation_status & progress_stage routing.",
        "question_pattern": (
            r"(?i)\b(awaiting|pending)\s*(for\s+)?(approval|review)\b.{0,40}\b(action|MA)s?\b"
            r"|\b(action|MA)s?\b.{0,40}\b(awaiting|pending)\s*(for\s+)?(approval|review)\b"
            r"|\bapproval\b.{0,30}\b(action|MA)s?\b"
            r"|\b(approved|sent\s+back|returned)\b.{0,30}\b(action|MA)s?\b"
            r"|\b(at\s+risk|behind\s+schedule|on\s+track|not\s+started)\b.{0,30}\b(action|MA)s?\b"
            r"|\b(action|MA)s?\b.{0,40}\b(at\s+risk|behind|on\s+track|approved?|pending|sent\s+back|returned)\b"
        ),
        "hint": (
            "\nMA WORKFLOW ROUTING — four independent ica columns, do not confuse:\n"
            "  'awaiting/pending approval/review' → implementation_status = 'PENDING_FOR_REVIEW'\n"
            "      (EXACT string — NOT 'PENDING'. NEVER capex_status — that is the financial track)\n"
            "  'approved'                         → implementation_status = 'APPROVE' (NOT 'APPROVED')\n"
            "  'sent back / returned' (MAs)       → ica.implementation_status = 'RETURN_FOR_MODIFICATION'\n"
            "  'returned' (INSPECTIONS)           → ir.status = 'RETURN_FOR_MODIFICATION' — different table!\n"
            "  'at risk / behind / on track'      → progress_stage (schedule health)\n"
            "      'at risk' = progress_stage='AT_RISK' — NOT risk_level='High' (severity)\n"
            "  'not started'                      → progress_stage = 'NOT_STARTED' OR progress_stage IS NULL\n"
            "  status='CLOSED' ⟺ implementation_status='APPROVE' — closure IS the approval\n"
        ),
    },

    # "Needs attention" dashboard questions — scalar subqueries, never UNION
    {
        "id":             "inject_needs_attention_dashboard",
        "type":           INJECT,
        "priority":       77,
        "description":    "Ambiguous dashboard questions — scalar subquery summary, never UNION ALL.",
        "question_pattern": (
            r"(?i)\b(needs?\s+attention|critical\s+issues?|urgent\s+items?"
            r"|what\s+should\s+i\s+(look|focus)|action\s+items?)\b"
        ),
        "hint": (
            "\nDASHBOARD QUESTION — return ONE ROW of scalar subqueries.\n"
            "NEVER UNION ALL heterogeneous tables (MA rows + schedule rows) — "
            "'N/A' placeholders clash with date/enum column types and fail at runtime.\n\n"
            "USE THIS PATTERN:\n"
            "SELECT\n"
            "    (SELECT COUNT(*) FROM inspection_corrective_action\n"
            "     WHERE status = 'OVERDUE' AND active = true)  AS overdue_mitigative_actions,\n"
            "    (SELECT COUNT(*) FROM inspection_corrective_action\n"
            "     WHERE status = 'OPEN' AND active = true)     AS open_mitigative_actions,\n"
            "    (SELECT COUNT(*) FROM inspection_schedule\n"
            "     WHERE status = 'OVERDUE')                    AS overdue_scheduled_inspections,\n"
            "    (SELECT COUNT(*) FROM inspection_corrective_action ica\n"
            "     JOIN risk_level rl ON ica.risk_level_id = rl.id\n"
            "     WHERE rl.name = 'High' AND ica.status IN ('OPEN','OVERDUE')\n"
            "       AND ica.active = true)                     AS high_risk_open_cas;\n"
        ),
    },

    # Schedule-status questions — inspection_schedule has its OWN status enum
    {
        "id":             "inject_schedule_status",
        "type":           INJECT,
        "priority":       76,
        "description":    "Schedule status questions — inspection_schedule enum, not inspection_report.",
        "question_pattern": (
            r"(?i)\b(completed|overdue|pending|cancelled)\b.{0,50}\bschedule\b"
            r"|\bschedule\b.{0,50}\b(completed|overdue|pending|cancelled)\b"
        ),
        "hint": (
            "\nSCHEDULE STATUS — this is an inspection_schedule question, NOT inspection_report:\n"
            "  inspection_schedule.status values: PENDING | COMPLETED | OVERDUE | CANCELLED\n"
            "  (different enum from inspection_report — 'COMPLETED' here, 'CLOSED' there)\n"
            "  inspection_schedule DOES have an active column — always filter isched.active = true\n"
            "  Columns: active, schedule_date (NOT scheduled_date), due_date, schedule_id, status\n\n"
            "PATTERN for completed vs overdue in the schedule this year:\n"
            "SELECT isched.status, COUNT(*) AS count\n"
            "FROM inspection_schedule isched\n"
            "WHERE isched.status IN ('COMPLETED','OVERDUE')\n"
            "  AND isched.active = true\n"
            "  AND EXTRACT(YEAR FROM isched.due_date) = EXTRACT(YEAR FROM CURRENT_DATE)\n"
            "GROUP BY isched.status;\n"
        ),
    },

    # Fault 3/4/9 — Single-row comparison queries
    {
        "id":             "inject_this_vs_last_comparison",
        "type":           INJECT,
        "priority":       85,
        "description":    "This X vs last X comparison — single-row FILTER aggregate, not GROUP BY.",
        "question_pattern": (
            r"\b(this|current)\s+(month|quarter|year|week).{0,20}"
            r"(vs\.?|versus|compared?\s+to|against).{0,20}"
            r"(last|previous|prior)\s+(month|quarter|year|week)\b"
            r"|\b(last|previous|prior)\s+(month|quarter|year|week).{0,20}"
            r"(vs\.?|versus|compared?\s+to)\b"
        ),
        "hint": (
            "\nSINGLE-ROW COMPARISON — use FILTER aggregate, NOT GROUP BY:\n"
            "SELECT\n"
            "  COUNT(*) FILTER (WHERE date_trunc('month', col) = date_trunc('month', NOW())) AS this_month,\n"
            "  COUNT(*) FILTER (WHERE date_trunc('month', col) = date_trunc('month', NOW()-INTERVAL '1 month')) AS last_month\n"
            "FROM table WHERE ...;\n"
            "CRITICAL: Do NOT add GROUP BY or TO_CHAR. This returns EXACTLY ONE ROW.\n"
        ),
    },

    # ICA risk level — prevent UUID drift (ICA-02/03 regression fix)
    {
        "id":             "inject_ica_risk_level",
        "type":           INJECT,
        "priority":       95,
        "description":    "ICA risk level queries — ALWAYS JOIN risk_level table, never select UUID.",
        "question_pattern": (
            # Must mention mitigative action or risk breakdown in context of actions/counts
            r"\b(high|medium|low)\s*risk\s*(mitigative.?action|action|count|breakdown|all\s+mitigative)\b"
            r"|\b(mitigative.?action).{0,40}(risk|high.?risk|medium.?risk|low.?risk)\b"
            r"|\brisk.?level.{0,30}(mitigative|action|breakdown|all\s+mitigative)\b"
            r"|\b(how\s+many|count|show|list).{0,20}(high|medium|low)\s*risk\s*(mitigative|action)\b"
        ),
        "hint": (
            "\nICA RISK LEVEL — risk_level_id is a UUID FK, NEVER a string:\n"
            "  ALWAYS: JOIN risk_level rl ON ica.risk_level_id = rl.id\n"
            "  ALWAYS: SELECT rl.name AS risk_level_name\n"
            "  ALWAYS: WHERE rl.name = 'High'  (for high-risk filter)\n"
            "  NEVER:  WHERE ica.risk_level_id = 'High'\n"
            "  NEVER:  SELECT ica.risk_level_id  ← outputs UUID garbage\n"
            "  NEVER:  GROUP BY ica.risk_level_id  ← UUID groups, not readable names\n"
            "\nCount by risk level (ICA-02 canonical form):\n"
            "SELECT rl.name AS risk_level_name, COUNT(*) AS count\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN risk_level rl ON ica.risk_level_id = rl.id\n"
            "GROUP BY rl.name ORDER BY count DESC;\n"
            "\nCount only High risk (ICA-03 canonical form):\n"
            "SELECT COUNT(*) AS high_risk_count\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN risk_level rl ON ica.risk_level_id = rl.id\n"
            "WHERE rl.name = 'High';\n"
        ),
    },

    # impact table columns — risk_level and impact are structurally IDENTICAL
    # (id, active, name, scale numeric, is_positive_impact bool, short_code) which
    # caused a 6-iteration, 2-error struggle on "highest impact scale rating"
    # (2026-07-08): neither table's scale/is_positive_impact column is exposed by
    # the normal lookup-table schema hint (which only ever shows .name), so the
    # model guessed plausible-sounding names (impact_scale, impact_scale_rating).
    {
        "id":             "inject_impact_scale",
        "type":           INJECT,
        "priority":       94,
        "description":    "Impact scale/rating/positive-impact questions — impact table has scale + is_positive_impact, NOT risk_level.",
        "required_tables": ["impact"],
        "question_pattern": (
            r"\bimpact.{0,20}(scale|rating|score)\b"
            r"|\b(scale|rating|score).{0,20}\bimpact\b"
            r"|\bpositive\s+impact\b|\bimpact.{0,15}positive\b"
        ),
        "hint": (
            "\nIMPACT SCALE/RATING — impact and risk_level are TWIN tables with the SAME columns "
            "(id, name, scale numeric, is_positive_impact bool, short_code) but are DIFFERENT lookup "
            "tables reached via DIFFERENT FKs — do not confuse them:\n"
            "  risk_level = severity classification, JOIN risk_level rl ON ica.risk_level_id = rl.id\n"
            "  impact     = impact classification,   JOIN impact im ON ica.impact_id = im.id\n"
            "ALWAYS: JOIN impact im ON ica.impact_id = im.id\n"
            "ALWAYS: SELECT im.scale, im.is_positive_impact, im.name AS impact_name\n"
            "NEVER:  im.impact_scale / im.impact_scale_rating / im.rating — these column names don't exist\n"
            "\nHighest impact scale (canonical form):\n"
            "SELECT ica.corrective_action_id, im.name AS impact_name, im.scale\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN impact im ON ica.impact_id = im.id\n"
            "WHERE ica.active = true AND im.active = true\n"
            "ORDER BY im.scale DESC LIMIT 50;\n"
            "\nCount with positive impact (canonical form):\n"
            "SELECT COUNT(*) AS positive_impact_count\n"
            "FROM inspection_corrective_action ica\n"
            "JOIN impact im ON ica.impact_id = im.id\n"
            "WHERE ica.active = true AND im.active = true AND im.is_positive_impact = true;\n"
        ),
    },

    # EDG-02 — "how many questions" routing disambiguation
    {
        "id":             "inject_count_questions_in_form",
        "type":           INJECT,
        "priority":       95,
        "description":    "Count questions in a form — route to COUNT(ai_questions), NOT list_forms.",
        "question_pattern": (
            r"\b(how\s+many|count|number\s+of).{0,25}"
            r"(question|field|item|section).{0,25}"
            r"(form|inspection|module)\b"
            r"|\b(form|inspection|module).{0,25}"
            r"(how\s+many|count|number\s+of).{0,25}"
            r"(question|field|item)\b"
        ),
        "hint": (
            "\nCOUNT QUESTIONS — use generate_sql with COUNT(*) on ai_questions, NOT list_forms.\n"
            "list_forms returns a summary table; it cannot answer 'how many questions' naturally.\n"
            "\nIf user names a specific form:\n"
            "SELECT COUNT(*) AS question_count\n"
            "FROM ai_questions\n"
            "WHERE module_name ILIKE '%<form_name>%';\n"
            "\nIf user did not name a form, list all forms and their counts:\n"
            "SELECT module_name, COUNT(*) AS question_count\n"
            "FROM ai_questions\n"
            "WHERE module_name IS NOT NULL AND module_name != ''\n"
            "GROUP BY module_name\n"
            "ORDER BY question_count DESC;\n"
            "NEVER call list_forms then generate_sql separately — one generate_sql call is enough.\n"
        ),
    },

    # Join registry — schedule/cycle/frequency tables
    # These tables are visible in the schema but their join paths are non-obvious.
    # Without explicit guidance deepseek will hallucinate joins or miss the chain.
    {
        "id":             "inject_inspection_schedule_joins",
        "type":           INJECT,
        "priority":       80,
        "description":    "Schedule/cycle/frequency queries — correct join chain and column names.",
        "question_pattern": (
            r"\b(schedule[ds]?|due\s+date|upcoming|overdue\s+inspection)\b"
            r"|\b(frequency|cycle|recurring|period)\s+(of\s+)?(inspection|visit|audit)\b"
            r"|\b(inspection|visit|audit).{0,30}(frequency|cycle|recurring|period)\b"
            r"|\bhow\s+(often|frequent)\b"
        ),
        "hint": (
            "\nINSPECTION CYCLE / FREQUENCY — EXACT column names (errors occur with wrong names):\n"
            "  inspection_schedule.inspection_cycle_id  → inspection_cycle.id\n"
            "  inspection_schedule.portfolio_details_id → inspector_portfolio_details.id\n"
            "    NEVER: isched.portfolio_id or isched.inspection_schedule_id (don't exist)\n"
            "  inspector_portfolio_details.frequency_definition_id → frequency_definition.id\n"
            "  inspector_portfolio_details.portfolio_id → inspector_portfolio.id\n"
            "    NEVER: ipd.cycle_id (does not exist — cycle goes through inspector_portfolio)\n"
            "  inspector_portfolio.cycle_id → inspection_cycle.id\n"
            "\nfrequency_definition columns: id, name, repeat_count, repeat_interval,\n"
            "  repeat_unit (DAY/WEEK/MONTH)\n"
            "  'Daily'=repeat_count=1,repeat_interval=1,DAY\n"
            "  'Alternate Days'=repeat_count=1,repeat_interval=2,DAY\n"
            "\nFor frequency breakdown of a specific cycle (use GROUP BY, not flat join):\n"
            "SELECT fd.name AS frequency_name, fd.repeat_count, fd.repeat_interval,\n"
            "       fd.repeat_unit, COUNT(*) AS schedule_count\n"
            "FROM inspection_schedule isched\n"
            "JOIN inspector_portfolio_details ipd ON isched.portfolio_details_id = ipd.id\n"
            "JOIN frequency_definition fd ON ipd.frequency_definition_id = fd.id\n"
            "WHERE isched.inspection_cycle_id = '<cycle_id>'\n"
            "GROUP BY fd.name, fd.repeat_count, fd.repeat_interval, fd.repeat_unit\n"
            "ORDER BY schedule_count DESC;\n"
            "\nFor overdue schedule count:\n"
            "SELECT COUNT(*) AS overdue_count FROM inspection_schedule\n"
            "WHERE status = 'OVERDUE'\n"
            "  AND due_date >= date_trunc('quarter', NOW())\n"
            "  AND due_date < date_trunc('quarter', NOW()) + INTERVAL '3 months';\n"
        ),
    },

    # Scores across inspection type categories — group ALL types, never filter by name
    {
        "id":             "inject_inspection_scores_by_type",
        "type":           INJECT,
        "priority":       88,
        "description":    "Inspection scores across named categories — group all inspection types without name-filter.",
        "question_pattern": (
            # "inspection scores across safety, hygiene, ..."
            r"(?i)\binspection\s+score\w*\b.{0,60}"
            r"\b(across|by|per|for\s+each|breakdown|compare|versus|vs)\b"
            # "scores across / by <anything> / breakdown"
            r"|\bscore\w*\b.{0,30}\b(across|by\s+\w+|breakdown)\b"
            # two or more inspection-domain keywords in the same sentence with a score word
            r"|\b(safety|hygiene|structure|compliance|documentation|cleanliness|fire"
            r"|electrical|mechanical|health)\b.{0,100}"
            r"\b(safety|hygiene|structure|compliance|documentation|cleanliness|fire"
            r"|electrical|mechanical|health)\b.{0,60}"
            r"\b(score\w*|rating\w*|inspection)\b"
            r"|\b(score\w*|rating\w*)\b.{0,40}"
            r"\b(safety|hygiene|structure|compliance|documentation|cleanliness|fire"
            r"|electrical|mechanical|health)\b.{0,80}"
            r"\b(safety|hygiene|structure|compliance|documentation|cleanliness|fire"
            r"|electrical|mechanical|health)\b"
        ),
        "hint": (
            "\nINSPECTION SCORES BY TYPE — CRITICAL RULE:\n"
            "NEVER filter WHERE it.name IN (...) or WHERE it.name ILIKE '%safety%' — "
            "those names may not match the DB and return ZERO rows.\n"
            "The user's category words (safety, hygiene, structure, compliance, documentation) "
            "are INTENT labels, NOT exact strings stored in inspection_type.name.\n"
            "\nALWAYS group ALL inspection types without a name filter:\n"
            "SELECT it.name AS inspection_type,\n"
            "       ROUND(AVG(ir.inspection_score)::numeric, 1) AS avg_score\n"
            "FROM inspection_report ir\n"
            "JOIN inspection_type it ON ir.inspection_type_id = it.id\n"
            "WHERE ir.active = true AND ir.status != 'DRAFT' AND it.active = true\n"
            "GROUP BY it.name\n"
            "ORDER BY avg_score DESC;\n"
            "Return ONLY these 2 columns (inspection_type, avg_score) — no count column.\n"
            "This gives exactly 1 categorical + 1 score column so the chart advisor auto-selects radar.\n"
        ),
    },

    # P&L summary — single-row waterfall trigger
    {
        "id":          "inject_pl_summary_waterfall",
        "type":        INJECT,
        "priority":    89,
        "description": "Overall P&L / income-cost-profit summary — force 3-column single-row pattern for waterfall.",
        "question_pattern": (
            r"(?i)\b(total|overall|summary)\b.{0,30}\b(income|revenue)\b"
            r"|\b(income|revenue).{0,30}\b(cost|expense|expenditure)\b"
            r"|\b(profit|net\s+profit|p&l|profit\s+and\s+loss|profit\s+loss)\b.{0,40}"
            r"\b(total|overall|summary|income|revenue|cost)\b"
            r"|\b(income|revenue).{0,20}\b(cost|expense).{0,20}\b(profit|net)\b"
            r"|\bfinancial\s+summary\b"
        ),
        "hint": (
            "\nP&L SUMMARY — WATERFALL PATTERN (single row, 3 exact column aliases):\n"
            "ALWAYS return exactly 1 row with these 3 column names — no other columns:\n"
            "  total_income  → SUM of po_details.po_amount  (client paid Khidmah)\n"
            "  total_cost    → SUM of pr_details.pr_amount  (Khidmah paid out)\n"
            "  net_profit    → total_income - total_cost\n"
            "NEVER alias as: po_total, pr_total, revenue, expenses, balance — use EXACT names above.\n"
            "NEVER GROUP BY anything — one global row only.\n"
            "NEVER add a category/label column — only the 3 numeric columns.\n"
            "\nCORRECT SQL:\n"
            "WITH po_agg AS (\n"
            "    SELECT mitigative_action_uuid AS ica_id, SUM(po_amount) AS total_income\n"
            "    FROM po_details WHERE active = true GROUP BY 1\n"
            "),\n"
            "pr_agg AS (\n"
            "    SELECT mitigative_action_uuid AS ica_id, SUM(pr_amount) AS total_cost\n"
            "    FROM pr_details WHERE active = true GROUP BY 1\n"
            ")\n"
            "SELECT\n"
            "    COALESCE(SUM(po.total_income), 0)                                   AS total_income,\n"
            "    COALESCE(SUM(pr.total_cost),  0)                                    AS total_cost,\n"
            "    COALESCE(SUM(po.total_income), 0) - COALESCE(SUM(pr.total_cost), 0) AS net_profit\n"
            "FROM inspection_corrective_action ica\n"
            "LEFT JOIN po_agg po ON po.ica_id = ica.id\n"
            "LEFT JOIN pr_agg pr ON pr.ica_id = ica.id\n"
            "WHERE ica.active = true;\n"
        ),
    },

]
