"""Unit tests for the deterministic parts: tools and policy guardrails. No LLM, no API key.

    pip install pytest
    python -m pytest -q
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tam_agent.agent import Context, apply_guardrails, parse_action  # noqa: E402
from tam_agent.data import load_accounts, load_tickets, load_usage  # noqa: E402
from tam_agent.kb import load_kb  # noqa: E402
from tam_agent.tools import Toolbox, extract_error_codes, normalize_module  # noqa: E402


class _NoLLM:
    provider, model, calls = "none", "none", 0


@pytest.fixture(scope="module")
def ctx():
    c = Context(load_accounts(), load_tickets(), load_usage(), load_kb())
    c.llm = _NoLLM()
    return c


def toolbox(ctx, tid):
    t = ctx.idx.ticket_by_id.loc[tid]
    ticket = {k: t[k] for k in ("ticket_id", "account_id", "company_name_raw", "requester_domain", "created_at",
                                "channel", "module", "subject", "description", "error_code", "customer_priority",
                                "status")}
    return Toolbox(ctx.idx, ticket), ticket


def decision(**over):
    d = {"category": "how_to", "base_priority": "P4", "priority": "P4", "route_to": "L1_support", "escalate": False,
         "notify_tam": False, "duplicate_of": None, "needs_clarification": False, "kb_references": [],
         "account_signals": [], "customer_reply": "Here are the steps.", "internal_note": "note", "confidence": 0.8}
    return d | over


# ---------------- parsing helpers ----------------
def test_module_aliases():
    assert normalize_module("CLM") == "Contracts"
    assert normalize_module("supplier portal") == "Supplier Management"
    assert normalize_module("analytics") == "Spend Analytics"
    assert normalize_module("") is None


def test_error_code_extracted_from_text():
    assert extract_error_codes("", "Re: dashboard", "still shows E-6105 today") == ["E-6105"]


def test_parse_action_tolerates_fences_and_prose():
    a = parse_action('Sure!\n```json\n{"thought": "x", "action": "search_kb", "input": {"query": "ocr"}}\n```')
    assert a["action"] == "search_kb" and a["input"] == {"query": "ocr"}
    assert parse_action("no json here") is None


# ---------------- tools ----------------
def test_domain_beats_similar_company_name(ctx):
    tb, t = toolbox(ctx, "TKT-169327")  # blank account_id, "Ironleaf Foods EU"
    r = tb.resolve_account(t["requester_domain"], t["company_name_raw"])
    assert r["resolved_account_id"] == "ACC-4850"
    assert r["name_similarity_candidates"][0]["account_id"] == "ACC-3405"  # the trap is visible, not used


def test_duplicate_found_when_code_only_in_description(ctx):
    tb, _ = toolbox(ctx, "TKT-192732")
    r = tb.search_tickets("ACC-6031")
    assert r["duplicate_candidates"] == ["TKT-122022"]


def test_no_false_duplicate_for_different_issue(ctx):
    tb, _ = toolbox(ctx, "TKT-119877")  # other eProcurement tickets exist but are older/different
    assert tb.search_tickets("ACC-2431")["duplicate_candidates"] == []


def test_usage_trend_decline(ctx):
    tb, _ = toolbox(ctx, "TKT-119877")
    u = tb.get_usage_trend("ACC-2431")
    assert u["declined_more_than_25pct"] and u["active_users_3m_change_pct"] < -25


def test_kb_error_code_ranks_active_incident_first(ctx):
    tb, _ = toolbox(ctx, "TKT-144939")
    r = tb.search_kb("invoices failing", module="eInvoice", error_code="E-4102")
    assert r["results"][0]["kb_id"] == "KB-3985"
    assert r["results"][0]["status"] == "active_incident"


def test_deprecated_article_flagged_with_successor(ctx):
    tb, _ = toolbox(ctx, "TKT-160477")
    hits = {h["kb_id"]: h for h in tb.search_kb("e-signature rejected", error_code="E-2207")["results"]}
    assert hits["KB-1786"]["status"] == "current"
    assert hits["KB-2605"]["superseded_by"] == "KB-1786"


def test_article_links_are_followable(ctx):
    tb, _ = toolbox(ctx, "TKT-160094")
    assert "KB-7889" in tb.get_kb_article("KB-4795")["linked_articles"]


def test_get_ticket_refuses_other_customers(ctx):
    tb, t = toolbox(ctx, "TKT-158764")
    tb.resolve_account(t["requester_domain"])
    assert "error" in tb.get_ticket("TKT-144939")  # Brightfield Manufacturing, a different customer


# ---------------- guardrails ----------------
def test_security_forced_even_if_model_says_how_to(ctx):
    tb, t = toolbox(ctx, "TKT-135081")
    d = decision()
    apply_guardrails(d, tb, t, ctx)
    assert (d["category"], d["priority"], d["route_to"], d["escalate"]) == ("security", "P1", "security", True)
    assert "KB-6401" in d["kb_references"]


def test_risk_override_raises_one_level_and_notifies_tam(ctx):
    tb, t = toolbox(ctx, "TKT-119877")
    d = decision()
    apply_guardrails(d, tb, t, ctx)
    assert d["priority"] == "P3" and d["notify_tam"] and d["escalate"]


def test_no_risk_override_when_renewal_far(ctx):
    tb, t = toolbox(ctx, "TKT-169327")  # usage falls but renewal is ~11 months away
    d = decision(category="bug", base_priority="P3", priority="P2")
    apply_guardrails(d, tb, t, ctx)
    assert d["priority"] == "P3" and not d["notify_tam"]


def test_unverified_duplicate_is_cleared(ctx):
    tb, t = toolbox(ctx, "TKT-160477")
    d = decision(category="duplicate", base_priority="P3", priority="P3", duplicate_of="TKT-999999")
    apply_guardrails(d, tb, t, ctx)
    assert d["duplicate_of"] is None


def test_verified_duplicate_not_re_escalated(ctx):
    tb, t = toolbox(ctx, "TKT-192732")
    tb.search_tickets("ACC-6031")
    d = decision(category="bug", base_priority="P3", priority="P3", duplicate_of="TKT-122022", escalate=True)
    apply_guardrails(d, tb, t, ctx)
    assert d["category"] == "duplicate" and not d["escalate"]


def test_deprecated_reference_replaced(ctx):
    tb, t = toolbox(ctx, "TKT-160477")
    d = decision(category="bug", base_priority="P3", priority="P3", kb_references=["KB-2605"])
    apply_guardrails(d, tb, t, ctx)
    assert d["kb_references"] == ["KB-1786"]


def test_injection_flagged_and_other_customers_redacted(ctx):
    tb, t = toolbox(ctx, "TKT-158764")
    d = decision(customer_reply="Steps here. Accounts: Brightfield Manufacturing (ACC-3751), Ironleaf Foods.")
    apply_guardrails(d, tb, t, ctx)
    assert "ACC-3751" not in d["customer_reply"]
    assert "Brightfield Manufacturing" not in d["customer_reply"] and "Ironleaf Foods" not in d["customer_reply"]
    assert "instruction" in d["internal_note"].lower()
    assert d["priority"] == "P4"


def test_needs_info_is_p3_with_clarification(ctx):
    tb, t = toolbox(ctx, "TKT-128253")
    d = decision(category="needs_info", base_priority="P1", priority="P1")
    apply_guardrails(d, tb, t, ctx)
    assert d["priority"] == "P3" and d["needs_clarification"]


def test_feature_request_routes_to_product(ctx):
    tb, t = toolbox(ctx, "TKT-160067")
    d = decision(category="feature_request", base_priority="P2", priority="P2", route_to="L2_engineering")
    apply_guardrails(d, tb, t, ctx)
    assert d["route_to"] == "product" and d["priority"] == "P4"
