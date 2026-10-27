"""Edge cases: one test per scenario that commonly breaks triage agents. No LLM, no API key.

    python -m pytest -q tests/test_edge_cases.py
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tam_agent.agent import MAX_TOOL_CALLS, Context, apply_guardrails, triage  # noqa: E402
from tam_agent.data import load_accounts, load_tickets, load_usage  # noqa: E402
from tam_agent.kb import load_kb  # noqa: E402
from tam_agent.schema import validate  # noqa: E402
from tam_agent.tools import Toolbox, _days_until  # noqa: E402


class Scripted:
    """Fake LLM that replays a fixed list of replies."""
    provider, model = "scripted", "test"

    def __init__(self, replies):
        self.replies, self.calls, self.input_tokens, self.output_tokens = list(replies), 0, 0, 0

    def chat(self, messages, system=None, **kw):
        self.calls += 1
        return self.replies.pop(0) if self.replies else json.dumps(
            {"thought": "", "action": "search_kb", "input": {"query": f"loop {self.calls}"}})


def act(action, **inp):
    return json.dumps({"thought": "t", "action": action, "input": inp})


def finish(**over):
    d = {"category": "how_to", "base_priority": "P4", "priority": "P4", "route_to": "L1_support", "escalate": False,
         "notify_tam": False, "duplicate_of": None, "needs_clarification": False, "kb_references": [],
         "account_signals": [], "customer_reply": "Steps.", "internal_note": "note", "confidence": 0.8} | over
    return json.dumps({"thought": "done", "action": "finish", "input": d})


@pytest.fixture(scope="module")
def data():
    return load_accounts(), load_tickets(), load_usage(), load_kb()


@pytest.fixture(scope="module")
def ctx(data):
    return Context(*data)


def ticket(ctx, tid):
    return ctx.idx.ticket_by_id.loc[tid][["ticket_id", "account_id", "company_name_raw", "requester_domain",
                                           "created_at", "channel", "module", "subject", "description",
                                           "error_code", "customer_priority", "status"]].to_dict()


def run(ctx, tid, replies):
    ctx.llm = Scripted(replies)
    return triage(ticket(ctx, tid), ctx)


# 1. Blank account ID ------------------------------------------------------------------
def test_blank_account_id_resolved_from_domain_even_if_model_never_looks(ctx):
    t = ticket(ctx, "TKT-169327")
    assert t["account_id"] == ""
    out = run(ctx, "TKT-169327", [finish(category="bug", base_priority="P3", priority="P3"), finish(
        category="bug", base_priority="P3", priority="P3")])
    assert out["account_id"] == "ACC-4850"


# 2. Similar company names ---------------------------------------------------------------
def test_similar_name_cannot_override_domain(ctx):
    tb = Toolbox(ctx.idx, ticket(ctx, "TKT-169327"))
    r = tb.resolve_account("ironleaf-foods-europe.example", company_name="Ironleaf Foods")  # exact name of ANOTHER account
    assert r["resolved_account_id"] == "ACC-4850"


# 3. Old KB article ---------------------------------------------------------------------
def test_deprecated_article_never_cited(ctx):
    out = run(ctx, "TKT-160477", [finish(category="bug", base_priority="P3", priority="P3", kb_references=["KB-2605"]),
                                  finish(category="bug", base_priority="P3", priority="P3", kb_references=["KB-2605"])])
    assert "KB-2605" not in out["kb_references"] and "KB-1786" in out["kb_references"]


# 4. Existing escalation / incident -------------------------------------------------------
def test_active_incident_referenced_not_duplicated(ctx):
    tb = Toolbox(ctx.idx, ticket(ctx, "TKT-144939"))
    d = json.loads(finish(category="incident", base_priority="P1", priority="P1", route_to="L2_engineering",
                          escalate=True, kb_references=["KB-3985"]))["input"]
    apply_guardrails(d, tb, ticket(ctx, "TKT-144939"), ctx)
    assert "INC-26-0418" in d["internal_note"] and d["route_to"] == "L2_engineering"


def test_duplicate_of_open_ticket_is_not_re_escalated(ctx):
    tb = Toolbox(ctx.idx, ticket(ctx, "TKT-192732"))
    tb.search_tickets("ACC-6031")
    d = json.loads(finish(category="bug", base_priority="P2", priority="P2", escalate=True,
                          duplicate_of="TKT-122022"))["input"]
    apply_guardrails(d, tb, ticket(ctx, "TKT-192732"), ctx)
    assert d["escalate"] is False


# 5. Conflicting evidence ------------------------------------------------------------------
def test_conflicting_account_id_and_domain_flags_and_trusts_domain(ctx):
    tb = Toolbox(ctx.idx, ticket(ctx, "TKT-192732"))
    r = tb.resolve_account("riverbend-health.example", account_id="ACC-2431")  # account_id of a different customer
    assert r["resolved_account_id"] == "ACC-6031" and "conflicts" in r["warning"]


# 6. Missing usage history -----------------------------------------------------------------
@pytest.mark.parametrize("keep_months", [0, 2])
def test_missing_usage_history_is_not_invented(data, keep_months):
    accounts, tickets, usage, kb = data
    u = usage[usage.account_id != "ACC-2431"]
    if keep_months:
        u = __import__("pandas").concat([u, usage[usage.account_id == "ACC-2431"].sort_values("month").tail(keep_months)])
    c = Context(accounts, tickets, u, kb)
    tb = Toolbox(c.idx, ticket(c, "TKT-119877"))
    trend = tb.get_usage_trend("ACC-2431")
    assert trend["trend_available"] is False and "active_users_3m_change_pct" not in trend
    d = json.loads(finish())["input"]
    notes = apply_guardrails(d, tb, ticket(c, "TKT-119877"), c)
    # Pinecrest still qualifies on its 2 open escalations, and the gap is stated, not filled in
    assert d["priority"] == "P3" and d["notify_tam"]
    assert any("usage trend unavailable" in n for n in notes)


# 7. Potential duplicate --------------------------------------------------------------------
def test_same_account_same_module_different_error_is_not_duplicate(ctx):
    tb = Toolbox(ctx.idx, ticket(ctx, "TKT-192732"))
    r = tb.search_tickets("ACC-6031")
    assert "TKT-152044" not in r["duplicate_candidates"]  # Spend Analytics too, but E-6169 and 28 days old


# 8. Tool returns no results / tool failure ---------------------------------------------------
def test_empty_kb_search_says_so(ctx):
    r = Toolbox(ctx.idx, ticket(ctx, "TKT-128253")).search_kb("qqxz zzvv", error_code="E-0000")
    assert r["results"] == [] and "No matching articles" in r["note"]


def test_bad_tool_input_does_not_crash_the_run(ctx):
    out = run(ctx, "TKT-158764", [act("get_account_profile", wrong_arg="x"), act("get_kb_article", kb_id="KB-0000"),
                                  act("not_a_tool"), finish(kb_references=["KB-4449"]), finish(kb_references=["KB-4449"])])
    assert not validate(out)
    obs = [s["observation"] for s in out["reasoning_trace"]]
    assert any("bad input" in o for o in obs) and any("unknown article" in o for o in obs)


# 9. LLM produces invalid JSON ------------------------------------------------------------------
def test_invalid_json_is_recovered(ctx):
    out = run(ctx, "TKT-158764", ["not json at all", '{"action": "finish", "input": {"category": "nope"}}',
                                  finish(kb_references=["KB-4449"]), finish(kb_references=["KB-4449"])])
    assert not validate(out)
    actions = [s["action"] for s in out["reasoning_trace"]]
    assert "parse_error" in actions and "finish_rejected" in actions


def test_tool_budget_exhaustion_gives_flagged_fallback_not_a_fabricated_answer(ctx):
    out = run(ctx, "TKT-128253", [])  # model loops on searches forever
    actions = [s["action"] for s in out["reasoning_trace"]]
    assert actions.count("search_kb") == MAX_TOOL_CALLS and "fallback" in actions
    assert out["needs_clarification"] and out["confidence"] <= 0.3 and "human review" in out["internal_note"]


# 10. Renewal date boundary (reference date 2026-10-20) --------------------------------------------
@pytest.mark.parametrize("renewal,expected", [("2027-01-18", True), ("2027-01-19", False), ("2026-10-20", True)])
def test_renewal_90_day_boundary(data, renewal, expected):
    accounts, tickets, usage, kb = data
    a = accounts.copy()
    a.loc[a.account_id == "ACC-2431", "renewal_date"] = renewal
    c = Context(a, tickets, usage, kb)
    prof = Toolbox(c.idx, ticket(c, "TKT-119877")).get_account_profile("ACC-2431")
    assert _days_until("2027-01-18") == 90
    assert prof["renewal_within_90_days"] is expected
