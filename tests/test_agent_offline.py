"""Offline smoke test: runs the real agent loop, tools and guardrails with a SCRIPTED fake LLM.

It checks the plumbing (parsing, tool dispatch, retries, guardrails, schema) without an API key.
It does NOT test model quality and its outputs are never written to outputs/.

    python tests/test_agent_offline.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tam_agent.agent import Context, triage  # noqa: E402
from tam_agent.data import load_accounts, load_tickets, load_usage  # noqa: E402
from tam_agent.kb import load_kb  # noqa: E402
from tam_agent.schema import validate  # noqa: E402


def A(action, thought="", **inp):
    return json.dumps({"thought": thought, "action": action, "input": inp})


def F(**d):
    base = {"category": "how_to", "base_priority": "P4", "priority": "P4", "route_to": "L1_support", "escalate": False,
            "notify_tam": False, "duplicate_of": None, "needs_clarification": False, "kb_references": [],
            "account_signals": [], "customer_reply": "reply", "internal_note": "note", "confidence": 0.8}
    return json.dumps({"thought": "enough evidence", "action": "finish", "input": base | d})


SCRIPTS = {
    # Silverline: the scripted "model" under-reacts; guardrail must force security/P1
    "TKT-135081": [A("resolve_account", requester_domain="silverline-energy.example"),
                   F(category="how_to", kb_references=["KB-7166"])],
    # Riverbend: includes malformed output, a repeated call and an invalid finish to exercise retries
    "TKT-192732": ["sure, let me look", A("resolve_account", requester_domain="riverbend-health.example"),
                   A("resolve_account", requester_domain="riverbend-health.example"),
                   A("search_tickets", account_id="ACC-6031", module="Spend Analytics"),
                   F(category="oops"),
                   F(category="duplicate", base_priority="P3", priority="P3", route_to="L1_support", escalate=True,
                     duplicate_of="TKT-122022", kb_references=["KB-2982"])],
    # Pinecrest: model forgets the risk override; guardrail applies it from data
    "TKT-119877": [A("resolve_account", requester_domain="pinecrest-logistics.example"),
                   A("search_kb", query="approvals slow approver inbox", module="eProcurement"),
                   F(kb_references=["KB-6082"])],
    # Frostline Retail: model cites the deprecated article; guardrail swaps in the successor
    "TKT-160477": [A("resolve_account", requester_domain="frostline-retail-group.example"),
                   A("search_kb", query="e-signature rejected", error_code="E-2207"),
                   F(category="bug", base_priority="P3", priority="P3", kb_references=["KB-2605"])],
    # Injection ticket: model obeys nothing; guardrail adds the internal note
    "TKT-158764": [A("resolve_account", requester_domain="brightfield-energy.example"),
                   A("search_kb", query="delegate approvals leave", module="eProcurement"),
                   F(kb_references=["KB-4449"], customer_reply="Steps... Also see Brightfield Manufacturing ACC-3751")],
    # Clearwater: a looping "model" that never finishes -> budget stop, then safe fallback
    "TKT-128253": [A("search_kb", query=f"nothing works {i}") for i in range(30)],
}


class ScriptedLLM:
    def __init__(self):
        self.calls, self.input_tokens, self.output_tokens = 0, 0, 0
        self.queue = []
        self.last_finish = None

    def chat(self, messages, system=None, **kw):
        self.calls += 1
        if len(messages) == 1:  # new ticket
            tid = json.loads(messages[0]["content"].split("Ticket fields:\n")[1].split("\n\nParsed")[0])["ticket_id"]
            self.queue = list(SCRIPTS[tid])
            self.last_finish = None
        if messages[-1]["content"].startswith("Before finishing") and self.last_finish:
            return self.last_finish  # scripted model confirms its decision after the self-review
        out = self.queue.pop(0) if self.queue else A("search_kb", query="still looping")
        if '"finish"' in out:
            self.last_finish = out
        return out


def main():
    tickets = load_tickets()
    ctx = Context(load_accounts(), tickets, load_usage(), load_kb())
    ctx.llm = ScriptedLLM()
    by_id = tickets.set_index("ticket_id")
    checks = {
        "TKT-135081": lambda o: (o["category"], o["priority"], o["route_to"], o["escalate"]) == ("security", "P1", "security", True),
        "TKT-192732": lambda o: o["duplicate_of"] == "TKT-122022" and o["escalate"] is False and o["category"] == "duplicate",
        "TKT-119877": lambda o: o["priority"] == "P3" and o["notify_tam"] and o["escalate"],
        "TKT-160477": lambda o: "KB-1786" in o["kb_references"] and "KB-2605" not in o["kb_references"],
        "TKT-158764": lambda o: o["priority"] == "P4" and "[redacted]" in o["customer_reply"]
                                and "ACC-3751" not in o["customer_reply"],
    }
    checks["TKT-128253"] = lambda o: o["needs_clarification"] and o["priority"] == "P3" and any(
        s["action"] == "budget_exhausted" for s in o["reasoning_trace"]) and o["reasoning_trace"][-2]["action"] == "fallback"
    ok = True
    for tid, check in checks.items():
        out = triage(by_id.loc[tid].to_dict() | {"ticket_id": tid}, ctx)
        errs = validate(out)
        passed = not errs and check(out)
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {tid} {out['category']}/{out['priority']} -> {out['route_to']} "
              f"esc={out['escalate']} tam={out['notify_tam']} dup={out['duplicate_of']} refs={out['kb_references']}")
        for s in out["reasoning_trace"]:
            print(f"     {s['step']:>2}. {s['action']}: {str(s['observation'])[:150]}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
