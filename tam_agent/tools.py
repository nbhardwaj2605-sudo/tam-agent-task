"""Tools the agent can call.

Every tool is plain, deterministic Python over the loaded data. No LLM calls
happen in here. The agent (LLM) decides WHICH tool to call and WHEN; the tools
only answer the question they were asked.

Indexes are built once in `Indexes` (per run) so each lookup is cheap even on
the full dataset.
"""
from __future__ import annotations

import difflib
import math
import re
from collections import Counter
from datetime import datetime, timedelta

import pandas as pd

from .data import AS_OF_DATE

TODAY = datetime.fromisoformat(AS_OF_DATE)
ERROR_CODE_RE = re.compile(r"\bE-\d{4}\b")

# ---------------------------------------------------------------------------
# Module normalisation: ticket `module` is free text. Map aliases to the
# canonical names used in accounts.modules_licensed and KB front matter.
# ---------------------------------------------------------------------------
CANONICAL_MODULES = ["Platform", "Contracts", "Sourcing", "Spend Analytics", "Supplier Management",
                     "eProcurement", "eInvoice", "Integrations"]
MODULE_ALIASES = {
    "platform": "Platform", "sso": "Platform", "login": "Platform",
    "contracts": "Contracts", "contract": "Contracts", "clm": "Contracts", "contract mgmt": "Contracts",
    "sourcing": "Sourcing", "rfx": "Sourcing", "esourcing": "Sourcing", "sourcing events": "Sourcing",
    "spend analytics": "Spend Analytics", "spend": "Spend Analytics", "analytics": "Spend Analytics",
    "reports": "Spend Analytics",
    "supplier management": "Supplier Management", "sim": "Supplier Management",
    "supplier mgmt": "Supplier Management", "supplier portal": "Supplier Management",
    "eprocurement": "eProcurement", "eproc": "eProcurement", "e-procurement": "eProcurement",
    "procurement": "eProcurement",
    "einvoice": "eInvoice", "e-invoice": "eInvoice", "einv": "eInvoice", "invoicing": "eInvoice",
    "ap": "eInvoice",
    "integrations": "Integrations", "erp integration": "Integrations", "api": "Integrations",
    "connector": "Integrations",
}


def normalize_module(raw: str | None) -> str | None:
    if not raw or not str(raw).strip():
        return None
    return MODULE_ALIASES.get(str(raw).strip().lower())


def extract_error_codes(*texts: str) -> list[str]:
    seen = []
    for t in texts:
        for c in ERROR_CODE_RE.findall(t or ""):
            if c not in seen:
                seen.append(c)
    return seen


def _days_until(date_str: str) -> int | None:
    if not date_str:
        return None
    return (datetime.fromisoformat(date_str) - TODAY).days


def _norm_company(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    name = re.sub(r"\b(inc|ltd|llc|gmbh|plc|corp|co|group|sa|ag)\b", " ", name)
    return " ".join(name.split())


# ---------------------------------------------------------------------------
# Small BM25 implementation (no dependency, easy to explain).
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[0-9]+)?")
_STOP = set("the a an and or of to in on for is are be with when this that it as by at from your our we you "
            "not no can does do how why what who if than then into after before".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(map(len, docs)) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, d in zip(self.tf, self.docs):
            s, dl = 0.0, len(d)
            for q in query:
                if q in tf:
                    f = tf[q]
                    s += self.idf[q] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


# ---------------------------------------------------------------------------
# Indexes built once per run
# ---------------------------------------------------------------------------
class Indexes:
    def __init__(self, accounts: pd.DataFrame, tickets: pd.DataFrame, usage: pd.DataFrame, kb: list[dict]):
        self.accounts = accounts.set_index("account_id", drop=False)
        self.domain_to_acc = dict(zip(accounts["domain"].str.lower(), accounts["account_id"]))
        self.company_norm = {aid: _norm_company(n) for aid, n in zip(accounts["account_id"], accounts["company_name"])}

        t = tickets.copy()
        t["created_dt"] = pd.to_datetime(t["created_at"])
        t["module_canon"] = t["module"].map(normalize_module)
        # resolved account for every ticket: account_id if present, else via domain (always maps)
        t["acc_resolved"] = [a or self.domain_to_acc.get(d.lower()) for a, d in zip(t["account_id"], t["requester_domain"])]
        t["codes"] = [extract_error_codes(c, s, d) for c, s, d in zip(t["error_code"], t["subject"], t["description"])]
        self.tickets = t
        self.tickets_by_acc = {k: g.sort_values("created_dt") for k, g in t.groupby("acc_resolved")}
        self.ticket_by_id = t.set_index("ticket_id", drop=False)

        self.usage = {k: g.sort_values("month") for k, g in usage.groupby("account_id")}

        self.kb = {a["id"]: a for a in kb}
        self.kb_ids = [a["id"] for a in kb]
        docs = []
        for a in kb:
            m = a["meta"]
            title = m.get("title", "")
            # title counted 3x, plus module, codes and body
            docs.append(tokenize(" ".join([title] * 3 + [m.get("module", ""), " ".join(m.get("error_codes") or []),
                                                          a["body"]])))
        self.bm25 = BM25(docs)
        self.kb_by_code: dict[str, list[str]] = {}
        for a in kb:
            for c in a["meta"].get("error_codes") or []:
                self.kb_by_code.setdefault(c, []).append(a["id"])
        self.policies = {a["id"]: a for a in kb if a["meta"].get("type") == "policy"}


# ---------------------------------------------------------------------------
# Tool implementations. Each returns a JSON-serialisable dict.
# `ticket` is the ticket being triaged (needed for date-relative logic).
# ---------------------------------------------------------------------------
class Toolbox:
    def __init__(self, idx: Indexes, ticket: dict):
        self.idx, self.ticket = idx, ticket
        self.ticket_dt = pd.to_datetime(ticket["created_at"])
        self.facts: dict = {}  # structured facts collected during the run (used by guardrails)

    # ---- tool registry (name -> (fn, description)) used by the prompt ----
    def specs(self) -> dict:
        return {
            "resolve_account": (self.resolve_account,
                                'Identify the customer account. input: {"requester_domain": str, "company_name": str?, '
                                '"account_id": str?}. Domain is authoritative; company names are often ambiguous.'),
            "get_account_profile": (self.get_account_profile,
                                    'Tier, renewal date/days, open escalations, licensed modules, SSO change date, NPS. '
                                    'input: {"account_id": str}'),
            "get_usage_trend": (self.get_usage_trend,
                                'Monthly active users / logins for the account and 3-month change. '
                                'input: {"account_id": str, "months": int?}'),
            "search_tickets": (self.search_tickets,
                               "Other tickets from this account (history, related issues, duplicate candidates per "
                               'policy). input: {"account_id": str, "module": str?, "error_code": str?, "days_back": int?}'),
            "get_ticket": (self.get_ticket, 'Full details of one ticket. input: {"ticket_id": str}'),
            "search_kb": (self.search_kb,
                          "Search knowledge-base articles (keyword/BM25 + exact error-code match). Returns ids, titles, "
                          'type and status (current/deprecated/active_incident). input: {"query": str, "module": str?, '
                          '"error_code": str?, "top_k": int?}'),
            "get_kb_article": (self.get_kb_article,
                               'Read one KB article in full, including links to related articles. input: {"kb_id": str}'),
            "get_policy": (self.get_policy,
                           'Read a support policy. input: {"topic": "priority" | "duplicates" | "data_exposure"}'),
        }

    # ---------------- accounts ----------------
    def resolve_account(self, requester_domain: str = "", company_name: str = "", account_id: str = "") -> dict:
        idx = self.idx
        dom = (requester_domain or "").strip().lower()
        by_domain = idx.domain_to_acc.get(dom)
        out: dict = {}
        if account_id and account_id in idx.accounts.index:
            out["account_id_field"] = account_id
            if by_domain and by_domain != account_id:
                out["warning"] = f"account_id {account_id} conflicts with domain {dom} ({by_domain}); trusting domain"
        if by_domain:
            row = idx.accounts.loc[by_domain]
            out |= {"resolved_account_id": by_domain, "company_name": row.company_name, "method": "exact domain match"}
        elif account_id and account_id in idx.accounts.index:
            row = idx.accounts.loc[account_id]
            out |= {"resolved_account_id": account_id, "company_name": row.company_name, "method": "ticket account_id"}
        # name candidates are shown so the agent can SEE the ambiguity, never used to override a domain match
        if company_name:
            q = _norm_company(company_name)
            scored = sorted(((difflib.SequenceMatcher(None, q, n).ratio(), aid) for aid, n in idx.company_norm.items()),
                            reverse=True)[:4]
            out["name_similarity_candidates"] = [
                {"account_id": aid, "company_name": idx.accounts.loc[aid].company_name,
                 "domain": idx.accounts.loc[aid].domain, "similarity": round(s, 2)} for s, aid in scored]
            if not by_domain:
                out["note"] = "No domain match. Name matches are ambiguous; only accept a candidate with a clearly unique match."
        if "resolved_account_id" not in out:
            out["resolved_account_id"] = None
        self.facts["account_id"] = out["resolved_account_id"]
        return out

    def get_account_profile(self, account_id: str) -> dict:
        if account_id not in self.idx.accounts.index:
            return {"error": f"unknown account {account_id}"}
        r = self.idx.accounts.loc[account_id]
        days = _days_until(r.renewal_date)
        sso_days = (TODAY - datetime.fromisoformat(r.sso_enabled_date)).days if r.sso_enabled_date else None
        esc = int(r.open_escalations or 0)
        out = {
            "account_id": account_id, "company_name": r.company_name, "tier": r.tier, "arr_band": r.arr_band,
            "industry": r.industry, "region": r.region,
            "renewal_date": r.renewal_date, "days_to_renewal": days, "renewal_within_90_days": days is not None and 0 <= days <= 90,
            "open_escalations": esc, "modules_licensed": r.modules_licensed.split(";") if r.modules_licensed else [],
            "sso_enabled_or_reconfigured": r.sso_enabled_date or None,
            "days_since_sso_change": sso_days,
            "last_qbr_date": r.last_qbr_date or None, "nps_last": r.nps_last or None,
        }
        if out["renewal_within_90_days"] and esc < 2:
            out["hint"] = "Renewal within 90 days: the risk override also depends on the 3-month active-user trend."
        self.facts["profile"] = out
        return out

    def get_usage_trend(self, account_id: str, months: int = 6) -> dict:
        u = self.idx.usage.get(account_id)
        if u is None or u.empty:
            return {"error": f"no usage for {account_id}"}
        u = u.tail(max(int(months), 4))
        rows = u[["month", "active_users", "login_rate_pct", "tickets_opened"]].to_dict("records")
        au = list(u["active_users"])
        # "declined over the last 3 months": latest month vs the month 3 months earlier
        base, latest = au[-4], au[-1]
        change = round((latest - base) / base * 100, 1) if base else None
        out = {"account_id": account_id, "monthly": rows,
               "active_users_3m_change_pct": change,
               "active_users_3m_window": f"{u['month'].iloc[-4]} -> {u['month'].iloc[-1]}",
               "declined_more_than_25pct": change is not None and change < -25}
        self.facts["usage"] = out
        return out

    # ---------------- tickets ----------------
    def _ticket_summary(self, r) -> dict:
        return {"ticket_id": r.ticket_id, "created_at": r.created_at, "status": r.status,
                "module": r.module or None, "module_canonical": r.module_canon, "error_codes": r.codes,
                "subject": r.subject, "customer_priority": r.customer_priority or None}

    def search_tickets(self, account_id: str, module: str = "", error_code: str = "", days_back: int = 30) -> dict:
        g = self.idx.tickets_by_acc.get(account_id)
        if g is None:
            return {"account_id": account_id, "tickets": [], "note": "no tickets for this account"}
        me = self.ticket
        my_mod = normalize_module(me.get("module")) or normalize_module(module)
        my_codes = extract_error_codes(me.get("error_code", ""), me.get("subject", ""), me.get("description", ""))
        if error_code and error_code not in my_codes:
            my_codes.append(error_code)
        lo = self.ticket_dt - timedelta(days=int(days_back or 30))
        g = g[(g.ticket_id != me["ticket_id"]) & (g.created_dt >= lo) & (g.created_dt <= self.ticket_dt)]
        want_mod = normalize_module(module) if module else None
        if want_mod:
            g = g[g.module_canon == want_mod]
        if error_code:
            g = g[g.codes.map(lambda cs: error_code in cs)]
        out, dups = [], []
        for _, r in g.sort_values("created_dt", ascending=False).iterrows():
            s = self._ticket_summary(r)
            # KB-5930: same account, open/pending, same module, same code/symptom, within previous 7 days
            age_days = (self.ticket_dt - r.created_dt).total_seconds() / 86400
            same_symptom = bool(set(r.codes) & set(my_codes)) if my_codes else _symptom_overlap(r.subject, me.get("subject", ""))
            if (r.status in ("open", "pending_customer") and age_days <= 7 and my_mod and r.module_canon == my_mod
                    and same_symptom):
                s["duplicate_candidate"] = True
                dups.append(r.ticket_id)
            s["age_days"] = round(age_days, 1)
            out.append(s)
        self.facts.setdefault("duplicate_candidates", [])
        self.facts["duplicate_candidates"] = sorted(set(self.facts["duplicate_candidates"]) | set(dups))
        return {"account_id": account_id, "current_ticket_module_canonical": my_mod,
                "current_ticket_error_codes": my_codes, "window_days": days_back,
                "count": len(out), "tickets": out[:15],
                "duplicate_candidates": dups,
                "duplicate_rule": "KB-5930: open/pending, same account+module, same error code or symptom, within 7 days"}

    def get_ticket(self, ticket_id: str) -> dict:
        if ticket_id not in self.idx.ticket_by_id.index:
            return {"error": f"unknown ticket {ticket_id}"}
        r = self.idx.ticket_by_id.loc[ticket_id]
        if r.acc_resolved != self.facts.get("account_id", r.acc_resolved):
            return {"error": "ticket belongs to a different customer; not shown"}
        return self._ticket_summary(r) | {"description": r.description[:1000], "channel": r.channel}

    # ---------------- knowledge base ----------------
    def _kb_brief(self, kid: str, score: float | None = None) -> dict:
        m = self.idx.kb[kid]["meta"]
        d = {"kb_id": kid, "title": m.get("title"), "module": m.get("module"), "type": m.get("type"),
             "status": m.get("status"), "error_codes": m.get("error_codes") or [], "last_updated": m.get("last_updated")}
        if m.get("superseded_by"):
            d["superseded_by"] = m["superseded_by"]
            d["warning"] = f"DEPRECATED - use {m['superseded_by']} instead"
        if m.get("status") == "active_incident":
            d["warning"] = "ACTIVE INCIDENT - read this article"
        if score is not None:
            d["score"] = round(score, 2)
        return d

    def search_kb(self, query: str = "", module: str = "", error_code: str = "", top_k: int = 5) -> dict:
        idx = self.idx
        top_k = max(1, min(int(top_k or 5), 8))
        q = tokenize(" ".join([query or "", error_code or ""]))
        scores = idx.bm25.scores(q) if q else [0.0] * len(idx.kb_ids)
        mod = normalize_module(module) if module else None
        codes = set(extract_error_codes(query or "", error_code or ""))
        ranked = []
        for kid, s in zip(idx.kb_ids, scores):
            m = idx.kb[kid]["meta"]
            code_hit = bool(codes & set(m.get("error_codes") or []))
            mod_hit = bool(mod and m.get("module") == mod)
            if code_hit:
                s += 20  # exact error-code match dominates keyword similarity
            if mod_hit:
                s += 2
            if m.get("status") == "active_incident" and (code_hit or mod_hit):
                s += 3  # surface live incidents, but only for the same module / code
            if m.get("status") == "deprecated":
                s -= 4  # still returned (flagged) so the agent can see it is outdated
            ranked.append((s, kid))
        ranked.sort(reverse=True)
        hits = [self._kb_brief(k, s) for s, k in ranked[:top_k] if s > 0]
        return {"query": query, "module": mod, "error_code": error_code or None, "results": hits}

    def get_kb_article(self, kb_id: str) -> dict:
        kb_id = (kb_id or "").strip().upper()
        if kb_id not in self.idx.kb:
            return {"error": f"unknown article {kb_id}"}
        a = self.idx.kb[kb_id]
        body = a["body"].strip()
        linked = sorted((set(re.findall(r"KB-\d+", body)) | set(a["meta"].get("related") or [])) - {kb_id})
        self.facts.setdefault("kb_read", [])
        if kb_id not in self.facts["kb_read"]:
            self.facts["kb_read"].append(kb_id)
        return self._kb_brief(kb_id) | {"body": body[:3500], "linked_articles": linked}

    def get_policy(self, topic: str = "priority") -> dict:
        topic = (topic or "").lower()
        keys = {"priority": ["priority", "escalation"], "duplicates": ["duplicate"],
                "data_exposure": ["exposure", "data"]}.get(topic)
        if not keys:
            return {"error": "topic must be one of priority, duplicates, data_exposure"}
        for kid, a in self.idx.policies.items():
            title = a["meta"].get("title", "").lower()
            if all(k in title for k in keys):
                return self.get_kb_article(kid)
        return {"error": f"no policy found for {topic}"}


def _symptom_overlap(a: str, b: str) -> bool:
    ta, tb = set(tokenize(a)) - {"re", "fw", "still"}, set(tokenize(b)) - {"re", "fw", "still"}
    if not ta or not tb:
        return False
    return len(ta & tb) / min(len(ta), len(tb)) >= 0.6
