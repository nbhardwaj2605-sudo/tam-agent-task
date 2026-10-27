# SOLUTION

## 1. Design
```
ticket ─► LLM ─{"thought","action","input"}─► tool ─observation─┐
           ▲                                                    │
           └──────────────── loop (≤10 tool calls) ◄────────────┘
           │ finish → schema check (errors sent back, model retries)
           ▼
  deterministic policy guardrails ─► output JSON + reasoning_trace
```
- **Loop (`agent.py`).** Each turn the model returns one JSON action; we run the tool and return the observation. The model decides what to check next (e.g. KB-4795 → KB-7889, renewal near → usage trend) and when to `finish`. JSON actions work the same on every provider. Stops on: a valid `finish`; a 10-call tool budget (further calls refused); 16 LLM turns. Repeated calls are refused. Before accepting `finish`, a one-time **self-review** sends the model back if it skipped a step (account profile, ticket history, reading the article, following an article's link) (added after evaluation showed ~2 lookups per ticket). No valid decision → a `needs_info`/P3 fallback for human review.
- **Tools (`tools.py`, no LLM, ~1 ms each).** `resolve_account` (domain is authoritative; look-alike names shown, never used), `get_account_profile`, `get_usage_trend`, `search_tickets` (history + duplicate candidates; codes read from the text, module aliases normalised), `get_ticket`, `search_kb` (BM25 + exact error-code boost; deprecated articles point to their successor; live incidents surfaced), `get_kb_article` (with links), `get_policy`.
- **Prompt.** Core policy KB-2914 is in the system prompt (every ticket needs it); duplicate and data-exposure policies are fetched only when relevant. Ticket text sits in `<ticket>` and is declared untrusted.
- **Guardrails.** The LLM decides; code enforces the non-negotiables and logs each correction: security → P1/Security/escalate; KB-5930 duplicates checked from data (an error-code match the model missed is applied; 3 of 3,005 tickets, all genuine), never re-escalated, original priority kept; needs-info → P3; **risk override computed from data** (model gives `base_priority`, code raises one level, max P2, + TAM + escalate); P1 → escalate; routing; deprecated refs → successor; other customers redacted from replies; injected instructions flagged.

## 2. Evaluation
- **Built:** `eval/gold_review.json` (expected labels + reasons) and `scripts/evaluate.py` (per-field accuracy, one must-not-fail check per ticket, privacy check, behaviour stats); 36 tests, one per edge case (blank/conflicting account, look-alike names, deprecated KB, missing usage, empty results, bad tool input, invalid JSON, budget exhaustion, 90-day boundary). Real runs on gemini-3.1-flash-lite: 8/10 → 9/10 → 10/10 must-not-fail checks (self-review, then KB-5930 checked from data) (`eval/`).
- Grow the gold set to ~300 tickets stratified by category, tier and trap type, double-labelled.
- Per-field accuracy: **account identification** (incl. blank and look-alike cases), category, route, **escalate**, notify_tam, duplicate_of. Priority also as a cost-weighted confusion matrix (a missed P1 costs far more than an over-rated P4).
- Hard-rule recall must be 100% (security, active incidents, risk overrides, no redundant escalation of duplicates), reported separately.
- **Output validity:** 100% schema-valid (`check_outputs.py`), invalid `finish` rejected and retried in the loop, fallback rate tracked.
- Retrieval: recall@5 / MRR for the correct current article; deprecated-citation rate = 0.
- Behaviour: tool calls per ticket, guardrail-correction rate, fallback rate; rubric-graded replies; shadow mode against human triage before go-live.

## 3. Scale (30k tickets, 4k articles)
- **Cost** does not grow with data: the model sees only top-k results and one account's slice. Measured: ~20k input / 1.2k output tokens over ~7 LLM calls per ticket. Levers: prompt caching of the fixed system prompt, a cheaper model for simple how-tos.
- **Latency:** tools stay in milliseconds (BM25 over 4k docs < 10 ms); LLM turns dominate (measured 15–60 s per ticket on a free tier, mostly rate-limit waits). Tickets are independent, so they run in parallel.
- **Retrieval quality is the risk:** more near-duplicate and outdated articles. Change to hybrid BM25 + embeddings with a reranker, metadata filters (module, status, release) before ranking, a persistent inverted index, and embedding similarity for paraphrased duplicates.

## 4. Limitations
- Duplicates without an error code rely on title overlap and miss paraphrases.
- Exposure and injection detectors are regexes: precise here (1 hit each in 3,000 tickets) but easy to evade.
- No tenant release data, so version-specific articles can't be checked.
- Smaller models investigate less deeply; guardrails keep outputs safe, but replies suffer.
- Policy lives partly in code, so a policy change needs a code change.

## 5. How to run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt pytest              # + openai / anthropic if used
cp .env.example .env    # e.g. LLM_PROVIDER=gemini, GEMINI_API_KEY=..., LLM_MODEL=gemini-3.1-flash-lite
python -m pytest -q                                 # 36 tests, no key needed
python run.py --set review && python scripts/check_outputs.py
python scripts/evaluate.py                          # accuracy + behaviour
python scripts/report.py                            # outputs/report.html trace viewer
```
