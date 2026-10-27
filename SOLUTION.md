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
- **Loop (`agent.py`).** Each turn the model returns one JSON action; we run the tool and return the observation. The model decides what to check next (e.g. KB-4795 → KB-7889, renewal near → usage trend) and when to `finish`. JSON actions keep the loop identical on Ollama, Anthropic, Gemini and OpenAI-compatible APIs. Stops on: a valid `finish`; a 10-call tool budget (further calls refused); 16 LLM turns. Repeated calls are refused. Before accepting `finish`, a one-time **self-review** sends the model back if it skipped a step (account profile, ticket history, reading the article, following an article's link); added after evaluation showed a fast model stopping after ~2 lookups. No valid decision → a `needs_info`/P3 fallback for human review.
- **Tools (`tools.py`, no LLM, ~1 ms each).** `resolve_account` (domain is authoritative; look-alike names shown, never used), `get_account_profile`, `get_usage_trend`, `search_tickets` (history + KB-5930 duplicate candidates; error codes taken from the text, module aliases normalised), `get_ticket`, `search_kb` (BM25 + exact error-code boost; deprecated articles point to their successor; live incidents surfaced), `get_kb_article` (with links), `get_policy`.
- **Prompt.** Core policy KB-2914 is in the system prompt (every ticket needs it); duplicate and data-exposure policies are fetched only when relevant. Ticket text sits in `<ticket>` and is declared untrusted.
- **Guardrails.** The LLM decides; code enforces the non-negotiables and logs each correction: security → P1/Security/escalate; KB-5930 duplicates checked from data (an error-code match the model missed is applied; fires on 3 of 3,005 tickets, all genuine), never re-escalated, original priority kept; needs-info → P3; **risk override computed from data** (model gives `base_priority`, code raises one level, max P2, + TAM + escalate); P1 → escalate; routing; deprecated refs → successor; other customers redacted from replies; injected instructions flagged.

## 2. Evaluation
- **Built:** `eval/gold_review.json` (expected labels + reasons) and `scripts/evaluate.py` (per-field accuracy, one must-not-fail check per ticket, privacy check, behaviour stats); 36 tests, one per edge case (blank/conflicting account, look-alike names, deprecated KB, missing usage, empty results, bad tool input, invalid JSON, budget exhaustion, 90-day boundary). Real runs on gemini-3.1-flash-lite: 8/10 → 9/10 → 10/10 must-not-fail checks (self-review, then KB-5930 checked from data) (`eval/`).
- Grow the gold set to ~300 tickets, stratified by category, tier and trap type, labelled by 2 engineers with agreement measured.
- Priority: accuracy plus a cost-weighted confusion matrix (a missed P1 costs far more than an over-rated P4).
- Hard-rule recall must be 100% (security, active incidents, risk overrides), reported separately.
- Retrieval: recall@5 / MRR for the correct current article; deprecated-citation rate = 0.
- Behaviour: tool calls per ticket, guardrail-correction rate, fallbacks; rubric-graded replies; shadow mode against human triage before go-live.

## 3. Scale (30k tickets, 4k articles)
- **Cost** does not grow with data: the model sees only top-k results and one account's slice (~15–40k input tokens over 3–7 calls). Levers: prompt caching, a cheaper model for simple how-tos.
- **Latency:** tools stay in milliseconds (BM25 over 4k docs < 10 ms); LLM turns (2–5 s each) dominate.
- **Retrieval quality is the risk:** more near-duplicate and outdated articles. Change to hybrid BM25 + embeddings with a reranker, metadata filters (module, status, release) before ranking, a persistent inverted index, and embedding similarity for paraphrased duplicates.

## 4. Limitations
- Duplicates without an error code rely on title overlap and miss paraphrases.
- Exposure and injection detectors are regexes: precise here (1 hit each in 3,000 tickets) but easy to evade.
- No tenant release data, so version-specific articles can't be checked.
- Small local models investigate less deeply; guardrails keep outputs safe, but KB choice and replies suffer.
- Policy lives partly in code, so a policy change needs a code change.

## 5. How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt openai pytest   # or anthropic
cp .env.example .env                            # provider, model, key
python -m pytest -q                             # unit tests, no key needed
python run.py --set review && python scripts/check_outputs.py
python scripts/evaluate.py                      # accuracy + behaviour
python scripts/report.py                        # outputs/report.html trace viewer
```
