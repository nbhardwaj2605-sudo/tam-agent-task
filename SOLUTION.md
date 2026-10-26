# SOLUTION

## 1. Design
```
ticket ──► LLM ──{"thought","action","input"}──► tool ──observation──┐
            ▲                                                         │
            └───────────────────── loop (≤10 tool calls) ◄────────────┘
            │ action = finish → schema check (errors sent back, model retries)
            ▼
   deterministic policy guardrails ──► output JSON + reasoning_trace
```
- **Loop (`agent.py`).** Each turn the model returns one JSON action. We run the tool and return the observation. The model then chooses the next tool, whether to dig further (e.g. KB-4795 → KB-7889, or renewal near → usage trend) or `finish`. JSON actions instead of native tool calling keep the loop identical on Ollama, Anthropic and OpenAI. Stop conditions: `finish` with a valid decision; a 10-call tool budget (further calls are refused and the model is told to finish); 16 LLM turns max; repeated identical calls are refused. If no valid decision ever arrives, a safe `needs_info`/P3 fallback is flagged for a human.
- **Tools (`tools.py`, deterministic, no LLM).** `resolve_account` (domain is authoritative; similar names are shown but never used), `get_account_profile` (days to renewal, escalations, SSO change date), `get_usage_trend` (3-month active-user change), `search_tickets` (account history plus KB-5930 duplicate candidates; error codes extracted from the description, module aliases normalised), `get_ticket`, `search_kb` (BM25 + exact error-code boost; deprecated articles flagged with their successor; live incidents surfaced), `get_kb_article` (with linked articles), `get_policy`. Indexes are built once per run (0.15 s); each lookup takes about 1 ms.
- **Prompt.** The core policy KB-2914 is loaded from the KB into the system prompt because every ticket needs it. Duplicate and data-exposure policies are fetched only when relevant. Ticket text is wrapped in `<ticket>` and declared untrusted.
- **Guardrails (`apply_guardrails`).** The LLM decides; code enforces the non-negotiables and logs every correction as a trace step: domain-resolved account; data-exposure detection → Security/P1/escalate; duplicates only if `search_tickets` verified them, never re-escalated; needs-info → P3; how-to/feature → P4 base; **risk override computed from data** (the model outputs `base_priority`; code raises one level, max P2, and sets TAM + escalate); P1 → escalate; routing; deprecated KB refs → successor; other customers' names/IDs redacted from replies; embedded instructions flagged. Each correction lowers confidence.

## 2. Evaluation
- Build a labelled gold set (~300 tickets, stratified by category, tier and trap type), with labels from 2 support engineers and inter-rater agreement measured.
- Per-field accuracy: category, route, escalate, notify_tam, duplicate_of. Priority as accuracy plus a cost-weighted confusion matrix (under-prioritising P1 costs far more than over-prioritising P4).
- Hard-rule recall must be 100%: security reports, active incidents, risk overrides. Report these separately, never averaged away.
- Retrieval: recall@5 / MRR of the correct current KB article, plus deprecated-article citation rate (target 0).
- Adversarial suite: prompt injections, look-alike company names, blank account_id, error code only in the text. Pass/fail per case.
- Trace quality: tool calls per ticket, guardrail-correction rate (high means the prompt or model is weak), budget-exhausted and fallback rate.
- Reply quality: rubric-graded sample (correct steps, no speculation, no cross-customer data), then shadow mode against human triage before go-live.

## 3. Scale (30k tickets, 4k articles)
- **Cost.** It does not grow with data size. The model sees only top-k results and per-account slices, so tokens per ticket stay at about 15–40k input across 3–7 calls. The main lever is prompt caching of the fixed system prompt and a cheaper model for easy how-tos.
- **Latency.** Tools stay in milliseconds (per-account groupby; BM25 over 4k docs is still under 10 ms). The LLM turns dominate at about 2–5 s each on hosted models.
- **Retrieval quality is the real risk.** With 10x the articles there are more near-duplicates and more outdated articles. Changes: hybrid BM25 + embeddings with a cross-encoder reranker, metadata filters (module, status, release) before ranking, an inverted index instead of a full scan, and a persistent index instead of a rebuild per run. For tickets, an account-scoped index plus embedding similarity for symptom-level duplicates (today: error code or title overlap).

## 4. Limitations
- Duplicate detection without an error code relies on title word overlap and misses paraphrases.
- The exposure and injection detectors are regexes. They are precise on this data (1 hit each across 3,000 tickets) but easy to evade, so a classifier is needed in production.
- No release/version awareness (articles say "applies to 26.1+"). Tenant release isn't in the data.
- Small local models (8B) sometimes produce weak JSON or shallow investigations. The guardrails keep outputs valid and safe, but the KB choice and reply quality drop.
- Guardrails encode KB-2914 in code, so a policy change means a code change.

## 5. How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # + anthropic or openai if used
cp .env.example .env                       # set LLM_PROVIDER / LLM_MODEL / key
python scripts/validate_data.py
python tests/test_agent_offline.py         # plumbing test with a scripted LLM, no key needed
python run.py --set review
python scripts/check_outputs.py            # expect 10/10 valid
```
