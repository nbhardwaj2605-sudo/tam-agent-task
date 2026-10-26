"""YOUR IMPLEMENTATION GOES HERE.

Implement `triage(ticket, ctx)` so that it returns a dict matching
tam_agent/schema.py. See README.md for the full requirements.

Requirements in short:
  - The agent decides which steps to take (which tools to call, in what order,
    when to stop). A fixed linear pipeline of LLM calls does not meet the bar.
  - Tools are yours to design: e.g. account lookup, usage trend, ticket history,
    KB search, policy lookup.
  - Every step goes into `reasoning_trace`.
  - Must work against the full dataset (3,000 tickets, ~390 KB articles).
"""
from .llm import LLMClient


class Context:
    """Shared resources, built once per run. Add indexes/tools here."""

    def __init__(self, accounts, tickets, usage, kb):
        self.accounts = accounts
        self.tickets = tickets
        self.usage = usage
        self.kb = kb
        self.llm = LLMClient()


def triage(ticket: dict, ctx: Context) -> dict:
    raise NotImplementedError("Implement the triage agent here.")
