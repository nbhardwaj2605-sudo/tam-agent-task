"""Build a single-file HTML trace viewer from the agent's outputs.

    python scripts/report.py                      # outputs/ -> outputs/report.html
    python scripts/report.py --outputs some/dir   # any folder of output JSONs

Open the HTML in a browser. For each ticket it shows the ticket text, every step the
agent took (thought -> tool -> observation), the guardrail check, the final decision,
and - when eval/gold_review.json has the ticket - whether each field matched.
"""
import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tam_agent.data import load_tickets  # noqa: E402

FIELDS = ["account_id", "category", "priority", "route_to", "escalate", "notify_tam", "duplicate_of",
          "needs_clarification"]
e = lambda s: html.escape(str(s))  # noqa: E731

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e4e3de;--accent:#2f5bea;
--ok:#1f7a4d;--okbg:#e6f4ec;--bad:#b42318;--badbg:#fdecea;--warn:#8a5a00;--warnbg:#fff4dc;--code:#f1f0ec}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--ink:#ecebe6;--muted:#a3a29b;--line:#33332f;
--accent:#7b9bff;--ok:#6fd3a0;--okbg:#17382a;--bad:#ff8a80;--badbg:#3d1a17;--warn:#f5c56b;--warnbg:#3a2d10;--code:#2a2a27}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 20px}table{width:100%;border-collapse:collapse;background:var(--card);
border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:14px}th,td{padding:8px 10px;text-align:left;
border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
.wrap{overflow-x:auto}a{color:var(--accent);text-decoration:none;white-space:nowrap}
details.t{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:16px 0;padding:0}
details.t>summary{cursor:pointer;padding:14px 16px;list-style:none;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
details.t>summary::-webkit-details-marker{display:none}.tid{font-weight:700}.subj{color:var(--muted);flex:1 1 260px}
.body{padding:0 16px 16px}.chip{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:600;
background:var(--code);white-space:nowrap}.ok{background:var(--okbg);color:var(--ok)}.bad{background:var(--badbg);color:var(--bad)}
.warn{background:var(--warnbg);color:var(--warn)}blockquote{margin:8px 0;padding:8px 12px;border-left:3px solid var(--line);
background:var(--code);border-radius:6px;white-space:pre-wrap}ol.steps{list-style:none;padding:0;margin:8px 0;counter-reset:s}
ol.steps li{position:relative;padding:8px 0 8px 36px;border-left:2px solid var(--line);margin-left:12px}
ol.steps li::before{counter-increment:s;content:counter(s);position:absolute;left:-13px;top:8px;width:24px;height:24px;
border-radius:50%;background:var(--card);border:2px solid var(--line);font-size:12px;display:grid;place-items:center;color:var(--muted)}
ol.steps li.guard::before{border-color:var(--warn);color:var(--warn)}ol.steps li.fin::before{border-color:var(--ok);color:var(--ok)}
.act{font-weight:700;font-family:ui-monospace,Consolas,monospace;font-size:13px}.th{color:var(--muted);font-style:italic}
code,.inp{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;background:var(--code);padding:1px 5px;border-radius:4px;word-break:break-word}
.obs{margin-top:2px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
h3{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:16px 0 6px}
"""


def chip(text, cls=""):
    return f'<span class="chip {cls}">{e(text)}</span>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--gold", default=str(ROOT / "eval" / "gold_review.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = Path(args.outputs)
    outs = {p.stem: json.loads(p.read_text()) for p in sorted(out_dir.glob("TKT-*.json"))}
    if not outs:
        sys.exit(f"no outputs in {out_dir}")
    gold = {}
    if Path(args.gold).exists():
        gold = {k: v for k, v in json.loads(Path(args.gold).read_text()).items() if k.startswith("TKT-")}
    tickets = load_tickets().set_index("ticket_id")

    def field_ok(tid, f):
        g = gold.get(tid, {})
        if f not in g:
            return None
        return outs[tid].get(f) in g[f] if isinstance(g[f], list) else outs[tid].get(f) == g[f]

    rows, cards = [], []
    for tid, o in outs.items():
        t = tickets.loc[tid] if tid in tickets.index else None
        n_tools = sum(s["action"] not in ("finish", "policy_guardrails", "parse_error", "finish_rejected",
                                          "fallback", "budget_exhausted", "unknown_tool", "llm_error")
                      for s in o["reasoning_trace"])
        checks = [field_ok(tid, f) for f in FIELDS]
        scored = [c for c in checks if c is not None]
        score = chip(f"{sum(scored)}/{len(scored)} fields", "ok" if all(scored) else "bad") if scored else ""
        decision = f"{o['category']} · {o['priority']} → {o['route_to']}"
        flags = "".join([chip("escalate", "warn") if o["escalate"] else "", chip("TAM", "warn") if o["notify_tam"] else "",
                         chip(f"dup of {o['duplicate_of']}") if o["duplicate_of"] else "",
                         chip("needs info") if o["needs_clarification"] else ""])
        rows.append(f"<tr><td><a href='#{tid}'>{tid}</a></td><td>{e(t.subject if t is not None else '')}</td>"
                    f"<td>{e(decision)} {flags}</td><td>{n_tools}</td><td>{score}</td></tr>")

        steps = []
        for s in o["reasoning_trace"]:
            cls = "guard" if s["action"] == "policy_guardrails" else "fin" if s["action"] == "finish" else ""
            inp = f' <span class="inp">{e(s.get("input", ""))[:300]}</span>' if s.get("input") else ""
            th = f'<div class="th">{e(s["thought"])}</div>' if s.get("thought") else ""
            steps.append(f'<li class="{cls}">{th}<span class="act">{e(s["action"])}</span>{inp}'
                         f'<div class="obs">→ {e(s["observation"])}</div></li>')

        fields = []
        for f in FIELDS:
            ok = field_ok(tid, f)
            mark = "" if ok is None else chip("✓", "ok") if ok else chip(f"expected {gold[tid][f]}", "bad")
            fields.append(f"<tr><td>{f}</td><td><code>{e(o.get(f))}</code> {mark}</td></tr>")
        why = f"<p><b>Expected because:</b> {e(gold[tid]['why'])}</p>" if tid in gold and gold[tid].get("why") else ""
        meta = (f"{e(t.company_name_raw)} · {e(t.requester_domain)} · {e(t.created_at)} · {e(t.channel)} · module "
                f"<code>{e(t.module or '(blank)')}</code> · customer priority <code>{e(t.customer_priority or '(blank)')}</code>"
                if t is not None else "")
        cards.append(f"""<details class="t" id="{tid}"><summary><span class="tid">{tid}</span>
<span class="subj">{e(t.subject if t is not None else '')}</span>{chip(decision)}{flags}{score}</summary><div class="body">
<p class="sub">{meta}</p><blockquote>{e(t.description if t is not None else '')}</blockquote>
<h3>Agent steps ({n_tools} tool calls)</h3><ol class="steps">{''.join(steps)}</ol>
<div class="grid"><div><h3>Decision</h3><div class="wrap"><table>{''.join(fields)}
<tr><td>kb_references</td><td>{' '.join(chip(k) for k in o['kb_references'])}</td></tr>
<tr><td>confidence</td><td>{o['confidence']}</td></tr></table></div>{why}</div>
<div><h3>Account signals</h3><ul>{''.join(f'<li>{e(x)}</li>' for x in o['account_signals']) or '<li>none</li>'}</ul>
<h3>Internal note</h3><blockquote>{e(o['internal_note'])}</blockquote></div></div>
<h3>Customer reply</h3><blockquote>{e(o['customer_reply'])}</blockquote></div></details>""")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Triage Trace Viewer</title>
<style>{CSS}</style></head><body><main><h1>Triage trace viewer</h1>
<p class="sub">{len(outs)} tickets from <code>{e(out_dir.name)}/</code>. Click a ticket to see each step the agent took.</p>
<div class="wrap"><table><tr><th>Ticket</th><th>Subject</th><th>Decision</th><th>Tool calls</th><th>vs expected</th></tr>
{''.join(rows)}</table></div>{''.join(cards)}</main></body></html>"""
    target = Path(args.out) if args.out else out_dir / "report.html"
    target.write_text(page, encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
