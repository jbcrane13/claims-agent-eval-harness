"""Writes the run report: a self-contained HTML page and a short Markdown summary for CI."""

from __future__ import annotations

import json
from html import escape
from typing import Any

from claims_eval.evals.gate import Check


def _pct(x: float) -> str:
    return f"{x:.1%}"


COUNT_CHECKS = {"safety_failures", "counterfactual_flips"}


def _fmt(name: str, value: float | None) -> str:
    if value is None:
        return "–"
    return str(int(value)) if name in COUNT_CHECKS else _pct(value)


def markdown_summary(meta: dict[str, Any], metrics: dict[str, Any], checks: list[Check] | None) -> str:
    verdict = "no baseline" if checks is None else ("PASS" if all(c.passed for c in checks) else "BLOCKED")
    lines = [
        f"## Claims agent eval: {verdict}",
        f"Model `{meta['model']}` · {metrics['cases']} cases × {metrics['runs_per_case']} runs",
        "",
        "| Metric | Value |", "|---|---|",
        f"| pass rate | {_pct(metrics['pass_rate'])} |",
        f"| pass^{metrics['runs_per_case']} | {_pct(metrics['pass_hat_k'])} |",
        f"| action accuracy | {_pct(metrics['action_accuracy'])} |",
        f"| safety failures | {metrics['safety_failures']} |",
        f"| counterfactual flips | {metrics['counterfactual']['flips']} |",
        f"| infra errors (excluded) | {metrics['infra_errors']} |",
    ]
    if checks:
        lines += ["", "| Gate check | Result | Current | Baseline | Limit | Note |", "|---|---|---|---|---|---|"]
        for c in checks:
            lines.append(f"| {c.name} | {'✅' if c.passed else '❌'} | {_fmt(c.name, c.current)} | "
                         f"{_fmt(c.name, c.baseline)} | {c.limit} | {c.note} |")
    if metrics["always_failing_cases"]:
        lines += ["", f"Always failing: {', '.join(metrics['always_failing_cases'])}"]
    if metrics["flaky_cases"]:
        lines += [f"Flaky: {', '.join(metrics['flaky_cases'])}"]
    return "\n".join(lines) + "\n"


CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d1f21;--muted:#62666d;--line:#e3e3df;--ok:#1f7a4d;--bad:#b42318;
--warn:#9a6700;--accent:#2f5d8a;--code:#f0f0ec}
@media (prefers-color-scheme:dark){:root{--bg:#141517;--panel:#1c1d20;--ink:#e8e8e6;--muted:#9a9ea5;
--line:#2d2f33;--ok:#4cc38a;--bad:#ff6b5e;--warn:#e3b341;--accent:#7fb0e0;--code:#25272b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:1100px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:32px 0 10px;text-transform:uppercase;
letter-spacing:.06em;color:var(--muted)}.sub{color:var(--muted)}
.verdict{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:600;margin-left:8px;font-size:13px}
.v-pass{background:color-mix(in srgb,var(--ok) 15%,transparent);color:var(--ok)}
.v-block{background:color-mix(in srgb,var(--bad) 15%,transparent);color:var(--bad)}
.v-none{background:var(--code);color:var(--muted)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:20px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile b{display:block;font-size:22px;font-variant-numeric:tabular-nums}.tile span{color:var(--muted);font-size:12px}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:10px;
overflow:hidden}th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--muted);font-weight:600}td.num{font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:0}.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
.warn{color:var(--warn)}code,pre{background:var(--code);border-radius:6px;font:12px/1.45 ui-monospace,Menlo,monospace}
code{padding:1px 5px}pre{padding:10px;overflow-x:auto;white-space:pre-wrap;word-break:break-word;margin:6px 0}
details{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin:8px 0;padding:8px 12px}
summary{cursor:pointer;font-weight:600}.runs{display:inline-flex;gap:4px;margin-left:8px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}.dot.p{background:var(--ok)}.dot.f{background:var(--bad)}
.dot.i{background:var(--warn)}
.wrap{overflow-x:auto}
"""


def _tile(label: str, value: str, cls: str = "") -> str:
    return f'<div class="tile"><b class="{cls}">{escape(value)}</b><span>{escape(label)}</span></div>'


def html_report(meta: dict[str, Any], metrics: dict[str, Any], records: list[dict[str, Any]],
                cases: list[dict[str, Any]], checks: list[Check] | None) -> str:
    case_index_all = {c["case_id"]: c for c in cases}
    if checks is None:
        verdict = '<span class="verdict v-none">no baseline</span>'
    elif all(c.passed for c in checks):
        verdict = '<span class="verdict v-pass">gate passed</span>'
    else:
        verdict = '<span class="verdict v-block">gate blocked</span>'

    k = metrics["runs_per_case"]
    tokens = f"{metrics['tokens_per_run']:,}" + (" est." if metrics["tokens_estimated"] else "")
    tiles = "".join([
        _tile("pass rate (all runs)", _pct(metrics["pass_rate"])),
        _tile(f"pass^{k} (every run passes)", _pct(metrics["pass_hat_k"])),
        _tile("action accuracy", _pct(metrics["action_accuracy"])),
        _tile("safety failures" + (f" (≤{metrics['safety_rate_upper_95']:.1%}/run at 95%)"
                                    if metrics.get("safety_rate_upper_95") else ""),
              str(metrics["safety_failures"]), "bad" if metrics["safety_failures"] else "ok"),
        _tile("counterfactual flips", str(metrics["counterfactual"]["flips"]),
              "bad" if metrics["counterfactual"]["flips"] else "ok"),
        _tile("infra errors (not scored)", str(metrics["infra_errors"]), "warn" if metrics["infra_errors"] else ""),
        _tile("latency p95", f"{metrics['latency_ms_p95']:,} ms"),
        _tile("tokens per run", tokens),
    ])

    gate_html = ""
    if checks:
        rows = "".join(
            f"<tr><td><code>{escape(c.name)}</code></td><td class='{'ok' if c.passed else 'bad'}'>"
            f"{'pass' if c.passed else 'fail'}</td><td class='num'>{_fmt(c.name, c.current)}</td>"
            f"<td class='num'>{_fmt(c.name, c.baseline)}</td><td>{escape(c.limit)}</td><td>{escape(c.note)}</td></tr>"
            for c in checks)
        gate_html = (f"<h2>Release gate</h2><div class='wrap'><table><tr><th>Check</th><th>Result</th>"
                     f"<th>Current</th><th>Baseline</th><th>Limit</th><th>Note</th></tr>{rows}</table></div>")

    crit = set(metrics["critical_scorers"])
    scorer_rows = "".join(
        f"<tr><td><code>{escape(n)}</code>{' <span class=warn>safety</span>' if n in crit else ''}</td>"
        f"<td class='num'>{_pct(v) if v is not None else '–'}</td></tr>" for n, v in metrics["scorers"].items())

    cf = metrics["counterfactual"]
    cf_rows = "".join(f"<tr><td>{escape(g)}</td><td>{escape(', '.join(f'{k}: {v}' for k, v in d.items()))}</td></tr>"
                      for g, d in cf["flipped_groups"].items())
    cf_html = (f"<p>{cf['groups_compared']} golden cases were each copied with the member moved to the other "
               f"lines of business; nothing else changed. A flip means the usual decision differed between copies.</p>"
               + (f"<table><tr><th>Case</th><th>Decision by line of business</th></tr>{cf_rows}</table>"
                  if cf_rows else "<p class='ok'>No flips.</p>"))

    slice_html = ""
    for dim, s in metrics["slices"].items():
        rows = "".join(f"<tr><td>{escape(v)}</td><td class='num'>{_pct(g['pass_rate'])}</td>"
                       f"<td class='num'>{g['cases']}</td></tr>" for v, g in s["groups"].items())
        slice_html += (f"<h3 style='font-size:14px;margin:14px 0 6px'>{escape(dim)} "
                       f"<span class='sub'>largest gap {_pct(s['max_gap'])} across {s['compared_groups']} "
                       f"comparable groups</span></h3>"
                       f"<table><tr><th>Group</th><th>Pass rate</th><th>Cases</th></tr>{rows}</table>")

    by_case: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_case.setdefault(r["case_id"], []).append(r)
    # Counterfactual copies only appear when one of their runs failed; otherwise they'd bury the rest.
    by_case = {cid: runs for cid, runs in by_case.items()
               if case_index_all[cid]["dataset"] != "counterfactual" or not all(r["passed"] for r in runs)}
    case_index = case_index_all
    details = []
    for cid in sorted(by_case, key=lambda c: (all(r["passed"] for r in by_case[c]), c)):
        runs = sorted(by_case[cid], key=lambda r: r["run"])
        case = case_index[cid]
        dots = "".join(f"<span class='dot {'p' if r['passed'] else ('i' if r['passed'] is None else 'f')}'></span>"
                       for r in runs)
        body = [f"<p class='sub'>{escape(case['instruction'])}<br>expected <code>{escape(case['expected']['action'])}"
                f"</code> {('→ <code>' + escape(case['expected']['queue']) + '</code>') if case['expected']['queue'] else ''}"
                f" · scenario <code>{escape(case['scenario'])}</code></p>"]
        for r in runs:
            fails = [s for s in r["scores"] if s["passed"] is False]
            if r.get("infra_error"):
                body.append(f"<p><b>Run {r['run'] + 1}</b> <span class='warn'>infrastructure error, not scored:"
                            f"</span> {escape(r['infra_error'])}</p>")
                continue
            fail_html = "".join(f"<li><code>{escape(s['name'])}</code>{' <b class=bad>critical</b>' if s['critical'] else ''}"
                                f" {escape(s['detail'])}</li>" for s in fails) or "<li class='ok'>all scorers passed</li>"
            trace = "\n".join(f"{i + 1}. {s['tool']}({json.dumps(s['args'])}){' → ERROR' if s['is_error'] else ''}"
                              for i, s in enumerate(r["trace"]))
            out = json.dumps(r["output"], indent=2) if r["output"] else (r.get("final_text") or r.get("error") or "")
            if r.get("stop_reason") and not r["output"]:
                out += f"\n[model stop reason: {r['stop_reason']}]"
            body.append(f"<p><b>Run {r['run'] + 1}</b></p><ul>{fail_html}</ul>"
                        f"<pre>{escape(trace) or '(no tool calls)'}</pre><pre>{escape(out)}</pre>")
        open_attr = " open" if not all(r["passed"] for r in runs) and len(details) < 3 else ""
        details.append(f"<details{open_attr}><summary>{escape(cid)} <span class='runs'>{dots}</span> "
                       f"<span class='sub'>{escape(case['dataset'])} · {escape(case['tags']['pend_code'])}</span>"
                       f"</summary>{''.join(body)}</details>")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Claims Agent Eval</title>
<style>{CSS}</style></head><body><main>
<h1>Claims agent eval {verdict}</h1>
<div class="sub">Model <code>{escape(meta['model'])}</code> · {metrics['cases']} cases × {k} runs
(+ {metrics['counterfactual']['groups_compared'] * 2} counterfactual copies) · commit {escape(str(meta.get('git_sha')))} ·
{escape(meta['started'])} · synthetic data only</div>
<div class="tiles">{tiles}</div>
{gate_html}
<h2>Scorers</h2><table><tr><th>Scorer</th><th>Pass rate</th></tr>{scorer_rows}</table>
<h2>Counterfactual fairness</h2>{cf_html}
<h2>Slices (golden set, descriptive)</h2>{slice_html}
<p class="sub">Observed pass rates by group. A gap here can come from bias or from the groups holding
different kinds of cases, and with this few cases it can't tell which. That is why the gate checks
counterfactual flips instead of these gaps.</p>
<h2>Cases</h2>{''.join(details)}
</main></body></html>"""
