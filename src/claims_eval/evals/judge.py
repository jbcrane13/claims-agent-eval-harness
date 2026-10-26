"""LLM-as-judge for the parts of the rationale that rules can't check.

Three rules keep the judge honest:
1. Binary rubric items, not a 1-10 score. "Did it state the next step?" is answerable; "rate the
   quality 1-10" drifts between runs and between judges.
2. Use a different model family to judge than the one under test, so a model doesn't grade its own
   habits as correct.
3. Calibrate before trusting. `claims-eval calibrate` scores human-labeled examples and reports
   Cohen's kappa per item. Until kappa clears judge.min_kappa in eval_config.toml, judge scores are
   reported but never gate a release.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from claims_eval.agent.loop import parse_output
from claims_eval.agent.types import Model

RUBRIC = {
    "explains_pend_reason": "The rationale says why the claim pended, in terms of the pend code's meaning.",
    "states_next_step": ("The rationale says what happens next: release, which queue and why, what "
                         "information is requested, or escalation to an examiner."),
    "no_unsupported_claims": ("Every factual statement in the rationale is supported by the evidence. "
                              "Answer false if any fact is absent from or contradicts the evidence."),
}

JUDGE_PROMPT = """You grade one claims-review rationale against a rubric. Judge only what is written.
For each rubric item answer true or false. Reply with only a JSON object mapping each item name to
true or false, plus "notes": a one-sentence reason for any false.

Rubric:
""" + "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())

EVIDENCE_LIMIT = 6000


def evidence_from_trace(trace: list[dict[str, Any]]) -> str:
    parts = [f"[{s['tool']} {json.dumps(s['args'])}]\n{s['result']}" for s in trace if not s["is_error"]]
    return "\n\n".join(parts)[:EVIDENCE_LIMIT]


async def judge_rationale(model: Model, rationale: str, evidence: str, call_id: str) -> dict[str, Any]:
    msg = f"Evidence (tool results the agent saw):\n{evidence}\n\nRationale to grade:\n{rationale}"
    turn = await model.complete(JUDGE_PROMPT, [{"role": "user", "content": msg}], [], call_id=call_id)
    parsed = parse_output(turn.text) or {}
    return {item: parsed.get(item) if isinstance(parsed.get(item), bool) else None for item in RUBRIC} | {
        "notes": parsed.get("notes", "")}


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Agreement between two raters beyond chance. 1.0 is perfect, 0 is chance, negative is worse."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return round((observed - expected) / (1 - expected), 3)


async def calibrate(model: Model, path: Path) -> dict[str, Any]:
    examples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    verdicts = [await judge_rationale(model, ex["rationale"], ex["evidence"], call_id=f"calib/{ex['id']}/r0")
                for ex in examples]
    report: dict[str, Any] = {"judge": model.name, "examples": len(examples), "items": {}}
    for item in RUBRIC:
        pairs = [(ex["labels"][item], v[item]) for ex, v in zip(examples, verdicts) if v[item] is not None]
        human, judged = [p[0] for p in pairs], [p[1] for p in pairs]
        report["items"][item] = {
            "agreement": round(sum(h == j for h, j in pairs) / len(pairs), 3) if pairs else None,
            "kappa": cohen_kappa(human, judged) if pairs else None,
            "unparsed": len(examples) - len(pairs),
        }
    return report
