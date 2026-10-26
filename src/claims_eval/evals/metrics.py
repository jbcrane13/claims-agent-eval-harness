"""Turns scored runs into the numbers the gate and the report use.

Two pass metrics, because agents are non-deterministic:
- pass_rate:  share of all runs that passed (pass@1 averaged over runs).
- pass_hat_k: share of cases where every one of the k runs passed (pass^k). This is the one that
              matters for mission-critical work: an agent that is right 2 times out of 3 on the same
              claim is not reliable, even though its pass_rate looks fine.

Headline metrics come from the golden and adversarial sets. The counterfactual set exists only to
measure whether changing a member's line of business changes the decision.

Runs that hit an infrastructure error are left out of every metric and counted on their own.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

HEADLINE_DATASETS = ("golden", "adversarial")


def _pct(values: list[bool]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _score(record: dict[str, Any], name: str) -> bool | None:
    return next((s["passed"] for s in record["scores"] if s["name"] == name), None)


def compute(records: list[dict[str, Any]], cases: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    case_index = {c["case_id"]: c for c in cases}
    scored = [r for r in records if r["infra_error"] is None]
    headline = [r for r in scored if case_index[r["case_id"]]["dataset"] in HEADLINE_DATASETS]

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in headline:
        by_case[r["case_id"]].append(r)

    scorer_names = list(dict.fromkeys(s["name"] for r in headline for s in r["scores"]))
    critical_names = list(dict.fromkeys(s["name"] for r in headline for s in r["scores"] if s["critical"]))

    def rate(name: str, recs: list[dict[str, Any]]) -> float | None:
        applicable = [v for v in (_score(r, name) for r in recs) if v is not None]
        return _pct(applicable) if applicable else None

    n_runs = len(headline)
    safety_failures = sum(1 for r in headline if r["critical_failures"])
    metrics: dict[str, Any] = {
        "cases": len(by_case),
        "runs_per_case": max((len(v) for v in by_case.values()), default=0),
        "scored_runs": n_runs,
        "pass_rate": _pct([r["passed"] for r in headline]),
        "pass_hat_k": _pct([all(r["passed"] for r in runs) for runs in by_case.values()]),
        "action_accuracy": _pct([_score(r, "action_correct") is True for r in headline]),
        "safety_failures": safety_failures,
        # Zero failures in n runs still allows a true per-run failure rate up to about 3/n (95%, rule of three).
        "safety_rate_upper_95": round(3 / n_runs, 4) if n_runs and safety_failures == 0 else None,
        "safety_failures_by_scorer": {
            name: n for name in critical_names
            if (n := sum(1 for r in headline if name in r["critical_failures"]))
        },
        "critical_scorers": critical_names,
        "scorers": {name: rate(name, headline) for name in scorer_names},
        "infra_errors": len(records) - len(scored),
        "infra_error_rate": round((len(records) - len(scored)) / len(records), 4) if records else 0.0,
        "flaky_cases": sorted(cid for cid, runs in by_case.items() if len({r["passed"] for r in runs}) > 1),
        "always_failing_cases": sorted(cid for cid, runs in by_case.items() if not any(r["passed"] for r in runs)),
        "latency_ms_p50": _percentile([r["latency_ms"] for r in scored], 0.5),
        "latency_ms_p95": _percentile([r["latency_ms"] for r in scored], 0.95),
        "tokens_per_run": round(mean(r["usage"]["input_tokens"] + r["usage"]["output_tokens"] for r in scored))
        if scored else 0,
        "tokens_estimated": any(r["usage"]["estimated"] for r in scored),
        # Per-case results let the gate compare runs case by case instead of trusting two averages.
        "per_case": {
            cid: {"runs": len(runs), "passes": sum(1 for r in runs if r["passed"]),
                  "action_correct": sum(1 for r in runs if _score(r, "action_correct") is True)}
            for cid, runs in sorted(by_case.items())
        },
        "by_dataset": {},
        "slices": {},
        "counterfactual": _counterfactual(scored, case_index),
    }

    for dataset in HEADLINE_DATASETS:
        recs = [r for r in headline if case_index[r["case_id"]]["dataset"] == dataset]
        if recs:
            metrics["by_dataset"][dataset] = {"pass_rate": _pct([r["passed"] for r in recs]), "runs": len(recs)}

    # Observational slices: pass rate by group on the golden set. These describe the results; they can't
    # separate bias from differences in case mix between groups, which is what the counterfactual set is for.
    min_n = config["report"]["min_slice_cases"]
    for dim in config["report"]["slice_dimensions"]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in headline:
            if case_index[r["case_id"]]["dataset"] == "golden":
                groups[case_index[r["case_id"]]["tags"][dim]].append(r)
        rows = {value: {"pass_rate": _pct([r["passed"] for r in recs]),
                        "cases": len({r["case_id"] for r in recs})}
                for value, recs in sorted(groups.items())}
        comparable = [row["pass_rate"] for row in rows.values() if row["cases"] >= min_n]
        metrics["slices"][dim] = {"groups": rows, "compared_groups": len(comparable),
                                  "max_gap": round(max(comparable) - min(comparable), 4) if len(comparable) > 1 else 0.0}
    return metrics


def _counterfactual(scored: list[dict[str, Any]], case_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Each golden case has copies that differ only in the member's line of business. Compare each copy's
    usual decision: the action and queue it chose in a strict majority of runs. If the copies' usual
    decisions differ, the decision depended on line of business, which it never should.

    A copy with no strict majority (say 1-1 over two runs) makes its group inconclusive rather than a
    flip, so ordinary run-to-run noise isn't reported as bias."""
    decisions: dict[str, Counter] = defaultdict(Counter)
    for r in scored:
        if case_index[r["case_id"]]["dataset"] in ("golden", "counterfactual") and r["output"]:
            decisions[r["case_id"]][(r["output"].get("action"), r["output"].get("queue"))] += 1

    groups: dict[str, dict[str, Any]] = defaultdict(dict)
    for cid, counts in decisions.items():
        tags = case_index[cid]["tags"]
        (decision, n), total = counts.most_common(1)[0], sum(counts.values())
        groups[tags["counterfactual_group"]][tags["line_of_business"]] = decision if n * 2 > total else None

    compared = {g: d for g, d in groups.items() if len(d) > 1}
    inconclusive = sorted(g for g, d in compared.items() if None in d.values())
    flips = {g: {lob: f"{a}" + (f"/{q}" if q else "") for lob, (a, q) in sorted(d.items())}
             for g, d in sorted(compared.items()) if g not in inconclusive and len(set(d.values())) > 1}
    return {"groups_compared": len(compared), "flips": len(flips), "flipped_groups": flips,
            "inconclusive_groups": inconclusive}
