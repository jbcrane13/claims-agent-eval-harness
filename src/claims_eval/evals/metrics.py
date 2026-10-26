"""Turns scored runs into the numbers the gate and the report use.

Two pass metrics, because agents are non-deterministic:
- pass_rate:  share of all runs that passed (pass@1 averaged over runs).
- pass_hat_k: share of cases where every one of the k runs passed (pass^k). This is the one that
              matters for mission-critical work: an agent that is right 2 times out of 3 on the same
              claim is not reliable, even though its pass_rate looks fine.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any


def _pct(values: list[bool]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def compute(records: list[dict[str, Any]], cases: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_case[r["case_id"]].append(r)
    case_index = {c["case_id"]: c for c in cases}

    def score_rate(name: str, recs: list[dict[str, Any]]) -> float:
        return _pct([next(s["passed"] for s in r["scores"] if s["name"] == name) for r in recs])

    scorer_names = [s["name"] for s in records[0]["scores"]] if records else []
    metrics: dict[str, Any] = {
        "cases": len(by_case),
        "runs_per_case": max((len(v) for v in by_case.values()), default=0),
        "pass_rate": _pct([r["passed"] for r in records]),
        "pass_hat_k": _pct([all(r["passed"] for r in runs) for runs in by_case.values()]),
        "action_accuracy": score_rate("action_correct", records),
        "safety_failures": sum(1 for r in records if r["critical_failures"]),
        "safety_failures_by_scorer": {
            name: sum(1 for r in records if name in r["critical_failures"]) for name in scorer_names
            if any(name in r["critical_failures"] for r in records)
        },
        "scorers": {name: score_rate(name, records) for name in scorer_names},
        "flaky_cases": sorted(cid for cid, runs in by_case.items()
                              if len({r["passed"] for r in runs}) > 1),
        "always_failing_cases": sorted(cid for cid, runs in by_case.items() if not any(r["passed"] for r in runs)),
        "latency_ms_p50": _percentile([r["latency_ms"] for r in records], 0.5),
        "latency_ms_p95": _percentile([r["latency_ms"] for r in records], 0.95),
        "tokens_per_run": round(mean(r["usage"]["input_tokens"] + r["usage"]["output_tokens"] for r in records))
        if records else 0,
        "tokens_estimated": any(r["usage"]["estimated"] for r in records),
        "by_dataset": {},
        "slices": {},
    }

    for dataset in sorted({c["dataset"] for c in cases}):
        recs = [r for r in records if case_index[r["case_id"]]["dataset"] == dataset]
        metrics["by_dataset"][dataset] = {"pass_rate": _pct([r["passed"] for r in recs]), "runs": len(recs)}

    # Fairness: pass rate per slice value. Only slices with enough cases are compared, and the
    # report says so, because a gap between two groups of three cases is noise, not bias.
    min_n = config["gate"]["min_slice_cases"]
    for dim in config["slices"]["dimensions"]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in records:
            if case_index[r["case_id"]]["dataset"] == "golden":
                groups[case_index[r["case_id"]]["tags"][dim]].append(r)
        rows = {
            value: {"pass_rate": _pct([r["passed"] for r in recs]),
                    "cases": len({r["case_id"] for r in recs}), "runs": len(recs)}
            for value, recs in sorted(groups.items())
        }
        comparable = [row["pass_rate"] for row in rows.values() if row["cases"] >= min_n]
        metrics["slices"][dim] = {
            "groups": rows,
            "max_gap": round(max(comparable) - min(comparable), 4) if len(comparable) > 1 else 0.0,
            "compared_groups": len(comparable),
        }
    return metrics
