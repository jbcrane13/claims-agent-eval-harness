"""The release gate: compare a run against the baseline and decide pass or block.

Rules, all set in eval_config.toml:
- Safety is absolute: more than max_safety_failures critical failures blocks, whatever the baseline.
- Quality is relative: each tracked metric may not drop more than its allowed amount below baseline.
  Gating on deltas (not fixed targets) keeps the bar honest as the dataset gets harder.
- Fairness: the largest pass-rate gap across comparable slices may not exceed max_slice_gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Check:
    name: str
    passed: bool
    current: float
    baseline: float | None
    limit: str
    note: str = ""


def evaluate(current: dict[str, Any], baseline: dict[str, Any] | None, config: dict[str, Any]) -> list[Check]:
    g = config["gate"]
    checks = [Check(
        "safety_failures", current["safety_failures"] <= g["max_safety_failures"], current["safety_failures"],
        baseline["safety_failures"] if baseline else None, f"<= {g['max_safety_failures']} (absolute)",
        ", ".join(f"{k}: {v}" for k, v in current["safety_failures_by_scorer"].items()),
    )]

    for metric, max_drop in g["max_drop"].items():
        cur = current[metric]
        if baseline is None:
            checks.append(Check(metric, True, cur, None, f"drop <= {max_drop:.0%}", "no baseline yet"))
            continue
        base = baseline[metric]
        checks.append(Check(metric, cur >= base - max_drop, cur, base, f"drop <= {max_drop:.0%}",
                            f"{cur - base:+.1%} vs baseline"))

    for dim, s in current["slices"].items():
        note = f"{s['compared_groups']} groups with >= {g['min_slice_cases']} cases"
        checks.append(Check(f"slice_gap:{dim}", s["max_gap"] <= g["max_slice_gap"], s["max_gap"],
                            baseline["slices"][dim]["max_gap"] if baseline and dim in baseline["slices"] else None,
                            f"<= {g['max_slice_gap']:.0%}", note))
    return checks


def passed(checks: list[Check]) -> bool:
    return all(c.passed for c in checks)
