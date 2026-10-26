"""The release gate: compare a run against the baseline and decide pass or block.

Rules, all set in eval_config.toml:
- Comparable or nothing. If the run used a different case set, run count, or config than the
  baseline, the gate refuses to compare at all. Dropping the adversarial set or running once instead
  of three times would otherwise make a worse agent look fine.
- Safety is absolute: more than max_safety_failures critical failures blocks, whatever the baseline.
- Counterfactual fairness is absolute: a decision that changes with line of business blocks.
- Infrastructure errors above a small rate block, because the run no longer measures the agent.
- Quality is relative and paired. Each metric is compared case by case against the baseline, and a
  drop blocks only if it is bigger than the allowed amount AND a bootstrap over cases says it is
  real (the 95% interval for the change sits entirely below zero). Comparing two averages without
  that step blocks good builds on noise.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from claims_eval.agent.types import HarnessError

COMPARABLE_FIELDS = ("case_set_hash", "runs", "config_hash")
BOOTSTRAP_SAMPLES = 2000


@dataclass
class Check:
    name: str
    passed: bool
    current: float
    baseline: float | None
    limit: str
    note: str = ""


def require_comparable(current_meta: dict[str, Any], baseline_meta: dict[str, Any]) -> None:
    diffs = [f for f in COMPARABLE_FIELDS if current_meta.get(f) != baseline_meta.get(f)]
    if diffs:
        detail = ", ".join(f"{f}: {baseline_meta.get(f)!r} -> {current_meta.get(f)!r}" for f in diffs)
        raise HarnessError(f"run is not comparable to the baseline ({detail}). Use the baseline's datasets, "
                           "runs, and config, or promote a new baseline in a reviewed PR.")


def _case_value(metric: str, row: dict[str, int]) -> float:
    if metric == "pass_rate":
        return row["passes"] / row["runs"]
    if metric == "pass_hat_k":
        return float(row["passes"] == row["runs"])
    if metric == "action_accuracy":
        return row["action_correct"] / row["runs"]
    raise KeyError(metric)


def paired_delta(metric: str, current: dict[str, Any], baseline: dict[str, Any]) -> tuple[float, float, float]:
    """Mean per-case change and its 95% bootstrap interval, resampling cases."""
    common = sorted(set(current["per_case"]) & set(baseline["per_case"]))
    diffs = [_case_value(metric, current["per_case"][c]) - _case_value(metric, baseline["per_case"][c])
             for c in common]
    if not diffs:
        return 0.0, 0.0, 0.0
    rng = random.Random(0)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(BOOTSTRAP_SAMPLES))
    return (sum(diffs) / len(diffs), means[int(0.025 * BOOTSTRAP_SAMPLES)],
            means[int(0.975 * BOOTSTRAP_SAMPLES) - 1])


def evaluate(current: dict[str, Any], baseline: dict[str, Any] | None, config: dict[str, Any]) -> list[Check]:
    g = config["gate"]
    upper = current.get("safety_rate_upper_95")
    checks = [
        Check("safety_failures", current["safety_failures"] <= g["max_safety_failures"], current["safety_failures"],
              baseline["safety_failures"] if baseline else None, f"<= {g['max_safety_failures']} (absolute)",
              ", ".join(f"{k}: {v}" for k, v in current["safety_failures_by_scorer"].items())
              or (f"true rate could still be up to {upper:.1%} per run" if upper else "")),
        Check("counterfactual_flips", current["counterfactual"]["flips"] <= g["max_counterfactual_flips"],
              current["counterfactual"]["flips"], baseline["counterfactual"]["flips"] if baseline else None,
              f"<= {g['max_counterfactual_flips']} (absolute)",
              f"{current['counterfactual']['groups_compared']} groups; flipped: "
              f"{', '.join(current['counterfactual']['flipped_groups']) or 'none'}; inconclusive: "
              f"{len(current['counterfactual'].get('inconclusive_groups', []))}"),
        Check("infra_error_rate", current["infra_error_rate"] <= g["max_infra_error_rate"], current["infra_error_rate"],
              baseline["infra_error_rate"] if baseline else None, f"<= {g['max_infra_error_rate']:.0%}",
              f"{current['infra_errors']} runs excluded from scoring"),
    ]

    for metric, max_drop in g["max_drop"].items():
        cur = current[metric]
        if baseline is None:
            checks.append(Check(metric, True, cur, None, f"drop <= {max_drop:.0%}", "no baseline yet"))
            continue
        delta, lo, hi = paired_delta(metric, current, baseline)
        blocked = delta < -max_drop and hi < 0
        checks.append(Check(metric, not blocked, cur, baseline[metric], f"drop <= {max_drop:.0%} and significant",
                            f"{delta:+.1%} per case (95% CI {lo:+.1%} to {hi:+.1%})"))
    return checks


def passed(checks: list[Check]) -> bool:
    return all(c.passed for c in checks)
