"""Loads the synthetic fixtures and answers queries against them.

This is the "system of record" behind the MCP server. It is plain Python with no
MCP dependency, so the business rules about what each tool may return
(minimum-necessary PHI, DLP redaction) can be unit-tested directly.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "fixtures"

SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


class NotFound(LookupError):
    """A record the caller asked for does not exist."""


@cache
def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def claims() -> dict[str, Any]:
    return _load("claims")


def members() -> dict[str, Any]:
    return _load("members")


def providers() -> dict[str, Any]:
    return _load("providers")


def pend_codes() -> dict[str, Any]:
    return _load("pend_codes")


def policies() -> dict[str, Any]:
    return _load("policies")


def notes() -> dict[str, Any]:
    return _load("notes")


def redact(text: str) -> str:
    """Pattern-based DLP. Catches SSNs; it cannot catch a name or a date of birth in free text.

    That gap is deliberate: it is why the agent itself is also tested for PHI leaks.
    """
    return SSN_PATTERN.sub("[REDACTED-SSN]", text)


def get_claim(claim_id: str) -> dict[str, Any]:
    claim = claims().get(claim_id)
    if claim is None:
        raise NotFound(f"claim {claim_id} not found")
    return claim


def get_claim_notes(claim_id: str) -> dict[str, Any]:
    if claim_id not in claims():
        raise NotFound(f"claim {claim_id} not found")
    return {
        "claim_id": claim_id,
        "notes": [{"author": n["author"], "text": redact(n["text"])} for n in notes().get(claim_id, [])],
    }


def _active(span: dict[str, Any], day: str) -> bool:
    # ISO dates compare correctly as strings.
    return span["effective"] <= day and (span.get("termination") is None or day <= span["termination"])


def get_member_eligibility(member_id: str, date_of_service: str) -> dict[str, Any]:
    member = members().get(member_id)
    if member is None:
        raise NotFound(f"member {member_id} not found")
    # Minimum necessary: no name, DOB, SSN, or address. Eligibility questions don't need them.
    return {
        "member_id": member_id,
        "plan": member["plan"],
        "line_of_business": member["line_of_business"],
        "date_of_service": date_of_service,
        "eligible_on_dos": any(_active(s, date_of_service) for s in member["coverage"]),
        "coverage_spans": member["coverage"],
        "other_coverage": [
            dict(oc, active_on_dos=_active(oc, date_of_service)) for oc in member["other_coverage"]
        ],
    }


def get_provider(npi: str) -> dict[str, Any]:
    provider = providers().get(npi)
    if provider is None:
        raise NotFound(f"NPI {npi} not on provider file")
    return {"npi": npi, **provider}


def lookup_pend_code(code: str) -> dict[str, Any]:
    entry = pend_codes().get(code)
    if entry is None:
        raise NotFound(f"pend code {code} is not in the pend code table")
    return {"code": code, **entry}


def search_policy(query: str, limit: int = 3) -> dict[str, Any]:
    """Keyword search over policy documents. Deliberately simple; a real system would use Vertex AI Search."""
    terms = {t for t in re.findall(r"[a-z0-9-]+", query.lower()) if len(t) > 2}
    scored = []
    for doc_id, doc in policies().items():
        haystack = set(re.findall(r"[a-z0-9-]+", f"{doc['title']} {doc['text']}".lower()))
        keywords = {k.lower() for k in doc["keywords"]}
        score = 3 * len(terms & keywords) + len(terms & haystack)
        if score:
            scored.append((score, doc_id))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return {
        "query": query,
        "results": [
            {"doc_id": doc_id, "title": policies()[doc_id]["title"],
             "effective_date": policies()[doc_id]["effective_date"], "text": policies()[doc_id]["text"]}
            for _, doc_id in scored[:limit]
        ],
    }


def phi_values() -> list[str]:
    """Every PHI value in the member master. The safety scorer checks agent output against this list."""
    values = []
    for m in members().values():
        values.extend([m["name"], m["dob"], m["ssn"], m["address"]])
    return values
