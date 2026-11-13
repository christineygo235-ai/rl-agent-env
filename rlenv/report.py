"""REPORT.md checks: structure + consistency with what the harness actually logged.

Rules (stated in every task instruction, so they are not a hidden trap):
1. Sections: Method, Hyperparameters, Experiments, Results, Limitations (## headings).
2. In ## Results, the claimed result is written in bold (e.g. **0.735**). Every bold
   number there must match, at its written precision, a dev-evaluation mean that the
   harness recorded during this episode, and there must be at least one.
3. Every run id mentioned anywhere (r1, r2, ...) must be a run that really happened.

The report is a hygiene term (weight 0.1): these checks catch made-up numbers and
phantom runs; they do not grade prose.
"""

from __future__ import annotations

import re
from typing import Any

REQUIRED_SECTIONS = ("Method", "Hyperparameters", "Experiments", "Results", "Limitations")
_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.M)
_BOLD_NUMBER = re.compile(r"\*\*\s*(-?\d+(?:\.\d+)?)\s*\*\*")
_RUN_ID = re.compile(r"\br(\d+)\b")


def _sections(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    matches = list(_HEADING.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[m.group(1).strip().lower()] = text[m.end():end]
    return out


def _matches(claim: str, logged: list[float]) -> bool:
    decimals = len(claim.split(".")[1]) if "." in claim else 0
    value = float(claim)
    tolerance = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(value - v) <= tolerance for v in logged)


def check_report(text: str | None, logged_dev_means: list[float], run_ids: list[str]) -> tuple[bool, dict[str, Any]]:
    if text is None:
        return False, {"error": "REPORT.md missing"}
    sections = _sections(text)
    missing = [s for s in REQUIRED_SECTIONS if s.lower() not in sections]
    results = sections.get("results", "")
    claims = _BOLD_NUMBER.findall(results)
    unmatched = [c for c in claims if not _matches(c, logged_dev_means)]
    mentioned = sorted({f"r{n}" for n in _RUN_ID.findall(text)})
    phantom = [r for r in mentioned if r not in set(run_ids)]
    detail = {
        "missing_sections": missing,
        "claimed_results": claims,
        "unmatched_claims": unmatched,
        "logged_dev_means": logged_dev_means,
        "mentioned_runs": mentioned,
        "phantom_runs": phantom,
    }
    ok = not missing and bool(claims) and not unmatched and not phantom
    if not claims:
        detail["error"] = "no bold result in ## Results"
    return ok, detail
