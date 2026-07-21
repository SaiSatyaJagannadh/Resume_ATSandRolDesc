"""Verify the tailored resume invented nothing that isn't in the base resume.

Two layers, and the ORDER matters. The deterministic set-comparison runs first
because it cannot be argued with: it is string containment, so it has no opinion
to be talked out of by a persuasive-sounding rewrite. The LLM layer runs second
and can only ADD findings — it is never allowed to clear a deterministic one.
Fabricated metrics in particular are the highest-risk failure mode and are
caught reliably by plain string comparison, so they are never delegated to a
model.
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import Fabrication, ParsedResume, ValidationResult
from tools.llm_factory import structured

SYSTEM = """You audit a tailored resume against the candidate's base resume for
truthfulness. The base resume is the ONLY source of fact.

A separate deterministic checker already compares companies, titles, dates,
degrees, institutions, certifications and numbers character by character. Do not
duplicate it. Your job is the semantic fabrication string matching cannot see:

- Inflated scope: "supported the rollout" rewritten as "owned the rollout";
  a team member presented as the team lead; a component presented as the system.
- Seniority creep: implying management, architecture ownership, or strategic
  authority the base resume never shows.
- Invented skills: a technology asserted as experience when the base resume only
  mentions it in passing, or not at all.
- Invented responsibilities or outcomes that no base bullet supports.

Rewording, reordering, tightening and using the job's vocabulary for the same
underlying work are LEGITIMATE — do not report them.

passed=false if and only if you find at least one fabrication. For each, give
the offending value, where it appears, and what the base resume actually says."""


# --- Layer 1: deterministic ------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")
# Digits not glued to a letter or another digit-word, so "S3" / "EC2" / "COVID19"
# don't register as the numbers 3, 2, 19 and produce false positives.
_NUM = re.compile(r"(?<![A-Za-z0-9.])\d[\d,]*(?:\.\d+)?")


def _norm(s: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", s or "")).strip().casefold()


def _numbers(text: str) -> set[str]:
    return {m.group().replace(",", "").rstrip(".") for m in _NUM.finditer(text or "")}


def find_fabrications(base: ParsedResume, tailored: ParsedResume) -> list[Fabrication]:
    """Every entity in `tailored` that has no counterpart in `base`.

    Importable without an LLM on purpose: this is the half that must be tested.
    """
    out: list[Fabrication] = []

    companies = {_norm(e.company) for e in base.experience}
    titles = {_norm(e.title) for e in base.experience}
    dates = {_norm(e.dates) for e in base.experience} | {
        _norm(e.dates) for e in base.education
    }
    institutions = {_norm(e.institution) for e in base.education}
    degrees = {_norm(e.degree) for e in base.education}
    certs = {_norm(c) for c in base.certifications}

    def check(value, known, kind, location, label):
        n = _norm(value)
        if n and n not in known:
            out.append(
                Fabrication(
                    kind=kind,
                    value=value,
                    location=location,
                    explanation=f"{label} '{value}' does not appear in the base resume.",
                )
            )

    for i, e in enumerate(tailored.experience):
        check(e.company, companies, "company", f"experience[{i}].company", "Employer")
        check(e.title, titles, "title", f"experience[{i}].title", "Job title")
        check(e.dates, dates, "date", f"experience[{i}].dates", "Employment date")

    for i, e in enumerate(tailored.education):
        check(e.institution, institutions, "degree", f"education[{i}].institution", "Institution")
        check(e.degree, degrees, "degree", f"education[{i}].degree", "Degree")

    for i, c in enumerate(tailored.certifications):
        check(c, certs, "certification", f"certifications[{i}]", "Certification")

    # Metrics: compare against every number anywhere in the base resume, not
    # per-bullet. A real number moved to a different bullet is legitimate
    # reordering, and flagging it would block valid tailoring.
    base_numbers = _numbers(base.model_dump_json())
    tailored_text = [
        (f"experience[{i}].bullets[{j}]", b)
        for i, e in enumerate(tailored.experience)
        for j, b in enumerate(e.bullets)
    ] + [
        (f"projects[{i}].bullets[{j}]", b)
        for i, p in enumerate(tailored.projects)
        for j, b in enumerate(p.bullets)
    ] + [("summary", tailored.summary)]

    for location, text in tailored_text:
        for num in sorted(_numbers(text) - base_numbers):
            out.append(
                Fabrication(
                    kind="metric",
                    value=num,
                    location=location,
                    explanation=(
                        f"The number {num} appears in '{text}' but nowhere in the "
                        f"base resume."
                    ),
                )
            )

    return out


# --- Layer 2 + merge -------------------------------------------------------


def _feedback(fabrications: list[Fabrication]) -> str:
    """Named, located, actionable — this text is fed straight back to the tailor.

    Vagueness here costs a whole retry, so each line says exactly what to delete
    and where it is.
    """
    if not fabrications:
        return ""
    lines = [
        f"- [{f.kind}] {f.value!r} at {f.location or 'unknown location'}: {f.explanation}"
        for f in fabrications
    ]
    return (
        "The tailored resume was REJECTED. The following content is not present "
        "in the base resume and must be removed:\n"
        + "\n".join(lines)
        + "\nReplace each with the wording the base resume actually supports, or "
        "drop the claim entirely. Do not substitute a different invented value."
    )


def truthfulness_validator_node(state) -> dict:
    base = state.get("parsed_resume")
    tailored = state.get("tailored_resume")
    if base is None or tailored is None:
        return {
            "validation": ValidationResult(passed=False),
            "validator_feedback": "Validation could not run: base or tailored resume missing.",
        }

    hard = find_fabrications(base, tailored)

    soft = structured(ValidationResult).invoke(
        [
            SystemMessage(SYSTEM),
            HumanMessage(
                f"BASE RESUME (ground truth):\n{base.model_dump_json(indent=2)}\n\n"
                f"TAILORED RESUME:\n{tailored.model_dump_json(indent=2)}"
            ),
        ]
    )

    merged = hard + list(soft.fabrications)
    passed = not hard and soft.passed
    return {
        "validation": ValidationResult(passed=passed, fabrications=merged),
        "validator_feedback": "" if passed else _feedback(merged),
    }
