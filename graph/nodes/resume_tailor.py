"""Rewrite the resume for the JD without inventing anything.

The guardrail below is the product's core safety property, not a style
preference: the output is a document a real person submits under their own name.
Anything the model adds that the candidate cannot defend in an interview is a
defect, however well it scores.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import ParsedResume, TailorResult
from tools.llm_factory import structured

SYSTEM = """You tailor an existing resume to a specific job description.

YOU MAY:
- Reorder sections and bullets so the most relevant experience comes first.
- Rewrite bullets to surface relevant impact and to use the job description's
  vocabulary for work the candidate actually did.
- Promote skills the candidate already lists into more prominent positions.
- Tighten wording, cut filler, sharpen verbs.
- Add a keyword ONLY where the candidate's real experience already supports it
  (they used the tool, did the methodology, worked in the domain).

YOU MUST NEVER:
- Invent or alter employers, job titles, employment dates, degrees,
  institutions, or certifications.
- Invent or alter metrics. Every number in the output must come from the input.
- Add a skill the candidate does not demonstrably have.
- Inflate scope or seniority: "contributed to" does not become "led"; "team of
  3" does not become "team of 30"; "assisted with migration" does not become
  "architected the migration".

HOW TO ACTUALLY MOVE THE MATCH
Reordering bullets between jobs changes nothing for a keyword scanner — the
same words are still on the page. What moves the match is REWRITING BULLET TEXT
so that work the candidate really did is described in the posting's vocabulary:

  Base:   "Built and scheduled Airflow DAGs loading into Redshift."
  Tailored: "Built ETL data pipelines with Apache Airflow, loading into Amazon
             Redshift."

Same facts, same scope, same tools — now stated in the words the employer
searches for. Do this for every keyword the experience genuinely supports. Also
add such terms to the skills list when the candidate demonstrably uses them.
Prefer the posting's exact phrasing ("Apache Airflow" over "Airflow") where the
resume already shows the tool.

If you find yourself only reordering and lightly rewording, you have not done
the job.

Every company, title, date, degree, certification and number in your output must
appear in the input resume. If the job description requires something the resume
does not support, it STAYS A GAP — do not write it in. Gaps flagged
unsupported_by_resume are absolutely off-limits.

OUTPUT FORMAT — READ CAREFULLY
You return a list of edits, and nothing else. Those edits ARE the rewrite: they
get applied to the base resume mechanically, so an edit you do not write is a
change that does not happen.

For each edit:
- `before`: the EXACT original text, copied character-for-character from the
  base resume field you are changing. It is what locates the field, so an
  approximate quote silently drops the edit. Leave it empty ONLY when adding a
  new entry to the skills or certifications list.
- `after`: the full replacement text for that field (not a fragment, not a diff).
- `location`: e.g. 'experience[0].bullets[2]', 'summary', 'skills'.
- `reason`: why it improves the match.

One edit per field changed. Rewrite every bullet whose wording can honestly
carry the posting's vocabulary — a handful of edits means you left points on
the table. Additions are limited to skills and certifications the candidate
demonstrably already has; you cannot add bullets, roles, or employers."""

RETRY_PREFIX = """*** PREVIOUS ATTEMPT REJECTED FOR FABRICATION ***

A truthfulness validator rejected your last output. The findings below name
content that does NOT exist in the base resume. Remove every one of these and do
not reintroduce them or any paraphrase of them. Where a rejected item was
covering a job-description requirement, that requirement stays an unmet gap.

{feedback}

*** END REJECTION NOTICE ***

"""

OPTIMIZE_PREFIX = """*** SCORE OPTIMIZATION PASS ***

Your previous version was truthful but scored below target. The analysis below
says exactly where the points are. Raise the score WITHOUT relaxing a single
rule above — a higher score obtained by inventing experience is a failure.

{feedback}

*** END OPTIMIZATION NOTICE ***

"""


def _targeting_block(state) -> str:
    """The specific terms an ATS looked for and did not find.

    Without this the first tailoring pass is blind: it sees the gap analysis
    but not the scorer's actual misses, so it optimizes for readability and
    the score does not move. Gaps flagged unsupported_by_resume are excluded —
    naming them here would be asking for a fabrication.
    """
    pre = state.get("pre_score")
    if not pre or not pre.missing_keywords:
        return ""

    gaps = state.get("gap_analysis")
    off_limits = (
        {g.item.strip().lower() for g in gaps.gaps if g.unsupported_by_resume}
        if gaps else set()
    )
    winnable = [
        k for k in pre.missing_keywords
        if k.keyword.strip().lower() not in off_limits
    ]
    if not winnable:
        return ""

    must = [k.keyword for k in winnable if k.is_must_have]
    rest = [k.keyword for k in winnable if not k.is_must_have]

    lines = [
        "",
        "",
        "KEYWORD TARGETING — an ATS keyword scan did NOT find these terms in "
        "the resume. For each one the candidate's experience genuinely "
        "supports, use the job description's exact wording in the bullet or "
        "skills entry where that work actually appears. If the resume says "
        "'Airflow' and the posting says 'Apache Airflow', prefer the "
        "posting's phrasing. Do NOT add any term the experience does not "
        "support.",
    ]
    if must:
        lines.append(f"MISSING (must-have): {', '.join(must)}")
    if rest:
        lines.append(f"MISSING (other): {', '.join(rest)}")
    return "\n".join(lines)


def _text_slots(node):
    """Yield (container, key) for every string leaf in a nested dict/list."""
    items = node.items() if isinstance(node, dict) else (
        enumerate(node) if isinstance(node, list) else ()
    )
    for key, value in items:
        if isinstance(value, str):
            yield node, key
        else:
            yield from _text_slots(value)


def _replace(data: dict, before: str, after: str) -> bool:
    """Rewrite the one field whose text is `before`. True if something changed.

    Matching on the text rather than parsing Edit.location ('experience[0].
    bullets[2]') because the location is prose the model composes freehand,
    while `before` is quoted from the resume we sent it. A stale index silently
    rewrites the wrong bullet; a stale quote just fails to match.
    """
    slots = list(_text_slots(data))
    for container, key in slots:
        if container[key].strip() == before:
            container[key] = after
            return True
    # Fallback for a `before` the model truncated or lightly reflowed.
    head = before[:40]
    if len(head) < 20:  # too short to identify a field unambiguously
        return False
    for container, key in slots:
        if container[key].strip().startswith(head):
            container[key] = after
            return True
    return False


def apply_edits(resume: ParsedResume, edits: list) -> tuple[ParsedResume, list]:
    """Build the tailored resume by applying `edits`. Returns (resume, applied).

    An edit that matches nothing is dropped rather than appended: unmatched
    `after` text is content with no home in the document, and inserting it
    blindly is how invented experience would get in.
    """
    data = resume.model_dump()
    applied = []
    for edit in edits:
        before, after = edit.before.strip(), edit.after.strip()
        if not after or before == after:
            continue
        if before:
            if _replace(data, before, after):
                applied.append(edit)
        # No `before` means an addition. Only the flat keyword lists take one;
        # a new bullet or role would be new experience, which the tailor is
        # never allowed to create.
        elif "skill" in edit.location.lower():
            if after not in data["skills"]:
                data["skills"].append(after)
                applied.append(edit)
        elif "cert" in edit.location.lower():
            if after not in data["certifications"]:
                data["certifications"].append(after)
                applied.append(edit)
    return ParsedResume.model_validate(data), applied


def resume_tailor_node(state) -> dict:
    resume = state.get("parsed_resume")
    jd = state.get("parsed_jd")
    gaps = state.get("gap_analysis")
    feedback = state.get("validator_feedback")
    optimize = state.get("optimizer_feedback")

    user = (
        f"BASE RESUME (the only source of facts):\n"
        f"{resume.model_dump_json(indent=2) if resume else '{}'}\n\n"
        f"JOB DESCRIPTION:\n{jd.model_dump_json(indent=2) if jd else '{}'}\n\n"
        f"GAP ANALYSIS:\n{gaps.model_dump_json(indent=2) if gaps else '{}'}"
        f"{_targeting_block(state)}"
    )
    # A fabrication rejection outranks a score shortfall: correctness first.
    if optimize and not feedback:
        user = OPTIMIZE_PREFIX.format(feedback=optimize) + user
    if feedback:
        user = RETRY_PREFIX.format(feedback=feedback) + user

    result = structured(TailorResult).invoke(
        [SystemMessage(SYSTEM), HumanMessage(user)]
    )
    tailored, applied = apply_edits(resume, result.edits)
    return {
        "tailored_resume": tailored,
        "edit_log": applied,
        "tailor_attempts": state.get("tailor_attempts", 0) + 1,
    }
