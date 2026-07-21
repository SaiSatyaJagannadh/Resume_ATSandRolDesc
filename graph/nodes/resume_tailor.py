"""Rewrite the resume for the JD without inventing anything.

The guardrail below is the product's core safety property, not a style
preference: the output is a document a real person submits under their own name.
Anything the model adds that the candidate cannot defend in an interview is a
defect, however well it scores.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import TailorResult
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

Every company, title, date, degree, certification and number in your output must
appear in the input resume. If the job description requires something the resume
does not support, it STAYS A GAP — do not write it in. Gaps flagged
unsupported_by_resume are absolutely off-limits.

Produce an Edit entry for every change you make, with the before text, the after
text, its location (e.g. 'experience[0].bullets[2]'), and the reason it improves
the match. An unlogged change is a change nobody can audit."""

RETRY_PREFIX = """*** PREVIOUS ATTEMPT REJECTED FOR FABRICATION ***

A truthfulness validator rejected your last output. The findings below name
content that does NOT exist in the base resume. Remove every one of these and do
not reintroduce them or any paraphrase of them. Where a rejected item was
covering a job-description requirement, that requirement stays an unmet gap.

{feedback}

*** END REJECTION NOTICE ***

"""


def resume_tailor_node(state) -> dict:
    resume = state.get("parsed_resume")
    jd = state.get("parsed_jd")
    gaps = state.get("gap_analysis")
    feedback = state.get("validator_feedback")

    user = (
        f"BASE RESUME (the only source of facts):\n"
        f"{resume.model_dump_json(indent=2) if resume else '{}'}\n\n"
        f"JOB DESCRIPTION:\n{jd.model_dump_json(indent=2) if jd else '{}'}\n\n"
        f"GAP ANALYSIS:\n{gaps.model_dump_json(indent=2) if gaps else '{}'}"
    )
    if feedback:
        user = RETRY_PREFIX.format(feedback=feedback) + user

    result = structured(TailorResult).invoke(
        [SystemMessage(SYSTEM), HumanMessage(user)]
    )
    return {
        "tailored_resume": result.resume,
        "edit_log": result.edits,
        "tailor_attempts": state.get("tailor_attempts", 0) + 1,
    }
