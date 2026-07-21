"""Compare parsed resume against parsed JD and produce a prioritized gap list."""

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import GapAnalysis
from tools.llm_factory import structured

SYSTEM = """You compare a candidate's resume against a job description and
report the gaps honestly.

For each gap:
- kind="missing" when the resume shows nothing at all on the requirement.
  kind="weak" when there is some related evidence but it is thin, dated, or
  peripheral.
- severity: "critical" for must-haves the candidate cannot claim, "important"
  for must-haves with weak evidence or high-signal nice-to-haves, "minor" for
  the rest.
- unsupported_by_resume=True when NOTHING in the resume — no bullet, skill,
  project, certification, or degree — supports the claim, even loosely.

That flag is a safety interlock, not a nuance. Downstream, a tailoring step
reads it as "off-limits: do not write this into the resume." Set it True
whenever you are not certain the resume backs the claim. A false True costs the
candidate one keyword; a false False lets a fabrication into a document they
will sign their name to.

Order gaps by severity, critical first. Recommendations are for the candidate to
act on in the real world (get the cert, build the project), not instructions to
reword the resume.

The pre-computed ATS missing keywords are a strong signal — every entry there is
a candidate gap — but they are keyword-level and mechanical. Merge related ones,
drop any the resume actually demonstrates under different wording, and add gaps
the keyword scan could not see (missing seniority, domain, scope)."""


def gap_analysis_node(state) -> dict:
    jd = state.get("parsed_jd")
    resume = state.get("parsed_resume")
    pre_score = state.get("pre_score")

    missing = ""
    if pre_score is not None:
        missing = "\n".join(
            f"- {m.keyword}" + (" (must-have)" if m.is_must_have else "")
            for m in pre_score.missing_keywords
        )

    user = (
        f"JOB DESCRIPTION:\n{jd.model_dump_json(indent=2) if jd else '{}'}\n\n"
        f"RESUME:\n{resume.model_dump_json(indent=2) if resume else '{}'}\n\n"
        f"ATS KEYWORDS SCORED AS MISSING:\n{missing or '(none reported)'}"
    )

    return {
        "gap_analysis": structured(GapAnalysis).invoke(
            [SystemMessage(SYSTEM), HumanMessage(user)]
        )
    }
