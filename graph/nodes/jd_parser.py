"""Parse a raw job description into the structured ParsedJD the rest of the graph scores against."""

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import ParsedJD
from tools.llm_factory import structured

SYSTEM = """You extract structured data from job postings for an ATS (Applicant \
Tracking System) matching pipeline.

Extract keywords the way an ATS indexes them: concrete, searchable nouns —
tools, technologies, platforms, languages, frameworks, certifications, named
methodologies (e.g. "Kubernetes", "SOC 2", "PySpark", "Scrum", "AWS Lambda",
"CPA"). Do NOT emit soft filler: "team player", "fast-paced environment",
"excellent communication", "self-starter". Those never match anything and only
dilute the score.

Split requirements by how the posting phrases them:
- must_have_skills: stated as required / must have / essential / minimum
  qualifications / "you have".
- nice_to_have_skills: preferred / bonus / plus / nice to have / "ideally".
When a posting does not mark a requirement either way, judge by whether the role
is plainly undoable without it; default to nice-to-have if genuinely ambiguous.

keywords is the union of everything ATS-indexable, including terms that appear
in the responsibilities but not in the requirements list.

Use the posting's own wording for skills and keywords — the downstream matcher
compares against resume text, so paraphrasing loses matches."""


def jd_parser_node(state) -> dict:
    jd = state.get("raw_jd_text", "")
    parsed = structured(ParsedJD).invoke(
        [SystemMessage(SYSTEM), HumanMessage(f"Job posting:\n\n{jd}")]
    )
    return {"parsed_jd": parsed}
