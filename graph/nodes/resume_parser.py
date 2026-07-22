"""Transcribe a raw resume into ParsedResume.

This parse is the ground truth the truthfulness validator diffs the tailored
resume against. A hallucination here is not a cosmetic error: it silently widens
the set of "facts" the tailor is allowed to keep, authorizing a fabrication that
the validator will then wave through. Hence the unusually blunt prompt.
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import ParsedResume
from tools.llm_factory import structured

SYSTEM = """You are a faithful transcriber. You convert resume text into
structured fields. You are NOT a writer, editor, or recruiter.

Rules, in priority order:
1. Transcribe only what is literally in the text. Do not invent, infer,
   embellish, summarize, or normalize.
2. Copy dates and job titles VERBATIM, exactly as written — including format,
   abbreviations, casing, and separators. "Jan 2020 - Present" stays
   "Jan 2020 - Present"; do not convert it to "January 2020 - Present" or
   "2020-Present". Same for company names, institutions, degrees, and
   certifications.
3. Copy bullets as written. Do not rewrite, merge, split, or "improve" them.
   Keep every number exactly as it appears.
4. If a field is absent from the text, leave it empty. An empty field is
   CORRECT. A guessed field is a bug. Never fill in a plausible email, a likely
   location, an implied seniority, or an assumed graduation year.
5. Do not deduplicate or reorder. Preserve the resume's own ordering.
6. SKILLS: capture EVERY item listed in any skills, technologies, tools, or
   technical-competencies section — one array entry per skill. These sections
   are usually dense delimited lists, sometimes grouped under sub-labels, e.g.
   "Languages: Python, SQL, Java | Cloud: AWS, GCP | Tools: Docker, Airflow".
   Split on every delimiter (commas, pipes, slashes, semicolons, bullets) and
   emit each skill separately: Python, SQL, Java, AWS, GCP, Docker, Airflow.
   Drop the group labels themselves, keep the skills. Returning a handful of
   entries when the resume lists dozens is a transcription failure — these
   entries are exactly what an ATS keyword scan reads first.
   When a vendor groups its products in brackets, emit the vendor and each
   product as separate WHOLE entries — never split through the bracket itself:
   "AWS (S3, EC2, Redshift)" becomes AWS, S3, EC2, Redshift. Every entry must
   read as a skill on its own; "AWS (S3" and "Redshift)" are broken fragments
   that match nothing.

If the text is ambiguous, prefer the more literal reading and leave the
uncertain part out."""


def repair_skills(skills: list[str]) -> list[str]:
    """Split skill entries whose brackets do not balance.

    Rule 6 tells the model to break dense skill lists on commas, and it applies
    that inside parentheses too: "AWS (S3, EC2, Redshift)" comes back as
    "AWS (S3", "EC2", "Redshift)". Those fragments match no keyword scan and
    poison the scorer's semantic candidates, so the stray bracket is treated as
    one more delimiter. Balanced entries ("Azure Data Factory (ADF)") are left
    alone.
    """
    out = []
    for skill in skills:
        parts = (
            re.split(r"[()]", skill)
            if skill.count("(") != skill.count(")")
            else [skill]
        )
        for part in parts:
            part = re.sub(r"\s+", " ", part).strip(" ,;|/")
            if part and part not in out:
                out.append(part)
    return out


def resume_parser_node(state) -> dict:
    text = state.get("raw_resume_text", "")
    parsed = structured(ParsedResume).invoke(
        [SystemMessage(SYSTEM), HumanMessage(f"Resume text:\n\n{text}")]
    )
    parsed.skills = repair_skills(parsed.skills)
    return {"parsed_resume": parsed}
