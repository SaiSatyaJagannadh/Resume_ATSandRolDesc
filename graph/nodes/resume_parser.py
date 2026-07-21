"""Transcribe a raw resume into ParsedResume.

This parse is the ground truth the truthfulness validator diffs the tailored
resume against. A hallucination here is not a cosmetic error: it silently widens
the set of "facts" the tailor is allowed to keep, authorizing a fabrication that
the validator will then wave through. Hence the unusually blunt prompt.
"""

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

If the text is ambiguous, prefer the more literal reading and leave the
uncertain part out."""


def resume_parser_node(state) -> dict:
    text = state.get("raw_resume_text", "")
    parsed = structured(ParsedResume).invoke(
        [SystemMessage(SYSTEM), HumanMessage(f"Resume text:\n\n{text}")]
    )
    return {"parsed_resume": parsed}
