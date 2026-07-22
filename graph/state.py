"""Shared schemas: the LangGraph state and every structured object flowing through it.

Every node reads and writes this state. The Pydantic models here are also the
structured-output schemas handed to the LLM, so they are the single source of
truth for the whole pipeline.
"""

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


# --- Parsed job description ------------------------------------------------


class ParsedJD(BaseModel):
    role_title: str = Field(description="The job title as written in the posting")
    seniority: str = Field(
        description="Seniority level, e.g. Junior, Mid, Senior, Staff, Principal"
    )
    must_have_skills: list[str] = Field(
        default_factory=list, description="Hard requirements; disqualifying if absent"
    )
    nice_to_have_skills: list[str] = Field(
        default_factory=list, description="Preferred but not required"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="ATS keywords: tools, technologies, methodologies, certifications",
    )
    responsibilities: list[str] = Field(
        default_factory=list, description="What the person will actually do"
    )
    domain: str = Field(description="Industry or domain, e.g. fintech, healthcare")


# --- Parsed resume ---------------------------------------------------------


class Contact(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    website: str = ""


class ExperienceEntry(BaseModel):
    company: str
    title: str
    dates: str = Field(description="As written on the resume, e.g. 'Jan 2020 - Present'")
    location: str = ""
    bullets: list[str] = Field(default_factory=list)


class EducationEntry(BaseModel):
    institution: str
    degree: str
    dates: str = ""
    details: str = ""


class ProjectEntry(BaseModel):
    name: str
    description: str = ""
    bullets: list[str] = Field(default_factory=list)


class ParsedResume(BaseModel):
    contact: Contact = Field(default_factory=Contact)
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


# --- Scoring ---------------------------------------------------------------


class KeywordMatch(BaseModel):
    keyword: str
    matched: bool
    match_type: Literal["exact", "semantic", "none"] = "none"
    # For semantic matches: what in the resume it matched against.
    matched_against: str = ""
    score: float = 0.0
    is_must_have: bool = False


class DimensionScore(BaseModel):
    name: str
    raw: float = Field(description="0.0-1.0 before weighting")
    weight: float
    weighted: float = Field(description="raw * weight * 100, i.e. points out of 100")
    detail: str = ""


class ATSScore(BaseModel):
    total: float = Field(description="0-100")
    dimensions: list[DimensionScore] = Field(default_factory=list)
    matched_keywords: list[KeywordMatch] = Field(default_factory=list)
    missing_keywords: list[KeywordMatch] = Field(default_factory=list)


# --- Gap analysis ----------------------------------------------------------


class Gap(BaseModel):
    item: str = Field(description="The missing or weak skill/keyword/requirement")
    severity: Literal["critical", "important", "minor"]
    kind: Literal["missing", "weak"]
    rationale: str
    # True when nothing in the base resume supports this. The tailor must NOT
    # invent it; it stays a genuine gap the candidate should know about.
    unsupported_by_resume: bool = False


class GapAnalysis(BaseModel):
    gaps: list[Gap] = Field(default_factory=list)
    recommendations: list[str] = Field(
        default_factory=list, description="Prioritized, actionable"
    )


# --- Tailoring -------------------------------------------------------------


class Edit(BaseModel):
    location: str = Field(description="Where the change was made, e.g. 'Experience[0].bullets[2]'")
    before: str = ""
    after: str = ""
    reason: str = Field(description="Why this change improves the JD match")


class TailorResult(BaseModel):
    """Only the edits. The tailored resume is built from them in code.

    Asking the model for the full rewritten ParsedResume *alongside* the edit
    list made it echo the input resume back unchanged while still describing
    edits it never applied — so the post-tailoring score was always identical
    to the pre-tailoring one. Dropping the redundant field also makes the edit
    log authoritative: what it says changed is exactly what changed.
    """

    edits: list[Edit] = Field(default_factory=list)


# --- Truthfulness validation ----------------------------------------------


class Fabrication(BaseModel):
    kind: Literal["company", "title", "date", "degree", "certification", "metric", "skill"]
    value: str
    location: str = ""
    explanation: str = ""


class ValidationResult(BaseModel):
    passed: bool
    fabrications: list[Fabrication] = Field(default_factory=list)


# --- Graph state -----------------------------------------------------------


class GraphState(TypedDict, total=False):
    """State threaded through every node.

    `total=False` because nodes fill this in progressively; early nodes must be
    able to run before later keys exist.
    """

    # Inputs
    raw_resume_text: str
    raw_jd_text: str
    # Only needed when no github.com link can be recovered from the resume file.
    github_username: str

    # Parsed
    parsed_resume: ParsedResume
    parsed_jd: ParsedJD

    # Scores
    pre_score: ATSScore
    post_score: ATSScore

    # Repos auto-added to the resume, by URL. Not cosmetic: appending content to
    # someone's resume without telling them is this feature's failure mode, so
    # the UI has to be able to show exactly what was added.
    github_projects_added: list[str]

    # Analysis and output
    gap_analysis: GapAnalysis
    tailored_resume: ParsedResume
    edit_log: list[Edit]

    # Truthfulness loop control
    validation: ValidationResult
    tailor_attempts: int
    validator_feedback: str

    # Score optimization loop. `best_*` exist because a re-tailoring round can
    # score WORSE than the one before it; without keeping the best-so-far, an
    # optimization pass could hand back a regression.
    optimize_rounds: int
    optimizer_feedback: str
    best_resume: ParsedResume
    best_score: ATSScore
    best_edit_log: list[Edit]
    score_history: list[float]
    target_met: bool
    # Why optimization stopped, in plain language for the user.
    ceiling_reason: str

    # Rendered artifact
    docx_path: str

    # Set when the graph hard-stops (e.g. fabrication could not be resolved).
    error: str
