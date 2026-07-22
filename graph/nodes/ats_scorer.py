"""Deterministic ATS scoring: pure Python + embeddings, no LLM.

Five weighted dimensions (weights from config.WEIGHTS). Every number here is
reproducible and explainable, which is the point: an LLM-guessed score is not
defensible to a user asking "why 63?".
"""

import re

import config
from graph.state import ATSScore, DimensionScore, KeywordMatch, ParsedJD, ParsedResume
from tools.embeddings import best_match, embed

# Anything that reads as a measurable outcome: digits, percent, currency, or a
# magnitude word ("cut costs by half a million").
# "zero downtime" / "no data loss" are quantified outcomes an employer reads as
# hard numbers; only the digit is missing.
_METRIC_RE = re.compile(
    r"\d|%|[$€£₹]|\b(?:\d+[kmb]n?|k|m|bn|million|billion|thousand|zero)\b", re.I
)


def _norm(text: str) -> str:
    """Lowercase, punctuation -> space, collapse whitespace.

    '+' and '#' survive because they carry meaning in skill names (C++, C#).
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+#]+", " ", text.lower())).strip()


def _contains(haystack: str, needle: str) -> bool:
    """Word-boundary substring test on already-normalized text.

    Plain `in` would make "Java" match "JavaScript" and "R" match everything.
    \\b is wrong here too: it would break "C++" (the trailing + is a non-word
    char, so \\b after it demands a word char). Lookarounds against an explicit
    boundary class handle both.
    """
    if not needle:
        return False
    return re.search(rf"(?<![\w+#]){re.escape(needle)}(?![\w+#])", haystack) is not None


# Vendor/qualifier words a JD routinely prefixes onto a tool name that resumes
# routinely omit: "Apache Airflow" vs "Airflow", "Amazon S3" vs "S3". Without
# stripping these the exact match fails and the semantic fallback dilutes below
# threshold against a long bullet, so a skill the candidate genuinely has is
# scored as missing.
_QUALIFIERS = {
    "apache", "amazon", "aws", "microsoft", "google", "oracle", "ibm",
    "adobe", "atlassian", "hashicorp", "elastic", "red", "hat", "ms",
}


def _keyword_variants(key: str) -> list[str]:
    """Normalized forms to try for a JD keyword, most specific first."""
    variants = [key]
    tokens = key.split()
    # Drop leading qualifier words ("apache airflow" -> "airflow").
    i = 0
    while i < len(tokens) - 1 and tokens[i] in _QUALIFIERS:
        i += 1
    if i:
        variants.append(" ".join(tokens[i:]))
    return variants


def _resume_chunks(resume: ParsedResume) -> list[str]:
    """Every free-text fragment of the resume, as separate semantic candidates."""
    chunks = list(resume.skills) + list(resume.certifications)
    if resume.summary:
        chunks.append(resume.summary)
    for exp in resume.experience:
        chunks.append(f"{exp.title} {exp.company}".strip())
        chunks += exp.bullets
    for proj in resume.projects:
        chunks.append(f"{proj.name} {proj.description}".strip())
        chunks += proj.bullets
    return [c for c in chunks if c.strip()]


def _bullets(resume: ParsedResume) -> list[str]:
    return [b for exp in resume.experience for b in exp.bullets if b.strip()]


# --- Dimensions ------------------------------------------------------------


def _score_keywords(resume: ParsedResume, jd: ParsedJD):
    chunks = _resume_chunks(resume)
    haystack = _norm(" ".join(chunks))

    terms = [(k, True) for k in jd.must_have_skills]
    terms += [(k, False) for k in jd.keywords + jd.nice_to_have_skills]

    seen, matches = set(), []
    for keyword, must in terms:
        key = _norm(keyword)
        if not key or key in seen:
            continue
        seen.add(key)

        if any(_contains(haystack, v) for v in _keyword_variants(key)):
            matches.append(
                KeywordMatch(
                    keyword=keyword, matched=True, match_type="exact",
                    score=1.0, is_must_have=must,
                )
            )
            continue

        # Partial credit only: a synonym in the resume helps a human reader but
        # a literal-string ATS filter will still miss it.
        #
        # Concept phrases only. A single-token keyword is a product or tool name
        # — you either have the tool or you do not — and embeddings rank those by
        # vibe rather than substance: measured against this corpus, "Go" sits
        # 0.491 from "Git" and "CloudFormation" 0.508 from "AWS", both ABOVE
        # genuine multi-word matches like "access control" -> the Azure AD bullet
        # at 0.383. No threshold separates them, so there is nothing to tune;
        # crediting them would both lie to the user and push the tailor toward
        # writing Terraform experience that does not exist.
        # ponytail: word count as a proxy for "concept vs product name" — a
        # two-word product ("Azure DevOps") can still slip through. Swap in a
        # capitalisation/known-vendor check if false positives show up in
        # practice; the truthfulness validator is the backstop meanwhile.
        against, sim = ("", 0.0)
        if len(key.split()) > 1:
            against, sim = best_match(keyword, chunks)
        if sim >= config.SEMANTIC_MATCH_THRESHOLD:
            matches.append(
                KeywordMatch(
                    keyword=keyword, matched=True, match_type="semantic",
                    matched_against=against, score=config.SEMANTIC_MATCH_CREDIT,
                    is_must_have=must,
                )
            )
        else:
            matches.append(
                KeywordMatch(keyword=keyword, matched=False, match_type="none",
                             score=0.0, is_must_have=must)
            )

    if not matches:
        return 0.0, "No keywords in the JD to match against.", []

    weights = [config.MUST_HAVE_MULTIPLIER if m.is_must_have else 1.0 for m in matches]
    raw = sum(m.score * w for m, w in zip(matches, weights)) / sum(weights)
    hits = sum(1 for m in matches if m.matched)
    detail = f"{hits}/{len(matches)} JD terms found in the resume."
    return raw, detail, matches


def _score_title(resume: ParsedResume, jd: ParsedJD):
    target = f"{jd.role_title} {jd.seniority}".strip()
    titles = [e.title for e in resume.experience[:2] if e.title.strip()]
    if not target or not titles:
        return 0.0, "No comparable job title on the resume."
    against, sim = best_match(target, titles)
    return max(0.0, min(1.0, sim)), f"Closest title: {against!r} vs JD {target!r}."


def _score_responsibilities(resume: ParsedResume, jd: ParsedJD):
    bullets = _bullets(resume)
    if not jd.responsibilities or not bullets:
        return 0.0, "No responsibilities or no experience bullets to compare."
    sims = [best_match(r, bullets)[1] for r in jd.responsibilities]
    raw = max(0.0, min(1.0, sum(sims) / len(sims)))
    return raw, f"Mean best-match similarity over {len(sims)} responsibilities."


def _score_quantified_impact(resume: ParsedResume, _jd: ParsedJD):
    bullets = _bullets(resume)
    if not bullets:
        return 0.0, "No experience bullets."
    hits = sum(1 for b in bullets if _METRIC_RE.search(b))
    return hits / len(bullets), f"{hits}/{len(bullets)} bullets contain a metric."


def _score_formatting(resume: ParsedResume, _jd: ParsedJD):
    checks = {
        "email": bool(re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", resume.contact.email.strip())),
        "phone": len(re.sub(r"\D", "", resume.contact.phone)) >= 7,
        "experience section": bool(resume.experience),
        "education section": bool(resume.education),
        "skills": bool(resume.skills),
        "complete experience entries (title, company, dates)": all(
            e.title.strip() and e.company.strip() and e.dates.strip()
            for e in resume.experience
        )
        and bool(resume.experience),
    }
    failed = [name for name, ok in checks.items() if not ok]
    detail = "All structural checks passed." if not failed else "Missing: " + ", ".join(failed)
    return sum(checks.values()) / len(checks), detail


_DIMENSIONS = {
    "keywords": _score_keywords,
    "title": _score_title,
    "responsibilities": _score_responsibilities,
    "quantified_impact": _score_quantified_impact,
    "formatting": _score_formatting,
}


# --- Public API ------------------------------------------------------------


def score(resume: ParsedResume, jd: ParsedJD) -> ATSScore:
    # Warm the cache in a single batched call so the per-comparison best_match
    # calls below are all cache hits instead of one API round-trip each.
    texts = (
        _resume_chunks(resume)
        + [e.title for e in resume.experience[:2]]
        + jd.must_have_skills + jd.nice_to_have_skills + jd.keywords
        + jd.responsibilities
        + [f"{jd.role_title} {jd.seniority}".strip()]
    )
    texts = [t for t in texts if t and t.strip()]
    if texts:
        embed(texts)

    dimensions, keyword_matches = [], []
    for name, fn in _DIMENSIONS.items():
        result = fn(resume, jd)
        if name == "keywords":
            raw, detail, keyword_matches = result
        else:
            raw, detail = result
        weight = config.WEIGHTS[name]
        dimensions.append(
            DimensionScore(
                name=name, raw=raw, weight=weight,
                weighted=raw * weight * 100, detail=detail,
            )
        )

    return ATSScore(
        total=round(sum(d.weighted for d in dimensions), 1),
        dimensions=dimensions,
        matched_keywords=[m for m in keyword_matches if m.match_type != "none"],
        missing_keywords=[m for m in keyword_matches if m.match_type == "none"],
    )


def ats_scorer_node(state) -> dict:
    """Runs twice in the graph; the state decides which resume it is scoring."""
    jd = state.get("parsed_jd")
    tailored = state.get("tailored_resume")
    validation = state.get("validation")
    if tailored is not None and validation is not None and validation.passed:
        return {"post_score": score(tailored, jd)}
    return {"pre_score": score(state.get("parsed_resume"), jd)}
