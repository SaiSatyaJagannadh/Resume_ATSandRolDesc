"""Surface the candidate's own GitHub projects that match the job description.

This exists because a resume under-reports its author. The scorer will call
PySpark and Delta Lake missing while a public repo of theirs has directories
named "PySpark & SQL operations" and "Delta Lake format" — the experience is
real and verifiable, it just never made it onto one page.

Two properties keep that honest. Nothing here is generated: every field is
copied verbatim from the candidate's own public repo metadata, and each added
project carries its Source URL, so any claim is one click from being checked.
And there is no LLM in this path at all — embeddings rank, string templates
write — so there is nothing to embellish and no prompt to subvert. Turning those
verbatim facts into prose is the tailor's job, where the truthfulness validator
already polices the result.
"""

import config
from graph.state import ProjectEntry
from tools.embeddings import cosine_sim, embed
from tools.github import fetch_repos, find_username, repo_profile, top_level_names


def _norm_name(text: str) -> set[str]:
    """Lowercased alphanumeric tokens of a project or repo name."""
    return {
        t for t in "".join(c if c.isalnum() else " " for c in text.lower()).split() if t
    }


def _is_duplicate(repo_name: str, existing) -> bool:
    """Cheap token test: True when the names plainly describe the same project."""
    repo_tokens = _norm_name(repo_name)
    if not repo_tokens:
        return True
    for project in existing:
        other = _norm_name(project.name)
        if not other:
            continue
        overlap = len(repo_tokens & other) / len(repo_tokens | other)
        if overlap >= 0.5 or repo_tokens <= other or other <= repo_tokens:
            return True
    return False


def _readable(name: str) -> str:
    return name.replace("-", " ").replace("_", " ").strip()


def dedupe(repos, existing) -> list[dict]:
    """Drop repos the resume already describes, by name tokens then meaning.

    Token overlap alone is not enough, and the failure is not hypothetical:
    "Azure-Data-Factory-Project-on-Covid19" against the resume's own "Covid-19
    Data Analysis on Azure" shares only a third of its tokens and sailed
    through, putting the same project on the page twice. Measured on this
    resume, real duplicates sit at 0.69-0.73 cosine and unrelated pairs at
    0.40-0.48, so 0.60 separates them with room on both sides.
    """
    if not existing or not repos:
        return list(repos)

    survivors = [r for r in repos if not _is_duplicate(r.get("name", ""), existing)]
    if not survivors:
        return []

    project_names = [p.name for p in existing if p.name.strip()]
    if not project_names:
        return survivors

    # One batched call: the embedding cache makes the repeat lookups free.
    vectors = embed([_readable(r.get("name", "")) for r in survivors] + project_names)
    repo_vecs = vectors[: len(survivors)]
    project_vecs = vectors[len(survivors):]

    return [
        r for r, rv in zip(survivors, repo_vecs)
        if max(cosine_sim(rv, pv) for pv in project_vecs) < config.GITHUB_DEDUP_THRESHOLD
    ]


def _to_project(repo: dict, dirs) -> ProjectEntry:
    """Verbatim repo facts as a ProjectEntry. No paraphrase, no invented outcome."""
    bullets = []
    if repo.get("language"):
        bullets.append(f"Primary language: {repo['language']}.")
    if repo.get("topics"):
        bullets.append(f"Topics: {', '.join(repo['topics'])}.")
    if dirs:
        bullets.append(f"Components: {', '.join(dirs)}.")
    if repo.get("html_url"):
        # Provenance: the reason this content is allowed to widen the
        # validator's fact set at all.
        bullets.append(f"Source: {repo['html_url']}")
    return ProjectEntry(
        name=repo.get("name", ""),
        description=repo.get("description") or "",
        bullets=bullets,
    )


def _rank(query_vec, repos, profiles):
    """(similarity, repo) pairs, best first."""
    vectors = embed(profiles)
    scored = [(cosine_sim(query_vec, v), r) for v, r in zip(vectors, repos)]
    return sorted(scored, key=lambda pair: pair[0], reverse=True)


def select_projects(jd, repos) -> list[tuple[dict, list]]:
    """Rank repos against the JD and return the best few, with their file trees.

    Two stages because listing repos costs one API call while reading a repo's
    file tree costs one call each: ranking on cheap metadata first means only a
    shortlist pays for the directory names, which is what keeps a 30-repo
    account inside the unauthenticated hourly allowance.
    """
    if not repos:
        return []

    query = " ".join(
        [jd.role_title, jd.seniority, *jd.must_have_skills, *jd.keywords]
    ).strip()
    if not query:
        return []
    query_vec = embed([query])[0]

    shortlist = _rank(query_vec, repos, [repo_profile(r) for r in repos])
    shortlist = [r for _, r in shortlist[: config.GITHUB_SHORTLIST]]

    # Second pass: the directory names carry the real signal.
    trees = {r["full_name"]: top_level_names(r["full_name"]) for r in shortlist}
    reranked = _rank(
        query_vec, shortlist, [repo_profile(r, trees[r["full_name"]]) for r in shortlist]
    )

    return [
        (r, trees[r["full_name"]])
        for sim, r in reranked[: config.GITHUB_MAX_PROJECTS]
        if sim >= config.GITHUB_MATCH_THRESHOLD
    ]


def github_enrich_node(state) -> dict:
    """Append JD-relevant GitHub projects to the parsed resume.

    Enrichment is a bonus, never a dependency: any failure returns {}, which
    LangGraph reads as "no state change", so the pipeline degrades to exactly
    its pre-GitHub behaviour rather than taking the run down with it.
    """
    try:
        jd = state.get("parsed_jd")
        resume = state.get("parsed_resume")
        if jd is None or resume is None:
            return {}

        username = (
            state.get("github_username")
            or find_username(state.get("raw_resume_text", ""))
            or config.GITHUB_USERNAME
        ).strip()
        if not username:
            return {}

        # Dedup before ranking, so a project the resume already covers cannot
        # consume one of the few slots.
        candidates = dedupe(fetch_repos(username), resume.projects)
        picks = select_projects(jd, candidates)
        if not picks:
            return {}

        enriched = resume.model_copy(deep=True)
        enriched.projects = list(enriched.projects) + [
            _to_project(r, dirs) for r, dirs in picks
        ]
        return {
            "parsed_resume": enriched,
            "github_projects_added": [r.get("html_url", "") for r, _ in picks],
        }
    except Exception:
        return {}
