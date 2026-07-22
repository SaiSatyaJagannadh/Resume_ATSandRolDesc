"""Ranking, dedup and the degrade-to-nothing contract. No network."""

import pytest

import config
from graph.nodes import github_enrich as ge
from graph.nodes.truthfulness_validator import find_fabrications
from graph.state import ParsedJD, ParsedResume, ProjectEntry


def _repo(name, desc="", lang="Python", topics=(), **kw):
    return {
        "name": name, "full_name": f"u/{name}", "description": desc,
        "language": lang, "topics": list(topics), "size": 10,
        "html_url": f"https://github.com/u/{name}", "fork": False, "archived": False,
    } | kw


JD = ParsedJD(
    role_title="Azure Data Engineer", seniority="Mid", domain="tech",
    must_have_skills=["PySpark", "Azure Databricks"], keywords=["Delta Lake", "ETL"],
)


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Similarity by keyword overlap — deterministic, no API, no cache."""
    vocab = ["pyspark", "databricks", "delta", "etl", "azure", "movie", "weather"]

    def embed(texts):
        out = []
        for t in texts:
            low = t.lower()
            out.append([1.0 if w in low else 0.0 for w in vocab])
        return out

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    monkeypatch.setattr(ge, "embed", embed)
    monkeypatch.setattr(ge, "cosine_sim", cosine)


# --- dedup -----------------------------------------------------------------


def test_repo_matching_an_existing_project_is_skipped():
    existing = [ProjectEntry(name="AI Healthcare Chatbot Website for Medical Queries")]
    assert ge._is_duplicate("AI-Healthcare-Chatbot", existing)
    assert not ge._is_duplicate("F1_Race_Databricks", existing)


def test_dedup_handles_empty_names():
    assert ge._is_duplicate("", [])
    assert not ge._is_duplicate("Repo", [ProjectEntry(name="")])


# --- verbatim conversion ---------------------------------------------------


def test_project_fields_are_copied_not_written():
    repo = _repo("F1_Race_Databricks", desc="Formula 1 pipeline", topics=["delta-lake"])
    p = ge._to_project(repo, ["PySpark & SQL operations"])
    assert p.name == "F1_Race_Databricks"
    assert p.description == "Formula 1 pipeline"
    assert "Primary language: Python." in p.bullets
    assert "Topics: delta-lake." in p.bullets
    assert "Components: PySpark & SQL operations." in p.bullets
    assert p.bullets[-1] == "Source: https://github.com/u/F1_Race_Databricks"


def test_missing_fields_produce_no_empty_bullets():
    p = ge._to_project(_repo("Bare", desc=None, lang=None), [])
    assert p.description == ""
    assert p.bullets == ["Source: https://github.com/u/Bare"]


# --- ranking ---------------------------------------------------------------


def test_relevant_repo_ranks_first_and_irrelevant_is_dropped(fake_embeddings, monkeypatch):
    repos = [
        _repo("MovieShopMvc", desc="movie shop"),
        _repo("F1_Race_Databricks", topics=["delta"]),
        _repo("node5-weather-website", desc="weather"),
    ]
    monkeypatch.setattr(ge, "top_level_names",
                        lambda fn: ["PySpark & SQL operations"] if "F1" in fn else ["src"])
    picks = ge.select_projects(JD, repos)
    assert picks, "the Databricks repo should clear the threshold"
    assert picks[0][0]["name"] == "F1_Race_Databricks"
    assert all(r["name"] not in ("MovieShopMvc", "node5-weather-website") for r, _ in picks)


def test_selection_is_capped(fake_embeddings, monkeypatch):
    repos = [_repo(f"pyspark-etl-{i}", topics=["delta"]) for i in range(10)]
    monkeypatch.setattr(ge, "top_level_names", lambda fn: ["pyspark", "etl"])
    assert len(ge.select_projects(JD, repos)) <= config.GITHUB_MAX_PROJECTS


def test_no_repos_or_empty_jd_selects_nothing(fake_embeddings):
    assert ge.select_projects(JD, []) == []
    blank = ParsedJD(role_title="", seniority="", domain="")
    assert ge.select_projects(blank, [_repo("x")]) == []


# --- the degrade contract --------------------------------------------------


@pytest.mark.parametrize("break_it", ["raises", "none", "no_user", "no_repos"])
def test_node_never_fails_the_pipeline(monkeypatch, break_it, fake_embeddings):
    resume = ParsedResume()
    state = {"parsed_jd": JD, "parsed_resume": resume,
             "raw_resume_text": "https://github.com/u", "github_username": ""}

    if break_it == "raises":
        monkeypatch.setattr(ge, "fetch_repos", lambda u: (_ for _ in ()).throw(RuntimeError))
    elif break_it == "none":
        monkeypatch.setattr(ge, "fetch_repos", lambda u: [])
    elif break_it == "no_user":
        state["raw_resume_text"] = "no links"
        monkeypatch.setattr(config, "GITHUB_USERNAME", "")
    elif break_it == "no_repos":
        monkeypatch.setattr(ge, "fetch_repos", lambda u: [])

    assert ge.github_enrich_node(state) == {}


def test_missing_state_keys_are_tolerated():
    assert ge.github_enrich_node({}) == {}


# --- the contract that matters: enriched projects must survive the validator


def test_enriched_projects_do_not_read_as_fabrication(fake_embeddings, monkeypatch):
    """Digits in repo names must not trip the number check."""
    base = ParsedResume(projects=[])
    repo = _repo("F1_Race_Databricks", desc="Formula 1 data pipeline", topics=["delta"])
    enriched = base.model_copy(deep=True)
    enriched.projects = [ge._to_project(repo, ["PySpark & SQL operations"])]

    # The tailor is allowed to reword these; nothing new is introduced.
    tailored = enriched.model_copy(deep=True)
    tailored.projects[0].bullets[0] = "Built with Python."

    assert find_fabrications(enriched, tailored) == []
