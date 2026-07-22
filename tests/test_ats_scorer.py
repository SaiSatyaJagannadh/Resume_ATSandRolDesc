"""ATS scorer tests. No network: tools.embeddings._embed_fn is always stubbed."""

import re

import pytest

import config
import tools.embeddings as emb
from graph.nodes.ats_scorer import score
from graph.state import (
    Contact,
    EducationEntry,
    ExperienceEntry,
    ParsedJD,
    ParsedResume,
)

DIM = 64


def _fake_vec(text: str) -> list[float]:
    """Deterministic bag-of-words vector: shared words -> high cosine, none -> 0."""
    vec = [0.0] * DIM
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[hash(word) % DIM] += 1.0
    return vec


@pytest.fixture(autouse=True)
def stub_embeddings(monkeypatch, tmp_path):
    monkeypatch.setattr(emb, "_embed_fn", lambda texts: [_fake_vec(t) for t in texts])
    monkeypatch.setattr(config, "EMBEDDING_CACHE", tmp_path / "cache.json")
    monkeypatch.setattr(emb, "_cache", None)


def make_resume(bullets=None, skills=("Python", "Django"), **kw):
    defaults = dict(
        contact=Contact(name="A", email="a@b.com", phone="555-123-4567"),
        summary="Backend engineer.",
        skills=list(skills),
        experience=[
            ExperienceEntry(
                company="Acme",
                title="Senior Backend Engineer",
                dates="2020 - Present",
                bullets=list(bullets if bullets is not None else ["Built Python services"]),
            )
        ],
        education=[EducationEntry(institution="Uni", degree="BS CS")],
    )
    defaults.update(kw)
    return ParsedResume(**defaults)


JD = ParsedJD(
    role_title="Senior Backend Engineer",
    seniority="Senior",
    must_have_skills=["Python", "Django"],
    nice_to_have_skills=["Kubernetes"],
    keywords=["PostgreSQL"],
    responsibilities=["Built Python services"],
    domain="fintech",
)


def dim(ats, name):
    return next(d for d in ats.dimensions if d.name == name)


def test_matching_resume_beats_unrelated_resume():
    good = make_resume(
        skills=["Python", "Django", "Kubernetes", "PostgreSQL"],
        bullets=["Built Python services on Kubernetes"],
    )
    bad = make_resume(
        skills=["Watercolor", "Ceramics"],
        bullets=["Taught pottery classes"],
        experience=[
            ExperienceEntry(company="Studio", title="Art Teacher", dates="2020 - Present",
                            bullets=["Taught pottery classes"])
        ],
    )
    assert score(good, JD).total > score(bad, JD).total + 25


def test_java_does_not_exact_match_javascript():
    """The subtle one: naive substring matching would score this as a hit."""
    jd = JD.model_copy(update={"must_have_skills": ["Java"], "nice_to_have_skills": [],
                               "keywords": []})
    resume = make_resume(skills=["JavaScript"], bullets=["Wrote JavaScript"])
    assert score(resume, jd).matched_keywords == []

    jd_py = jd.model_copy(update={"must_have_skills": ["Python"]})
    hit = score(make_resume(bullets=["Used Python 3 in production"]), jd_py)
    assert [m.match_type for m in hit.matched_keywords] == ["exact"]


def test_must_haves_outweigh_nice_to_haves():
    resume = make_resume(skills=["Python", "Django"], bullets=["Built things"])
    missing_must = ParsedJD(
        role_title="X", seniority="Senior", domain="d",
        must_have_skills=["Rust", "Elixir"], keywords=["Python", "Django"],
    )
    missing_nice = ParsedJD(
        role_title="X", seniority="Senior", domain="d",
        must_have_skills=["Python", "Django"], keywords=["Rust", "Elixir"],
    )
    assert dim(score(resume, missing_nice), "keywords").raw > dim(
        score(resume, missing_must), "keywords"
    ).raw


def test_quantified_impact():
    numeric = make_resume(bullets=["Cut latency 40%", "Saved $2M", "Grew users 3x"])
    assert dim(score(numeric, JD), "quantified_impact").raw == 1.0

    prose = make_resume(bullets=["Led the team", "Improved the codebase"])
    assert dim(score(prose, JD), "quantified_impact").raw == 0.0

    empty = make_resume(bullets=[])
    assert dim(score(empty, JD), "quantified_impact").raw == 0.0  # no ZeroDivisionError


def test_total_is_sum_of_weighted_and_in_range():
    for resume in (make_resume(skills=["Python", "Django", "Kubernetes", "PostgreSQL"]),
                   ParsedResume()):
        ats = score(resume, JD)
        assert ats.total == pytest.approx(sum(d.weighted for d in ats.dimensions), abs=0.05)
        assert 0.0 <= ats.total <= 100.0


def test_embedding_cache_dedupes_calls():
    calls = []

    def counting(texts):
        calls.append(list(texts))
        return [_fake_vec(t) for t in texts]

    emb._embed_fn = counting
    emb.embed(["hello world"])
    emb.embed(["hello world"])
    assert len(calls) == 1


# --- Vendor-qualifier keyword matching -------------------------------------
# Regression: a JD saying "Apache Airflow" against a resume saying "Airflow"
# scored as MISSING. Exact-phrase match failed and the semantic fallback
# diluted below threshold against a long bullet, understating the score on a
# skill the candidate genuinely has.


def _jd(**kw):
    from graph.state import ParsedJD
    base = dict(role_title="Data Engineer", seniority="Senior", must_have_skills=[],
                nice_to_have_skills=[], keywords=[], responsibilities=[], domain="data")
    base.update(kw)
    return ParsedJD(**base)


def _resume(skills, bullets=("Did work.",)):
    from graph.state import Contact, ExperienceEntry, ParsedResume
    return ParsedResume(
        contact=Contact(name="A", email="a@b.com", phone="5550100"),
        skills=list(skills),
        experience=[ExperienceEntry(company="C", title="Data Engineer",
                                    dates="2020-2024", bullets=list(bullets))],
    )


def test_vendor_qualifier_is_stripped_when_matching():
    from graph.nodes.ats_scorer import score
    s = score(_resume(["Airflow", "S3"]), _jd(must_have_skills=["Apache Airflow", "Amazon S3"]))
    missing = {k.keyword for k in s.missing_keywords}
    assert "Apache Airflow" not in missing, "resume has Airflow; must not score as missing"
    assert "Amazon S3" not in missing
    assert all(k.match_type == "exact" for k in s.matched_keywords)


def test_qualifier_stripping_does_not_invent_matches():
    """Stripping a qualifier must not make an absent skill look present."""
    from graph.nodes.ats_scorer import score
    s = score(_resume(["Python"]), _jd(must_have_skills=["Apache Kafka"]))
    assert "Apache Kafka" in {k.keyword for k in s.missing_keywords}


def test_bare_qualifier_word_is_not_stripped_to_nothing():
    from graph.nodes.ats_scorer import _keyword_variants
    assert _keyword_variants("aws") == ["aws"]
    assert _keyword_variants("apache airflow") == ["apache airflow", "airflow"]


# --- Cache poisoning -------------------------------------------------------
# Regression: a single cached vector of the wrong dimension (from a model
# switch, an interrupted write, or a shared test path) reached cosine_sim and
# raised a shape error, taking down the whole scorer mid-analysis.


def test_mismatched_cache_entries_are_dropped_not_fatal(tmp_path, monkeypatch):
    import json
    import tools.embeddings as emb

    cache_file = tmp_path / "cache.json"
    good = [0.1] * 1536
    cache_file.write_text(json.dumps({
        emb._key("alpha"): good,
        emb._key("beta"): good,
        emb._key("poison"): [1.0, 2.0],      # wrong dimension
        emb._key("junk"): "not-a-vector",     # wrong type entirely
    }))
    monkeypatch.setattr(emb.config, "EMBEDDING_CACHE", cache_file)
    monkeypatch.setattr(emb, "_cache", None)

    loaded = emb._load_cache()
    assert len(loaded) == 2, "bad entries should be dropped"
    assert all(len(v) == 1536 for v in loaded.values())


def test_cosine_sim_returns_zero_on_shape_mismatch():
    from tools.embeddings import cosine_sim
    assert cosine_sim([1.0] * 1536, [1.0, 2.0]) == 0.0


def test_single_token_keywords_never_match_semantically():
    """Tool names are exact-match only.

    Embeddings rank single tokens by vibe: "Go" scores 0.491 against "Git" and
    "CloudFormation" 0.508 against "AWS" — above genuine multi-word matches. No
    threshold separates them, so they are excluded from the semantic tier
    outright.
    """
    import config
    from graph.nodes import ats_scorer
    from graph.state import ExperienceEntry, ParsedJD, ParsedResume

    resume = ParsedResume(
        skills=["Git", "AWS"],
        experience=[ExperienceEntry(
            company="C", title="T", dates="2020",
            bullets=["Maintained automated data pipelines in the cloud."],
        )],
    )
    jd = ParsedJD(
        role_title="Engineer", seniority="Mid", domain="tech",
        must_have_skills=["Go", "CloudFormation", "ETL/ELT pipelines"],
    )

    # Force every semantic lookup to land well above the threshold; only the
    # multi-word keyword may take advantage of it.
    ats_scorer.best_match = lambda q, c: (c[0], 0.99)
    try:
        result = ats_scorer.score(resume, jd)
    finally:
        from tools.embeddings import best_match as real
        ats_scorer.best_match = real

    by_kw = {k.keyword: k for k in result.matched_keywords + result.missing_keywords}
    assert by_kw["Go"].match_type == "none"
    assert by_kw["CloudFormation"].match_type == "none"
    assert by_kw["ETL/ELT pipelines"].match_type == "semantic"
    assert by_kw["ETL/ELT pipelines"].score == config.SEMANTIC_MATCH_CREDIT
