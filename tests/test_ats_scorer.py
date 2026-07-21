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
