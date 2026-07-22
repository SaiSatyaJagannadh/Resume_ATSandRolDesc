"""Link recovery and the GitHub client. No network: _get is stubbed."""

import pytest

from tools import github
from tools.extract_text import _append_links


# --- username discovery ----------------------------------------------------


def test_finds_username_in_a_links_block():
    text = "Links:\nhttps://www.linkedin.com/in/x/\nhttps://github.com/SaiSatyaJagannadh"
    assert github.find_username(text) == "SaiSatyaJagannadh"


def test_trailing_slash_and_repo_path_still_yield_the_user():
    assert github.find_username("https://github.com/octocat/") == "octocat"
    assert github.find_username("https://github.com/octocat/some-repo") == "octocat"


def test_github_io_portfolio_is_not_a_username():
    """The portfolio URL sits next to the GitHub one on this resume."""
    assert github.find_username("https://saisatyajagannadh.github.io/Portfolio/") == ""


def test_reserved_paths_are_skipped():
    text = "https://github.com/features/actions and https://github.com/realuser"
    assert github.find_username(text) == "realuser"


def test_no_link_returns_empty():
    assert github.find_username("no links here") == ""
    assert github.find_username("") == ""


# --- link block ------------------------------------------------------------


def test_append_links_dedupes_and_preserves_order():
    out = _append_links("resume text", [
        "https://b.com", "https://a.com", "https://b.com",
    ])
    assert out.index("https://b.com") < out.index("https://a.com")
    assert out.count("https://b.com") == 1


def test_append_links_skips_non_http_and_already_present():
    out = _append_links("see https://a.com", ["mailto:x@y.com", "https://a.com"])
    assert out == "see https://a.com"  # nothing new -> unchanged


def test_append_links_noop_returns_text_unchanged():
    assert _append_links("body", []) == "body"


# --- repo fetching ---------------------------------------------------------


def _repo(name, **kw):
    base = {"name": name, "full_name": f"u/{name}", "size": 10,
            "fork": False, "archived": False, "description": "", "language": "Python",
            "topics": [], "html_url": f"https://github.com/u/{name}"}
    return base | kw


@pytest.fixture
def stub_get(monkeypatch):
    calls = {}

    def fake(url):
        calls["last"] = url
        return calls.get("payload")

    monkeypatch.setattr(github, "_get", fake)
    return calls


def test_fetch_repos_drops_forks_archived_and_empty(stub_get):
    stub_get["payload"] = [
        _repo("keep"),
        _repo("a-fork", fork=True),
        _repo("stale", archived=True),
        _repo("empty", size=0),
    ]
    assert [r["name"] for r in github.fetch_repos("u")] == ["keep"]


def test_fetch_repos_survives_failure_and_bad_shapes(stub_get):
    stub_get["payload"] = None          # rate limited / offline
    assert github.fetch_repos("u") == []
    stub_get["payload"] = {"message": "Not Found"}
    assert github.fetch_repos("u") == []
    assert github.fetch_repos("") == []  # no username, no call


def test_top_level_names_returns_entry_names(stub_get):
    stub_get["payload"] = [{"name": "PySpark & SQL operations"}, {"name": "README.md"}]
    assert github.top_level_names("u/r") == ["PySpark & SQL operations", "README.md"]
    stub_get["payload"] = None
    assert github.top_level_names("u/r") == []


def test_repo_profile_includes_dirs_and_skips_empty_fields():
    profile = github.repo_profile(
        _repo("F1_Race_Databricks", description=None, topics=["delta-lake"]),
        ["PySpark & SQL operations", "Delta Lake format"],
    )
    assert "F1 Race Databricks" in profile      # separators normalized for embedding
    assert "delta-lake" in profile
    assert "PySpark & SQL operations" in profile
    assert "None" not in profile
