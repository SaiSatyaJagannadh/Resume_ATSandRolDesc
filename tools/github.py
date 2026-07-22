"""Read-only GitHub client for surfacing a candidate's own public projects.

Everything here is stdlib: `requests` is not a dependency of this project and a
handful of GETs does not justify adding one. Every call is best-effort — a
failure returns empty rather than raising, because enrichment is a bonus and must
never take down the pipeline.
"""

import json
import re
import time
import urllib.request

import config

API = "https://api.github.com"

# github.com/<user>. Deliberately anchored on the api-bearing host so a
# `*.github.io` portfolio URL — which sits right next to it on a resume — does
# not get mistaken for a username.
_USER_RE = re.compile(r"github\.com/([A-Za-z0-9][A-Za-z0-9-]*)", re.I)

# Paths that look like a username but are GitHub's own routes.
_RESERVED = {
    "orgs", "features", "topics", "sponsors", "about", "pricing", "enterprise",
    "collections", "events", "explore", "marketplace", "settings", "login",
    "signup", "apps", "search", "trending", "readme",
}

_cache: dict | None = None


def find_username(text: str) -> str:
    """First real github.com username in `text`, or ''."""
    for match in _USER_RE.finditer(text or ""):
        user = match.group(1)
        if user.lower() not in _RESERVED:
            return user
    return ""


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(config.GITHUB_CACHE.read_text())
        except Exception:
            # Missing or corrupt cache is never fatal: start empty and rewrite.
            _cache = {}
    return _cache


def _get(url: str):
    """Cached GET returning parsed JSON, or None on any failure.

    None is the single failure signal: no network, a 403 from the 60/hour
    unauthenticated rate limit, a 404, or unparseable JSON all look the same to
    callers, and all mean "carry on without enrichment".
    """
    cache = _load_cache()
    hit = cache.get(url)
    if hit and time.time() - hit.get("t", 0) < config.GITHUB_CACHE_TTL_HOURS * 3600:
        return hit.get("v")

    headers = {
        "Accept": "application/vnd.github+json",
        # GitHub rejects API requests that don't identify themselves.
        "User-Agent": "resume-ats-optimizer",
    }
    if config.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"

    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except Exception:
        return None

    cache[url] = {"t": time.time(), "v": payload}
    try:
        config.GITHUB_CACHE.parent.mkdir(parents=True, exist_ok=True)
        config.GITHUB_CACHE.write_text(json.dumps(cache))
    except Exception:
        pass  # An unwritable cache costs a re-fetch, nothing more.
    return payload


def fetch_repos(username: str) -> list[dict]:
    """The user's own public repos, newest first.

    Forks, archived repos and empty repos are dropped: none of them evidence
    work the candidate can be questioned about in an interview.
    """
    if not username:
        return []
    data = _get(f"{API}/users/{username}/repos?per_page=100&sort=updated&type=owner")
    if not isinstance(data, list):
        return []
    return [
        r for r in data
        if isinstance(r, dict)
        and not r.get("fork")
        and not r.get("archived")
        and r.get("size")
    ]


def top_level_names(full_name: str) -> list[str]:
    """Top-level file and directory names of a repo.

    Worth a call because directory names carry the real signal: the repo that
    proves PySpark experience says so in a folder called "PySpark & SQL
    operations", not in its description.
    """
    data = _get(f"{API}/repos/{full_name}/contents")
    if not isinstance(data, list):
        return []
    return [e["name"] for e in data if isinstance(e, dict) and e.get("name")]


def repo_profile(repo: dict, dirs=()) -> str:
    """Text blob describing a repo, for embedding against a job description."""
    parts = [
        repo.get("name", "").replace("-", " ").replace("_", " "),
        repo.get("description") or "",
        repo.get("language") or "",
        " ".join(repo.get("topics") or []),
        " ".join(dirs),
    ]
    return " ".join(p for p in parts if p).strip()
