"""Central configuration: provider switch, model names, scoring weights, thresholds."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Providers -------------------------------------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()

MODELS = {
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o"),
    "google": os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
}

# Which env var each provider needs. Used for fail-fast validation.
PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}

# Embeddings always come from OpenAI: Anthropic has no embeddings API, and a
# local sentence-transformers model pulls ~800MB of torch that will not fit in
# Streamlit Community Cloud's memory limit.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_KEY = "OPENAI_API_KEY"

# --- ATS scoring weights (must sum to 1.0) ---------------------------------

WEIGHTS = {
    "keywords": 0.45,
    "title": 0.15,
    "responsibilities": 0.20,
    "quantified_impact": 0.10,
    "formatting": 0.10,
}

# Must-have skills count this much more than nice-to-haves in the keyword
# dimension.
MUST_HAVE_MULTIPLIER = 2.0

# Cosine similarity at or above this counts as a semantic (partial-credit)
# match. Below it, no credit. Applies to multi-word keywords only — see
# _score_keywords for why single tokens are excluded entirely.
#
# 0.50, not 0.72: text-embedding-3-small returns a compressed range, and 0.72
# was never reached by anything on a real resume, so the semantic tier was dead
# code and every near-miss scored zero. Calibrated against 36 hand-labelled
# terms from three live postings: at 0.50 it credits 5 genuine matches
# ("ETL/ELT pipelines" -> "automated data pipelines") with no false positives;
# the first wrong answer appears at 0.436.
SEMANTIC_MATCH_THRESHOLD = 0.50

# A semantic match earns this fraction of the credit an exact match earns.
SEMANTIC_MATCH_CREDIT = 0.6

# --- Tailoring loop --------------------------------------------------------

# Fabrication retries, per optimization round.
MAX_TAILOR_RETRIES = 2

# --- Score optimization loop -----------------------------------------------

# The score the tailored resume aims to clear.
#
# 75, not 85, because 85 was never calibrated against what this scorer can
# actually emit. Measured on a real resume:
#   - JD copied verbatim from the resume  -> 91.7  (degenerate; literal identity)
#   - ideal candidate, employer's wording -> 70.5  (every requirement genuinely met)
# Two dimensions score raw cosine similarity, and paraphrase costs ~0.35 of it
# however well-qualified the candidate is: `title` cannot exceed ~0.63 without
# an almost identical job title, and the tailor is rightly forbidden from
# editing titles. So 85 was reachable only by copying the posting's wording
# wholesale — the exact behaviour the truthfulness guardrail exists to prevent.
# The result was that every honest run reported failure.
TARGET_ATS_SCORE = float(os.getenv("TARGET_ATS_SCORE", "75"))

# Extra re-tailoring passes allowed when the target is missed. Each round costs
# a full tailor + validate + score cycle, so this is deliberately small.
MAX_OPTIMIZE_ROUNDS = int(os.getenv("MAX_OPTIMIZE_ROUNDS", "2"))

# Below this share of must-have keywords present, the target is treated as
# unreachable honestly: the remaining points live almost entirely in the 45%
# keyword dimension, and the only way to close that gap is to claim skills the
# candidate does not have. Stopping early here is the guardrail doing its job,
# not a failure to optimize.
MUST_HAVE_FLOOR = 0.5

# --- GitHub project enrichment ---------------------------------------------

# Normally discovered from a github.com link recovered out of the resume file;
# these are the fallbacks for when the resume carries no such link.
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "")
# Optional. Lifts the API allowance from 60 requests/hour to 5000.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# How many repos are appended to the resume, and how many get a second look.
# Ranking happens in two stages because listing repos costs one call while
# reading a repo's file tree costs one call *each* — 37 repos would blow the
# unauthenticated hourly allowance on a single run.
GITHUB_MAX_PROJECTS = int(os.getenv("GITHUB_MAX_PROJECTS", "3"))
GITHUB_SHORTLIST = int(os.getenv("GITHUB_SHORTLIST", "8"))

# Minimum JD-to-repo similarity for a project to be worth adding. Deliberately
# NOT SEMANTIC_MATCH_THRESHOLD (0.50): that was calibrated for a short keyword
# against a resume fragment, whereas this compares a whole job description
# against a whole repo profile — a different distribution entirely. Its only job
# is to stop an account with dozens of repos from always contributing the full
# quota regardless of relevance.
GITHUB_MATCH_THRESHOLD = float(os.getenv("GITHUB_MATCH_THRESHOLD", "0.30"))

# Above this similarity, a repo is treated as the project the resume already
# describes. Name-token overlap alone let "Azure-Data-Factory-Project-on-Covid19"
# through against "Covid-19 Data Analysis on Azure" — a third of its tokens, and
# the same project listed twice. Measured here, real duplicates sit at 0.69-0.73
# and unrelated pairs at 0.40-0.48.
GITHUB_DEDUP_THRESHOLD = float(os.getenv("GITHUB_DEDUP_THRESHOLD", "0.60"))

GITHUB_CACHE_TTL_HOURS = 24

# --- Persistence -----------------------------------------------------------

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
BASE_RESUME_JSON = DATA_DIR / "base_resume.json"
BASE_RESUME_ORIGINAL = DATA_DIR / "base_resume_original"  # extension appended
EMBEDDING_CACHE = DATA_DIR / "embedding_cache.json"
GITHUB_CACHE = DATA_DIR / "github_cache.json"


def require_key(provider: str = None) -> None:
    """Raise with a clear message if the selected provider's key is missing."""
    provider = provider or LLM_PROVIDER
    if provider not in PROVIDER_KEYS:
        raise ValueError(
            f"Unknown LLM_PROVIDER {provider!r}. "
            f"Expected one of: {', '.join(PROVIDER_KEYS)}"
        )
    key = PROVIDER_KEYS[provider]
    if not os.getenv(key):
        raise RuntimeError(
            f"LLM_PROVIDER is {provider!r} but {key} is not set. "
            f"Copy .env.example to .env and add your key."
        )


def require_embedding_key() -> None:
    if not os.getenv(EMBEDDING_KEY):
        raise RuntimeError(
            f"{EMBEDDING_KEY} is not set. It is required for semantic matching "
            f"in the ATS scorer (embeddings), independently of LLM_PROVIDER."
        )
