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
# match. Below it, no credit.
SEMANTIC_MATCH_THRESHOLD = 0.72

# A semantic match earns this fraction of the credit an exact match earns.
SEMANTIC_MATCH_CREDIT = 0.6

# --- Tailoring loop --------------------------------------------------------

MAX_TAILOR_RETRIES = 2

# --- Persistence -----------------------------------------------------------

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
BASE_RESUME_JSON = DATA_DIR / "base_resume.json"
BASE_RESUME_ORIGINAL = DATA_DIR / "base_resume_original"  # extension appended
EMBEDDING_CACHE = DATA_DIR / "embedding_cache.json"


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
