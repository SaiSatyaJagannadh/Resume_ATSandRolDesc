"""Streamlit Community Cloud entry point — the app lives in app.py.

Cloud defaults the main-file path to streamlit_app.py; this re-exports app.main
so that default just works and app.py stays the single source of truth.

It also bridges the Streamlit secrets manager into environment variables before
app/config load, so config.py's os.getenv(...) calls find the keys you set in
the Cloud "Secrets" panel. No secret values live in this file — they are read
at runtime from the secrets manager. Locally (no secrets file) this is a no-op
and the app falls back to your .env, exactly as before.
"""

import os

try:
    import streamlit as st

    for key, value in st.secrets.items():
        # Only flat string entries map cleanly to env vars; setdefault never
        # clobbers a value the platform already exported.
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:
    # No secrets.toml (local dev) or secrets unavailable — .env / real env
    # vars are used instead. Never fatal.
    pass

from app import main

main()
