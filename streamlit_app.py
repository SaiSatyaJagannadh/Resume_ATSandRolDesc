"""Streamlit Community Cloud entry point — the app lives in app.py.

Cloud defaults the main-file path to streamlit_app.py; this re-exports it so
that default just works and app.py stays the single source of truth. Importing
app runs its module-level st.set_page_config first, before main()'s widgets.
"""

from app import main

main()
