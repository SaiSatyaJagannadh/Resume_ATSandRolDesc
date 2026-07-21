"""Puts the repo root on sys.path so tests can import `graph` and `tools`.

The project runs as a script (streamlit run app.py), not an installed package,
so there is no distribution metadata to make these importable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
