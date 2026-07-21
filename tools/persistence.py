"""Persist the base resume to disk so it survives across Streamlit sessions."""

import json
from pathlib import Path

import config
from graph.state import ParsedResume


def save_base_resume(resume: ParsedResume, original: bytes = None, filename: str = "") -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.BASE_RESUME_JSON.write_text(resume.model_dump_json(indent=2))
    if original is not None and filename:
        suffix = Path(filename).suffix
        config.BASE_RESUME_ORIGINAL.with_suffix(suffix).write_bytes(original)


def load_base_resume() -> ParsedResume | None:
    if not config.BASE_RESUME_JSON.exists():
        return None
    try:
        return ParsedResume.model_validate_json(config.BASE_RESUME_JSON.read_text())
    except Exception:
        # A corrupt or schema-drifted file should prompt a re-upload, not crash
        # the app on startup.
        return None


def clear_base_resume() -> None:
    config.BASE_RESUME_JSON.unlink(missing_ok=True)
    for f in config.DATA_DIR.glob("base_resume_original.*"):
        f.unlink(missing_ok=True)
