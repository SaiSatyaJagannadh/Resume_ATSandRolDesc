"""Terminal node: write the tailored resume to an ATS-clean .docx."""

import tempfile
from pathlib import Path

from tools.render_docx import render_docx


def resume_renderer_node(state) -> dict:
    # best_resume, not tailored_resume: an optimization round can score worse
    # than an earlier one, and the user should get the best truthful version
    # produced, not whichever happened to run last.
    resume = (
        state.get("best_resume")
        or state.get("tailored_resume")
        or state.get("parsed_resume")
    )
    if resume is None:
        return {"error": "Nothing to render: no resume in state."}

    name = (resume.contact.name or "resume").strip().replace(" ", "_")
    # Rendered into a temp dir rather than ./data: this is a per-run artifact
    # streamed straight to the user's download, not persisted state.
    out = Path(tempfile.mkdtemp(prefix="tailored_")) / f"{name}_tailored.docx"
    return {"docx_path": str(render_docx(resume, out))}
