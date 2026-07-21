"""Text extraction from resume files (.pdf, .docx, .txt/.md)."""

import os
import re
import tempfile
from pathlib import Path

SUPPORTED = (".pdf", ".docx", ".txt", ".md")


def extract_text(path: str | Path) -> str:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        text = _from_pdf(path)
    elif ext == ".docx":
        text = _from_docx(path)
    elif ext in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(
            f"Unsupported file type {ext!r}. Supported types: {', '.join(SUPPORTED)}"
        )
    return _normalize(text)


def extract_text_from_bytes(data: bytes, filename: str) -> str:
    """Streamlit hands uploads over as bytes, so round-trip through a temp file."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(
            f"Unsupported file type {suffix!r}. Supported types: {', '.join(SUPPORTED)}"
        )
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        return extract_text(tmp.name)
    finally:
        os.unlink(tmp.name)


def _from_pdf(path: Path) -> str:
    # pdfplumber has better layout handling, but chokes on some malformed PDFs;
    # pypdf is more permissive, so it gets a second shot before we give up.
    text = ""
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception:
        pass

    if not text.strip():
        try:
            from pypdf import PdfReader

            text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        except Exception:
            text = ""

    if not text.strip():
        # Returning "" here would let the LLM confidently parse an empty resume.
        raise ValueError(
            f"No text could be extracted from {path.name}. It appears to be an "
            f"image-only/scanned PDF and needs OCR before it can be parsed."
        )
    return text


def _from_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs]
    # Many real resumes lay content out in tables; body order isn't exposed by
    # python-docx without XML walking, so tables land after the paragraphs.
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            parts.append(" | ".join(c for c in cells if c))
    return "\n".join(parts)


def _normalize(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
