# Resume ATS Optimizer

Store your master resume once. Paste any job description. Get a before → after
ATS score, a gap report, an explanation of every edit, and a downloadable
tailored `.docx`.

The tailoring agent rephrases, reframes, reorders, and resurfaces your real
experience. It **never** invents employers, titles, dates, degrees,
certifications, metrics, or skills — a validation step checks the output
against your base resume and blocks the run if anything was fabricated.

## Architecture

[LangGraph](https://langchain-ai.github.io/langgraph/) orchestrates a stateful
agent graph. Shared state (`graph/state.py`) carries the raw and parsed resume
and JD, the pre- and post-scores, the gaps, and the edit log.

```
resume_parser → jd_parser → ats_scorer (pre) → gap_analysis → resume_tailor
                                                                    ↓
                          ┌──── retry (max 2) ──── truthfulness_validator
                          │                               ↓ pass
                          │                       ats_scorer (post)
                          │                               ↓
                          └──── optimize ──────────── score_gate
                                                              ↓ target met / ceiling
                                                        resume_renderer → END
```

| Node | Type | Does |
| --- | --- | --- |
| `resume_parser` | tool + LLM | File → text → structured sections |
| `jd_parser` | LLM | JD → skills, keywords, seniority, responsibilities |
| `ats_scorer` | deterministic | Score + per-dimension breakdown (no LLM) |
| `gap_analysis` | LLM | What the resume is missing or weak on |
| `resume_tailor` | LLM | Rewrites under the no-fabrication guardrail |
| `truthfulness_validator` | deterministic + LLM | Blocks fabricated entities |
| `score_gate` | deterministic | Drives toward the target score, or stops honestly |
| `resume_renderer` | tool | ATS-clean `.docx` |

If the validator finds a fabrication it routes back to the tailor with specific
feedback, up to 2 retries, then hard-stops rather than shipping a resume with
invented details.

## The 85% target, and why it isn't a guarantee

`score_gate` aims for `TARGET_ATS_SCORE` (default 85, set in `config.py`). When
the tailored resume lands short, it re-tailors with score-derived feedback —
which keywords are missing, which dimensions are losing the most points — for up
to `MAX_OPTIMIZE_ROUNDS` extra passes.

**It will not always reach 85, by design.** Keyword coverage is 45% of the
score. If the candidate genuinely lacks the job's must-have skills, the only way
to close that gap is to claim skills they don't have — which is precisely what
the guardrail exists to prevent. A loop that guaranteed 85 would be a loop that
eventually fabricates.

So the gate stops and explains itself when:

- **Must-have coverage is below `MUST_HAVE_FLOOR`** (default 50%). The missing
  points are unreachable honestly; the user is told which skills are missing,
  because that's genuinely useful signal.
- **Rounds are exhausted**, or **a pass moved the score less than 0.5 points**.

Two safeguards worth knowing:

- Feedback to the tailor **excludes** any gap flagged `unsupported_by_resume`.
  The optimizer is never allowed to ask for a fabrication.
- The gate tracks the **best** result, not the latest. A re-tailoring pass can
  score worse than the one before it, and the rendered `.docx` is always the
  highest-scoring truthful version produced.

A truthful resume at 78 beats a fabricated one at 92 that collapses in an
interview.

## ATS scoring

A weighted 0–100 score. Weights live in `config.py` and are tunable.

| Dimension | Weight | Measures |
| --- | ---: | --- |
| Keyword & skills coverage | 45% | JD must-haves + keywords present in the resume. Exact match = full credit; semantic match ≥ threshold = partial. Must-haves count double. |
| Responsibility match | 20% | Semantic overlap between JD responsibilities and resume bullets |
| Role/title alignment | 15% | JD title/seniority vs. your recent titles |
| Quantified impact | 10% | Share of bullets containing metrics |
| ATS formatting | 10% | Single column, standard headings, parseable contact info, no tables/images |

**This is a heuristic estimate.** Real ATS platforms (Workday, Greenhouse,
Taleo, iCIMS) each score differently and none publish their algorithm. What
holds universally is that keyword coverage and clean, single-column formatting
are what get a resume parsed and surfaced — those are what this optimizes.

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then add your key(s)
streamlit run app.py
```

### Keys

`LLM_PROVIDER` selects `anthropic` (default), `openai`, or `google` — switching
is one line in `.env`, no code changes. Only the selected provider's key is
required.

**`OPENAI_API_KEY` is required regardless of provider.** The ATS scorer uses
OpenAI embeddings for semantic keyword matching. Anthropic has no embeddings
API, and a local `sentence-transformers` model pulls ~800MB of torch — which
does not fit in Streamlit Community Cloud's memory limit. Embeddings are cached
on disk, so cost is a fraction of a cent per analysis.

If `LLM_PROVIDER=openai`, one key covers both.

The app fails fast with a clear message if a required key is missing.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), connect the repo and set
   `app.py` as the entry point.
3. In **Settings → Secrets**, mirror your `.env` keys:
   ```toml
   LLM_PROVIDER = "anthropic"
   ANTHROPIC_API_KEY = "sk-ant-..."
   OPENAI_API_KEY = "sk-..."
   ```
4. Deploy.

Cloud-compatibility choices, and why:

- **No `sentence-transformers`.** Torch alone would exceed the memory limit.
  Embeddings go over HTTP instead, so cold start stays fast.
- **No `docx2pdf`.** It requires a desktop Word install and cannot work on
  Linux. Export is `.docx` only — which most ATS platforms prefer anyway; open
  it and "Save as PDF" if a posting demands PDF.
- **`./data` is ephemeral on Streamlit Cloud.** The base resume persists across
  browser sessions locally, but a cloud container restart clears it and you
  re-upload. Persisting across restarts would need external storage (S3, a DB).

## Tests

```bash
.venv/bin/pytest -q
```

Covers the text extractors, the docx round-trip, the ATS scorer's weighting and
word-boundary matching, and the deterministic fabrication detector. No network
calls — the embedding backend is stubbed.

## Layout

```
app.py                  Streamlit UI
config.py               Weights, thresholds, models, provider switch
conftest.py             Puts repo root on sys.path for tests
graph/
  state.py              Shared state + all Pydantic schemas
  build_graph.py        Graph assembly + conditional retry edge
  nodes/                One file per node
tools/
  extract_text.py       pdf/docx/txt → text
  embeddings.py         Embeddings + cosine similarity, disk-cached
  llm_factory.py        Provider switch
  render_docx.py        Structured resume → ATS-clean .docx
  persistence.py        Base resume on disk
data/                   Persisted base resume (gitignored)
tests/
```

## Known limits

- Scanned/image-only PDFs are rejected with a clear message; there is no OCR.
- `.docx` table extraction appends table text after paragraph text, so a resume
  built entirely inside a table may parse in a slightly odd order.
- The metric detector counts a bare `k`/`m`/`bn` token as a number, so an odd
  bullet can false-positive on quantified impact.
- The scorer is a heuristic, not a simulation of any specific ATS.
