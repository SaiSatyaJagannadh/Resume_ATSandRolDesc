# Claude Code Build Prompt — Agentic Resume ATS Optimizer


---

You are building a production-quality **agentic resume optimization app**. Build it end to end, in a fresh repo, and make it runnable and deployable. Do not stub things out with `pass` — implement each piece.

## 1. What the app does

A user stores a **base (master) resume** once. Then, for any job, they paste a **job description (JD)** into the UI. The app runs an agent pipeline that:

1. Parses the JD into structured requirements (hard skills, soft skills, keywords, seniority, responsibilities).
2. Parses the base resume into structured sections.
3. Computes an **ATS match score (0–100)** with a per-dimension breakdown, before any editing.
4. Does a **gap analysis** — which JD keywords/skills/requirements are missing or weak in the resume.
5. **Rewrites the resume** to maximize the ATS score for that specific JD: reorders sections, rewrites bullets to surface relevant impact, injects missing keywords **that the candidate legitimately has**, and fixes ATS-unfriendly formatting.
6. Re-scores the tailored resume and shows a **before → after** comparison.
7. Lets the user **download the tailored resume** as a clean, ATS-parseable `.docx` (and optionally `.pdf`).

This is meant to be used for real job applications, so quality and correctness matter.

## 2. Hard guardrail (non-negotiable)

The tailoring agent may **rephrase, reframe, reorder, and surface** existing experience, and may add keywords **only where the candidate's real experience supports them**. It must **never fabricate** employers, job titles, dates, degrees, certifications, metrics, or skills the candidate does not have. If the JD requires something absent from the base resume, report it in the gap analysis as a genuine gap — do not invent it. Bake this rule into the tailoring agent's system prompt and add a validation step that flags any newly introduced company/title/date/credential not present in the base resume.

## 3. Architecture

Use **LangGraph** to orchestrate a stateful agent graph. Use **LangChain** for LLM + tool plumbing. Make the LLM provider configurable via env (`OpenAI`, `Anthropic`, or `Gemini`) behind a single factory function so switching providers is one line.

Graph state (shared `TypedDict` / Pydantic model) carries: raw resume text, raw JD text, parsed resume (structured), parsed JD (structured), pre-score, gaps, tailored resume (structured), post-score, and a log of edits made.

Nodes (each is an agent or tool step):

- **`jd_parser`** — LLM node. Input: raw JD. Output: `{role_title, seniority, must_have_skills[], nice_to_have_skills[], keywords[], responsibilities[], domain}`. Force structured JSON output.
- **`resume_parser`** — tool + LLM node. Extract text from uploaded file, then structure into `{contact, summary, skills[], experience[{company,title,dates,bullets[]}], education[], projects[], certifications[]}`.
- **`ats_scorer`** — deterministic tool node (see §5). Runs before and after tailoring. Returns a score + dimension breakdown + matched/missing keyword lists.
- **`gap_analysis`** — LLM node. Compares parsed JD vs parsed resume; outputs missing/weak keywords and a prioritized recommendation list.
- **`resume_tailor`** — LLM node with the §2 guardrail. Produces the tailored structured resume + an edit log explaining each change.
- **`truthfulness_validator`** — deterministic + LLM check. Verifies no fabricated entities were introduced. If it finds fabrication, route back to `resume_tailor` with feedback (max 2 retries), then hard-stop and surface the issue.
- **`resume_renderer`** — tool node. Renders the tailored structured resume into an ATS-clean `.docx` (and `.pdf`).

Edges: `resume_parser` → `jd_parser` → `ats_scorer` (pre) → `gap_analysis` → `resume_tailor` → `truthfulness_validator` → (conditional loop back or) `ats_scorer` (post) → `resume_renderer` → END.

## 4. Tools each agent needs (implement these)

- **File text extraction**: `pdfplumber` (primary) with `pypdf` fallback for PDFs; `python-docx` for `.docx`. Handle both upload types.
- **Structured LLM output**: use LangChain structured output / Pydantic parsers so parser and JD nodes return validated schemas, not free text.
- **ATS scorer** (pure Python, no LLM): keyword matching (exact + normalized), plus **semantic matching** via embeddings (`sentence-transformers` `all-MiniLM-L6-v2` locally, or the provider's embedding API) to catch synonyms (e.g. "PyTorch" ↔ "deep learning framework"). Cache embeddings.
- **Resume renderer**: `python-docx` writing a single-column, standard-heading, standard-font layout (no tables, text boxes, images, or header/footer content that breaks ATS parsing). Optional `.pdf` via `docx2pdf` or a LaTeX/reportlab path — pick the most reliable on Streamlit Cloud and note the choice.
- **Persistence**: save the base resume + parsed structure to local disk (`./data/base_resume.json` + original file) so it survives across sessions; expose "replace base resume" in the UI.

## 5. ATS scoring method (implement concretely)

Score 0–100 as a weighted sum. Use these starting weights and expose them as constants so they're tunable:

- **Keyword & skills coverage — 45%.** Fraction of JD `must_have_skills` + `keywords` present in the resume (exact match = full credit, semantic match ≥ threshold = partial credit). Weight must-haves higher than nice-to-haves.
- **Role/title alignment — 15%.** Similarity between JD `role_title`/`seniority` and the resume's most recent titles.
- **Responsibility/experience match — 20%.** Semantic overlap between JD responsibilities and resume bullets.
- **Quantified impact — 10%.** Share of experience bullets containing metrics/numbers.
- **ATS formatting/parse-ability — 10%.** Deterministic checks: single column, standard section headings present (Experience, Education, Skills), no tables/images/text-boxes, standard fonts, common file type, contact info parseable.

Return the total plus the per-dimension subscores and the matched/missing keyword lists so the UI can explain the number. Make clear in the UI copy that this is a **heuristic estimate** — real ATS platforms (Workday, Greenhouse, Taleo, iCIMS) score differently — but that keyword coverage and clean formatting are the reliable, universal wins.

## 6. Tech stack & dependencies

- Python 3.11
- `langgraph`, `langchain`, `langchain-openai`, `langchain-anthropic`, `langchain-google-genai`
- `streamlit`
- `pydantic`
- `pdfplumber`, `pypdf`, `python-docx`
- `sentence-transformers` (or provider embeddings), `numpy`
- `python-dotenv`
- (optional pdf export) `docx2pdf` or `reportlab`

**Pin every version in `requirements.txt`** and develop inside a fresh virtualenv — resolve dependency conflicts up front rather than letting them surface at deploy time. If any package won't install cleanly on Streamlit Community Cloud (e.g. `docx2pdf` needing Word, or heavy torch for sentence-transformers), choose the lighter/cloud-compatible alternative and document it in the README.

## 7. Project structure

```
resume-ats-optimizer/
├── app.py                    # Streamlit entry point
├── requirements.txt
├── .env.example              # documents required keys, no secrets
├── .gitignore                # ignores .env, ./data, __pycache__
├── README.md
├── config.py                 # weights, thresholds, model names, provider switch
├── graph/
│   ├── state.py              # shared state schema
│   ├── build_graph.py        # LangGraph assembly + conditional edges
│   └── nodes/
│       ├── jd_parser.py
│       ├── resume_parser.py
│       ├── ats_scorer.py
│       ├── gap_analysis.py
│       ├── resume_tailor.py
│       ├── truthfulness_validator.py
│       └── resume_renderer.py
├── tools/
│   ├── extract_text.py       # pdf/docx → text
│   ├── embeddings.py         # embedding + cosine sim, cached
│   ├── llm_factory.py        # provider switch
│   └── render_docx.py        # structured resume → ATS-clean docx/pdf
├── data/                     # persisted base resume (gitignored)
└── tests/                    # unit tests for scorer + extractors
```

## 8. Environment setup

- Create `.env.example` listing `LLM_PROVIDER`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` (only the selected provider's key is required), and any embedding config.
- Load with `python-dotenv`.
- In the README, give exact setup steps: create venv, `pip install -r requirements.txt`, copy `.env.example` to `.env`, add key, `streamlit run app.py`.
- Fail fast with a clear error if the required key for the selected provider is missing.

## 9. Streamlit UI spec

- **Sidebar**: provider selector, API-key status indicator, "Base resume" section — upload/replace master resume (PDF or docx), shows parsed summary once stored.
- **Main panel**:
  - A large text area to paste the JD (plus optional JD file upload).
  - A "Analyze & Tailor" button that runs the graph with a progress spinner / status per node.
  - **Results**: a before → after ATS score display (two big numbers + delta), an expandable per-dimension breakdown, matched vs missing keywords as chips/lists, the gap-analysis recommendations, and the edit log (what changed and why).
  - **Download** buttons for the tailored `.docx` (and `.pdf` if enabled).
- Keep session state clean; don't re-run the whole graph on every widget interaction.
- Do **not** use browser localStorage; use Streamlit session state + the on-disk `./data` persistence.

## 10. Deployment

Target **Streamlit Community Cloud**. Ensure: a clean `requirements.txt` that installs on their environment, secrets configured via Streamlit's secrets manager (mirror the `.env` keys), and no dependency that requires desktop Word or a GPU. In the README, include a "Deploy to Streamlit Cloud" section: push to GitHub, connect the repo, set secrets, deploy. Confirm the app cold-starts within Streamlit Cloud's memory limits (lazy-load the embedding model).

## 11. Definition of done

- I can upload a base resume, paste a JD, click one button, and get a before/after ATS score with a breakdown, a gap report, an edit log, and a downloadable tailored `.docx`.
- The truthfulness validator provably blocks fabricated entities.
- Provider is switchable via `.env` with no code changes.
- `requirements.txt` installs cleanly in a fresh venv and on Streamlit Cloud.
- README covers local run + deploy.
- Basic unit tests pass for the text extractors and the ATS scorer.

## 12. Stretch goals (only after the above works)

- Cover-letter generator node reusing the parsed JD + tailored resume.
- Support multiple saved base-resume variants (e.g. "AI Engineer" vs "ML Platform").
- A/B: show two tailoring variants and let the user pick.
- Persist a history of JD → score runs.

Build incrementally: scaffold the repo and env first, then extractors + scorer with tests, then the graph nodes one at a time wiring each into a minimal Streamlit view, then rendering, then deployment polish. After each milestone, tell me what you built and how to run it.