"""Streamlit entry point for the agentic resume ATS optimizer."""

import os
from pathlib import Path

import streamlit as st

import config
from tools.extract_text import extract_text_from_bytes
from tools.persistence import clear_base_resume, load_base_resume, save_base_resume

st.set_page_config(page_title="Resume ATS Optimizer", page_icon="🎯", layout="wide")

RESUME_TYPES = ["pdf", "docx", "txt", "md"]


# --- Sidebar ---------------------------------------------------------------


def render_sidebar():
    st.sidebar.header("Configuration")

    providers = list(config.PROVIDER_KEYS)
    provider = st.sidebar.selectbox(
        "LLM provider",
        providers,
        index=providers.index(config.LLM_PROVIDER)
        if config.LLM_PROVIDER in providers
        else 0,
        help="Switch providers without code changes. Set the matching key in .env.",
    )
    # Selecting a provider in the UI overrides the .env default for this run.
    config.LLM_PROVIDER = provider
    os.environ["LLM_PROVIDER"] = provider

    key_name = config.PROVIDER_KEYS[provider]
    if os.getenv(key_name):
        st.sidebar.success(f"{key_name} found")
    else:
        st.sidebar.error(f"{key_name} missing — add it to .env")

    if os.getenv(config.EMBEDDING_KEY):
        st.sidebar.success(f"{config.EMBEDDING_KEY} found (embeddings)")
    else:
        st.sidebar.error(
            f"{config.EMBEDDING_KEY} missing — required for semantic scoring, "
            "regardless of the provider above."
        )

    st.sidebar.divider()
    st.sidebar.header("Base resume")

    base = st.session_state.get("base_resume")
    if base:
        st.sidebar.success(f"Stored: {base.contact.name or 'unnamed'}")
        st.sidebar.caption(
            f"{len(base.experience)} roles · {len(base.skills)} skills · "
            f"{len(base.education)} education entries"
        )
        with st.sidebar.expander("Parsed summary"):
            for e in base.experience:
                st.markdown(f"**{e.title}** — {e.company}  \n_{e.dates}_")
        if st.sidebar.button("Remove base resume"):
            clear_base_resume()
            st.session_state.pop("base_resume", None)
            st.rerun()

    st.sidebar.text_input(
        "GitHub username (optional)",
        key="github_username",
        help="Only needed if your resume has no GitHub link. Relevant public "
        "repos get added as projects, with their URL shown in the results.",
    )

    upload = st.sidebar.file_uploader(
        "Upload / replace master resume", type=RESUME_TYPES
    )
    if upload is not None and st.sidebar.button("Save base resume", type="primary"):
        save_uploaded_resume(upload)


def save_uploaded_resume(upload):
    from graph.nodes.resume_parser import resume_parser_node

    with st.spinner("Parsing your resume…"):
        try:
            data = upload.getvalue()
            text = extract_text_from_bytes(data, upload.name)
            parsed = resume_parser_node({"raw_resume_text": text})["parsed_resume"]
            save_base_resume(parsed, original=data, filename=upload.name)
            st.session_state["base_resume"] = parsed
            st.session_state["base_resume_text"] = text
        except Exception as exc:
            st.sidebar.error(f"Could not read that file: {exc}")
            return
    st.rerun()


# --- Results ---------------------------------------------------------------


def render_target_banner(result, final_score):
    target = config.TARGET_ATS_SCORE
    if result.get("target_met"):
        st.success(f"Target met — scored {final_score.total:.1f} (target {target:.0f}).")
        return
    reason = result.get("ceiling_reason")
    if reason:
        st.warning(f"**Target of {target:.0f} not reached.** {reason}")
        st.caption(
            "This is the highest score achievable without claiming experience "
            "you don't have. A truthful resume you can defend in an interview "
            "beats a higher-scoring one you can't."
        )


def render_github_projects(urls):
    """Say plainly which repos were added — this content goes out under the
    user's name, so it cannot be added silently."""
    if not urls:
        return
    st.info(
        f"Added {len(urls)} project(s) from your GitHub, chosen by relevance to "
        "this job description. Details come straight from the repo — review "
        "them before you send the resume out."
    )
    for url in urls:
        st.caption(f"- {url}")


def render_scores(pre, post, history=None):
    st.subheader("ATS match score")
    c1, c2, c3 = st.columns(3)
    c1.metric("Before", f"{pre.total:.1f}")
    c2.metric(
        "After",
        f"{post.total:.1f}",
        delta=f"{post.total - pre.total:+.1f}",
    )
    if history and len(history) > 1:
        c3.caption(
            "Tailoring passes: "
            + " → ".join(f"{h:.1f}" for h in history)
        )
    c3.caption(
        "Heuristic estimate. Real ATS platforms (Workday, Greenhouse, Taleo, "
        "iCIMS) each score differently — but keyword coverage and clean "
        "formatting are the reliable, universal wins."
    )

    with st.expander("Per-dimension breakdown"):
        before = {d.name: d for d in pre.dimensions}
        for d in post.dimensions:
            b = before.get(d.name)
            st.markdown(
                f"**{d.name.replace('_', ' ').title()}** "
                f"({d.weight * 100:.0f}% of total)"
            )
            st.progress(min(d.raw, 1.0))
            delta = f" (was {b.weighted:.1f})" if b else ""
            st.caption(f"{d.weighted:.1f} / {d.weight * 100:.0f} points{delta}")
            if d.detail:
                st.caption(d.detail)


def render_keywords(score):
    st.subheader("Keyword coverage")
    matched = [k for k in score.matched_keywords]
    missing = [k for k in score.missing_keywords]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Matched ({len(matched)})**")
        st.markdown(
            " ".join(
                f"`{k.keyword}{'*' if k.is_must_have else ''}"
                f"{'~' if k.match_type == 'semantic' else ''}`"
                for k in matched
            )
            or "_None_"
        )
    with c2:
        st.markdown(f"**Missing ({len(missing)})**")
        st.markdown(
            " ".join(f"`{k.keyword}{'*' if k.is_must_have else ''}`" for k in missing)
            or "_None_"
        )
    st.caption("`*` = must-have · `~` = semantic (partial credit) match")


def render_gaps(gaps):
    st.subheader("Gap analysis")
    order = {"critical": 0, "important": 1, "minor": 2}
    icon = {"critical": "🔴", "important": "🟠", "minor": "🟡"}
    for g in sorted(gaps.gaps, key=lambda g: order.get(g.severity, 3)):
        flag = " — **not supported by your resume**" if g.unsupported_by_resume else ""
        st.markdown(f"{icon.get(g.severity, '•')} **{g.item}**{flag}")
        st.caption(g.rationale)
    if gaps.recommendations:
        st.markdown("**Recommendations**")
        for r in gaps.recommendations:
            st.markdown(f"- {r}")


def render_edits(edits):
    st.subheader("What changed")
    if not edits:
        st.caption("No edits were made.")
        return
    for e in edits:
        with st.expander(e.location):
            if e.before:
                st.markdown("**Before**")
                st.text(e.before)
            st.markdown("**After**")
            st.text(e.after)
            st.caption(e.reason)


def render_results(result):
    if result.get("error"):
        st.error(result["error"])
        return

    # best_* is what actually got rendered into the .docx; fall back for
    # robustness if the gate never ran.
    final_score = result.get("best_score") or result["post_score"]

    render_github_projects(result.get("github_projects_added"))
    render_target_banner(result, final_score)
    render_scores(result["pre_score"], final_score, result.get("score_history"))
    render_keywords(final_score)
    st.divider()
    render_gaps(result["gap_analysis"])
    st.divider()
    render_edits(result.get("best_edit_log") or result.get("edit_log", []))
    st.divider()

    path = result.get("docx_path")
    if path and Path(path).exists():
        st.download_button(
            "Download tailored resume (.docx)",
            data=Path(path).read_bytes(),
            file_name=Path(path).name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary",
        )


# --- Main ------------------------------------------------------------------


def run_pipeline(resume_text, jd_text, github_username=""):
    from graph.build_graph import build_graph

    status = st.status("Running the agent pipeline…", expanded=True)
    labels = {
        "resume_parser": "Parsing resume",
        "jd_parser": "Parsing job description",
        "github_enrich": "Finding relevant GitHub projects",
        "score_pre": "Scoring your current resume",
        "gap_analysis": "Analyzing gaps",
        "resume_tailor": "Tailoring resume",
        "truthfulness_validator": "Checking for fabrications",
        "score_post": "Re-scoring tailored resume",
        "score_gate": f"Checking against target ({config.TARGET_ATS_SCORE:.0f})",
        "resume_renderer": "Building .docx",
        "fail": "Stopped: fabrication detected",
    }

    final = {}
    graph = build_graph()
    # stream() surfaces per-node progress; the last write of each key wins.
    for chunk in graph.stream(
        {
            "raw_resume_text": resume_text,
            "raw_jd_text": jd_text,
            "github_username": github_username,
        },
        # Headroom for the optimization loop: each round can run a full
        # tailor + fabrication-retry + score cycle.
        {"recursion_limit": 60},
    ):
        for node, update in chunk.items():
            status.write(f"✓ {labels.get(node, node)}")
            final.update(update)
    status.update(label="Done", state="complete", expanded=False)
    return final


def main():
    st.title("🎯 Resume ATS Optimizer")
    st.caption(
        "Tailors your master resume to a specific job description — rephrasing "
        "and resurfacing real experience, never inventing it."
    )

    if "base_resume" not in st.session_state:
        stored = load_base_resume()
        if stored:
            st.session_state["base_resume"] = stored

    render_sidebar()

    base = st.session_state.get("base_resume")
    if not base:
        st.info("Upload your master resume in the sidebar to get started.")
        return

    jd_text = st.text_area("Paste the job description", height=280)
    jd_file = st.file_uploader(
        "…or upload it", type=RESUME_TYPES, key="jd_upload"
    )
    if jd_file is not None:
        try:
            jd_text = extract_text_from_bytes(jd_file.getvalue(), jd_file.name)
            st.success(f"Loaded JD from {jd_file.name}")
        except Exception as exc:
            st.error(f"Could not read that file: {exc}")

    if st.button("Analyze & Tailor", type="primary", disabled=not jd_text.strip()):
        try:
            config.require_key()
            config.require_embedding_key()
        except Exception as exc:
            st.error(str(exc))
            return
        # Cached in session state so widget interactions below (expanders,
        # download button) don't re-run the whole graph.
        st.session_state["result"] = run_pipeline(
            st.session_state.get("base_resume_text")
            or base.model_dump_json(),
            jd_text,
            st.session_state.get("github_username", ""),
        )

    if st.session_state.get("result"):
        render_results(st.session_state["result"])


if __name__ == "__main__":
    main()
