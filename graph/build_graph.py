"""LangGraph assembly: the agent pipeline and its conditional retry loop."""

from langgraph.graph import END, StateGraph

import config

from graph.nodes.ats_scorer import ats_scorer_node
from graph.nodes.gap_analysis import gap_analysis_node
from graph.nodes.github_enrich import github_enrich_node
from graph.nodes.jd_parser import jd_parser_node
from graph.nodes.optimizer import route_after_score, score_gate_node
from graph.nodes.resume_parser import resume_parser_node
from graph.nodes.resume_renderer import resume_renderer_node
from graph.nodes.resume_tailor import resume_tailor_node
from graph.nodes.route import route_after_validation
from graph.nodes.truthfulness_validator import truthfulness_validator_node
from graph.state import GraphState


def _fabrication_detail(state) -> str:
    fabs = state.get("validation").fabrications if state.get("validation") else []
    return "; ".join(f"{f.kind}: {f.value}" for f in fabs) or "unspecified"


def _fail_node(state) -> dict:
    """Hard stop: nothing truthful exists to fall back to.

    Only reachable when scoring never ran, so there is genuinely nothing to
    show. A fabricating tailor with a scored base resume behind it goes to
    `salvage` instead.
    """
    return {
        "error": (
            "Tailoring was stopped: the model introduced details not present in "
            f"your base resume and did not correct them after retries "
            f"({_fabrication_detail(state)}). Your original resume is unchanged."
        )
    }


def _salvage_node(state) -> dict:
    """The tailor could not produce a truthful rewrite. Ship the analysis anyway.

    Discarding the whole run here was costing far more than it protected: the
    score, the keyword coverage, the gap analysis and a rendered document all
    exist and are all truthful, and the user was getting an error instead of
    any of them because one skill line could not be worded safely. The base
    resume is by definition fabrication-free, so it renders; only the tailoring
    is lost, and the reason says so.
    """
    base = state.get("parsed_resume")
    pre = state.get("pre_score")
    return {
        "best_resume": base,
        "best_score": pre,
        "best_edit_log": [],
        "target_met": False,
        "ceiling_reason": (
            f"Scored {pre.total:.1f}. Tailoring was skipped: every rewrite "
            f"attempt claimed something your resume does not support "
            f"({_fabrication_detail(state)}), so your own wording was kept "
            "instead. The analysis below is still accurate — the gaps it names "
            "are real, and closing them is what would raise the score."
        ),
    }


def _keep_best_node(state) -> dict:
    """An optimization round fabricated after a good one already succeeded.

    The offending rewrite is dropped; the last validated resume still renders.
    """
    best = state.get("best_score")
    return {
        "target_met": False,
        "ceiling_reason": (
            f"Scored {best.total:.1f}, short of {config.TARGET_ATS_SCORE:.0f}. "
            "A further rewrite introduced details your base resume does not "
            "support, so it was discarded and this — the best truthful version "
            "— was kept."
        ),
    }


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("resume_parser", resume_parser_node)
    g.add_node("jd_parser", jd_parser_node)
    # Same callable registered twice: it inspects state to decide whether this
    # is the pre- or post-tailoring pass.
    g.add_node("score_pre", ats_scorer_node)
    g.add_node("github_enrich", github_enrich_node)
    g.add_node("gap_analysis", gap_analysis_node)
    g.add_node("resume_tailor", resume_tailor_node)
    g.add_node("truthfulness_validator", truthfulness_validator_node)
    g.add_node("score_post", ats_scorer_node)
    g.add_node("score_gate", score_gate_node)
    g.add_node("resume_renderer", resume_renderer_node)
    g.add_node("keep_best", _keep_best_node)
    g.add_node("salvage", _salvage_node)
    g.add_node("fail", _fail_node)

    g.set_entry_point("resume_parser")
    g.add_edge("resume_parser", "jd_parser")
    # Enrichment sits here so parsed_jd exists to rank against, and the
    # resume is enriched before score_pre, gap_analysis, the tailor and the
    # validator ever read it. The tailoring loop never routes back through
    # this node, so it runs exactly once.
    g.add_edge("jd_parser", "github_enrich")
    g.add_edge("github_enrich", "score_pre")
    g.add_edge("score_pre", "gap_analysis")
    g.add_edge("gap_analysis", "resume_tailor")
    g.add_edge("resume_tailor", "truthfulness_validator")

    g.add_conditional_edges(
        "truthfulness_validator",
        route_after_validation,
        {
            "retry": "resume_tailor",
            "ok": "score_post",
            "keep_best": "keep_best",
            "salvage": "salvage",
            "fail": "fail",
        },
    )

    # Score gate: re-tailor toward the target, or accept the best truthful
    # result and render it.
    g.add_edge("score_post", "score_gate")
    g.add_conditional_edges(
        "score_gate",
        route_after_score,
        {"optimize": "resume_tailor", "done": "resume_renderer"},
    )

    g.add_edge("keep_best", "resume_renderer")
    g.add_edge("salvage", "resume_renderer")
    g.add_edge("resume_renderer", END)
    g.add_edge("fail", END)

    return g.compile()
