"""LangGraph assembly: the agent pipeline and its conditional retry loop."""

from langgraph.graph import END, StateGraph

from graph.nodes.ats_scorer import ats_scorer_node
from graph.nodes.gap_analysis import gap_analysis_node
from graph.nodes.jd_parser import jd_parser_node
from graph.nodes.resume_parser import resume_parser_node
from graph.nodes.resume_renderer import resume_renderer_node
from graph.nodes.resume_tailor import resume_tailor_node
from graph.nodes.route import route_after_validation
from graph.nodes.truthfulness_validator import truthfulness_validator_node
from graph.state import GraphState


def _fail_node(state) -> dict:
    """Hard stop: the tailor kept fabricating after its retries were spent.

    Surfacing the tailored resume anyway would defeat the entire guardrail, so
    the run ends with an error and no download.
    """
    fabs = state.get("validation").fabrications if state.get("validation") else []
    detail = "; ".join(f"{f.kind}: {f.value}" for f in fabs) or "unspecified"
    return {
        "error": (
            "Tailoring was stopped: the model introduced details not present in "
            f"your base resume and did not correct them after retries ({detail}). "
            "Your original resume is unchanged."
        )
    }


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("resume_parser", resume_parser_node)
    g.add_node("jd_parser", jd_parser_node)
    # Same callable registered twice: it inspects state to decide whether this
    # is the pre- or post-tailoring pass.
    g.add_node("score_pre", ats_scorer_node)
    g.add_node("gap_analysis", gap_analysis_node)
    g.add_node("resume_tailor", resume_tailor_node)
    g.add_node("truthfulness_validator", truthfulness_validator_node)
    g.add_node("score_post", ats_scorer_node)
    g.add_node("resume_renderer", resume_renderer_node)
    g.add_node("fail", _fail_node)

    g.set_entry_point("resume_parser")
    g.add_edge("resume_parser", "jd_parser")
    g.add_edge("jd_parser", "score_pre")
    g.add_edge("score_pre", "gap_analysis")
    g.add_edge("gap_analysis", "resume_tailor")
    g.add_edge("resume_tailor", "truthfulness_validator")

    g.add_conditional_edges(
        "truthfulness_validator",
        route_after_validation,
        {"retry": "resume_tailor", "ok": "score_post", "fail": "fail"},
    )

    g.add_edge("score_post", "resume_renderer")
    g.add_edge("resume_renderer", END)
    g.add_edge("fail", END)

    return g.compile()
