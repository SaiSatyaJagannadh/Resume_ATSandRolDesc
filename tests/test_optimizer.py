"""Tests for the score optimization loop.

No LLM, no network: score_gate_node and route_after_score are pure functions
over state.
"""

import config
from graph.nodes.optimizer import build_feedback, route_after_score, score_gate_node
from graph.state import (
    ATSScore,
    DimensionScore,
    Gap,
    GapAnalysis,
    KeywordMatch,
    ParsedResume,
)


def make_score(total, must_matched=3, must_missing=0, other_missing=()):
    matched = [
        KeywordMatch(keyword=f"mh{i}", matched=True, match_type="exact",
                     score=1.0, is_must_have=True)
        for i in range(must_matched)
    ]
    missing = [
        KeywordMatch(keyword=f"gap{i}", matched=False, match_type="none",
                     score=0.0, is_must_have=True)
        for i in range(must_missing)
    ] + [
        KeywordMatch(keyword=k, matched=False, match_type="none",
                     score=0.0, is_must_have=False)
        for k in other_missing
    ]
    return ATSScore(
        total=total,
        dimensions=[
            DimensionScore(name="keywords", raw=0.5, weight=0.45, weighted=22.5),
            DimensionScore(name="formatting", raw=1.0, weight=0.10, weighted=10.0),
        ],
        matched_keywords=matched,
        missing_keywords=missing,
    )


def base_state(score, **kw):
    return {"post_score": score, "tailored_resume": ParsedResume(), **kw}


def test_target_met_routes_done():
    out = score_gate_node(base_state(make_score(88.0)))
    assert out["target_met"] is True
    assert route_after_score({**out}) == "done"


def test_below_target_with_room_routes_optimize():
    out = score_gate_node(base_state(make_score(70.0, must_matched=4, must_missing=1)))
    assert out["target_met"] is False
    assert out["optimize_rounds"] == 1
    assert out["optimizer_feedback"]
    assert route_after_score(out) == "optimize"


def test_optimize_round_resets_fabrication_budget():
    """Each optimization round is a fresh attempt and gets a full retry budget."""
    out = score_gate_node(
        base_state(make_score(70.0, must_matched=4, must_missing=1),
                   tailor_attempts=2, validator_feedback="stale")
    )
    assert out["tailor_attempts"] == 0
    assert out["validator_feedback"] == ""


def test_missing_must_haves_is_an_honest_ceiling():
    """Too few must-haves: stop and explain, rather than pressure a fabrication."""
    out = score_gate_node(base_state(make_score(60.0, must_matched=1, must_missing=5)))
    assert out["target_met"] is False
    assert "must-have" in out["ceiling_reason"]
    assert route_after_score(out) == "done"
    assert "optimizer_feedback" not in out


def test_rounds_exhausted_stops():
    out = score_gate_node(
        base_state(make_score(80.0, must_matched=4, must_missing=1),
                   optimize_rounds=config.MAX_OPTIMIZE_ROUNDS)
    )
    assert route_after_score(out) == "done"
    assert "short of" in out["ceiling_reason"]


def test_diminishing_returns_stops():
    out = score_gate_node(
        base_state(make_score(80.2, must_matched=4, must_missing=1),
                   score_history=[80.0], optimize_rounds=0)
    )
    assert route_after_score(out) == "done"
    assert "almost no change" in out["ceiling_reason"]


def test_best_result_survives_a_regression():
    """A later, worse round must not overwrite the best-so-far."""
    good = make_score(82.0, must_matched=4, must_missing=1)
    first = score_gate_node(base_state(good))
    assert first["best_score"].total == 82.0

    worse = make_score(74.0, must_matched=4, must_missing=1)
    second = score_gate_node(
        base_state(worse, best_score=first["best_score"],
                   best_resume=first["best_resume"],
                   optimize_rounds=config.MAX_OPTIMIZE_ROUNDS)
    )
    # No best_* keys written back means the earlier, better result stands.
    assert "best_score" not in second
    assert second["score_history"] == [74.0]


def test_feedback_never_asks_for_unsupported_skills():
    """The one thing this must never do: tell the tailor to cover a real gap."""
    score = make_score(70.0, must_matched=2, must_missing=0,
                       other_missing=["Kubernetes", "Airflow"])
    state = {
        "gap_analysis": GapAnalysis(
            gaps=[Gap(item="Kubernetes", severity="critical", kind="missing",
                      rationale="No exposure.", unsupported_by_resume=True)]
        )
    }
    fb = build_feedback(score, state)
    assert "Airflow" in fb                    # supported -> fair game
    assert "DO NOT attempt to cover" in fb
    # Kubernetes may only appear inside the off-limits block, never as a target.
    before_prohibition = fb.split("DO NOT attempt to cover")[0]
    assert "Kubernetes" not in before_prohibition


def test_missing_post_score_does_not_crash():
    out = score_gate_node({"tailored_resume": None, "post_score": None})
    assert out["target_met"] is False
    assert route_after_score(out) == "done"
