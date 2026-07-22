"""Drive the tailored resume toward the target ATS score — honestly.

This node exists because a single tailoring pass often lands short of the
target. It re-tailors with specific, score-derived feedback about what is
actually costing points.

The hard part is knowing when to STOP. Keyword coverage is 45% of the score, so
when the candidate genuinely lacks the job's must-have skills the remaining
points are only reachable by claiming skills they do not have. An optimizer that
keeps pushing in that situation is not optimizing, it is applying pressure to
fabricate — and the truthfulness validator would then be the only thing standing
between the user and a resume they cannot defend in an interview. So the loop
stops on a real ceiling and explains it, rather than grinding against it.
"""

import config
from graph.state import ATSScore


def _must_have_coverage(score: ATSScore) -> tuple[int, int]:
    """(matched must-haves, total must-haves)."""
    matched = sum(1 for k in score.matched_keywords if k.is_must_have)
    missing = sum(1 for k in score.missing_keywords if k.is_must_have)
    return matched, matched + missing


def _unsupported_items(state) -> set[str]:
    """Gap items the analyst flagged as unsupported by the base resume.

    These are off-limits: telling the tailor to "cover" them is telling it to
    invent them.
    """
    gaps = state.get("gap_analysis")
    if not gaps:
        return set()
    return {g.item.strip().lower() for g in gaps.gaps if g.unsupported_by_resume}


def build_feedback(score: ATSScore, state) -> str:
    """Score-derived instructions for the next tailoring pass.

    Only names keywords the candidate can legitimately claim, and points at the
    weakest weighted dimensions so the model spends effort where the points are.
    """
    off_limits = _unsupported_items(state)
    winnable = [
        k.keyword
        for k in score.missing_keywords
        if k.keyword.strip().lower() not in off_limits
    ]
    must_first = [k for k in winnable if any(
        m.keyword == k and m.is_must_have for m in score.missing_keywords
    )]
    others = [k for k in winnable if k not in must_first]

    # Biggest point losses first: (weight - earned) is the headroom per dimension.
    headroom = sorted(
        score.dimensions, key=lambda d: (d.weight * 100) - d.weighted, reverse=True
    )
    weak = [
        f"- {d.name}: {d.weighted:.1f} of {d.weight * 100:.0f} points"
        + (f" ({d.detail})" if d.detail else "")
        for d in headroom[:3]
        if (d.weight * 100) - d.weighted > 1.0
    ]

    parts = [
        f"The tailored resume scored {score.total:.1f}. The target is "
        f"{config.TARGET_ATS_SCORE:.0f}. Revise to close the gap.",
        "",
        "WHERE THE POINTS ARE:",
        *(weak or ["- (no single dimension dominates the shortfall)"]),
    ]

    if must_first:
        parts += [
            "",
            "MISSING MUST-HAVE KEYWORDS the base resume DOES support — surface "
            "these using the job description's exact wording, in the bullets "
            "where that work actually happened:",
            *(f"- {k}" for k in must_first[:12]),
        ]
    if others:
        parts += [
            "",
            "Other missing keywords the resume supports:",
            *(f"- {k}" for k in others[:12]),
        ]
    if off_limits:
        parts += [
            "",
            "DO NOT attempt to cover these — the base resume does not support "
            "them, and claiming them would be fabrication:",
            *(f"- {g}" for g in sorted(off_limits)),
        ]

    parts += [
        "",
        "Every rule from your original instructions still applies. Raising the "
        "score by inventing experience is a failure, not a success. If a "
        "keyword cannot be honestly supported, leave it missing.",
    ]
    return "\n".join(parts)


def score_gate_node(state) -> dict:
    """Record the best result so far and decide whether another round is worth it."""
    score = state.get("post_score")
    resume = state.get("tailored_resume")
    if score is None or resume is None:
        return {"ceiling_reason": "No tailored resume was produced.", "target_met": False}

    history = list(state.get("score_history", [])) + [score.total]
    rounds = state.get("optimize_rounds", 0)

    # Keep the best, not the latest: a later round can regress.
    best = state.get("best_score")
    out = {"score_history": history}
    if best is None or score.total > best.total:
        out |= {
            "best_score": score,
            "best_resume": resume,
            "best_edit_log": state.get("edit_log", []),
        }
        best_total = score.total
    else:
        best_total = best.total

    if best_total >= config.TARGET_ATS_SCORE:
        return out | {
            "target_met": True,
            "ceiling_reason": "",
        }

    matched_mh, total_mh = _must_have_coverage(score)
    coverage = (matched_mh / total_mh) if total_mh else 1.0

    # Ceiling 1: too few must-haves present. More rounds would only pressure the
    # model to claim skills the candidate lacks.
    if total_mh and coverage < config.MUST_HAVE_FLOOR:
        missing = [k.keyword for k in score.missing_keywords if k.is_must_have]
        return out | {
            "target_met": False,
            "ceiling_reason": (
                f"Scored {best_total:.1f}, short of {config.TARGET_ATS_SCORE:.0f}. "
                f"Only {matched_mh} of {total_mh} must-have requirements are "
                f"supported by your experience. Missing: {', '.join(missing[:8])}. "
                "Wording cannot close this gap — the score is honest, and the "
                "real fix is acquiring or documenting these skills."
            ),
        }

    # Ceiling 2: rounds exhausted.
    if rounds >= config.MAX_OPTIMIZE_ROUNDS:
        return out | {
            "target_met": False,
            "ceiling_reason": (
                f"Scored {best_total:.1f} after {rounds + 1} tailoring passes, "
                f"short of {config.TARGET_ATS_SCORE:.0f}. Further rewriting "
                "stopped producing gains. This is the best truthful version."
            ),
        }

    # Ceiling 3: diminishing returns — a round that moved the score < 0.5 points
    # is not going to be rescued by another one.
    if len(history) >= 2 and abs(history[-1] - history[-2]) < 0.5:
        return out | {
            "target_met": False,
            "ceiling_reason": (
                f"Scored {best_total:.1f}, short of {config.TARGET_ATS_SCORE:.0f}. "
                "The last rewrite produced almost no change, so additional "
                "passes were skipped. This is the best truthful version."
            ),
        }

    # Worth another round. Reset the fabrication-retry budget so the new attempt
    # gets a full allowance, and clear stale validator feedback.
    return out | {
        "target_met": False,
        "optimize_rounds": rounds + 1,
        "optimizer_feedback": build_feedback(score, state),
        "tailor_attempts": 0,
        "validator_feedback": "",
    }


def route_after_score(state) -> str:
    """"optimize" for another tailoring round, else "done"."""
    if state.get("target_met"):
        return "done"
    if state.get("ceiling_reason"):
        return "done"
    return "optimize"
