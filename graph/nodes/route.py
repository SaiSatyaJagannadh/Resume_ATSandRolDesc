"""Conditional edge after truthfulness validation."""

import config


def route_after_validation(state) -> str:
    validation = state.get("validation")
    if validation is not None and validation.passed:
        return "ok"
    if state.get("tailor_attempts", 0) < config.MAX_TAILOR_RETRIES:
        return "retry"
    # A fabricating OPTIMIZATION round is a dead end, not a dead run: an earlier
    # round already banked a validated resume, so keep that rather than throwing
    # the whole analysis away. Only a first pass with nothing banked is fatal.
    if state.get("best_resume") is not None:
        return "keep_best"
    return "fail"
