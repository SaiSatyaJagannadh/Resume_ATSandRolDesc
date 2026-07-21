"""Conditional edge after truthfulness validation."""

import config


def route_after_validation(state) -> str:
    validation = state.get("validation")
    if validation is not None and validation.passed:
        return "ok"
    if state.get("tailor_attempts", 0) < config.MAX_TAILOR_RETRIES:
        return "retry"
    return "fail"
