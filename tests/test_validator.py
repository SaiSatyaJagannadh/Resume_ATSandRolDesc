"""Deterministic half of the truthfulness validator. No network, no LLM."""

import config
from graph.nodes.route import route_after_validation
from graph.nodes.truthfulness_validator import find_fabrications
from graph.state import (
    EducationEntry,
    ExperienceEntry,
    ParsedResume,
    ValidationResult,
)


def base_resume() -> ParsedResume:
    return ParsedResume(
        summary="Backend engineer.",
        skills=["Python", "AWS"],
        experience=[
            ExperienceEntry(
                company="Acme Corp",
                title="Senior Software Engineer",
                dates="Jan 2020 - Present",
                bullets=[
                    "Cut p99 latency by 35% across the billing service.",
                    "Mentored 4 engineers on the payments team.",
                ],
            ),
            ExperienceEntry(
                company="Globex",
                title="Software Engineer",
                dates="Jun 2017 - Dec 2019",
                bullets=["Built an internal reporting tool."],
            ),
        ],
        education=[
            EducationEntry(
                institution="State University", degree="BS Computer Science", dates="2013 - 2017"
            )
        ],
        certifications=["AWS Certified Developer"],
    )


def mutate(**changes) -> ParsedResume:
    """Base resume with the first experience entry's fields overridden."""
    r = base_resume()
    for k, v in changes.items():
        setattr(r.experience[0], k, v)
    return r


def kinds(fabs):
    return [f.kind for f in fabs]


def test_reworded_bullets_only_is_clean():
    tailored = mutate(
        bullets=[
            "Reduced p99 latency 35% on the billing service, using AWS.",
            "Mentored 4 engineers across the payments team.",
        ]
    )
    assert find_fabrications(base_resume(), tailored) == []


def test_invented_company_caught():
    fabs = find_fabrications(base_resume(), mutate(company="Initech"))
    assert kinds(fabs) == ["company"]
    assert fabs[0].value == "Initech"


def test_altered_title_caught():
    fabs = find_fabrications(base_resume(), mutate(title="Principal Software Engineer"))
    assert kinds(fabs) == ["title"]


def test_changed_dates_caught():
    fabs = find_fabrications(base_resume(), mutate(dates="Jan 2018 - Present"))
    assert kinds(fabs) == ["date"]


def test_fabricated_metric_caught():
    fabs = find_fabrications(
        base_resume(), mutate(bullets=["Increased revenue 40% year over year."])
    )
    assert kinds(fabs) == ["metric"]
    assert fabs[0].value == "40"


def test_existing_number_moved_to_another_bullet_is_not_flagged():
    # 35 and 4 both exist in the base, just on different bullets. Legitimate
    # reordering: a false positive here would block valid tailoring.
    fabs = find_fabrications(
        base_resume(),
        mutate(
            bullets=[
                "Mentored 4 engineers while cutting p99 latency 35%.",
                "Owned the billing service.",
            ]
        ),
    )
    assert fabs == []


def test_invented_certification_and_degree_caught():
    tailored = base_resume()
    tailored.certifications = ["AWS Certified Developer", "CISSP"]
    tailored.education[0].institution = "Ivy University"
    assert sorted(kinds(find_fabrications(base_resume(), tailored))) == [
        "certification",
        "degree",
    ]


def test_punctuation_and_case_differences_are_not_fabrications():
    fabs = find_fabrications(
        base_resume(), mutate(company="ACME  CORP.", title="senior software engineer")
    )
    assert fabs == []


def test_route_passed():
    assert route_after_validation({"validation": ValidationResult(passed=True)}) == "ok"


def test_route_failed_under_limit():
    state = {"validation": ValidationResult(passed=False), "tailor_attempts": 1}
    assert config.MAX_TAILOR_RETRIES > 1
    assert route_after_validation(state) == "retry"


def test_route_failed_at_limit():
    state = {
        "validation": ValidationResult(passed=False),
        "tailor_attempts": config.MAX_TAILOR_RETRIES,
    }
    assert route_after_validation(state) == "fail"
