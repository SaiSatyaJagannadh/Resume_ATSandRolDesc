"""The tailor's edits must actually land on the resume.

Regression guard: TailorResult used to carry the full rewritten resume, and the
model reliably echoed the input back unchanged while still filling in `edits`,
so post-tailoring scores were identical to pre-tailoring ones.
"""

from graph.nodes.resume_tailor import apply_edits
from graph.state import Edit, ExperienceEntry, ParsedResume, ProjectEntry

BASE = ParsedResume(
    summary="System Engineer building scalable services.",
    skills=["Python", "Airflow"],
    experience=[
        ExperienceEntry(
            company="UBS",
            title="Tech Engineer",
            dates="Jun 2025 - Present",
            bullets=[
                "Resolved critical production issues under tight timelines.",
                "Utilized Azure Application Insights to analyze logs.",
            ],
        )
    ],
    projects=[ProjectEntry(name="Stocks", description="Kafka pipeline.")],
)


def _edit(before, after, location="experience[0].bullets[0]"):
    return Edit(location=location, before=before, after=after, reason="test")


def test_bullet_is_rewritten():
    out, applied = apply_edits(
        BASE,
        [_edit(
            "Resolved critical production issues under tight timelines.",
            "Led incident response for critical production issues.",
        )],
    )
    assert out.experience[0].bullets[0] == "Led incident response for critical production issues."
    assert len(applied) == 1
    # The base is never mutated in place.
    assert BASE.experience[0].bullets[0].startswith("Resolved")


def test_edits_across_sections_all_land():
    out, applied = apply_edits(
        BASE,
        [
            _edit("System Engineer building scalable services.",
                  "Site Reliability Engineer building scalable services.", "summary"),
            _edit("Kafka pipeline.", "Event-driven Kafka pipeline.", "projects[0].description"),
            _edit("Airflow", "Apache Airflow", "skills[1]"),
        ],
    )
    assert out.summary.startswith("Site Reliability")
    assert out.projects[0].description == "Event-driven Kafka pipeline."
    assert "Apache Airflow" in out.skills
    assert len(applied) == 3


def test_truncated_before_still_matches():
    out, applied = apply_edits(
        BASE,
        [_edit("Utilized Azure Application Insights to analyze",  # truncated quote
               "Utilized Azure Application Insights for monitoring and alerting.",
               "experience[0].bullets[1]")],
    )
    assert out.experience[0].bullets[1].endswith("monitoring and alerting.")
    assert len(applied) == 1


def test_addition_only_allowed_for_skills_and_certs():
    out, applied = apply_edits(
        BASE,
        [
            _edit("", "Terraform", "skills"),
            _edit("", "Ran the whole platform single-handedly.", "experience[0].bullets[9]"),
        ],
    )
    assert "Terraform" in out.skills
    # A bodiless bullet addition has no home and must be dropped, not appended.
    assert len(out.experience[0].bullets) == 2
    assert len(applied) == 1


def test_unmatched_and_noop_edits_are_dropped():
    out, applied = apply_edits(
        BASE,
        [
            _edit("Text that does not exist anywhere in this resume.", "Something new."),
            _edit("Python", "Python", "skills[0]"),  # no-op
        ],
    )
    assert applied == []
    assert out.model_dump() == BASE.model_dump()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
