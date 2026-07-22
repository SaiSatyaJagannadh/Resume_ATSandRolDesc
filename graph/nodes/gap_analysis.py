"""Compare parsed resume against parsed JD and produce a prioritized gap list."""

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import GapAnalysis
from tools.llm_factory import structured

SYSTEM = """You compare a candidate's resume against a job description and
report the gaps honestly.

For each gap:
- kind="missing" when the resume shows nothing at all on the requirement.
  kind="weak" when there is some related evidence but it is thin, dated, or
  peripheral.
- severity: "critical" for must-haves the candidate cannot claim, "important"
  for must-haves with weak evidence or high-signal nice-to-haves, "minor" for
  the rest.
- unsupported_by_resume: see the calibration below. This is the single most
  consequential field you set.

CALIBRATING unsupported_by_resume
This flag answers ONE question: is there concrete evidence in the resume that
the candidate has actually DONE this work? It does NOT ask whether the exact
word appears. Those are different questions, and conflating them is the most
common way this analysis goes wrong.

Set it False (the term is fair game) when the resume evidences the work under
different wording. The candidate did the thing; the resume just names it
differently. Examples:
- JD wants "ETL"; resume shows Airflow DAGs loading into Redshift. That IS ETL.
- JD wants "CI/CD"; resume shows Jenkins pipelines and automated deploys.
- JD wants "data modeling"; resume shows designing warehouse schemas.
- JD wants "REST APIs"; resume shows building Flask endpoints.
Using the posting's vocabulary for work the candidate genuinely performed is
accurate resume writing, not fabrication. Flagging these as unsupported blocks
the tailoring step from stating true things in the words the employer used —
which costs the candidate real interviews for no safety benefit.

A NAMED PRODUCT IMPLIES ITS CATEGORY. This is the single most common mistake:
seeing that a generic term is absent while the specific product that IS that
term sits in the skills list. Before flagging any term unsupported, check
whether the resume names a concrete tool that performs it:
- Azure Data Factory, AWS Glue, Informatica, Talend, SSIS -> ETL and ELT
- Azure Databricks, EMR -> Apache Spark
- Azure Synapse, Redshift, Snowflake, BigQuery -> data warehousing, data modeling
- Airflow, Azure Data Factory, Step Functions -> orchestration, data pipelines
- Kafka, Event Hubs, Kinesis -> streaming, real-time processing
- Azure ML, SageMaker, MLflow -> ML engineering, model deployment
- EKS, AKS -> Kubernetes; ECS, ACI -> containers
If the resume names the product, the candidate has done the category of work.
Mark it supported (False) so the tailoring step may state it in the employer's
words. Do NOT extend this to tools the resume never names.

Set it True ONLY when nothing in the resume — no bullet, skill, project,
certification, or degree — evidences the underlying work. Examples:
- JD wants Kubernetes; nothing in the resume touches containers or orchestration.
- JD wants a security clearance the resume never mentions.
- JD wants 10 years; the resume shows 3.
Downstream, True means "off-limits: never write this in." A false True costs
the candidate a keyword they earned. A false False lets a fabrication into a
document they sign their name to. Judge on evidence of the work, and when there
is genuinely no evidence, say True without hesitation.

Order gaps by severity, critical first. Recommendations are for the candidate to
act on in the real world (get the cert, build the project), not instructions to
reword the resume.

The pre-computed ATS missing keywords are a strong signal — every entry there is
a candidate gap — but they are keyword-level and mechanical. Merge related ones,
drop any the resume actually demonstrates under different wording, and add gaps
the keyword scan could not see (missing seniority, domain, scope)."""


def gap_analysis_node(state) -> dict:
    jd = state.get("parsed_jd")
    resume = state.get("parsed_resume")
    pre_score = state.get("pre_score")

    missing = ""
    if pre_score is not None:
        missing = "\n".join(
            f"- {m.keyword}" + (" (must-have)" if m.is_must_have else "")
            for m in pre_score.missing_keywords
        )

    user = (
        f"JOB DESCRIPTION:\n{jd.model_dump_json(indent=2) if jd else '{}'}\n\n"
        f"RESUME:\n{resume.model_dump_json(indent=2) if resume else '{}'}\n\n"
        f"ATS KEYWORDS SCORED AS MISSING:\n{missing or '(none reported)'}"
    )

    return {
        "gap_analysis": structured(GapAnalysis).invoke(
            [SystemMessage(SYSTEM), HumanMessage(user)]
        )
    }
