"""
The evaluation gate in CI.

Every pull request replays the committed recordings through the real driver against a
throwaway Postgres, scores them and runs `python -m evals gate`. CI has no model and needs
none: nothing is installed but the Python dependencies, no secret is used, and the job can
only read the repository. A pull request that makes the agent worse, or unsafe, or that
changes a prompt without recording again, fails this job.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def evals_job() -> dict:
    return workflow()["jobs"]["evals"]


def steps_text() -> str:
    return "\n".join(str(step.get("run", "")) for step in evals_job()["steps"])


def test_there_is_an_evals_job_that_runs_the_gate():
    assert "python -m evals gate" in steps_text()


def test_it_runs_on_every_pull_request_and_push_to_main():
    triggers = workflow()[True]  # YAML reads the bare key `on` as true
    assert "pull_request" in triggers
    assert triggers["push"]["branches"] == ["main"]


def test_it_has_a_postgres_with_pgvector_and_points_the_gate_at_it():
    job = evals_job()

    assert job["services"]["db"]["image"] == "pgvector/pgvector:pg16"
    assert job["env"]["OPSAGENT_EVAL_ADMIN_URL"] == "postgresql://opsagent:dev@localhost:5432/postgres"


def test_it_has_a_time_limit():
    assert 0 < evals_job()["timeout-minutes"] <= 30


def test_it_needs_no_model_and_no_secret():
    text = WORKFLOW.read_text()
    job_text = yaml.safe_dump(evals_job())

    assert "ollama" not in job_text.lower()
    assert "secrets." not in job_text
    assert "pull_request_target" not in text


def test_the_workflow_can_only_read_the_repository():
    assert workflow()["permissions"] == {"contents": "read"}
    assert "permissions" not in evals_job() or evals_job()["permissions"] == {"contents": "read"}
