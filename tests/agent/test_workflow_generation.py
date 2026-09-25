"""Background workflow compilation is single-flight and ownership-aware."""

import json
import os
import time
from datetime import datetime, timezone


def test_enqueue_is_single_flight_and_external_generation_stays_candidate(tmp_path, monkeypatch):
    import agent.workflow_generation as generation
    import agent.workflows as workflows
    import hermes_cli.workflow_cmd as workflow_cmd
    import tools.skill_usage as usage

    generation._ACTIVE.clear()
    submitted, records, generated = [], [], []
    monkeypatch.setattr(generation, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(workflows, "workflow_source_fingerprint", lambda _skill: "fingerprint")
    monkeypatch.setattr(generation, "_in_failure_backoff", lambda *_args: False)
    monkeypatch.setattr(generation, "_lock_path", lambda *_args: tmp_path / "job.lock")
    monkeypatch.setattr(generation, "_claim", lambda path, job: path.write_text(json.dumps({"job_id": job})) is not None)
    monkeypatch.setattr(generation._POOL, "submit", lambda fn: submitted.append(fn))
    monkeypatch.setattr(usage, "record_workflow_generation", lambda skill, **data: records.append((skill, data)))
    monkeypatch.setattr(usage, "workflow_eligibility", lambda _skill: "external")
    monkeypatch.setattr(workflow_cmd, "_generate", lambda skill, name, **kwargs:
                        generated.append((skill, name, kwargs)) or {"path": "candidate.yaml"})

    first = generation.enqueue_workflow_generation("demo")
    second = generation.enqueue_workflow_generation("demo")
    submitted[0]()

    assert first == second and len(submitted) == 1
    assert generated[0][2]["activation"] == "candidate"
    assert records[-1][1]["status"] == "candidate"


def test_generation_lock_reclaims_abandoned_owner_and_failure_backoff(monkeypatch, tmp_path):
    import agent.workflow_generation as generation
    import tools.skill_usage as usage

    lock = tmp_path / "generation.lock"
    lock.write_text("{}", encoding="utf-8")
    old = time.time() - generation._ABANDONED_SECONDS - 1
    os.utime(lock, (old, old))
    assert generation._claim(lock, "replacement")
    assert json.loads(lock.read_text(encoding="utf-8"))["job_id"] == "replacement"

    monkeypatch.setattr(usage, "load_usage", lambda: {"demo": {"workflow_generation": {
        "status": "failed", "source_fingerprint": "same",
        "updated_at": datetime.now(timezone.utc).isoformat()}}})
    assert generation._in_failure_backoff("demo", "same")
    assert not generation._in_failure_backoff("demo", "changed")


def test_generation_single_flight_is_profile_scoped_in_a_b_a_order(tmp_path, monkeypatch):
    import agent.workflow_generation as generation
    import agent.workflows as workflows
    import tools.skill_usage as usage

    generation._ACTIVE.clear()
    homes = {"a": tmp_path / "a", "b": tmp_path / "b"}
    current = ["a"]
    submitted = []
    monkeypatch.setattr(generation, "get_hermes_home", lambda: homes[current[0]])
    monkeypatch.setattr(workflows, "workflow_source_fingerprint", lambda _skill: "fingerprint")
    monkeypatch.setattr(generation, "_in_failure_backoff", lambda *_args: False)
    monkeypatch.setattr(generation._POOL, "submit", lambda fn: submitted.append(fn))
    monkeypatch.setattr(usage, "record_workflow_generation", lambda *_args, **_kwargs: None)

    job_a = generation.enqueue_workflow_generation("demo")
    current[0] = "b"
    job_b = generation.enqueue_workflow_generation("demo")
    current[0] = "a"

    assert generation.enqueue_workflow_generation("demo") == job_a
    assert job_a != job_b and len(submitted) == 2
