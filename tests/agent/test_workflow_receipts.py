"""Auditable workflow runs keep the event ledger canonical and child attempts singular."""

import json
import os
from types import SimpleNamespace

import pytest

from agent.workflow_receipts import WorkflowReceipt, fallback_receipt, inspect_workflow_run, list_workflow_runs, prune_workflow_runs
from agent.workflows import WorkflowError


def _workflow():
    return {
        "schema": "hermes/workflow@2", "id": "default", "skill": "demo", "entry": "answer",
        "max_transitions": 2, "input_schema": {"type": "object", "required": ["request"]},
        "steps": {"answer": {"kind": "task", "instructions": "Answer.",
                  "output_schema": {"type": "object", "required": ["answer"]}, "next": "END"}},
        "output": {"answer": "$steps.answer.output.answer"},
    }


def test_receipt_replays_redacted_ledger_with_secure_permissions(tmp_path, monkeypatch):
    import agent.workflow_receipts as receipts
    monkeypatch.setattr(receipts, "get_hermes_home", lambda: tmp_path)
    run = WorkflowReceipt(_workflow(), {"request": "use api_key=sk_test_12345678901234567890"}, skill="demo")
    run.event("step_started", {"step": "answer", "input": {"token": "secret-value"}})
    run.event("step_completed", {"step": "answer", "transition": "END", "output": {"answer": "done"},
                                     "execution": {"toolsets": ["safe"], "used_tools": []}})
    run.finish("conformant", output={"answer": "done"})

    replayed = inspect_workflow_run(run.run_id)
    ledger = run.events_path.read_text(encoding="utf-8")
    assert replayed["status"] == "conformant"
    assert replayed["control_flow"] == [{"step": "answer", "state": "completed", "transition": "END"}]
    assert "sk_test_" not in ledger and "secret-value" not in ledger
    assert os.stat(run.path).st_mode & 0o777 == 0o700
    assert os.stat(run.events_path).st_mode & 0o777 == 0o600


def test_terminal_receipt_closes_an_active_step(tmp_path, monkeypatch):
    import agent.workflow_receipts as receipts
    monkeypatch.setattr(receipts, "get_hermes_home", lambda: tmp_path)
    run = WorkflowReceipt(_workflow(), {"request": "hello"}, skill="demo")
    run.event("step_started", {"step": "answer", "attempt": 1})
    with pytest.raises(ValueError, match="running step"):
        run.finish("conformant", output={"answer": "incorrect"})
    run.finish("cancelled", error="interrupted")
    inspected = inspect_workflow_run(run.run_id)
    assert inspected["status"] == "cancelled"
    assert inspected["current_step"] is None
    assert inspected["control_flow"][-1]["state"] == "cancelled"


def test_large_partial_checkpoint_survives_bounded_ledger(tmp_path, monkeypatch):
    import agent.workflow_receipts as receipts
    monkeypatch.setattr(receipts, "get_hermes_home", lambda: tmp_path)
    run = WorkflowReceipt(_workflow(), {"request": "many results"}, skill="demo")
    action = {"outcome": "partial", "code": "goal_unmet", "choices": {"accept": "answer"},
              "output": {"evidence": {"rows": "x" * 300_000}}}
    run.checkpoint({"state": {"input": {"request": "many results"}, "steps": {}},
                    "current": "answer", "pending": action})
    run.pause("partial", action)

    inspected = inspect_workflow_run(run.run_id)
    assert inspected["status"] == "paused"
    assert inspected["pending_action"]["choices"] == {"accept": "answer"}
    assert json.loads((run.path / "checkpoint.json").read_text())["pending"]["output"]["evidence"]["rows"] == "x" * 300_000
    assert max(len(line.encode()) for line in run.events_path.read_text().splitlines()) <= 256 * 1024


def test_invalid_step_output_fails_without_retry_or_prose_fallback(tmp_path, monkeypatch):
    import agent.workflow_receipts as receipts
    import agent.workflow_runner as runner
    import run_agent

    monkeypatch.setattr(receipts, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(runner, "load_workflow", lambda *_args: (_workflow(), type("Source", (), {"source": "packaged"})()))
    calls = []

    class Child:
        provider = "test"
        model = "test"
        def __init__(self, **_kwargs):
            calls.append("created")
        def run_conversation(self, _request):
            return {"final_response": json.dumps({"wrong": True}), "messages": []}
        def get_activity_summary(self):
            return {}
        def close(self):
            pass

    monkeypatch.setattr(run_agent, "AIAgent", Child)
    with pytest.raises(WorkflowError, match="does not satisfy"):
        runner.run_workflow("demo", "default", {"request": "hello"})

    assert calls == ["created"]
    summary = next((tmp_path / "workflow-runs").glob("*/*/run.json"))
    assert json.loads(summary.read_text(encoding="utf-8"))["status"] == "failed"


def test_runner_retries_invalid_step_output_before_advancing(tmp_path, monkeypatch):
    import agent.workflow_receipts as receipts
    import agent.workflow_runner as runner
    import run_agent
    monkeypatch.setattr(receipts, "get_hermes_home", lambda: tmp_path)
    graph = _workflow()
    graph["steps"]["answer"]["max_retries"] = 1
    monkeypatch.setattr(runner, "load_workflow", lambda *_args: (graph, SimpleNamespace(source="packaged")))
    requests = []

    class Child:
        provider, model = "test", "test"
        def __init__(self, **_kwargs):
            pass
        def run_conversation(self, request):
            requests.append(json.loads(request))
            return {"final_response": '{"wrong":true}' if len(requests) == 1 else '{"answer":"done"}',
                    "messages": []}
        def close(self):
            pass

    monkeypatch.setattr(run_agent, "AIAgent", Child)
    result = runner.run_workflow("demo", "default", {"request": "hello"}, retry_of="a" * 32)
    inspected = inspect_workflow_run(result["run_id"])
    assert len(requests) == 2
    assert "does not satisfy" in requests[1]["previous_attempt_error"]
    assert [row["state"] for row in inspected["control_flow"]] == ["failed", "completed"]
    assert inspected["outcome"] == "success"
    assert inspected["retry_of"] == "a" * 32


def test_receipts_remain_profile_scoped_in_a_b_a_order(tmp_path, monkeypatch):
    import agent.workflow_receipts as receipts
    homes = {"a": tmp_path / "a", "b": tmp_path / "b"}
    current = ["a"]
    monkeypatch.setattr(receipts, "get_hermes_home", lambda: homes[current[0]])

    fallback_receipt("alpha", {}, reason="missing", generation_job_id="job-a1")
    current[0] = "b"
    fallback_receipt("beta", {}, reason="missing", generation_job_id="job-b")
    current[0] = "a"
    fallback_receipt("alpha", {}, reason="stale", generation_job_id="job-a2")

    assert [run["skill"] for run in list_workflow_runs()] == ["alpha", "alpha"]
    current[0] = "b"
    assert [run["skill"] for run in list_workflow_runs()] == ["beta"]


def test_runner_records_observed_tools_and_parent_accounting(tmp_path, monkeypatch):
    import agent.workflow_receipts as receipts
    import agent.workflow_runner as runner
    import run_agent
    monkeypatch.setattr(receipts, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(runner, "load_workflow", lambda *_args: (_workflow(), type("Source", (), {"source": "packaged"})()))
    accounting = []

    class Child:
        provider, model, base_url = "test", "model", "https://example.test"
        session_input_tokens, session_output_tokens, session_api_calls = 10, 4, 1
        session_cache_read_tokens = session_cache_write_tokens = session_reasoning_tokens = 0
        session_estimated_cost_usd = 0.01
        def __init__(self, **_kwargs):
            pass
        def run_conversation(self, _request):
            return {"final_response": '{"answer":"done"}', "messages": [
                {"role": "assistant", "tool_calls": [{"function": {"name": "read_file"}}]}]}
        def get_activity_summary(self):
            return {}
        def close(self):
            pass

    monkeypatch.setattr(run_agent, "AIAgent", Child)
    db = SimpleNamespace(record_auxiliary_usage=lambda *args, **kwargs: accounting.append((args, kwargs)))
    parent = SimpleNamespace(provider="test", model="model", session_id="parent", _session_db=db,
                             request_overrides={}, status_callback=None, event_callback=None)
    result = runner.run_workflow("demo", "default", {"request": "hello"}, parent_agent=parent)
    inspected = inspect_workflow_run(result["run_id"])

    assert inspected["tool_policy"]["answer"]["used"] == ["read_file"]
    assert accounting[0][1]["task"] == "skill_workflow"
    assert accounting[0][1]["input_tokens"] == 10


def test_event_size_cap_and_retention_preserve_incomplete_runs(tmp_path, monkeypatch):
    import agent.workflow_receipts as receipts
    monkeypatch.setattr(receipts, "get_hermes_home", lambda: tmp_path)
    oversized = WorkflowReceipt(_workflow(), {}, skill="demo")
    oversized.event("step_started", {"step": "answer", "input": {"body": "x" * 300_000}})
    event = json.loads(oversized.events_path.read_text(encoding="utf-8").splitlines()[-1])
    assert len(oversized.events_path.read_bytes().splitlines()[-1]) <= receipts._EVENT_LIMIT
    assert event["omitted_payload"]["omitted"] is True

    old = fallback_receipt("old", {}, reason="missing", generation_job_id=None)
    summary_path = next((tmp_path / "workflow-runs").glob(f"*/{old['run_id']}/run.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["started_at"] = "2000-01-01T00:00:00+00:00"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    prune_workflow_runs(retention_days=30, max_runs=0)

    assert not summary_path.parent.exists()
    assert oversized.path.exists()  # incomplete runs are never pruned
