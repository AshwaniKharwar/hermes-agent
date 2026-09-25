"""Durable partial and blocked workflows resume from their saved graph and outputs."""

import json
import os
import time
from copy import deepcopy
from types import SimpleNamespace

import yaml

from agent.workflow_receipts import inspect_workflow_run
from agent.workflows import save_overlay


def _step(kind="task", **fields):
    return {"kind": kind, "instructions": "Execute the declared step.", "effect": "read",
            "max_retries": 1, "recovery": {"method": "retry"}, **fields}


def _install(tmp_path, monkeypatch, graph):
    import agent.workflow_receipts as receipts
    import agent.workflows as workflows
    import tools.skill_manager_tool as manager

    skill_dir = tmp_path / "skills" / graph["skill"]
    skill_dir.mkdir(parents=True)
    contract = {"entry": graph["entry"], "steps": {},
                **({"goal_gate": graph["goal_gate"]} if graph.get("goal_gate") else {})}
    for name, step in graph["steps"].items():
        contract["steps"][name] = {key: step[key] for key in (
            "kind", "next", "branches", "repeat", "max_iterations", "on_exhausted",
            "terminal_outcome", "resume_step", "on_error") if key in step}
    (skill_dir / "SKILL.md").write_text(
        "## Workflow contract\n```yaml\n" + yaml.safe_dump(contract) + "```\n", encoding="utf-8")
    monkeypatch.setattr(manager, "_find_skill", lambda _name: {"path": skill_dir})
    monkeypatch.setattr(workflows, "get_hermes_home", lambda: tmp_path / "home")
    monkeypatch.setattr(receipts, "get_hermes_home", lambda: tmp_path / "home")
    save_overlay(graph["skill"], graph)


def _child(monkeypatch, answer):
    import run_agent

    class Child:
        provider, model = "test", "test"
        def __init__(self, **_kwargs):
            pass
        def run_conversation(self, request):
            return {"final_response": json.dumps(answer(json.loads(request))), "messages": []}
        def close(self):
            pass

    monkeypatch.setattr(run_agent, "AIAgent", Child)


def test_reliable_write_graph_executes_with_file_toolset():
    from agent.workflows import execute_workflow

    graph = {"schema": "hermes/workflow@2", "id": "default", "skill": "demo",
             "entry": "write", "max_transitions": 2,
             "steps": {"write": _step(next="END", terminal_outcome="success",
                                      effect="write", toolsets=["file"], max_retries=1,
                                      recovery={"method": "reconcile", "instructions": "Check the operation ID."})}}
    result = execute_workflow(graph, {}, lambda _name, _step, _context: {"saved": True})
    assert result["outcome"] == "success"
    assert result["output"] == {"saved": True}


def test_shortfall_pauses_and_accept_or_continue_resumes_from_checkpoint(tmp_path, monkeypatch):
    from agent.workflow_runner import resume_workflow, run_workflow

    graph = {"schema": "hermes/workflow@2", "id": "default", "skill": "demo", "entry": "scope",
             "max_transitions": 8, "input_schema": {"type": "object", "required": ["request"],
             "properties": {"request": {"type": "string"}}},
             "steps": {
                 "scope": _step(next="collect", output_schema={"type": "object", "required": ["target"]}),
                 "collect": _step(next="gate", output_schema={"type": "object", "required": ["count"]}),
                 "gate": _step("decision", branches={"enough": "deliver", "short": "partial"},
                               output_schema={"type": "object", "required": ["transition"],
                                              "properties": {"transition": {"enum": ["enough", "short"]}}}),
                 "partial": _step(next="END", terminal_outcome="partial",
                                  input={"target": "$steps.scope.output.target",
                                         "count": "$steps.collect.output.count"},
                                  terminal_reason={"code": "goal_unmet", "evidence": ["target", "count"]}),
                 "deliver": _step(next="END", terminal_outcome="success",
                                  effect="external_action", max_retries=1,
                                  recovery={"method": "reconcile", "instructions": "Check operation ID."},
                                  output_schema={"type": "object", "required": ["sent"]}),
             },
             "goal_gate": {"gate_step": "gate", "target": "$steps.scope.output.target",
                           "actual": "$steps.collect.output.count", "operator": "gte",
                           "partial_step": "partial", "accept_step": "deliver",
                           "continue_step": "collect"}}
    _install(tmp_path, monkeypatch, graph)
    calls, counts = [], iter([15, 15, 20, 15])

    def answer(request):
        calls.append(request["step"])
        return {"scope": {"target": 20}, "collect": lambda: {"count": next(counts)},
                "gate": {"transition": "enough"}, "deliver": {"sent": True}}[request["step"]]() if request["step"] == "collect" else {
                    "scope": {"target": 20}, "gate": {"transition": "enough"},
                    "deliver": {"sent": True}}[request["step"]]

    _child(monkeypatch, answer)
    first = run_workflow("demo", "default", {"request": "20"})
    assert first["status"] == "paused" and first["outcome"] == "partial"
    paused_receipt = inspect_workflow_run(first["run_id"])
    assert paused_receipt["pending_action"]["code"] == "goal_unmet"
    assert paused_receipt["reason_code"] == "goal_unmet"
    assert paused_receipt["observed"] == {"target": 20, "count": 15}
    regenerated = deepcopy(graph)
    regenerated["steps"]["deliver"]["instructions"] = "Newly generated delivery procedure."
    save_overlay("demo", regenerated)
    accepted = resume_workflow(first["run_id"], "accept")
    assert accepted["outcome"] == "partial"
    assert accepted["source"] == "snapshot"
    assert calls == ["scope", "collect", "deliver"]

    second = run_workflow("demo", "default", {"request": "20"})
    continued = resume_workflow(second["run_id"], "continue")
    assert continued["outcome"] == "success"
    assert calls[-4:] == ["scope", "collect", "collect", "deliver"] or calls[-5:] == [
        "scope", "collect", "collect", "gate", "deliver"]

    third = run_workflow("demo", "default", {"request": "20"})
    stopped = resume_workflow(third["run_id"], "stop")
    assert stopped["outcome"] == "partial"
    assert inspect_workflow_run(third["run_id"])["status"] == "conformant"
    assert calls.count("deliver") == 2


def test_unreadable_external_check_repair_resumes_without_repeating_send(tmp_path, monkeypatch):
    from agent.workflow_runner import resume_workflow, run_workflow

    graph = {"schema": "hermes/workflow@2", "id": "default", "skill": "mail", "entry": "check",
             "max_transitions": 6, "steps": {
                 "check": _step(next="decide", output_schema={"type": "object", "required": ["state"],
                                       "properties": {"state": {"enum": ["clear", "duplicate", "unreadable"]}}}),
                 "decide": _step("decision", input={"state": "$steps.check.output.state"},
                                 branches={"send": "send", "blocked": "blocked"},
                                 output_schema={"type": "object", "required": ["transition"],
                                                "properties": {"transition": {"enum": ["send", "blocked"]}}}),
                 "blocked": _step(next="END", terminal_outcome="blocked", resume_step="check",
                                  input={"state": "$steps.check.output.state"},
                                  terminal_reason={"code": "external_unreadable", "evidence": ["state"],
                                                   "action": "Repair mailbox access."}),
                 "send": _step(next="verify", effect="external_action", max_retries=1,
                               recovery={"method": "reconcile", "instructions": "Check operation ID."},
                               output_schema={"type": "object", "required": ["sent"]}),
                 "verify": _step(next="done", output_schema={"type": "object", "required": ["verified"]}),
                 "done": _step(next="END", terminal_outcome="success",
                               input={"verified": "$steps.verify.output.verified"}),
             }}
    _install(tmp_path, monkeypatch, graph)
    calls, reads = [], iter(["unreadable", "clear"])

    def answer(request):
        calls.append(request["step"])
        if request["step"] == "check":
            return {"state": next(reads)}
        return {"decide": {"transition": "blocked" if calls.count("check") == 1 else "send"},
                "send": {"sent": True}, "verify": {"verified": True}, "done": {}}[request["step"]]

    _child(monkeypatch, answer)
    first = run_workflow("mail", "default", {})
    assert first["status"] == "paused" and first["outcome"] == "blocked"
    final = resume_workflow(first["run_id"], "retry")
    assert final["status"] == "conformant" and final["outcome"] == "success"
    assert calls.count("send") == 1


def test_unknown_write_state_pauses_before_a_second_write(tmp_path, monkeypatch):
    from agent.workflow_runner import resume_workflow, run_workflow

    graph = {"schema": "hermes/workflow@2", "id": "default", "skill": "write", "entry": "create",
             "max_transitions": 2, "steps": {
                 "create": _step(next="END", terminal_outcome="success", effect="write", max_retries=1,
                                 recovery={"method": "reconcile", "instructions": "Check operation ID."},
                                 output_schema={"type": "object", "required": ["created"]}),
             }}
    _install(tmp_path, monkeypatch, graph)
    calls, repaired = [], [False]

    def answer(request):
        calls.append((request["step"], request["reconcile"], request["operation_id"]))
        if not request["reconcile"]:
            return {"wrong": True}  # Invalid output; the first write may still have happened.
        return {"state": "done", "output": {"created": True}} if repaired[0] else {"state": "unknown"}

    _child(monkeypatch, answer)
    result = run_workflow("write", "default", {})
    assert result["status"] == "paused"
    assert result["pending_action"]["code"] == "side_effect_unknown"
    assert [reconciling for _, reconciling, _ in calls] == [False, True]
    assert calls[0][2] == calls[1][2]
    repaired[0] = True
    resumed = resume_workflow(result["run_id"], "retry")
    assert resumed["outcome"] == "success"
    assert [reconciling for _, reconciling, _ in calls] == [False, True, True]
    assert len({operation_id for _, _, operation_id in calls}) == 1


def test_receipts_and_overlays_stay_in_the_selected_profile_a_b_a(tmp_path, monkeypatch):
    from pathlib import Path
    from agent import workflow_receipts, workflows
    from agent.workflow_runner import run_workflow
    from hermes_constants import get_hermes_home

    graph = {"schema": "hermes/workflow@2", "id": "default", "skill": "profiles", "entry": "finish",
             "max_transitions": 1, "steps": {
                 "finish": _step(next="END", terminal_outcome="success",
                                 output_schema={"type": "object", "required": ["profile"]}),
             }}
    _install(tmp_path, monkeypatch, graph)
    monkeypatch.setattr(workflows, "get_hermes_home", get_hermes_home)
    monkeypatch.setattr(workflow_receipts, "get_hermes_home", get_hermes_home)
    _child(monkeypatch, lambda _request: {"profile": Path(os.environ["HERMES_HOME"]).name})
    runs = []
    for name in ("A", "B", "A"):
        home = tmp_path / name
        monkeypatch.setenv("HERMES_HOME", str(home))
        save_overlay("profiles", graph)
        result = run_workflow("profiles", "default", {})
        runs.append(result)
        assert result["output"]["profile"] == name
        assert Path(result["receipt"]).is_relative_to(home)
    assert runs[0]["run_id"] != runs[2]["run_id"]


def test_parent_activity_tracks_child_progress_but_not_silence(tmp_path, monkeypatch):
    import run_agent
    from agent import workflow_runner

    graph = {"schema": "hermes/workflow@2", "id": "default", "skill": "activity", "entry": "finish",
             "max_transitions": 1, "steps": {
                 "finish": _step(next="END", terminal_outcome="success"),
             }}
    _install(tmp_path, monkeypatch, graph)
    monkeypatch.setattr(workflow_runner, "_CHILD_ACTIVITY_POLL_SECONDS", 0.005)
    mode = ["progress"]

    class Child:
        provider, model = "test", "test"
        def __init__(self, **_kwargs):
            self._last_activity_ts = 1
        def run_conversation(self, _request):
            if mode[0] == "progress":
                self._last_activity_ts = 2
            time.sleep(0.05)
            return {"final_response": "{}", "messages": []}
        def close(self):
            pass

    monkeypatch.setattr(run_agent, "AIAgent", Child)
    touches = []
    parent = SimpleNamespace(session_id="same", provider="test", model="test",
                             _touch_activity=touches.append)
    workflow_runner.run_workflow("activity", "default", {}, parent_agent=parent)
    assert any("progressing" in touch for touch in touches)
    touches.clear()
    mode[0] = "silent"
    workflow_runner.run_workflow("activity", "default", {}, parent_agent=parent)
    assert not any("progressing" in touch for touch in touches)


def test_exhausted_read_recovery_records_typed_block(tmp_path, monkeypatch):
    from agent.workflow_runner import run_workflow

    graph = {"schema": "hermes/workflow@2", "id": "default", "skill": "recover", "entry": "read",
             "max_transitions": 3, "input_schema": {"type": "object", "required": ["request"],
             "properties": {"request": {"type": "string"}}},
             "steps": {
                 "read": _step(next="done", on_error="blocked",
                               output_schema={"type": "object", "required": ["value"]}),
                 "done": _step(next="END", terminal_outcome="success"),
                 "blocked": _step(next="END", terminal_outcome="blocked",
                                  input={"request": "$input.request"},
                                  terminal_reason={"code": "recovery_exhausted", "evidence": ["request"]}),
             }}
    _install(tmp_path, monkeypatch, graph)
    calls = []

    def answer(request):
        calls.append(request["step"])
        return {}  # Read output is invalid on both attempts.

    _child(monkeypatch, answer)
    result = run_workflow("recover", "default", {"request": "find"})
    assert result["outcome"] == "blocked"
    assert calls == ["read", "read"]
    receipt = inspect_workflow_run(result["run_id"])
    assert receipt["reason_code"] == "recovery_exhausted"
    assert receipt["output"]["evidence"] == {"request": "find"}
