"""Behavior contracts for executable skill workflow graphs."""

import json

import pytest

from agent.workflows import WorkflowError, _fingerprint, execute_workflow, render_mermaid, validate_workflow


def _workflow():
    return {
        "schema": "hermes/workflow@1", "id": "default", "skill": "research",
        "entry": "collect", "max_transitions": 3,
        "input_schema": {"type": "object"},
        "steps": {
            "collect": {
                "instructions": "Collect facts.", "input": {"request": "$input.request"},
                "output_schema": {"type": "object", "required": ["facts"]}, "next": "write",
            },
            "write": {
                "instructions": "Write the answer.", "input": {"facts": "$steps.collect.output.facts"},
                "output_schema": {"type": "object", "required": ["answer"]}, "next": "END",
            },
        },
        "output": {"answer": "$steps.write.output.answer"},
    }


def test_workflow_hands_structured_output_between_steps_and_writes_receipts(tmp_path):
    calls = []

    def run_step(step_id, _step, context):
        calls.append((step_id, context["input"]))
        return {"facts": ["one"]} if step_id == "collect" else {"answer": "done"}

    receipt = tmp_path / "run.jsonl"
    result = execute_workflow(_workflow(), {"request": "topic"}, run_step, receipt_path=receipt)

    assert calls == [("collect", {"request": "topic"}), ("write", {"facts": ["one"]})]
    assert result["output"] == {"answer": "done"}
    events = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [event["step"] for event in events] == ["collect", "write"]


def test_workflow_rejects_unsafe_bindings_and_dead_steps():
    unsafe = _workflow()
    unsafe["steps"]["collect"]["input"] = {"request": "${env.SECRET}"}
    with pytest.raises(WorkflowError, match="Unsafe binding"):
        validate_workflow(unsafe)

    dead = _workflow()
    dead["steps"]["unused"] = dict(dead["steps"]["write"])
    with pytest.raises(WorkflowError, match="Unreachable"):
        validate_workflow(dead)


def test_mermaid_contains_routes_and_model_labels():
    graph = _workflow()
    graph["steps"]["collect"]["route"] = {"model": "fast"}
    diagram = render_mermaid(graph)
    assert "collect" in diagram and "fast" in diagram and "END" in diagram


def test_workflow_fingerprint_includes_procedure_support_files(tmp_path):
    skill = tmp_path / "skill"
    scripts = skill / "scripts"
    scripts.mkdir(parents=True)
    (skill / "SKILL.md").write_text("skill", encoding="utf-8")
    (scripts / "run.py").write_text("first", encoding="utf-8")
    before = _fingerprint(skill)

    (scripts / "run.py").write_text("second", encoding="utf-8")

    assert _fingerprint(skill) != before
