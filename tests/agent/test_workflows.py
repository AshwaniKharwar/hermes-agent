"""Behavior contracts for executable skill workflow graphs."""

import json

import pytest

from agent.workflows import WorkflowError, _fingerprint, activate_workflow, execute_workflow, load_workflow, render_mermaid, save_overlay, validate_workflow, validate_workflow_contract


def _workflow():
    return {
        "schema": "hermes/workflow@2", "id": "default", "skill": "research",
        "entry": "collect", "max_transitions": 3,
        "input_schema": {"type": "object", "properties": {"request": {"type": "string"}},
                         "required": ["request"]},
        "steps": {
            "collect": {
                "kind": "task", "instructions": "Collect facts.", "input": {"request": "$input.request"},
                "effect": "read", "max_retries": 1, "recovery": {"method": "retry"},
                "output_schema": {"type": "object", "required": ["facts"]}, "next": "write",
            },
            "write": {
                "kind": "task", "instructions": "Write the answer.", "input": {"facts": "$steps.collect.output.facts"},
                "effect": "read", "max_retries": 1, "recovery": {"method": "retry"},
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


def test_workflow_rejects_unavailable_top_level_binding_before_running():
    graph = _workflow()
    graph["steps"]["write"]["input"]["previous_output"] = "$input.previous_output"
    with pytest.raises(WorkflowError, match="first visit"):
        execute_workflow(graph, {"request": "topic"}, lambda *_args: pytest.fail("step ran"))

    graph["steps"]["write"]["input"] = {"missing": "$input.missing"}
    with pytest.raises(WorkflowError, match="not required by input_schema"):
        validate_workflow(graph)


def test_workflow_precondition_prevents_a_side_effect_step_from_running():
    graph = _workflow()
    graph["steps"]["write"]["input"]["clear_to_send"] = "$steps.collect.output.facts"
    graph["steps"]["write"]["preconditions"] = {"clear_to_send": "clear"}
    calls = []

    with pytest.raises(WorkflowError, match="precondition"):
        execute_workflow(graph, {"request": "topic"}, lambda step_id, *_args:
                         calls.append(step_id) or {"facts": ["ambiguous"]})

    assert calls == ["collect"]


def test_workflow_precondition_rejects_binding_as_expected_value():
    graph = _workflow()
    graph["steps"]["write"]["preconditions"] = {"facts": "$steps.collect.output.facts"}
    with pytest.raises(WorkflowError, match="must compare a literal value"):
        validate_workflow(graph, reliable=True)


def test_reliable_goal_gate_cannot_be_bypassed_before_delivery():
    def step(kind="task", **fields):
        return {"kind": kind, "instructions": "Check.", "effect": "read",
                "max_retries": 1, "recovery": {"method": "retry"}, **fields}

    graph = {"schema": "hermes/workflow@2", "id": "default", "skill": "gate",
             "entry": "choose", "max_transitions": 5,
             "input_schema": {"type": "object", "required": ["target", "actual"],
                              "properties": {"target": {"type": "integer"}, "actual": {"type": "integer"}}},
             "steps": {
                 "choose": step("decision", branches={"bypass": "deliver", "checked": "gate"},
                                output_schema={"type": "object", "required": ["transition"],
                                               "properties": {"transition": {"enum": ["bypass", "checked"]}}}),
                 "gate": step("decision", branches={"met": "deliver", "short": "partial"},
                              output_schema={"type": "object", "required": ["transition"],
                                             "properties": {"transition": {"enum": ["met", "short"]}}}),
                 "partial": step(next="END", terminal_outcome="partial",
                                 input={"target": "$input.target", "actual": "$input.actual"},
                                 terminal_reason={"code": "goal_unmet", "evidence": ["target", "actual"]}),
                 "deliver": step(next="END", terminal_outcome="success"),
             },
             "goal_gate": {"gate_step": "gate", "target": "$input.target", "actual": "$input.actual",
                           "operator": "gte", "partial_step": "partial", "accept_step": "deliver",
                           "continue_step": "choose"}}
    with pytest.raises(WorkflowError, match="precede every path"):
        validate_workflow(graph, reliable=True)


def test_reliable_target_count_requires_a_goal_gate():
    graph = _workflow()
    graph["input_schema"]["properties"]["target_count"] = {"type": "integer"}
    graph["input_schema"]["required"].append("target_count")
    with pytest.raises(WorkflowError, match="requires goal_gate"):
        validate_workflow(graph, reliable=True)


def test_reliable_partial_requires_a_durable_approval_route():
    graph = _workflow()
    graph["steps"]["write"]["terminal_outcome"] = "partial"
    graph["steps"]["write"]["terminal_reason"] = {"code": "goal_unmet", "evidence": ["facts"]}
    with pytest.raises(WorkflowError, match="partial terminal requires goal_gate"):
        validate_workflow(graph, reliable=True)


def test_contract_rejects_merged_leadfinder_checkpoints(tmp_path):
    (tmp_path / "SKILL.md").write_text("""## Workflow contract
```yaml
entry: discover_candidates
steps:
  discover_candidates: {kind: task, next: qualify_candidates}
  qualify_candidates: {kind: task, next: draft_report}
  draft_report: {kind: task, next: create_docx}
  create_docx: {kind: task, next: validate_docx}
  validate_docx: {kind: task, next: check_sent_before}
  check_sent_before: {kind: task, next: send_mail}
  send_mail: {kind: task, next: check_sent_after, max_retries: 1}
  check_sent_after: {kind: task, next: END, terminal_outcome: success}
```
""")
    merged = {"entry": "discover_candidates", "steps": {
        "discover_candidates": {"kind": "task", "next": "draft_report"},
        "draft_report": {"kind": "task", "next": "create_docx"},
        "create_docx": {"kind": "task", "next": "send_mail"},
        "send_mail": {"kind": "task", "next": "END"},
    }}
    with pytest.raises(WorkflowError, match="missing .*qualify_candidates.*validate_docx"):
        validate_workflow_contract(merged, tmp_path)


def test_contract_requires_terminal_task_for_each_branch(tmp_path):
    (tmp_path / "SKILL.md").write_text("""## Workflow contract
```yaml
entry: choose
steps:
  choose: {kind: decision, branches: {ready: finish, blocked: END}}
  finish: {kind: task, next: END, terminal_outcome: success}
```
""")
    with pytest.raises(WorkflowError, match="terminal paths through a task"):
        validate_workflow_contract({}, tmp_path)


def test_contract_rejects_a_declared_zero_recovery_count():
    from agent.workflows import parse_workflow_contract

    content = """## Workflow contract
```yaml
entry: finish
steps:
  finish: {kind: task, next: END, terminal_outcome: success, max_retries: 0}
```
"""
    with pytest.raises(WorkflowError, match="max_retries must be between 1 and 5"):
        parse_workflow_contract(content)


def test_failed_attempts_follow_declared_error_path_after_limit():
    graph = {"schema": "hermes/workflow@2", "id": "default", "skill": "research",
             "entry": "collect", "max_transitions": 3, "steps": {
                 "collect": {"kind": "task", "instructions": "Collect.", "next": "deliver",
                             "max_retries": 1, "on_error": "blocked", "output_schema": {"type": "object"}},
                 "deliver": {"kind": "task", "instructions": "Deliver.", "next": "END"},
                 "blocked": {"kind": "task", "instructions": "Report blocker.", "next": "END",
                             "terminal_outcome": "blocked", "output_schema": {"type": "object"}},
             }, "output": {}}
    attempts, events = [], []

    def run_step(step_id, _step, context):
        attempts.append((step_id, context["attempt"], context["previous_error"]))
        if step_id == "collect":
            raise WorkflowError("No usable output")
        return {"reason": "Research unavailable"}

    result = execute_workflow(graph, {}, run_step, event_callback=lambda kind, data: events.append((kind, data)))
    assert [name for name, *_ in attempts] == ["collect", "collect", "blocked"]
    assert attempts[1][2] == "No usable output"
    assert result["outcome"] == "blocked"
    assert [data["attempt"] for kind, data in events if kind == "step_started" and data["step"] == "collect"] == [1, 2]
    assert [data.get("transition") for kind, data in events if kind == "step_failed"] == [None, "blocked"]


def test_side_effect_step_cannot_automatically_retry():
    graph = _workflow()
    graph["steps"]["collect"]["toolsets"] = ["terminal"]
    graph["steps"]["collect"]["max_retries"] = 1
    with pytest.raises(WorkflowError, match="possible side effects"):
        validate_workflow(graph)


def test_workflow_rejects_unsafe_bindings_and_dead_steps():
    unsafe = _workflow()
    unsafe["steps"]["collect"]["input"] = {"request": "${env.SECRET}"}
    with pytest.raises(WorkflowError, match="Unsafe binding"):
        validate_workflow(unsafe)

    dead = _workflow()
    dead["steps"]["unused"] = dict(dead["steps"]["write"])
    with pytest.raises(WorkflowError, match="Unreachable"):
        validate_workflow(dead)

    legacy = _workflow()
    legacy["schema"] = "hermes/workflow@1"
    with pytest.raises(WorkflowError, match="hermes/workflow@2"):
        validate_workflow(legacy)


def test_workflow_rejects_optional_and_branch_only_step_inputs():
    optional = _workflow()
    optional["steps"]["collect"]["output_schema"]["required"] = []
    with pytest.raises(WorkflowError, match="optional or undeclared"):
        validate_workflow(optional)

    branch_only = {
        "schema": "hermes/workflow@2", "id": "default", "skill": "research",
        "entry": "choose", "max_transitions": 4, "input_schema": {"type": "object"},
        "steps": {
            "choose": {
                "kind": "decision", "instructions": "Choose.",
                "branches": {"left": "collect", "right": "merge"},
                "output_schema": {"type": "object", "required": ["transition"],
                                  "properties": {"transition": {"enum": ["left", "right"]}}},
            },
            "collect": {"kind": "task", "instructions": "Collect.", "next": "merge",
                        "output_schema": {"type": "object", "required": ["facts"],
                                          "properties": {"facts": {"type": "array"}}}},
            "merge": {"kind": "task", "instructions": "Merge.", "next": "END",
                      "input": {"facts": "$steps.collect.output.facts"}},
        },
        "output": {},
    }
    with pytest.raises(WorkflowError, match="not guaranteed") as exc:
        validate_workflow(branch_only)
    assert "$steps.collect.output.facts" in str(exc.value)
    assert "choose --right--> merge" in str(exc.value)


def test_mail_paths_keep_distinct_blocked_evidence():
    graph = {"schema": "hermes/workflow@2", "id": "default", "skill": "mail",
             "entry": "check_before", "max_transitions": 8, "steps": {
                 "check_before": {"kind": "task", "instructions": "Check Sent.", "next": "decide_mail",
                                  "output_schema": {"type": "object", "required": ["reason"]}},
                 "decide_mail": {"kind": "decision", "instructions": "Decide.",
                                 "branches": {"blocked": "finish_precheck_blocked", "send_now": "send_mail"},
                                 "output_schema": {"type": "object", "required": ["transition"],
                                                   "properties": {"transition": {"enum": ["blocked", "send_now"]}}}},
                 "send_mail": {"kind": "task", "instructions": "Send once.", "next": "check_after",
                               "max_retries": 0},
                 "check_after": {"kind": "task", "instructions": "Verify Sent.", "next": "decide_after",
                                 "output_schema": {"type": "object", "required": ["reason"]}},
                 "decide_after": {"kind": "decision", "instructions": "Decide.",
                                  "branches": {"verified": "finish_sent", "unverified": "finish_unverified"},
                                  "output_schema": {"type": "object", "required": ["transition"],
                                                    "properties": {"transition": {"enum": ["verified", "unverified"]}}}},
                 "finish_precheck_blocked": {"kind": "task", "instructions": "Report precheck block.",
                                             "input": {"reason": "$steps.check_before.output.reason"},
                                             "next": "END", "terminal_outcome": "blocked"},
                 "finish_unverified": {"kind": "task", "instructions": "Report unverified send.",
                                       "input": {"reason": "$steps.check_after.output.reason"},
                                       "next": "END", "terminal_outcome": "blocked"},
                 "finish_sent": {"kind": "task", "instructions": "Report verified send.",
                                 "input": {"reason": "$steps.check_after.output.reason"},
                                 "next": "END", "terminal_outcome": "success"},
             }, "output": {}}
    validate_workflow(graph)

    for scenario, expected_steps, expected_outcome, expected_reason in (
        ("precheck", ["check_before", "decide_mail", "finish_precheck_blocked"], "blocked", "duplicate"),
        ("unverified", ["check_before", "decide_mail", "send_mail", "check_after",
                         "decide_after", "finish_unverified"], "blocked", "not in Sent"),
        ("sent", ["check_before", "decide_mail", "send_mail", "check_after",
                  "decide_after", "finish_sent"], "success", "matched in Sent"),
    ):
        visited = []

        def run_step(step_id, _step, context):
            visited.append(step_id)
            if step_id == "check_before":
                return {"reason": "duplicate"}
            if step_id == "decide_mail":
                return {"transition": "blocked" if scenario == "precheck" else "send_now"}
            if step_id == "check_after":
                return {"reason": "not in Sent" if scenario == "unverified" else "matched in Sent"}
            if step_id == "decide_after":
                return {"transition": "unverified" if scenario == "unverified" else "verified"}
            return {"reason": context["input"]["reason"]} if step_id.startswith("finish_") else {}

        result = execute_workflow(graph, {}, run_step)
        assert visited == expected_steps
        assert result["outcome"] == expected_outcome
        assert result["output"]["reason"] == expected_reason


def test_workflow_rejects_transition_budget_smaller_than_declared_graph_bounds():
    workflow = _workflow()
    workflow["max_transitions"] = 1

    with pytest.raises(WorkflowError, match="max_transitions must be at least 2"):
        validate_workflow(workflow)


def test_mermaid_contains_routes_and_model_labels():
    graph = _workflow()
    graph["steps"]["collect"]["route"] = {"model": "fast"}
    diagram = render_mermaid(graph)
    assert "collect" in diagram and "fast" in diagram and "END" in diagram


def test_loop_routes_retries_to_an_explicit_exhausted_path():
    workflow = {
        "schema": "hermes/workflow@2", "id": "default", "skill": "research", "entry": "collect", "max_transitions": 8,
        "steps": {
            "collect": {"kind": "task", "instructions": "Collect.", "next": "enough"},
            "enough": {
                "kind": "loop", "instructions": "Check whether enough facts exist.", "max_iterations": 2,
                "repeat": "retry", "on_exhausted": "shortfall",
                "output_schema": {"type": "object", "required": ["transition"], "properties": {"transition": {"enum": ["done", "retry"]}}},
                "branches": {"done": "deliver", "retry": "collect"},
            },
            "deliver": {"kind": "task", "instructions": "Deliver.", "next": "END"},
            "shortfall": {"kind": "task", "instructions": "Report shortfall.", "next": "END"},
        },
        "output": {},
    }
    calls = []
    contexts = []

    def run_step(step_id, _step, context):
        calls.append(step_id)
        contexts.append(context["input"])
        return {"transition": "retry"} if step_id == "enough" else {}

    result = execute_workflow(workflow, {}, run_step)

    assert calls == ["collect", "enough", "collect", "enough", "shortfall"]
    assert "previous_output" not in contexts[1]
    assert contexts[3]["previous_output"] == {"transition": "retry"}
    assert result["output"] == {}
    diagram = render_mermaid(workflow)
    assert "enough{" in diagram and "repeat_max_2" in diagram and "exhausted" in diagram


def test_empty_declared_output_returns_actual_terminal_step_output():
    workflow = _workflow()
    workflow["output"] = {}

    result = execute_workflow(
        workflow, {"request": "topic"},
        lambda step_id, _step, _context: {"facts": []} if step_id == "collect" else {"answer": "done"},
    )

    assert result["output"] == {"answer": "done"}


def test_final_output_uses_the_terminal_branch_when_other_branch_binding_did_not_run():
    workflow = {
        "schema": "hermes/workflow@2", "id": "default", "skill": "delivery",
        "entry": "choose", "max_transitions": 3, "input_schema": {"type": "object"},
        "steps": {
            "choose": {
                "kind": "decision", "instructions": "Choose a path.",
                "output_schema": {"type": "object", "required": ["transition"],
                                  "properties": {"transition": {"enum": ["sent", "blocked"]}}},
                "branches": {"sent": "verify", "blocked": "report"},
            },
            "verify": {
                "kind": "task", "instructions": "Verify delivery.", "next": "END",
                "output_schema": {"type": "object", "required": ["status"]},
            },
            "report": {
                "kind": "task", "instructions": "Report why delivery stopped.", "next": "END",
                "output_schema": {"type": "object", "required": ["status", "reason"]},
            },
        },
        "output": {"status": "$steps.verify.output.status"},
    }

    def run_step(step_id, _step, _context):
        if step_id == "choose":
            return {"transition": "blocked"}
        return {"status": "not_sent", "reason": "mailbox unavailable"}

    result = execute_workflow(workflow, {}, run_step)

    assert result["output"] == {"status": "not_sent", "reason": "mailbox unavailable"}


def test_workflow_fingerprint_includes_procedure_support_files(tmp_path):
    skill = tmp_path / "skill"
    scripts = skill / "scripts"
    scripts.mkdir(parents=True)
    (skill / "SKILL.md").write_text("skill", encoding="utf-8")
    (scripts / "run.py").write_text("first", encoding="utf-8")
    before = _fingerprint(skill)

    (scripts / "run.py").write_text("second", encoding="utf-8")

    assert _fingerprint(skill) != before


def test_candidate_overlay_never_shadows_packaged_workflow(tmp_path, monkeypatch):
    import agent.workflows as workflows
    import tools.skill_manager_tool as skill_manager
    skill_dir = tmp_path / "skill"
    (skill_dir / "workflows").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("demo", encoding="utf-8")
    packaged = _workflow()
    packaged["steps"]["collect"]["instructions"] = "Packaged"
    import yaml
    (skill_dir / "workflows" / "default.yaml").write_text(yaml.safe_dump(packaged), encoding="utf-8")
    monkeypatch.setattr(skill_manager, "_find_skill", lambda _name: {"path": skill_dir})
    monkeypatch.setattr(workflows, "get_hermes_home", lambda: tmp_path / "home")
    candidate = _workflow()
    candidate["steps"]["collect"]["instructions"] = "Candidate"
    save_overlay("research", candidate, activation="candidate")

    selected, source = load_workflow("research")

    assert source.source == "packaged"
    assert selected["steps"]["collect"]["instructions"] == "Packaged"
    activate_workflow("research")
    selected, source = load_workflow("research")
    assert source.source == "overlay"
    assert selected["steps"]["collect"]["instructions"] == "Candidate"


def test_packaged_contract_workflow_stales_after_skill_edit(tmp_path, monkeypatch):
    import yaml
    import agent.workflows as workflows
    import tools.skill_manager_tool as skill_manager

    skill_dir = tmp_path / "skill"
    (skill_dir / "workflows").mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text("""## Workflow contract
```yaml
entry: collect
steps:
  collect: {kind: task, next: write}
  write: {kind: task, next: END, terminal_outcome: success}
```
""", encoding="utf-8")
    packaged = _workflow()
    packaged["steps"]["write"]["terminal_outcome"] = "success"
    packaged["skill_fingerprint"] = _fingerprint(skill_dir)
    (skill_dir / "workflows" / "default.yaml").write_text(yaml.safe_dump(packaged), encoding="utf-8")
    monkeypatch.setattr(skill_manager, "_find_skill", lambda _name: {"path": skill_dir})
    monkeypatch.setattr(workflows, "get_hermes_home", lambda: tmp_path / "home")

    assert load_workflow("research")[1].source == "packaged"
    skill_file.write_text(skill_file.read_text(encoding="utf-8") + "\nNew qualification instruction.\n",
                          encoding="utf-8")

    with pytest.raises(WorkflowError, match="stale"):
        load_workflow("research")


def test_overlay_selection_remains_profile_scoped_in_a_b_a_order(tmp_path, monkeypatch):
    import agent.workflows as workflows
    import tools.skill_manager_tool as skill_manager

    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("""## Workflow contract
```yaml
entry: collect
steps:
  collect: {kind: task, next: write}
  write: {kind: task, next: END, terminal_outcome: success}
```
""", encoding="utf-8")
    homes = {"a": tmp_path / "a", "b": tmp_path / "b"}
    current = ["a"]
    monkeypatch.setattr(skill_manager, "_find_skill", lambda _name: {"path": skill_dir})
    monkeypatch.setattr(workflows, "get_hermes_home", lambda: homes[current[0]])

    workflow_a = _workflow()
    workflow_a["steps"]["collect"]["instructions"] = "Profile A"
    workflow_a["steps"]["write"]["terminal_outcome"] = "success"
    save_overlay("research", workflow_a)
    current[0] = "b"
    assert workflows.workflow_sources("research") == []
    workflow_b = _workflow()
    workflow_b["steps"]["collect"]["instructions"] = "Profile B"
    workflow_b["steps"]["write"]["terminal_outcome"] = "success"
    save_overlay("research", workflow_b)
    current[0] = "a"

    selected, _source = load_workflow("research")
    assert selected["steps"]["collect"]["instructions"] == "Profile A"
