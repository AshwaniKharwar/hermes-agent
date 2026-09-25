"""Bulk workflow generation chooses background-created skills by provenance."""

import json
from types import SimpleNamespace

import pytest

from agent.workflows import WorkflowError


def test_generator_retries_contract_mismatch_before_activating(tmp_path, monkeypatch):
    from hermes_cli import workflow_cmd
    from agent import workflows
    from agent import auxiliary_client
    from tools import skill_manager_tool, skill_usage

    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("""## Workflow contract
```yaml
entry: discover_candidates
steps:
  discover_candidates: {kind: task, next: qualify_candidates}
  qualify_candidates: {kind: task, next: END, terminal_outcome: success}
```
""")
    monkeypatch.setattr(skill_manager_tool, "_find_skill", lambda _name: {"path": skill_dir})
    monkeypatch.setattr(skill_usage, "workflow_eligibility", lambda _name: "eligible")
    monkeypatch.setattr(workflows, "get_hermes_home", lambda: tmp_path / "home")
    bad = {"schema": "hermes/workflow@2", "id": "default", "skill": "demo",
           "entry": "research", "max_transitions": 2, "steps": {
               "research": {"kind": "task", "instructions": "Research.", "next": "END",
                            "effect": "read", "max_retries": 1, "recovery": {"method": "retry"}}}}
    good = {"schema": "hermes/workflow@2", "id": "default", "skill": "demo",
            "entry": "discover_candidates", "max_transitions": 2, "steps": {
                "discover_candidates": {"kind": "task", "instructions": "Discover.", "next": "qualify_candidates",
                                        "effect": "read", "max_retries": 1, "recovery": {"method": "retry"}},
                "qualify_candidates": {"kind": "task", "instructions": "Qualify.", "next": "END",
                                       "terminal_outcome": "success", "effect": "read",
                                       "max_retries": 1, "recovery": {"method": "retry"}}}}
    responses = iter((bad, good))
    prompts = []
    formats = []
    progress = []

    def call_llm(**kwargs):
        prompts.append(kwargs["messages"])
        formats.append(kwargs["extra_body"]["response_format"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(responses))))])

    monkeypatch.setattr(auxiliary_client, "call_llm", call_llm)
    result = workflow_cmd._generate("demo", "default", progress=progress.append)
    assert len(prompts) == 2
    assert all(fmt["type"] == "json_schema" for fmt in formats)
    assert all(
        set(fmt["json_schema"]["schema"]["properties"]["steps"]["properties"])
        == {"discover_candidates", "qualify_candidates"}
        for fmt in formats
    )
    assert "Workflow entry must be" in prompts[1][-1]["content"]
    assert result["activation"] == "active"
    assert progress[0] == "Reading skill and workflow contract"
    assert "Attempt 1/3: waiting for model response" in progress
    assert "Attempt 1/3: validating draft" in progress
    assert any(message.startswith("Attempt 1/3 rejected: Workflow entry must be") for message in progress)
    assert "Attempt 2/3: waiting for model response" in progress
    assert progress[-1].startswith("Workflow saved:")

    monkeypatch.setattr(workflows, "get_hermes_home", lambda: tmp_path / "failed-home")
    responses = iter((bad, bad, bad))
    with pytest.raises(WorkflowError, match="Workflow entry must be"):
        workflow_cmd._generate("demo", "default")
    assert len(prompts) == 5
    assert not (tmp_path / "failed-home" / "workflow-overlays").exists()


def test_contract_branch_labels_define_generated_transition_schema():
    from hermes_cli.workflow_cmd import _normalize_generated_contract_fields

    graph = {"steps": {"decide": {"max_retries": 0, "output_schema": {"type": "object", "properties": {
        "transition": {"enum": ["invented"]}, "observed": {"type": "string"}},
        "required": ["observed"]}}}}
    contract = {"steps": {"decide": {"kind": "decision", "branches": {
        "clear": "send", "unreadable": "blocked"}}}}
    _normalize_generated_contract_fields(graph, contract)
    schema = graph["steps"]["decide"]["output_schema"]
    assert graph["steps"]["decide"]["max_retries"] == 1
    assert graph["steps"]["decide"]["effect"] == "read"
    assert graph["steps"]["decide"]["recovery"] == {"method": "retry"}
    assert schema["properties"]["transition"]["enum"] == list(contract["steps"]["decide"]["branches"])
    assert set(schema["required"]) == {"observed", "transition"}


def test_generator_repairs_branch_join_binding_with_prior_draft(tmp_path, monkeypatch):
    from hermes_cli import workflow_cmd
    from agent import workflows, auxiliary_client
    from tools import skill_manager_tool

    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("""## Workflow contract
```yaml
entry: choose
steps:
  choose: {kind: decision, branches: {before: blocked, after: verify}}
  verify: {kind: task, next: blocked}
  blocked: {kind: task, next: END, terminal_outcome: blocked}
```
""")
    monkeypatch.setattr(skill_manager_tool, "_find_skill", lambda _name: {"path": skill_dir})
    monkeypatch.setattr(workflows, "get_hermes_home", lambda: tmp_path / "home")
    graph = {"schema": "hermes/workflow@2", "id": "default", "skill": "demo",
             "entry": "choose", "max_transitions": 3, "steps": {
                 "choose": {"kind": "decision", "instructions": "Choose.",
                            "branches": {"before": "blocked", "after": "verify"},
                            "output_schema": {"type": "object", "required": ["transition"],
                                              "properties": {"transition": {"enum": ["before", "after"]}}}},
                 "verify": {"kind": "task", "instructions": "Verify.", "next": "blocked",
                            "output_schema": {"type": "object", "required": ["checked"]}},
                 "blocked": {"kind": "task", "instructions": "Report.", "next": "END",
                             "terminal_outcome": "blocked",
                             "input": {"checked": "$steps.verify.output.checked"}},
                 }}
    for step in graph["steps"].values():
        step.update(effect="read", max_retries=1, recovery={"method": "retry"})
    graph["steps"]["blocked"]["terminal_reason"] = {"code": "blocked_condition", "evidence": ["checked"]}
    bad = json.dumps(graph)
    repaired = json.loads(bad)
    repaired["steps"]["blocked"]["input"] = {"reason": "$steps.choose.output.transition"}
    repaired["steps"]["blocked"]["terminal_reason"]["evidence"] = ["reason"]
    replies = iter((bad, json.dumps(repaired)))
    calls = []

    def call_llm(**kwargs):
        calls.append(list(kwargs["messages"]))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(replies)))])

    monkeypatch.setattr(auxiliary_client, "call_llm", call_llm)
    result = workflow_cmd._generate("demo", "default")

    assert result["workflow"]["steps"]["blocked"]["input"] == repaired["steps"]["blocked"]["input"]
    assert calls[1][-2] == {"role": "assistant", "content": bad}
    feedback = calls[1][-1]["content"]
    assert "$steps.verify.output.checked" in feedback
    assert "choose --before--> blocked" in feedback
    assert "choose --after--> verify --next--> blocked" in feedback
    assert "every incoming path" in feedback

    replies = iter((bad, bad, bad))
    calls.clear()
    with pytest.raises(WorkflowError) as exc:
        workflow_cmd._generate("demo", "default")
    assert len(calls) == 3
    assert "steps.blocked.input binding '$steps.verify.output.checked'" in str(exc.value)
    assert "choose --before--> blocked" in str(exc.value)
    assert "choose --after--> verify --next--> blocked" in str(exc.value)


def test_regenerating_an_uncontracted_skill_requires_a_contract(tmp_path, monkeypatch):
    from hermes_cli import workflow_cmd
    from agent import auxiliary_client
    from tools import skill_manager_tool

    (tmp_path / "SKILL.md").write_text("Research and send a report.")
    monkeypatch.setattr(skill_manager_tool, "_find_skill", lambda _name: {"path": tmp_path})
    monkeypatch.setattr(auxiliary_client, "call_llm", lambda **_kwargs:
                        (_ for _ in ()).throw(AssertionError("must not call model")))

    with pytest.raises(WorkflowError, match="Workflow contract"):
        workflow_cmd._generate("legacy", "default")


def test_generate_cli_streams_progress_without_changing_json(monkeypatch, capsys):
    from hermes_cli import workflow_cmd

    def generate(skill, name, *, progress):
        progress("Attempt 1/3: waiting for model response")
        return {"success": True, "skill": skill, "name": name}

    monkeypatch.setattr(workflow_cmd, "_generate", generate)
    args = SimpleNamespace(workflow_action="generate", all=False, skill="demo", name="default")
    assert workflow_cmd.cmd_workflow(args) == 0
    captured = capsys.readouterr()
    assert "demo: Attempt 1/3: waiting for model response" in captured.err
    assert json.loads(captured.out) == {"success": True, "skill": "demo", "name": "default"}


def test_bulk_generation_skips_prebuilt_and_user_owned_skills(monkeypatch):
    from hermes_cli import workflow_cmd
    from tools import skill_usage, skills_tool

    rows = [{"name": name} for name in ("background", "learned", "manual", "bundled", "external")]
    statuses = {
        "background": "eligible", "learned": "eligible", "manual": "user_owned",
        "bundled": "prebuilt", "external": "external",
    }
    generated = []
    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda: rows)
    monkeypatch.setattr(skill_usage, "workflow_eligibility", lambda name: statuses[name])
    monkeypatch.setattr(workflow_cmd, "load_workflow", lambda *_args: (_ for _ in ()).throw(WorkflowError("has no workflow")))
    monkeypatch.setattr(workflow_cmd, "_generate", lambda skill, _name, **_kwargs: generated.append(skill) or {"path": skill})

    results, failed = workflow_cmd._bulk_generate(force=False, dry_run=False, continue_on_error=False)

    assert not failed
    assert generated == ["background", "learned"]
    assert {row["skill"]: row["status"] for row in results} == {
        "background": "generated", "learned": "generated", "manual": "skipped_user_owned",
        "bundled": "skipped_prebuilt", "external": "skipped_external",
    }


def test_bulk_generation_opt_ins_expand_the_candidate_set(monkeypatch):
    from hermes_cli import workflow_cmd
    from tools import skill_usage, skills_tool

    rows = [{"name": name} for name in ("background", "learned", "manual", "bundled", "external")]
    statuses = {
        "background": "eligible", "learned": "eligible", "manual": "user_owned",
        "bundled": "prebuilt", "external": "external",
    }
    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda: rows)
    monkeypatch.setattr(skill_usage, "workflow_eligibility", lambda name: statuses[name])
    monkeypatch.setattr(workflow_cmd, "load_workflow", lambda *_args: (_ for _ in ()).throw(WorkflowError("has no workflow")))

    local, _ = workflow_cmd._bulk_generate(force=False, dry_run=True, continue_on_error=False, include_local=True)
    prebuilt, _ = workflow_cmd._bulk_generate(force=False, dry_run=True, continue_on_error=False, include_prebuilt=True)

    assert {row["skill"] for row in local if row["status"] == "would_generate"} == {"background", "learned", "manual"}
    assert {row["skill"] for row in prebuilt if row["status"] == "would_generate"} == {
        "background", "learned", "bundled", "external"}


def test_bulk_generation_with_no_eligible_skills_never_calls_a_model(monkeypatch):
    from hermes_cli import workflow_cmd
    from tools import skill_usage, skills_tool

    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda: [{"name": "bundled"}, {"name": "learned"}])
    monkeypatch.setattr(skill_usage, "workflow_eligibility", lambda name: "prebuilt" if name == "bundled" else "user_owned")
    monkeypatch.setattr(workflow_cmd, "_generate", lambda *_args: (_ for _ in ()).throw(AssertionError("must not generate")))

    results, failed = workflow_cmd._bulk_generate(force=False, dry_run=False, continue_on_error=False)

    assert not failed
    assert workflow_cmd._summary(results) == {"skipped_prebuilt": 1, "skipped_user_owned": 1}
