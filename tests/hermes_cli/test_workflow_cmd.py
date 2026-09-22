"""Bulk workflow generation chooses background-created skills by provenance."""

from agent.workflows import WorkflowError


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
    monkeypatch.setattr(workflow_cmd, "_generate", lambda skill, _name: generated.append(skill) or {"path": skill})

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
    assert {row["skill"] for row in prebuilt if row["status"] == "would_generate"} == {"background", "bundled", "external"}


def test_bulk_generation_with_no_eligible_skills_never_calls_a_model(monkeypatch):
    from hermes_cli import workflow_cmd
    from tools import skill_usage, skills_tool

    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda: [{"name": "bundled"}, {"name": "learned"}])
    monkeypatch.setattr(skill_usage, "workflow_eligibility", lambda name: "prebuilt" if name == "bundled" else "user_owned")
    monkeypatch.setattr(workflow_cmd, "_generate", lambda *_args: (_ for _ in ()).throw(AssertionError("must not generate")))

    results, failed = workflow_cmd._bulk_generate(force=False, dry_run=False, continue_on_error=False)

    assert not failed
    assert workflow_cmd._summary(results) == {"skipped_prebuilt": 1, "skipped_user_owned": 1}
