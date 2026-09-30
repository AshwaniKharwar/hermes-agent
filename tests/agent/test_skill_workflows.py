"""Skill workflow contracts through real profile paths and LangGraph compilation."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.skill_workflow_runtime import run_workflow, select_workflow
from agent.skill_workflow_store import graph_state, required_records, workflow_dir
from tools.skill_manager_tool import skill_manage


def _content(name, description):
    return f"---\nname: {name}\ndescription: {description}\n---\n\n# Procedure\n\n1. Inspect.\n2. Finish.\n"


def _checklist():
    return [{"id": "r1", "quote": "Inspect.", "description": "Inspect the task"},
            {"id": "r2", "quote": "Finish.", "description": "Finish the task"}]


def _graph(digest):
    from agent.skill_workflow_runtime import checklist_digest

    return {
        "version": 4, "source_sha256": digest, "requirements_sha256": checklist_digest(_checklist()),
        "description": "Run a sample task.",
        "entry": "inspect", "max_steps": 4,
        "nodes": [
            {"id": "inspect", "goal": "Inspect the task", "covers": ["r1"],
             "inputs": [], "outputs": [], "effects": [],
             "next": {"success": "finish", "blocked": "END", "error": "END"}},
            {"id": "finish", "goal": "Finish the task", "covers": ["r2"],
             "inputs": [], "outputs": [], "effects": [],
             "next": {"success": "END", "blocked": "END", "error": "END"}},
        ],
    }


def test_committed_skills_are_profile_scoped_and_batch_rollback_has_no_graph(tmp_path, monkeypatch, request):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from agent.secret_scope import set_multiplex_active

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    homes = [tmp_path / name for name in ("a", "b")]
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    set_multiplex_active(True)
    request.addfinalizer(lambda: set_multiplex_active(False))
    for home in homes:
        home.mkdir()
        token = set_hermes_home_override(home)
        try:
            result = json.loads(skill_manage("create", "sample", content=_content("sample", f"Sample task in {home.name}.")))
            assert result["success"] and required_records(home)
            failed = json.loads(skill_manage("", "", operations=[
                {"name": "rolled-back", "action": "create", "content": _content("rolled-back", "Temporary skill.")},
                {"name": "rolled-back", "action": "patch", "old_string": "absent", "new_string": "new"},
            ]))
            assert not failed["success"]
            assert len(required_records(home)) == 1
        finally:
            reset_hermes_home_override(token)
    try:
        for home, expected in ((homes[0], "Sample task in a."), (homes[1], "Sample task in b."),
                               (homes[0], "Sample task in a.")):
            token = set_hermes_home_override(home)
            try:
                assert required_records()[0][1]["description"] == expected
                if home == homes[0]:
                    directory, record = required_records()[0]
                    assert graph_state(directory, record)[0] == "missing"
            finally:
                reset_hermes_home_override(token)
        token = set_hermes_home_override(homes[0])
        try:
            (Path(record["skill_path"]) / "SKILL.md").write_text(_content("sample", "Sample task in a after edit."))
            assert graph_state(directory, record)[0] == "stale"
        finally:
            reset_hermes_home_override(token)
    finally:
        set_multiplex_active(False)


def test_explicit_route_and_approved_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    assert json.loads(skill_manage("create", "sample", content=_content("sample", "Run a sample task.")))["success"]
    directory, record = required_records()[0]
    agent = SimpleNamespace(platform="cli")
    selected = select_workflow('/sample run the task', agent)
    assert selected is not None and selected[2] == "missing"
    from agent.skill_workflow_generation import publish_graph
    from agent.skill_workflow_runtime import checklist_digest, validate_graph

    graph = _graph(record["source_sha256"])
    with pytest.raises(ValueError, match="version 4"):
        validate_graph({**graph, "version": 3}, _checklist())
    uncovered = copy.deepcopy(graph)
    uncovered["nodes"][1]["covers"] = []
    with pytest.raises(ValueError, match="does not cover requirements"):
        validate_graph(uncovered, _checklist())
    described = copy.deepcopy(graph)
    described["nodes"][0]["outputs"] = [{"key": "finding", "description": "x" * 4_000}]
    validate_graph(described, _checklist())
    described["nodes"][0]["inputs"] = [{"key": "query", "description": "x" * 4_000,
                                           "source": "request"}]
    validate_graph(described, _checklist())
    named = [{**_checklist()[0], "quote": "Use `web_search` for current facts."}, _checklist()[1]]
    named_graph = copy.deepcopy(graph)
    named_graph["requirements_sha256"] = checklist_digest(named)
    with pytest.raises(ValueError, match="requirement r1 must name web_search"):
        validate_graph(named_graph, named)
    named_graph["nodes"][0]["goal"] = "Inspect current facts with web_search"
    validate_graph(named_graph, named)
    assert not publish_graph({"name": "sample", "skill_path": record["skill_path"],
                              "source_sha256": record["source_sha256"], "graph": graph,
                              "checklist": [{**_checklist()[0], "quote": "not in skill"}, _checklist()[1]]})["success"]
    result = publish_graph({"name": "sample", "skill_path": record["skill_path"],
                            "source_sha256": record["source_sha256"], "graph": graph,
                            "checklist": _checklist()})
    assert result["success"]
    assert select_workflow('/sample run the task', agent)[2] == "ready"
    def classify(**kwargs):
        choices = json.loads(kwargs["messages"][1]["content"])["workflows"]
        assert choices[0]["workflow_description"] == graph["description"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="sample"))])

    monkeypatch.setattr("agent.auxiliary_client.call_llm", classify)
    assert select_workflow("Please run a sample task", agent)[2] == "ready"
    assert workflow_dir(Path(record["skill_path"])) == directory
    requirements_path = directory / "requirements.json"
    requirements_path.write_text(json.dumps([{**_checklist()[0], "description": "Changed"}, _checklist()[1]]))
    assert graph_state(directory, required_records()[0][1])[0] == "invalid"
    requirements_path.write_text(json.dumps(_checklist()))
    (directory / "graph.json").write_text(json.dumps({**graph, "description": "Changed without approval."}))
    assert graph_state(directory, required_records()[0][1])[0] == "invalid"
    graph["source_sha256"] = "wrong"
    assert not publish_graph({"name": "sample", "skill_path": record["skill_path"],
                              "source_sha256": record["source_sha256"], "graph": graph,
                              "checklist": _checklist()})["success"]
    monkeypatch.setattr("agent.skill_utils.get_disabled_skill_names", lambda _platform: {"sample"})
    assert select_workflow("Please run a sample task", agent) is None


def test_langgraph_runs_one_hermes_child_per_node(tmp_path, monkeypatch):
    import agent.subagent_lifecycle as lifecycle
    import tools.lazy_deps as lazy_deps
    from agent.skill_workflow_store import save_record

    monkeypatch.setattr(lazy_deps, "ensure", lambda *_args, **_kwargs: None)
    launched = []

    class Service:
        def __init__(self, parent_resolver):
            assert parent_resolver() is parent

        def launch(self, request):
            launched.append(request)
            return len(launched)

        def wait(self, handle):
            return SimpleNamespace(completed=True)

        def result(self, handle):
            return SimpleNamespace(terminal_state=lifecycle.SubagentState.SUCCEEDED,
                                   summary=json.dumps({"status": "success", "summary": f"node {handle} done"}),
                                   error_message=None)

    monkeypatch.setattr(lifecycle, "SubagentLifecycleService", Service)
    parent = SimpleNamespace(session_id="20260929_155922_9c9917")
    directory = tmp_path / "profile" / "workflows" / "graph-id"
    save_record(directory, {"required": True, "name": "sample"})
    graph = _graph("source")
    graph["nodes"][0]["toolsets"] = ["web_search"]  # Old graphs no longer restrict execution.
    from agent.skill_workflow_runtime import validate_graph
    validate_graph(graph, _checklist())
    output, success = run_workflow(graph, "Run the task", parent, directory)
    assert success and output == "node 2 done"
    assert ["Inspect the task", "Finish the task"] == [r.goal.split("\n")[0] for r in launched]
    assert all(r.allowed_toolsets is None for r in launched)
    assert (directory / "runs.sqlite").exists()
    event_files = list((tmp_path / "profile" / "workflows" / "sample" / "runs").glob(
        "*-20260929_155922_9c9917-*.jsonl"))
    assert len(event_files) == 1
    events = [json.loads(line) for line in event_files[0].read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "run_started", "node_started", "node_finished", "node_started", "node_finished", "run_finished",
    ]
    assert [event["node"] for event in events if event["event"] == "node_started"] == ["inspect", "finish"]


def test_workflow_scopes_node_inputs_and_declares_generic_effects(tmp_path, monkeypatch):
    import agent.subagent_lifecycle as lifecycle
    import tools.lazy_deps as lazy_deps
    from agent.skill_workflow_runtime import validate_graph
    from agent.skill_workflow_store import save_record

    monkeypatch.setattr(lazy_deps, "ensure", lambda *_args, **_kwargs: None)
    request_text = "Inspect item alpha using option beta, then deliver the result to destination gamma"
    from agent.skill_workflow_runtime import checklist_digest
    graph = {"version": 4, "requirements_sha256": checklist_digest(_checklist()),
             "description": "Inspect and deliver an item", "entry": "inspect", "max_steps": 4,
             "nodes": [
                 {"id": "inspect", "goal": "Inspect the item",
                  "covers": ["r1"], "outputs": [{"key": "finding", "description": "Inspection result"}],
                  "inputs": [{"key": "item", "description": "Item to inspect", "source": "request"},
                             {"key": "option", "description": "Inspection option", "source": "request"}],
                  "effects": [],
                  "next": {"success": "prepare", "blocked": "END", "error": "END"}},
                 {"id": "prepare", "goal": "Prepare for delivery",
                  "covers": [], "inputs": [], "outputs": [], "effects": [],
                  "next": {"success": "deliver", "blocked": "END", "error": "END"}},
                 {"id": "deliver", "goal": "Deliver the result",
                  "covers": ["r2"], "outputs": [],
                  "inputs": [{"key": "destination", "description": "Delivery destination", "source": "request"},
                             {"key": "finding", "description": "Inspection result", "source": "inspect.finding"}],
                  "effects": ["deliver_result"], "next": {"success": "END", "blocked": "END", "error": "END"}},
             ]}
    validate_graph(graph, _checklist())
    unknown_source = copy.deepcopy(graph)
    unknown_source["nodes"][2]["inputs"][1]["source"] = "prepare.finding"
    with pytest.raises(ValueError, match="unknown input source"):
        validate_graph(unknown_source, _checklist())

    def extract(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        assert payload["request"] == request_text
        values = ({"item": [{"source": "request", "quote": "alpha"}],
                   "option": [{"source": "request", "quote": "beta"}]}
                  if payload["inputs"][0]["key"] == "item" else
                  {"destination": [{"source": "request", "quote": "gamma"}]})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(values)))])

    monkeypatch.setattr("agent.auxiliary_client.call_llm", extract)
    seen = []

    class Service:
        def __init__(self, _parent_resolver):
            pass

        def launch(self, request):
            seen.append(request)
            return len(seen)

        def wait(self, _handle):
            return SimpleNamespace(completed=True)

        def result(self, handle):
            return SimpleNamespace(terminal_state=lifecycle.SubagentState.SUCCEEDED,
                                   summary=json.dumps({"status": "success", "summary": f"node {handle} done",
                                                       "outputs": {"finding": "alpha checked"} if handle == 1 else {}}),
                                   error_message=None)

    monkeypatch.setattr(lifecycle, "SubagentLifecycleService", Service)
    directory = tmp_path / "profile" / "workflows" / "graph-id"
    save_record(directory, {"required": True, "name": "sample"})
    output, success = run_workflow(graph, request_text, SimpleNamespace(session_id="one"), directory)
    assert success and output == "node 3 done"
    assert len(seen) == 3
    assert "gamma" not in seen[0].context
    assert "deliver the result" not in seen[0].context
    assert "alpha" in seen[0].context and "beta" in seen[0].context
    assert "gamma" not in seen[1].context and "alpha checked" not in seen[1].context
    assert "gamma" in seen[2].context
    assert "alpha checked" in seen[2].context
    assert "deliver_result" in seen[2].context
    assert all(not req.include_workspace_context for req in seen)
    assert all(req.allowed_toolsets is None for req in seen)
    from tools.delegate_tool_progress import _build_child_system_prompt
    monkeypatch.setattr("agent.prompt_builder.build_context_files_prompt", lambda **_kw: (_ for _ in ()).throw(
        AssertionError("workspace context was loaded")))
    prompt = _build_child_system_prompt(seen[0].goal, seen[0].context,
                                        workspace_path=str(tmp_path), include_workspace_context=False)
    assert "alpha" in prompt and "gamma" not in prompt
    repeated = copy.deepcopy(graph)
    repeated["nodes"][2]["next"]["error"] = "inspect"
    with pytest.raises(ValueError, match="effect node deliver can repeat via deliver -> inspect -> prepare -> deliver"):
        validate_graph(repeated, _checklist())
    read_only_retry = copy.deepcopy(graph)
    read_only_retry["nodes"][0]["next"]["error"] = "inspect"
    validate_graph(read_only_retry, _checklist())


def test_missing_upstream_output_uses_error_edge_without_launching_child(tmp_path, monkeypatch):
    import agent.subagent_lifecycle as lifecycle
    import tools.lazy_deps as lazy_deps
    from agent.skill_workflow_runtime import validate_graph
    from agent.skill_workflow_store import save_record

    monkeypatch.setattr(lazy_deps, "ensure", lambda *_args, **_kwargs: None)
    graph = _graph("source")
    graph["nodes"][0]["outputs"] = [{"key": "finding", "description": "Inspection result"}]
    graph["nodes"][0]["next"]["blocked"] = "finish"
    graph["nodes"][1]["inputs"] = [{"key": "finding", "description": "Inspection result",
                                     "source": "inspect.finding"}]
    validate_graph(graph, _checklist())
    launched = []

    class Service:
        def __init__(self, _parent_resolver):
            pass

        def launch(self, request):
            launched.append(request)
            return len(launched)

        def wait(self, _handle):
            return SimpleNamespace(completed=True)

        def result(self, _handle):
            return SimpleNamespace(terminal_state=lifecycle.SubagentState.SUCCEEDED,
                                   summary=json.dumps({"status": "blocked", "summary": "no finding"}),
                                   error_message=None)

    monkeypatch.setattr(lifecycle, "SubagentLifecycleService", Service)
    directory = tmp_path / "profile" / "workflows" / "graph-id"
    save_record(directory, {"required": True, "name": "sample"})
    output, success = run_workflow(graph, "Inspect the task", SimpleNamespace(session_id="one"), directory)
    assert not success and "missing from inspect.finding" in output
    assert len(launched) == 1


def test_workflow_step_limit_finishes_without_retry(tmp_path, monkeypatch):
    import agent.subagent_lifecycle as lifecycle
    import tools.lazy_deps as lazy_deps
    from agent.skill_workflow_store import save_record

    monkeypatch.setattr(lazy_deps, "ensure", lambda *_args, **_kwargs: None)
    launches = []

    class Service:
        def __init__(self, _parent_resolver):
            pass

        def launch(self, request):
            launches.append(request)
            return 1

        def wait(self, _handle):
            return SimpleNamespace(completed=True)

        def result(self, _handle):
            return SimpleNamespace(terminal_state=lifecycle.SubagentState.SUCCEEDED,
                                   summary=json.dumps({"status": "error", "summary": "try again"}),
                                   error_message=None)

    monkeypatch.setattr(lifecycle, "SubagentLifecycleService", Service)
    from agent.skill_workflow_runtime import checklist_digest
    graph = {"version": 4, "requirements_sha256": checklist_digest(_checklist()),
            "description": "Retry once", "entry": "work", "max_steps": 1,
             "nodes": [{"id": "work", "goal": "Do work", "covers": ["r1", "r2"],
                        "inputs": [], "outputs": [], "effects": [],
                        "next": {"success": "END", "blocked": "END", "error": "work"}}]}
    directory = tmp_path / "profile" / "workflows" / "graph-id"
    save_record(directory, {"required": True, "name": "retry"})
    output, success = run_workflow(graph, "Do work", SimpleNamespace(session_id="one"), directory)
    assert not success and "step limit" in output
    assert len(launches) == 1


def test_unavailable_graph_announces_before_direct_execution(monkeypatch, tmp_path):
    from agent.skill_workflow_turn import route_skill_workflow
    import agent.skill_workflow_generation as generation
    import agent.skill_workflow_runtime as runtime

    events = []
    agent = SimpleNamespace(platform="cli", _emit_status=lambda message: events.append(("announce", message)))
    prior = [{"role": "system", "content": "stable prompt"},
             {"role": "user", "content": "earlier request"},
             {"role": "assistant", "content": "earlier reply"}]
    prior_bytes = json.dumps(prior)
    state = SimpleNamespace(user_message="/sample task", messages=[*prior, {"role": "user", "content": "/sample task"}])
    record = {"name": "sample"}
    monkeypatch.setattr(generation, "resume_pending", lambda _agent: None)
    monkeypatch.setattr(runtime, "select_workflow", lambda *_args: (tmp_path, record, "missing", None))
    monkeypatch.setattr(generation, "schedule_repair", lambda *_args: events.append(("repair", "sample")))
    assert route_skill_workflow(agent, state) is False
    assert [event[0] for event in events] == ["announce", "repair"]
    assert "directly using the skill" in events[0][1]
    assert "skill_view(name='sample')" in state.messages[-1]["api_content"]
    assert json.dumps(state.messages[:3]) == prior_bytes
    assert [message["role"] for message in state.messages] == ["system", "user", "assistant", "user"]
    events.clear()
    record["completed_failures"] = 3
    exhausted = SimpleNamespace(user_message="/sample task", messages=[{"role": "user", "content": "/sample task"}])
    assert route_skill_workflow(agent, exhausted) is False
    assert [event[0] for event in events] == ["announce"]
    assert "automatic repair attempts are exhausted" in events[0][1]


def test_started_graph_failure_never_reenters_direct_loop(monkeypatch, tmp_path):
    from agent.skill_workflow_turn import route_skill_workflow
    import agent.skill_workflow_generation as generation
    import agent.skill_workflow_runtime as runtime

    monkeypatch.setattr(generation, "resume_pending", lambda _agent: None)
    monkeypatch.setattr(runtime, "select_workflow", lambda *_args: (tmp_path, {"name": "sample"}, "ready", _graph("x")))
    monkeypatch.setattr(runtime, "run_workflow", lambda *_args: (_ for _ in ()).throw(RuntimeError("node failed")))
    state = SimpleNamespace(user_message="/sample task", messages=[{"role": "user", "content": "/sample task"}])
    assert route_skill_workflow(SimpleNamespace(platform="cli"), state) is True
    assert state.failed and "node failed" in state.final_response
    assert state.messages == [{"role": "user", "content": "/sample task"}]


def test_route_error_stops_before_direct_execution(monkeypatch):
    from agent.skill_workflow_turn import route_skill_workflow
    import agent.skill_workflow_generation as generation
    import agent.skill_workflow_runtime as runtime

    monkeypatch.setattr(generation, "resume_pending", lambda _agent: None)
    monkeypatch.setattr(runtime, "select_workflow", lambda *_args: (_ for _ in ()).throw(RuntimeError("router offline")))
    state = SimpleNamespace(user_message="sample task", messages=[{"role": "user", "content": "sample task"}])
    assert route_skill_workflow(SimpleNamespace(platform="cli"), state) is True
    assert state.failed and "router offline" in state.final_response
    assert state.messages == [{"role": "user", "content": "sample task"}]


def test_unapproved_graph_does_not_execute_directly(monkeypatch, tmp_path):
    from agent.skill_workflow_turn import route_skill_workflow
    import agent.skill_workflow_generation as generation
    import agent.skill_workflow_runtime as runtime

    monkeypatch.setattr(generation, "resume_pending", lambda _agent: None)
    monkeypatch.setattr(runtime, "select_workflow", lambda *_args: (
        tmp_path, {"name": "sample"}, "awaiting_approval", None))
    state = SimpleNamespace(user_message="/sample task", messages=[{"role": "user", "content": "/sample task"}])
    assert route_skill_workflow(SimpleNamespace(platform="cli"), state) is True
    assert state.failed and "awaits approval" in state.final_response
    assert state.messages == [{"role": "user", "content": "/sample task"}]


def test_rejected_graph_is_not_requeued_without_a_skill_change(tmp_path, monkeypatch):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    from agent.skill_workflow_store import save_record

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    assert json.loads(skill_manage("create", "sample", content=_content("sample", "Run a sample task.")))["success"]
    directory, record = required_records()[0]
    payload = {"action": "workflow_publish", "name": "sample", "skill_path": record["skill_path"],
               "source_sha256": record["source_sha256"], "graph": _graph(record["source_sha256"]),
               "checklist": _checklist()}
    pending = wa.stage_write(wa.SKILLS, payload, summary="approve workflow", origin="background_review")
    diff = json.loads(wa.skill_pending_diff(pending))
    assert diff["requirements"] == _checklist() and diff["graph"] == payload["graph"]
    record.update(status="awaiting_approval", pending_id=pending["id"])
    save_record(directory, record)
    assert "Rejected pending skills write" in handle_pending_subcommand(wa.SKILLS, ["reject", pending["id"]])
    rejected = required_records()[0][1]
    assert graph_state(directory, rejected)[0] == "rejected"
    from agent.skill_workflow_generation import schedule_repair
    assert schedule_repair(rejected, object()) is False


def test_old_pending_workflow_is_retired_before_regeneration(tmp_path, monkeypatch):
    import agent.skill_workflow_generation as generation
    from agent.skill_workflow_store import save_record
    from tools import write_approval as wa

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    assert json.loads(skill_manage("create", "sample", content=_content("sample", "Run a task.")))["success"]
    directory, record = required_records()[0]
    old_graph = {**_graph(record["source_sha256"]), "version": 3}
    pending = wa.stage_write(wa.SKILLS, {"action": "workflow_publish", "name": "sample",
                                           "skill_path": record["skill_path"],
                                           "source_sha256": record["source_sha256"],
                                           "graph": old_graph}, summary="old graph", origin="background_review")
    save_record(directory, {**record, "status": "awaiting_approval", "pending_id": pending["id"]})
    queued = []
    monkeypatch.setattr(generation, "schedule_repair", lambda item, parent: queued.append(item["status"]))
    generation.resume_pending(object())
    assert wa.get_pending(wa.SKILLS, pending["id"]) is None
    assert required_records()[0][1]["status"] == "pending"
    assert queued == ["pending"]


def test_approval_replay_publishes_checklist_with_graph(tmp_path, monkeypatch):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from agent.skill_workflow_store import save_record
    from tools import write_approval as wa

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    assert json.loads(skill_manage("create", "sample", content=_content("sample", "Run a task.")))["success"]
    directory, record = required_records()[0]
    payload = {"action": "workflow_publish", "name": "sample", "skill_path": record["skill_path"],
               "source_sha256": record["source_sha256"], "graph": _graph(record["source_sha256"]),
               "checklist": _checklist()}
    pending = wa.stage_write(wa.SKILLS, payload, summary="approve workflow", origin="background_review")
    save_record(directory, {**record, "status": "awaiting_approval", "pending_id": pending["id"]})
    assert "Approved 1 skills write" in handle_pending_subcommand(wa.SKILLS, ["approve", pending["id"]])
    assert graph_state(directory, required_records()[0][1])[0] == "ready"
    assert json.loads((directory / "requirements.json").read_text()) == _checklist()


def test_approval_replay_marks_only_committed_skill(tmp_path, monkeypatch):
    from hermes_cli.config import load_config, save_config
    from tools import write_approval as wa
    from tools.skill_manager_tool import apply_skill_pending

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    config = load_config()
    config.setdefault("skills", {})["write_approval"] = True
    save_config(config)
    staged = json.loads(skill_manage("create", "sample", content=_content("sample", "Run a task.")))
    assert staged["staged"] and required_records() == []
    pending = wa.get_pending(wa.SKILLS, staged["pending_id"])
    assert json.loads(apply_skill_pending(pending["payload"]))["success"]
    assert [record["name"] for _, record in required_records()] == ["sample"]

    staged_batch = json.loads(skill_manage("", "", operations=[
        {"name": "inspect", "action": "create", "content": _content("inspect", "Inspect a task.")},
        {"name": "inspect", "action": "write_file", "file_path": "references/note.md", "file_content": "Check logs."},
    ]))
    assert staged_batch["staged"] and len(required_records()) == 1
    pending_batch = wa.get_pending(wa.SKILLS, staged_batch["pending_id"])
    assert json.loads(apply_skill_pending(pending_batch["payload"]))["success"]
    assert {record["name"] for _, record in required_records()} == {"sample", "inspect"}


def test_workflow_child_inherits_parent_toolsets():
    import tools.delegate_tool_toolsets as toolsets

    parent = SimpleNamespace(enabled_toolsets=["terminal", "web", "mcp-example"], disabled_toolsets=[])
    enabled, _disabled = toolsets._resolve_child_toolsets(parent, None, "leaf")
    assert set(enabled) == set(parent.enabled_toolsets)


def test_generation_children_receive_only_supplied_skill_context(monkeypatch):
    import agent.subagent_lifecycle as lifecycle
    from agent.skill_workflow_generation import _ask_child

    seen = []

    class Service:
        def __init__(self, _parent_resolver):
            pass

        def launch(self, request):
            seen.append(request)
            return 1

        def wait(self, _handle):
            return SimpleNamespace(completed=True)

        def result(self, _handle):
            return SimpleNamespace(terminal_state=lifecycle.SubagentState.SUCCEEDED,
                                   summary="{}", error_message=None)

    monkeypatch.setattr(lifecycle, "SubagentLifecycleService", Service)
    assert _ask_child(object(), "Review skill", "SKILL.md content") == {}
    assert seen[0].context == "SKILL.md content" and not seen[0].include_workspace_context


def test_writer_and_independent_reviewer_activate_a_valid_graph(tmp_path, monkeypatch):
    import agent.skill_workflow_generation as generation

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    assert json.loads(skill_manage("create", "sample", content=_content("sample", "Run a sample task.")))["success"]
    directory, record = required_records()[0]
    calls = []

    def child(_parent, goal, context):
        calls.append((goal, context))
        if len(calls) == 1:
            return {"requirements": [{"quote": item["quote"], "description": item["description"]}
                                     for item in _checklist()]}
        if len(calls) == 2:
            return _graph(record["source_sha256"])
        return {"aligned": True, "checklist_complete": True, "missing_ids": [],
                "missing_requirements": [], "reason": "covers both steps",
                "coverage_assignments": [{"id": "r1", "node": "inspect", "evidence": "Inspect the task"},
                                         {"id": "r2", "node": "finish", "evidence": "Finish the task"}]}

    monkeypatch.setattr(generation, "_ask_child", child)
    generation._build(directory, record["source_sha256"], object())
    assert len(calls) == 3
    assert "do not declare toolsets" in calls[1][1]
    assert "retry cycle may contain only" in calls[1][1]
    assert "name it exactly in the covering node goal" in calls[1][1]
    assert "Do not require or review per-node toolsets" in calls[2][1]
    assert all("previous graph" not in context.lower() for _, context in calls)
    assert graph_state(directory, required_records()[0][1])[0] == "ready"


def test_missing_checklist_requirement_rebuilds_from_skill_without_prior_graph(tmp_path, monkeypatch):
    import agent.skill_workflow_generation as generation
    from agent.skill_workflow_runtime import checklist_digest

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    assert json.loads(skill_manage("create", "sample", content=_content("sample", "Run a task.")))["success"]
    directory, record = required_records()[0]
    calls = []

    def child(_parent, goal, context):
        calls.append((goal, context))
        if goal.startswith("Extract"):
            return {"requirements": [{"quote": "Inspect.", "description": "Inspect the task"}]}
        if goal.startswith("Design"):
            if len(calls) == 4:
                assert '"id": "r2"' in context
                assert '"goal": "Inspect the task"' not in context
                assert not (directory / "graph.json").exists()
                return _graph(record["source_sha256"])
            first = _graph(record["source_sha256"])
            first["nodes"] = [{**first["nodes"][0],
                               "next": {"success": "END", "blocked": "END", "error": "END"}}]
            first["requirements_sha256"] = checklist_digest(_checklist()[:1])
            return first
        return ({"aligned": False, "checklist_complete": False, "missing_ids": [],
                 "missing_requirements": [{"quote": "Finish.", "description": "Finish the task"}],
                 "reason": "missing final verification"}
                if len(calls) == 3 else
                {"aligned": True, "checklist_complete": True, "missing_ids": [],
                 "missing_requirements": [], "reason": "complete"})

    monkeypatch.setattr(generation, "_ask_child", child)
    generation._build(directory, record["source_sha256"], object())
    assert [goal.startswith("Design") for goal, _ in calls] == [False, True, False, True, False]
    assert [item["id"] for item in json.loads((directory / "requirements.json").read_text())] == ["r1", "r2"]
    assert graph_state(directory, required_records()[0][1])[0] == "ready"


def test_duplicate_review_requirement_and_missing_cover_are_repaired(tmp_path, monkeypatch):
    import agent.skill_workflow_generation as generation

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    assert json.loads(skill_manage("create", "sample", content=_content("sample", "Run a task.")))["success"]
    directory, record = required_records()[0]
    calls = []

    def child(_parent, goal, context):
        calls.append((goal, context))
        if goal.startswith("Extract"):
            return {"requirements": [{"quote": item["quote"], "description": item["description"]}
                                     for item in _checklist()]}
        if goal.startswith("Design"):
            assert len(json.loads(context.split("REQUIREMENTS:\n", 1)[1].split("\n\nPREVIOUS FEEDBACK", 1)[0])) == 2
            graph = _graph(record["source_sha256"])
            if len(calls) == 4:
                graph["nodes"][1]["covers"] = []
            return graph
        if len(calls) == 3:
            return {"aligned": False, "checklist_complete": False, "missing_ids": ["r1"],
                    "missing_requirements": [{"quote": "Inspect.", "description": "Inspect node is too vague"}],
                    "reason": "make the inspection goal specific"}
        return {"aligned": True, "checklist_complete": True, "missing_ids": [],
                "missing_requirements": [],
                "coverage_assignments": [{"id": "r1", "node": "inspect", "evidence": "Inspect the task"},
                                         {"id": "r2", "node": "finish", "evidence": "Finish the task"}],
                "reason": "the finish goal covers r2"}

    monkeypatch.setattr(generation, "_ask_child", child)
    generation._build(directory, record["source_sha256"], object())
    assert len(calls) == 5
    assert [item["id"] for item in json.loads((directory / "requirements.json").read_text())] == ["r1", "r2"]
    assert "r2" in json.loads((directory / "graph.json").read_text())["nodes"][1]["covers"]
    assert graph_state(directory, required_records()[0][1])[0] == "ready"
    events = [json.loads(line) for line in (directory / "generation.jsonl").read_text().splitlines()]
    assert next(event for event in events if event["event"] == "draft_rejected")["added_requirements"] == 0


def test_unverifiable_cover_assignment_never_activates(tmp_path, monkeypatch):
    import agent.skill_workflow_generation as generation

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    assert json.loads(skill_manage("create", "sample", content=_content("sample", "Run a task.")))["success"]
    directory, record = required_records()[0]

    def child(_parent, goal, _context):
        if goal.startswith("Extract"):
            return {"requirements": [{"quote": item["quote"], "description": item["description"]}
                                     for item in _checklist()]}
        if goal.startswith("Design"):
            graph = _graph(record["source_sha256"])
            graph["nodes"][1]["covers"] = []
            return graph
        return {"aligned": True, "checklist_complete": True, "missing_ids": [],
                "missing_requirements": [],
                "coverage_assignments": [{"id": "r2", "node": "inspect", "evidence": "Finish the task"}],
                "reason": "claimed coverage"}

    monkeypatch.setattr(generation, "_ask_child", child)
    with pytest.raises(ValueError, match="exhausted three drafts"):
        generation._build(directory, record["source_sha256"], object())
    assert not (directory / "graph.json").exists()
    events = [json.loads(line) for line in (directory / "generation.jsonl").read_text().splitlines()]
    assert [event["stage"] for event in events if event["event"] == "draft_rejected"] == ["coverage_review"] * 3


def test_skill_change_during_review_does_not_publish_old_graph(tmp_path, monkeypatch):
    import agent.skill_workflow_generation as generation

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    assert json.loads(skill_manage("create", "sample", content=_content("sample", "Run a task.")))["success"]
    directory, record = required_records()[0]

    def child(_parent, goal, _context):
        if goal.startswith("Extract"):
            return {"requirements": [{"quote": item["quote"], "description": item["description"]}
                                     for item in _checklist()]}
        if goal.startswith("Design"):
            return _graph(record["source_sha256"])
        skill_path = Path(record["skill_path"]) / "SKILL.md"
        skill_path.write_text(skill_path.read_text() + "\n3. Verify.\n")
        return {"aligned": True, "checklist_complete": True, "missing_ids": [],
                "missing_requirements": [], "reason": "complete"}

    monkeypatch.setattr(generation, "_ask_child", child)
    generation._build(directory, record["source_sha256"], object())
    assert not (directory / "graph.json").exists()
    assert graph_state(directory, record)[0] == "stale"


def test_failed_workflow_is_retried_on_next_profile_turn(tmp_path, monkeypatch):
    import agent.skill_workflow_generation as generation
    from agent.skill_workflow_store import save_record

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    assert json.loads(skill_manage("create", "sample", content=_content("sample", "Run a task.")))["success"]
    directory, record = required_records()[0]
    record.update(status="failed", error="workflow generation failed")
    save_record(directory, record)
    calls = []
    monkeypatch.setattr(generation, "schedule_repair", lambda item, parent: calls.append((item, parent)))
    parent = object()
    generation.resume_pending(parent)
    assert len(calls) == 1 and calls[0][0]["name"] == "sample" and calls[0][1] is parent


def test_generation_stops_after_two_later_failures_and_manual_queue_resets(tmp_path, monkeypatch):
    import agent.skill_workflow_generation as generation
    from agent.skill_workflow_store import record_committed_skill

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    assert json.loads(skill_manage("create", "sample", content=_content("sample", "Run a task.")))["success"]
    directory, record = required_records()[0]
    monkeypatch.setattr(generation, "_build", lambda *_args: (_ for _ in ()).throw(ValueError("review rejected")))
    key = (str(directory), record["source_sha256"])
    for completed in range(1, 4):
        generation._worker(directory, record["source_sha256"], object(), key)
        assert required_records()[0][1]["completed_failures"] == completed
    generation._worker(directory, record["source_sha256"], object(), key)
    assert required_records()[0][1]["completed_failures"] == 3
    assert generation.schedule_repair(required_records()[0][1], object()) is False
    events = [json.loads(line) for line in (directory / "generation.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["event"] == "generation_failed"]) == 3
    assert record_committed_skill("sample", "create")["completed_failures"] == 0


def test_interrupted_generation_does_not_consume_retry(tmp_path, monkeypatch):
    import agent.skill_workflow_generation as generation

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    assert json.loads(skill_manage("create", "sample", content=_content("sample", "Run a task.")))["success"]
    directory, record = required_records()[0]
    monkeypatch.setattr(generation, "_build", lambda *_args: (_ for _ in ()).throw(
        generation.InterruptedGeneration("stopped")))
    generation._worker(directory, record["source_sha256"], object(), (str(directory), record["source_sha256"]))
    updated = required_records()[0][1]
    assert updated["status"] == "pending" and updated["completed_failures"] == 0
