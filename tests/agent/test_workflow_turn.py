"""Slash and model-selected skills prefer executable workflows."""

from types import SimpleNamespace

from agent.workflows import WorkflowError


SINGLE = ('[IMPORTANT: The user has invoked the "demo" skill. '
          'The full skill content is loaded below.]\n[Invocation source: explicit slash skill]\n\nprocedure\n'
          'The user has provided the following instruction alongside the skill invocation: do it')
BUNDLE = '[IMPORTANT: The user has invoked the "/demo /other" skill bundle, containing 2 skills.]'


def _state(message):
    return SimpleNamespace(user_message=message, effective_task_id="task", final_response=None,
                           failed=False, interrupted=False, _turn_exit_reason="unknown")


def test_turn_hook_executes_workflow_but_leaves_natural_and_bundle_paths_unchanged(monkeypatch):
    import agent.workflow_runner as runner
    import agent.workflow_turn as turn
    calls = []
    monkeypatch.setattr(turn, "_enabled", lambda: True)
    monkeypatch.setattr(turn, "load_workflow", lambda *_args: ({}, object()))
    monkeypatch.setattr(runner, "run_workflow", lambda skill, name, inputs, **_kwargs:
                        calls.append((skill, name, inputs)) or {"output": {"answer": "done"}})
    agent = SimpleNamespace(session_id="session")

    assert turn.prefer_skill_workflow(agent, _state("use demo naturally")) is None
    assert turn.prefer_skill_workflow(agent, _state(BUNDLE)) is None
    state = _state(SINGLE)
    assert turn.prefer_skill_workflow(agent, state) is not None
    assert calls == [("demo", "default", {"request": "do it"})]
    assert state.final_response == "done" and state._turn_exit_reason == "workflow_completed"


def test_blocked_terminal_path_is_reported_as_blocked(monkeypatch):
    import agent.workflow_runner as runner
    import agent.workflow_turn as turn

    monkeypatch.setattr(turn, "_enabled", lambda: True)
    monkeypatch.setattr(turn, "load_workflow", lambda *_args: ({}, object()))
    monkeypatch.setattr(runner, "run_workflow", lambda *_args, **_kwargs:
                        {"output": {"reason": "Mail delivery could not be verified"}, "outcome": "blocked"})
    state = _state(SINGLE)

    turn.prefer_skill_workflow(SimpleNamespace(session_id="session"), state)

    assert state._turn_exit_reason == "workflow_blocked"
    assert state.final_response.startswith("Workflow blocked:")


def test_successful_skill_view_routes_plain_text_to_workflow_with_conversation_context(monkeypatch):
    import agent.workflow_runner as runner
    import agent.workflow_turn as turn
    calls = []
    workflow = {
        "input_schema": {
            "type": "object",
            "properties": {"request": {"type": "string"}, "recipient": {"type": "string"}},
            "required": ["request", "recipient"],
        },
        "steps": {"first": {"input": {"request": "$input.request"}}},
    }
    monkeypatch.setattr(turn, "_enabled", lambda: True)
    monkeypatch.setattr(turn, "load_workflow", lambda *_args: (workflow, object()))
    monkeypatch.setattr(runner, "run_workflow", lambda skill, name, inputs, **_kwargs:
                        calls.append((skill, name, inputs)) or {"output": {"answer": "workflow done"}})
    tool_call = SimpleNamespace(
        id="call-1", function=SimpleNamespace(name="skill_view", arguments='{"name":"demo"}'))
    assistant = SimpleNamespace(tool_calls=[tool_call])
    messages = [
        {"role": "user", "content": "send it to buyer@example.com"},
        {"role": "user", "content": "do today's lead generation"},
        {"role": "tool", "tool_call_id": "call-1", "content": '{"success":true,"name":"demo"}'},
    ]
    state = _state("do today's lead generation")

    result = turn.prefer_loaded_skill_workflow(
        SimpleNamespace(session_id="session", platform="desktop"), state, assistant, messages, state.user_message)

    assert result is not None
    assert calls[0][0:2] == ("demo", "default")
    assert "buyer@example.com" in calls[0][2]["request"]
    assert calls[0][2]["recipient"] == ""
    assert state.final_response == "workflow done"


def test_skill_view_errors_linked_files_and_workflow_children_do_not_route(monkeypatch):
    import agent.workflow_turn as turn
    monkeypatch.setattr(turn, "_enabled", lambda: True)
    monkeypatch.setattr(turn, "load_workflow", lambda *_args: (_ for _ in ()).throw(AssertionError("must not load")))
    state = _state("plain request")
    cases = [
        (SimpleNamespace(id="a", function=SimpleNamespace(name="skill_view", arguments='{"name":"demo"}')),
         {"role": "tool", "tool_call_id": "a", "content": '{"success":false}'}),
        (SimpleNamespace(id="b", function=SimpleNamespace(name="skill_view", arguments='{"name":"demo","file_path":"references/x.md"}')),
         {"role": "tool", "tool_call_id": "b", "content": '{"success":true,"name":"demo"}'}),
    ]
    for call, result in cases:
        assistant = SimpleNamespace(tool_calls=[call])
        assert turn.prefer_loaded_skill_workflow(
            SimpleNamespace(platform="desktop"), state, assistant, [result], state.user_message) is None
    assistant = SimpleNamespace(tool_calls=[cases[0][0]])
    assert turn.prefer_loaded_skill_workflow(
        SimpleNamespace(platform="workflow"), state, assistant,
        [{"role": "tool", "tool_call_id": "a", "content": '{"success":true,"name":"demo"}'}],
        state.user_message) is None


def test_missing_workflow_records_fallback_and_queues_generation(monkeypatch):
    import agent.workflow_turn as turn
    recorded = []
    monkeypatch.setattr(turn, "_enabled", lambda: True)
    monkeypatch.setattr(turn, "load_workflow", lambda *_args: (_ for _ in ()).throw(WorkflowError("missing")))
    monkeypatch.setattr(turn, "enqueue_workflow_generation", lambda _skill: "job-1")
    monkeypatch.setattr(turn, "fallback_receipt", lambda skill, inputs, **details:
                        recorded.append((skill, inputs, details)))

    assert turn.prefer_skill_workflow(SimpleNamespace(session_id="session"), _state(SINGLE)) is None
    assert recorded[0][0:2] == ("demo", {"request": "do it"})
    assert recorded[0][2]["generation_job_id"] == "job-1"


def test_shared_turn_boundary_finalizes_workflow_before_parent_model_loop(monkeypatch):
    import agent.conversation_loop as loop
    import agent.workflow_runner as runner
    import agent.workflow_turn as turn

    context = SimpleNamespace(
        user_message=SINGLE, original_user_message="/demo do it", conversation_history=None,
        effective_task_id="task", turn_id="turn", should_review_memory=False,
        plugin_user_context="", ext_prefetch_cache="", messages=[{"role": "user", "content": SINGLE}],
        active_system_prompt="system", current_turn_user_idx=0, preflight_compression_blocked=False,
    )
    agent = SimpleNamespace(
        session_id="session", max_compression_attempts=3, api_mode="openai",
        _try_refresh_env_client_credentials=lambda: None,
    )
    finalized = []

    monkeypatch.setattr(loop, "begin_fast_mode_turn", lambda *_args: None)
    monkeypatch.setattr(loop, "build_turn_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(turn, "_enabled", lambda: True)
    monkeypatch.setattr(turn, "load_workflow", lambda *_args: ({}, object()))
    monkeypatch.setattr(runner, "run_workflow", lambda *_args, **_kwargs: {"output": {"answer": "done"}})

    def finalize(agent, final_response, messages, _turn_exit_reason):
        finalized.append((final_response, messages, _turn_exit_reason))
        return {"final_response": final_response, "messages": messages}

    monkeypatch.setattr(loop, "finalize_turn", finalize)
    result = loop._run_conversation_turn(agent, SINGLE)

    assert result["final_response"] == "done"
    assert finalized == [("done", context.messages, "workflow_completed")]


def test_try_again_uses_failed_receipt_request_and_links_new_run(tmp_path, monkeypatch):
    import agent.workflow_receipts as receipts
    import agent.workflow_runner as runner
    import agent.workflow_turn as turn

    monkeypatch.setattr(receipts, "get_hermes_home", lambda: tmp_path)
    graph = {"schema": "hermes/workflow@2", "id": "default", "skill": "demo", "entry": "answer",
             "max_transitions": 1, "input_schema": {"type": "object", "required": ["request"]},
             "steps": {"answer": {"kind": "task", "instructions": "Answer.", "next": "END"}}}
    old = receipts.WorkflowReceipt(graph, {"request": "find 20 companies"}, skill="demo", session_id="session")
    old.event("step_started", {"step": "answer", "allowed_toolsets": ["web"]})
    old.finish("failed", error="broken binding")
    monkeypatch.setattr(turn, "_enabled", lambda: True)
    monkeypatch.setattr(turn, "load_workflow", lambda *_args: (graph, object()))
    calls = []
    monkeypatch.setattr(runner, "run_workflow", lambda *args, **kwargs:
                        calls.append((args, kwargs)) or {"output": {"answer": "done"}})
    state = _state("try again")
    state.messages = [{"role": "assistant", "content": "Workflow execution failed: broken binding"},
                      {"role": "user", "content": "try again"}]

    turn.prefer_skill_workflow(SimpleNamespace(session_id="session", platform="desktop"), state)

    assert calls[0][0] == ("demo", "default", {"request": "find 20 companies"})
    assert calls[0][1]["retry_of"] == old.run_id
    assert state.final_response == "done"


def test_try_again_stops_after_a_possible_side_effect(tmp_path, monkeypatch):
    import agent.workflow_receipts as receipts
    import agent.workflow_turn as turn

    monkeypatch.setattr(receipts, "get_hermes_home", lambda: tmp_path)
    graph = {"schema": "hermes/workflow@2", "id": "default", "skill": "demo", "entry": "send",
             "max_transitions": 1, "steps": {"send": {"kind": "task", "instructions": "Send.", "next": "END"}}}
    old = receipts.WorkflowReceipt(graph, {"request": "send report"}, skill="demo", session_id="session")
    old.event("step_started", {"step": "send", "allowed_toolsets": ["terminal"]})
    old.finish("failed", error="uncertain send")
    state = _state("try again")
    state.messages = [{"role": "assistant", "content": "Workflow execution failed: uncertain send"}]
    monkeypatch.setattr(turn, "_enabled", lambda: True)
    monkeypatch.setattr(turn, "load_workflow", lambda *_args: (_ for _ in ()).throw(AssertionError("reran")))

    turn.prefer_skill_workflow(SimpleNamespace(session_id="session", platform="desktop"), state)

    assert state._turn_exit_reason == "workflow_retry_blocked"
    assert "Sent" in state.final_response


def test_try_again_continues_a_paused_partial_run(monkeypatch):
    import agent.workflow_receipts as receipts
    import agent.workflow_runner as runner
    import agent.workflow_turn as turn

    run_id = "a" * 32
    monkeypatch.setattr(receipts, "list_workflow_runs", lambda **_kwargs: [{
        "run_id": run_id, "session_id": "session", "status": "paused", "outcome": "partial",
        "pending_action": {"choices": {"accept": "deliver", "continue": "collect", "stop": "END"}}}])
    called = []
    monkeypatch.setattr(runner, "resume_workflow", lambda *args, **kwargs:
                        called.append((args, kwargs)) or {"run_id": run_id, "outcome": "success",
                                                         "output": {"answer": "done"}})
    state = _state("try again")
    turn._resume_paused_workflow(SimpleNamespace(session_id="session"), state)
    assert called[0][0] == (run_id, "continue")
    assert state.final_response == "done"


def test_background_review_loading_a_skill_never_runs_workflow(monkeypatch):
    import agent.workflow_turn as turn

    monkeypatch.setattr(turn, "load_workflow", lambda *_args: (_ for _ in ()).throw(AssertionError("ran")))
    call = SimpleNamespace(id="call-1", function=SimpleNamespace(name="skill_view", arguments='{"name":"demo"}'))
    state = _state("review skills")
    agent = SimpleNamespace(platform="desktop", _memory_write_origin="background_review")
    assert turn.prefer_loaded_skill_workflow(agent, state, SimpleNamespace(tool_calls=[call]),
                                              [{"role": "tool", "tool_call_id": "call-1",
                                                "content": '{"success":true,"name":"demo"}'}],
                                              state.user_message) is None
