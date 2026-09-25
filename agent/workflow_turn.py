"""Automatic workflow preference for slash and model-selected skills."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Optional

from agent.message_content import flatten_message_text
from agent.skill_commands import extract_user_instruction_from_skill_message, parse_single_skill_invocation
from agent.workflow_generation import enqueue_workflow_generation
from agent.workflow_receipts import fallback_receipt
from agent.workflows import WorkflowError, has_valid_candidate, load_workflow


def _enabled() -> bool:
    try:
        from hermes_cli.config import load_config_readonly
        return load_config_readonly().get("skills", {}).get("workflow_execution", "prefer") == "prefer"
    except Exception:
        return True


def _render_output(output: Any) -> str:
    if isinstance(output, dict):
        for key in ("answer", "response", "result"):
            if isinstance(output.get(key), str):
                return output[key]
    if isinstance(output, str):
        return output
    return json.dumps(output, indent=2, ensure_ascii=False, default=str)


def _render_workflow_result(result: Mapping[str, Any]) -> str:
    outcome = result.get("outcome", "success")
    rendered = _render_output(result["output"])
    if result.get("paused"):
        choices = ", ".join(result["pending_action"]["choices"])
        return (f"Workflow {outcome} (run {result['run_id']}): {rendered}\n"
                f"Reply `workflow resume {result['run_id']} <{choices}>` to continue.")
    return rendered if outcome == "success" else f"Workflow {outcome}: {rendered}"


def _placeholder(rule: Any) -> Any:
    """Supply a harmless value for a required input the graph never reads."""
    if not isinstance(rule, Mapping):
        return None
    if "default" in rule:
        return rule["default"]
    if rule.get("enum"):
        return rule["enum"][0]
    kind = rule.get("type")
    if isinstance(kind, list) and "null" in kind:
        return None
    return {"string": "", "array": [], "object": {}, "integer": 0,
            "number": 0, "boolean": False}.get(kind)


def _input_roots(value: Any) -> set[str]:
    roots: set[str] = set()
    if isinstance(value, str) and value.startswith("$input."):
        roots.add(value.removeprefix("$input.").split(".", 1)[0])
    elif isinstance(value, Mapping):
        for child in value.values():
            roots.update(_input_roots(child))
    elif isinstance(value, list):
        for child in value:
            roots.update(_input_roots(child))
    return roots


def _automatic_inputs(workflow: Mapping[str, Any], request: str) -> dict[str, Any]:
    """Build automatic inputs without inventing values consumed by the graph.

    Older generated workflows sometimes declared unused top-level fields as required.
    Neutral placeholders keep those overlays runnable; a missing field actually bound by
    a step remains missing and is rejected by the normal schema validator.
    """
    inputs: dict[str, Any] = {"request": request}
    schema = workflow.get("input_schema", {})
    referenced = _input_roots(workflow.get("steps", {}))
    properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
    for name in schema.get("required", []) if isinstance(schema, Mapping) else []:
        if name not in inputs and name not in referenced:
            inputs[name] = _placeholder(properties.get(name, {}))
    return inputs


def _run_selected_workflow(agent: Any, state: Any, skill: str, request: str, *,
                           retry_of: str = "") -> Optional[dict[str, Any]]:
    if not _enabled():
        return None
    try:
        workflow, _source = load_workflow(skill, "default")
    except WorkflowError as exc:
        if retry_of:
            state.failed = True
            state._turn_exit_reason = "workflow_failed"
            state.final_response = f"Workflow retry cannot start: {exc}"
            return {"workflow_run": None}
        job_id = None
        try:
            candidate_ready = has_valid_candidate(skill)
        except WorkflowError:
            candidate_ready = False
        if not candidate_ready:
            try:
                job_id = enqueue_workflow_generation(skill)
            except Exception:
                pass
        fallback_receipt(
            skill, {"request": request}, reason=str(exc), generation_job_id=job_id,
            session_id=str(getattr(agent, "session_id", "") or ""),
            task_id=str(state.effective_task_id or ""),
        )
        return None

    inputs = _automatic_inputs(workflow, request)
    from agent.workflow_runner import run_workflow
    try:
        result = run_workflow(skill, "default", inputs, parent_agent=agent,
                              task_id=str(state.effective_task_id or ""), retry_of=retry_of)
        outcome = result.get("outcome", "success")
        state.final_response = _render_workflow_result(result)
        state._turn_exit_reason = "workflow_completed" if outcome == "success" else f"workflow_{outcome}"
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            state.interrupted = True
            state._turn_exit_reason = "workflow_cancelled"
            state.final_response = "Workflow execution was cancelled."
        else:
            state.failed = True
            state._turn_exit_reason = "workflow_failed"
            state.final_response = f"Workflow execution failed: {exc}"
    return {"workflow_run": result if 'result' in locals() else None}


def _failed_workflow_retry(agent: Any, state: Any) -> Optional[dict[str, Any]]:
    if flatten_message_text(state.user_message).strip().casefold().rstrip(".! ") not in {
        "try again", "please try again", "retry",
    }:
        return None
    messages = getattr(state, "messages", []) or []
    previous = next((flatten_message_text(message.get("content")) for message in reversed(messages)
                     if isinstance(message, Mapping) and message.get("role") == "assistant"
                     and flatten_message_text(message.get("content")).strip()), "")
    if not previous.startswith("Workflow execution failed: "):
        return None
    from agent.workflow_receipts import list_workflow_runs
    session_id = str(getattr(agent, "session_id", "") or "")
    for receipt in list_workflow_runs(limit=1000):
        if (receipt.get("session_id") != session_id or receipt.get("status") != "failed"
                or previous != f"Workflow execution failed: {receipt.get('error')}"):
            continue
        policies = receipt.get("tool_policy") or {}
        if not policies or any(not set(policy.get("allowed") or []) <= {"web"} for policy in policies.values()):
            state.failed = True
            state._turn_exit_reason = "workflow_retry_blocked"
            state.final_response = ("The previous workflow may have changed a file or sent a message. "
                                    "Inspect its artifact or Sent result before retrying.")
            return {"workflow_run": None}
        request = (receipt.get("inputs") or {}).get("request")
        if isinstance(request, str) and request:
            return _run_selected_workflow(agent, state, receipt["skill"], request,
                                          retry_of=receipt["run_id"])
    state.failed = True
    state._turn_exit_reason = "workflow_retry_blocked"
    state.final_response = "The failed workflow receipt is unavailable; I cannot safely retry it."
    return {"workflow_run": None}


def prefer_skill_workflow(agent: Any, state: Any) -> Optional[dict[str, Any]]:
    """Run an explicit single-skill slash scaffold before the parent model loop."""
    if getattr(agent, "platform", "") == "workflow" or getattr(agent, "_memory_write_origin", "") == "background_review":
        return None
    if _enabled():
        resumed = _resume_paused_workflow(agent, state)
        if resumed is not None:
            return resumed
    invocation = parse_single_skill_invocation(state.user_message)
    if not invocation:
        return _failed_workflow_retry(agent, state) if _enabled() else None
    skill, instruction = invocation
    request = _conversation_request(getattr(state, "messages", []) or [], instruction)
    return _run_selected_workflow(agent, state, skill, request)


def _resume_paused_workflow(agent: Any, state: Any) -> Optional[dict[str, Any]]:
    message = flatten_message_text(state.user_message).strip()
    explicit = re.fullmatch(r"workflow resume ([0-9a-f]{32}) (accept|continue|stop|retry)",
                            message, re.I)
    from agent.workflow_receipts import list_workflow_runs
    session_id = str(getattr(agent, "session_id", "") or "")
    if explicit:
        run_id, choice = explicit.group(1), explicit.group(2).lower()
    elif message.casefold().rstrip(".! ") in {"try again", "please try again", "retry"}:
        candidates = [row for row in list_workflow_runs(limit=1000)
                      if row.get("session_id") == session_id and row.get("status") == "paused"
                      and ("retry" if row.get("outcome") == "blocked" else "continue")
                      in (row.get("pending_action") or {}).get("choices", {})]
        if not candidates:
            return None
        run_id = candidates[0]["run_id"]
        choice = "retry" if candidates[0].get("outcome") == "blocked" else "continue"
    else:
        return None
    from agent.workflow_runner import resume_workflow
    try:
        result = resume_workflow(run_id, choice, parent_agent=agent)
        state.final_response = _render_workflow_result(result)
        state._turn_exit_reason = f"workflow_{result['outcome']}"
    except (WorkflowError, FileNotFoundError) as exc:
        state.failed = True
        state._turn_exit_reason = "workflow_resume_blocked"
        state.final_response = f"Workflow resume failed: {exc}"
        result = None
    return {"workflow_run": result}


def _tool_call(call: Any) -> tuple[str, dict[str, Any], str]:
    function = call.get("function", {}) if isinstance(call, Mapping) else getattr(call, "function", None)
    name = function.get("name", "") if isinstance(function, Mapping) else getattr(function, "name", "")
    raw = function.get("arguments", {}) if isinstance(function, Mapping) else getattr(function, "arguments", {})
    try:
        args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        args = {}
    call_id = call.get("id", "") if isinstance(call, Mapping) else getattr(call, "id", "")
    return str(name), args if isinstance(args, dict) else {}, str(call_id or "")


def _loaded_skill(assistant_message: Any, messages: list[Any]) -> Optional[str]:
    """Return one successfully loaded main skill from this tool round."""
    results = {str(message.get("tool_call_id") or ""): message for message in messages
               if isinstance(message, Mapping) and message.get("role") == "tool"}
    loaded: list[str] = []
    for call in getattr(assistant_message, "tool_calls", []) or []:
        name, args, call_id = _tool_call(call)
        if name != "skill_view" or args.get("file_path") or not args.get("name"):
            continue
        content = flatten_message_text(results.get(call_id, {}).get("content"))
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("success"):
            loaded.append(str(payload.get("name") or args["name"]))
    return loaded[0] if len(loaded) == 1 else None


def _conversation_request(messages: list[Any], current: Any) -> str:
    """Give a selected workflow the current request plus bounded prior user context."""
    current_text = flatten_message_text(current).strip()
    prior: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        text = flatten_message_text(message.get("content")).strip()
        text = extract_user_instruction_from_skill_message(text) or text
        if text and text != current_text:
            prior.append(text)
    prior = prior[-5:]
    if not prior:
        return current_text
    context = "\n\n".join(f"Earlier user message: {text}" for text in prior)
    return (context + f"\n\nCurrent user request: {current_text}")[-16_000:]


def prefer_loaded_skill_workflow(
    agent: Any, state: Any, assistant_message: Any, messages: list[Any], current_request: Any,
) -> Optional[dict[str, Any]]:
    """Take over after the parent model successfully selects one skill via skill_view."""
    if (getattr(agent, "platform", "") == "workflow"
            or getattr(agent, "_memory_write_origin", "") == "background_review"):
        return None
    skill = _loaded_skill(assistant_message, messages)
    if not skill:
        return None
    return _run_selected_workflow(agent, state, skill, _conversation_request(messages, current_request))
