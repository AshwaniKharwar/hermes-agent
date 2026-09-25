"""Reusable isolated child-agent runner for executable skill workflows."""

from __future__ import annotations

import json
import hashlib
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Mapping, Optional

import yaml
import psutil

from agent.workflow_receipts import WorkflowReceipt, inspect_workflow_run
from agent.workflows import WorkflowError, execute_workflow, load_workflow, validate_workflow

_USAGE_COUNTERS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "api_calls",
)
_CHILD_ACTIVITY_POLL_SECONDS = 15


def _json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise WorkflowError(f"Step output is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkflowError("Step output must be a JSON object.")
    return value


def _validate_value(value: Mapping[str, Any], schema: Mapping[str, Any], label: str) -> None:
    try:
        from jsonschema.validators import validator_for
    except ImportError as exc:
        raise WorkflowError("Workflow validation requires jsonschema.") from exc
    validator = validator_for(schema)(schema)
    errors = sorted(validator.iter_errors(value), key=lambda error: list(map(str, error.absolute_path)))
    if errors:
        raise WorkflowError(f"{label} does not satisfy its JSON Schema: " +
                            "; ".join(error.message for error in errors[:10]))


def _status(parent_agent: Any, state: str, message: str, *, skill: str, step: str) -> None:
    callback = getattr(parent_agent, "status_callback", None) if parent_agent is not None else None
    if callback:
        try:
            callback(state, message)
        except Exception:
            pass
    event_callback = getattr(parent_agent, "event_callback", None) if parent_agent is not None else None
    if event_callback:
        try:
            event_callback(state, {"skill": skill, "step": step, "message": message})
        except Exception:
            pass


def _child_runtime(parent_agent: Any, route: Mapping[str, Any]) -> dict[str, Any]:
    provider = route.get("provider") or getattr(parent_agent, "provider", None)
    inherits_provider = parent_agent is not None and provider == getattr(parent_agent, "provider", None)
    runtime = {"provider": provider,
               "model": route.get("model") or (getattr(parent_agent, "model", "") if inherits_provider else "")}
    if inherits_provider:
        api_key = getattr(parent_agent, "api_key", None) or (
            getattr(parent_agent, "_client_kwargs", {}) or {}).get("api_key")
        runtime.update(
            api_key=api_key,
            base_url=getattr(parent_agent, "base_url", None), api_mode=getattr(parent_agent, "api_mode", None),
            credential_pool=getattr(parent_agent, "_credential_pool", None),
            request_overrides=dict(getattr(parent_agent, "request_overrides", {}) or {}),
        )
    return runtime


def _used_tools(messages: Any) -> list[str]:
    names = []
    for message in messages or []:
        for call in message.get("tool_calls", []) if isinstance(message, Mapping) else []:
            function = call.get("function", {}) if isinstance(call, Mapping) else getattr(call, "function", None)
            name = function.get("name") if isinstance(function, Mapping) else getattr(function, "name", None)
            if isinstance(name, str) and name and name not in names:
                names.append(name)
    return names


def _record_usage(parent_agent: Any, child: Any) -> None:
    try:
        session_db = getattr(parent_agent, "_session_db", None)
        session_id = getattr(parent_agent, "session_id", None)
        counts = {key: int(getattr(child, f"session_{key}", 0) or 0) for key in _USAGE_COUNTERS}
        if session_db is None or not session_id or not any(counts.values()):
            return
        session_db.record_auxiliary_usage(
            session_id, task="skill_workflow", model=getattr(child, "model", None),
            billing_provider=getattr(child, "provider", None), billing_base_url=getattr(child, "base_url", None),
            estimated_cost_usd=getattr(child, "session_estimated_cost_usd", None),
            api_call_count=counts.pop("api_calls"), **counts,
        )
    except Exception:
        pass


def run_workflow(skill: str, name: str, inputs: Mapping[str, Any], *, parent_agent: Any = None,
                 task_id: str = "",
                 retry_of: str = "",
                 resume_run_id: str = "", choice: str = "",
                 model_overrides: Optional[Mapping[str, str]] = None,
                 reasoning_overrides: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """Execute one active workflow. A started run never falls back to prose."""
    if resume_run_id:
        prior = inspect_workflow_run(resume_run_id)
        if prior.get("status") != "paused" or prior.get("skill") != skill:
            raise WorkflowError("Run is not a paused workflow for this skill.")
        if parent_agent is not None and prior.get("session_id") != getattr(parent_agent, "session_id", None):
            raise WorkflowError("Paused workflow belongs to another session.")
        snapshot = prior["workflow_snapshot"]
        if hashlib.sha256(snapshot.encode()).hexdigest() != prior.get("workflow_sha256"):
            raise WorkflowError("Saved workflow snapshot failed integrity verification.")
        workflow = validate_workflow(yaml.safe_load(snapshot), reliable=True)
        receipt = WorkflowReceipt.reopen(resume_run_id)
        resume_state = json.loads((receipt.path / "checkpoint.json").read_text(encoding="utf-8"))
        if choice not in (resume_state.get("pending") or {}).get("choices", {}):
            raise WorkflowError("Decision is not offered by this paused workflow.")
        inputs = resume_state["state"]["input"]
        source = type("SnapshotSource", (), {"source": "snapshot"})()
        receipt.resume(choice)
    else:
        workflow, source = load_workflow(skill, name)
        receipt = WorkflowReceipt(workflow, inputs, skill=skill, workflow_name=name,
                                  session_id=str(getattr(parent_agent, "session_id", "") or ""),
                                  task_id=str(task_id or getattr(parent_agent, "task_id", "") or ""),
                                  retry_of=retry_of)
        resume_state = None
    model_overrides, reasoning_overrides = dict(model_overrides or {}), dict(reasoning_overrides or {})

    def run_step(step_id: str, step: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        route = dict(context["route"])
        if step_id in model_overrides:
            provider, separator, model = model_overrides[step_id].partition("/")
            if not separator or not provider or not model:
                raise WorkflowError("Step model overrides must be PROVIDER/MODEL.")
            route.update(provider=provider, model=model)
        if step_id in reasoning_overrides:
            route["reasoning_effort"] = reasoning_overrides[step_id]
        toolsets = list(step.get("toolsets", workflow.get("defaults", {}).get("toolsets", [])))
        reconciling = bool(context.get("reconcile"))
        if reconciling:
            recovery = step["recovery"]
            toolsets = list(recovery.get("toolsets", toolsets))
        output_schema = ({"type": "object", "required": ["state"], "properties": {
            "state": {"enum": ["done", "not_done", "unknown"]}, "output": {"type": "object"}}}
            if reconciling else step.get("output_schema", {}))
        request = {"skill": skill, "step": step_id,
                   "instructions": recovery["instructions"] if reconciling else step["instructions"],
                   "input": context["input"], "operation_id": context["operation_id"],
                   "reconcile": reconciling, "output_schema": output_schema, "toolsets": toolsets}
        if context.get("previous_error"):
            request["previous_attempt_error"] = context["previous_error"]
        from run_agent import AIAgent
        child = AIAgent(
            **_child_runtime(parent_agent, route), max_iterations=20,
            enabled_toolsets=toolsets, quiet_mode=True, platform="workflow",
            skip_context_files=True, skip_memory=True, skip_background_review=True,
            ephemeral_system_prompt=("Execute one workflow step. Return only JSON matching output_schema. "
                                     + ("Reconciliation is inspection only: do not create, send, or change anything. "
                                        "If the observed state is uncertain, return unknown."
                                        if reconciling else "")),
            reasoning_config={"effort": route["reasoning_effort"]} if route.get("reasoning_effort") else None,
        )
        child._persist_disabled = True
        child.suppress_status_output = True
        stop_monitor = threading.Event()
        monitor = None
        touch = getattr(parent_agent, "_touch_activity", None)
        if callable(touch):
            from agent.memory_provider import spawn_context_thread
            def monitor_child() -> None:
                last = getattr(child, "_last_activity_ts", None)
                while not stop_monitor.wait(_CHILD_ACTIVITY_POLL_SECONDS):
                    activity = getattr(child, "_last_activity_ts", None)
                    if activity is not None and activity != last:
                        last = activity
                        touch(f"workflow step {step_id} progressing")
            monitor = spawn_context_thread(monitor_child, name=f"workflow-activity-{step_id}")
            monitor.start()
        try:
            child_result = child.run_conversation(json.dumps(request, ensure_ascii=False))
            text = child_result["final_response"]
            activity = child.get_activity_summary() if hasattr(child, "get_activity_summary") else {}
            output = _json_object(text)
            _validate_value(output, output_schema, f"Step {step_id!r} output")
            if reconciling and output["state"] == "done":
                _validate_value(output.get("output", {}), step.get("output_schema", {}),
                                f"Step {step_id!r} reconciled output")
        finally:
            stop_monitor.set()
            if monitor is not None:
                monitor.join(timeout=1)
            _record_usage(parent_agent, child)
            child.close()
        if reconciling:
            return output
        used_tools = _used_tools(child_result.get("messages"))
        output["__workflow_meta__"] = {
            "provider": getattr(child, "provider", route.get("provider")),
            "model": getattr(child, "model", route.get("model")), "toolsets": toolsets,
            "used_tools": used_tools, "activity": activity,
        }
        return output

    def record_event(kind: str, payload: Mapping[str, Any]) -> None:
        receipt.event(kind, payload)
        touch = getattr(parent_agent, "_touch_activity", None)
        if callable(touch) and kind in {"step_started", "step_completed", "step_failed"}:
            touch(f"workflow {kind}: {payload.get('step')}")
        if kind in {"step_started", "step_completed", "step_failed", "step_cancelled"}:
            label = kind.removeprefix("step_")
            _status(parent_agent, f"workflow_step_{label}",
                    f"Workflow {skill}: {label} {payload.get('step')}",
                    skill=skill, step=str(payload.get("step", "")))

    try:
        _validate_value(inputs, workflow.get("input_schema", {}), "Workflow input")
        result = execute_workflow(workflow, inputs, run_step, event_callback=record_event,
                                  run_id=receipt.run_id, resume_state=resume_state,
                                  resume_choice=choice, checkpoint_callback=receipt.checkpoint)
        if result.get("paused"):
            receipt.pause(result["outcome"], result["pending_action"])
        else:
            code = result["output"].get("reason_code") if isinstance(result["output"], Mapping) else None
            receipt.finish("conformant", output=result["output"], extra={"outcome": result["outcome"],
                           "reason_code": code or ("completed" if result["outcome"] == "success"
                                                   else "partial_accepted" if result["outcome"] == "partial"
                                                   else "blocked_condition")})
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        receipt.finish(status, error=str(exc), extra={"outcome": status,
                       "reason_code": "execution_cancelled" if status == "cancelled" else "execution_error"})
        raise
    result.update(source=source.source, receipt=str(receipt.path),
                  status="paused" if result.get("paused") else "conformant")
    return result


def resume_workflow(run_id: str, choice: str, *, parent_agent: Any = None) -> dict[str, Any]:
    prior = inspect_workflow_run(run_id)
    with _resume_claim(WorkflowReceipt.reopen(run_id).path):
        return run_workflow(prior["skill"], prior["workflow"], {}, parent_agent=parent_agent,
                            resume_run_id=run_id, choice=choice)


@contextmanager
def _resume_claim(path):
    lock = path / ".resume.lock"
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError:
            try:
                owner = json.loads(lock.read_text(encoding="utf-8"))
                process = psutil.Process(owner["pid"])
                if abs(process.create_time() - owner["started_at"]) < 0.01:
                    raise WorkflowError("Workflow is already being resumed.")
            except (OSError, ValueError, KeyError, psutil.NoSuchProcess):
                # A fresh empty file may still be getting its owner written.
                if lock.exists() and time.time() - lock.stat().st_mtime < 60:
                    raise WorkflowError("Workflow resume is already being claimed.")
            lock.unlink(missing_ok=True)
    else:
        raise WorkflowError("Could not claim the paused workflow.")
    with os.fdopen(fd, "w") as handle:
        json.dump({"pid": os.getpid(), "started_at": psutil.Process().create_time()}, handle)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)
