"""Profile-scoped, replayable receipts for executable skill workflows."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from agent.redact import redact_sensitive_text
from agent.workflows import validate_workflow
from hermes_constants import get_hermes_home
from utils import atomic_write_text

_EVENT_LIMIT = 256 * 1024
_EVENT_BODY_LIMIT = _EVENT_LIMIT - 1  # Reserve the newline in the JSONL ledger.
_SECRET_KEYS = frozenset({"authorization", "cookie", "password", "secret", "token", "api_key", "apikey"})
_TERMINAL = frozenset({"conformant", "failed", "cancelled", "unverifiable"})
_LAST_PRUNE_BY_HOME: dict[str, float] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): "[REDACTED]" if str(key).lower() in _SECRET_KEYS else _safe(child)
                for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(child) for child in value]
    if isinstance(value, str):
        return redact_sensitive_text(value, force=True)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_sensitive_text(str(value), force=True)


def _secure_dir(path: Path) -> None:
    root = get_hermes_home() / "workflow-runs"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for directory in (root, path.parent, path):
        directory.mkdir(exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
                      preserve_mode=True, create_mode=0o600)
    os.chmod(path, 0o600)


def _bounded_event(event: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict(_safe(event))
    encoded = json.dumps(clean, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    if len(encoded) <= _EVENT_BODY_LIMIT:
        return clean
    omitted = {key: clean.pop(key) for key in ("input", "output", "structured_data", "execution") if key in clean}
    raw = json.dumps(omitted, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    clean["omitted_payload"] = {"omitted": True, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    encoded = json.dumps(clean, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    if len(encoded) > _EVENT_BODY_LIMIT:
        raw = encoded
        clean = {key: clean.get(key) for key in ("sequence", "type", "timestamp", "step") if key in clean}
        clean["omitted_payload"] = {"omitted": True, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return clean


class WorkflowReceipt:
    def __init__(self, workflow: Optional[Mapping[str, Any]], inputs: Mapping[str, Any], *,
                 skill: str, workflow_name: str = "default", execution_mode: str = "workflow",
                 session_id: str = "", task_id: str = "", run_id: Optional[str] = None,
                 retry_of: str = ""):
        self.run_id = run_id or uuid.uuid4().hex
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        self.path = get_hermes_home() / "workflow-runs" / month / self.run_id
        _secure_dir(self.path)
        _prune_daily_best_effort()
        self.events_path = self.path / "events.jsonl"
        self.events_path.touch(mode=0o600, exist_ok=True)
        os.chmod(self.events_path, 0o600)
        snapshot = validate_workflow(workflow) if workflow is not None else None
        snapshot_text = yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False) if snapshot else ""
        if snapshot:
            atomic_write_text(self.path / "workflow.yaml", snapshot_text, preserve_mode=True, create_mode=0o600)
            os.chmod(self.path / "workflow.yaml", 0o600)
        self.summary: dict[str, Any] = {
            "run_id": self.run_id, "skill": skill, "workflow": workflow_name,
            "execution_mode": execution_mode, "status": "incomplete", "started_at": _now(),
            "finished_at": None, "session_id": session_id, "task_id": task_id,
            "workflow_sha256": hashlib.sha256(snapshot_text.encode()).hexdigest() if snapshot else None,
            "sequence": 0, "current_step": None, "control_flow": [], "schemas": {}, "tool_policy": {},
            "inputs": _safe(inputs), "receipt_path": str(self.path),
            "retry_of": retry_of or None,
        }
        self._write_summary()
        self.event("run_started", {"inputs": inputs, "execution_mode": execution_mode})

    @classmethod
    def reopen(cls, run_id: str) -> "WorkflowReceipt":
        inspected = inspect_workflow_run(run_id)
        receipt = cls.__new__(cls)
        receipt.run_id = run_id
        receipt.path = Path(inspected["receipt_path"])
        receipt.events_path = receipt.path / "events.jsonl"
        receipt.summary = {key: value for key, value in inspected.items()
                           if key not in {"events", "workflow_snapshot"}}
        return receipt

    def checkpoint(self, value: Mapping[str, Any]) -> None:
        _write_json(self.path / "checkpoint.json", value)

    def pause(self, outcome: str, action: Mapping[str, Any]) -> None:
        self.event("run_paused", {"outcome": outcome, "pending_action": action,
                                  "reason_code": action.get("code")})
        self.summary.update(status="paused", outcome=outcome, pending_action=_safe(action),
                            reason_code=action.get("code"),
                            observed=_safe((action.get("output") or {}).get("evidence", {})),
                            current_step=None)
        self._write_summary()

    def resume(self, choice: str) -> None:
        self.event("run_resumed", {"choice": choice})
        self.summary.update(status="incomplete", pending_action=None)
        self._write_summary()

    def _write_summary(self) -> None:
        _write_json(self.path / "run.json", self.summary)

    def event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        self.summary["sequence"] += 1
        event = _bounded_event({"sequence": self.summary["sequence"], "type": event_type,
                                "timestamp": _now(), **dict(payload)})
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if event_type == "step_started":
            self.summary["current_step"] = event.get("step")
            self.summary["control_flow"].append({"step": event.get("step"), "state": "running"})
            self.summary["schemas"][str(event.get("step"))] = {"input_valid": True}
            self.summary["tool_policy"][str(event.get("step"))] = {
                "allowed": event.get("allowed_toolsets", []), "used": []}
        elif event_type == "step_completed":
            self.summary["current_step"] = None
            self.summary["control_flow"][-1].update(
                state="exhausted" if event.get("exhausted") else "completed",
                transition=event.get("transition"))
            self.summary["schemas"].setdefault(str(event.get("step")), {})["output_valid"] = True
            execution = event.get("execution") or {}
            self.summary["tool_policy"][str(event.get("step"))] = {
                "allowed": execution.get("toolsets", []), "used": execution.get("used_tools", [])}
        elif event_type in {"step_failed", "step_cancelled"}:
            self.summary["current_step"] = None
            if self.summary["control_flow"]:
                self.summary["control_flow"][-1]["state"] = "cancelled" if event_type == "step_cancelled" else "failed"
                if event.get("transition"):
                    self.summary["control_flow"][-1]["transition"] = event["transition"]
            self.summary["schemas"].setdefault(str(event.get("step")), {})["output_valid"] = False
        self._write_summary()

    def finish(self, status: str, *, output: Any = None, error: str = "", extra: Optional[Mapping[str, Any]] = None) -> None:
        if status == "conformant" and (self.summary["current_step"] or
                                       any(row["state"] == "running" for row in self.summary["control_flow"])):
            raise ValueError("Cannot complete a workflow with a running step.")
        if self.summary["current_step"]:
            self.event("step_cancelled" if status == "cancelled" else "step_failed",
                       {"step": self.summary["current_step"], "error": error})
        event_type = {"conformant": "run_completed", "cancelled": "run_cancelled"}.get(status, "run_failed")
        payload = {"output": output, "error": error, **dict(extra or {})}
        self.event(event_type, payload)
        self.summary.update(status=status, finished_at=_now(), output=_safe(output), error=_safe(error))
        if extra:
            self.summary.update(_safe(extra))
        self._write_summary()


def fallback_receipt(skill: str, inputs: Mapping[str, Any], *, reason: str, generation_job_id: Optional[str],
                     session_id: str = "", task_id: str = "") -> dict[str, Any]:
    receipt = WorkflowReceipt(None, inputs, skill=skill, execution_mode="skill_fallback",
                              session_id=session_id, task_id=task_id)
    receipt.finish("unverifiable", output=None, extra={"fallback_reason": reason,
                   "generation_job_id": generation_job_id})
    return receipt.summary


def inspect_workflow_run(run_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise FileNotFoundError(f"Workflow run {run_id!r} was not found.")
    root = get_hermes_home() / "workflow-runs"
    matches = list(root.glob(f"*/{run_id}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Workflow run {run_id!r} was not found.")
    path = matches[0]
    summary = json.loads((path / "run.json").read_text(encoding="utf-8"))
    events = []
    for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            break  # A concurrently appended final line is not complete yet.
    summary = _replay(summary, events)
    if summary["status"] != "incomplete":
        _write_json(path / "run.json", summary)
    summary["events"] = events
    summary["workflow_snapshot"] = (path / "workflow.yaml").read_text(encoding="utf-8") if (path / "workflow.yaml").exists() else None
    return summary


def _replay(base: Mapping[str, Any], events: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Rebuild auditable fields from the canonical ledger, not mutable summary state."""
    result = dict(base)
    result.update(status="incomplete", current_step=None, control_flow=[], schemas={}, tool_policy={})
    result["sequence"] = max((int(event.get("sequence", 0) or 0) for event in events), default=0)
    positions: dict[str, int] = {}
    for event in events:
        kind, step = event.get("type"), event.get("step")
        if kind == "step_started" and step:
            positions[str(step)] = len(result["control_flow"])
            result["control_flow"].append({"step": step, "state": "running"})
            result["schemas"].setdefault(str(step), {})["input_valid"] = True
            result["tool_policy"][str(step)] = {"allowed": event.get("allowed_toolsets", []), "used": []}
            result["current_step"] = step
        elif kind in {"step_completed", "step_failed", "step_cancelled"} and step:
            index = positions.get(str(step), len(result["control_flow"]) - 1)
            if index >= 0:
                result["control_flow"][index].update(
                    state=("exhausted" if event.get("exhausted") else "completed") if kind == "step_completed" else
                          ("cancelled" if kind == "step_cancelled" else "failed"),
                    **({"transition": event.get("transition")} if event.get("transition") else {}),
                )
            result["schemas"].setdefault(str(step), {})["output_valid"] = kind == "step_completed"
            execution = event.get("execution") or {}
            result["tool_policy"][str(step)] = {
                "allowed": execution.get("toolsets", event.get("allowed_toolsets", [])),
                "used": execution.get("used_tools", [])}
            result["current_step"] = None
        elif kind == "run_completed":
            result["status"], result["finished_at"] = "conformant", event.get("timestamp")
            result["output"] = event.get("output")
            result["outcome"] = event.get("outcome", "success")
            result["reason_code"] = event.get("reason_code")
            result["pending_action"] = None
        elif kind == "run_paused":
            result["status"], result["outcome"] = "paused", event.get("outcome")
            pending = event.get("pending_action") or base.get("pending_action")
            result["pending_action"] = pending
            result["outcome"] = event.get("outcome") or base.get("outcome")
            result["reason_code"] = event.get("reason_code") or base.get("reason_code")
            result["observed"] = (pending or {}).get("output", {}).get("evidence", base.get("observed", {}))
            result["current_step"] = None
        elif kind == "run_resumed":
            result["status"], result["pending_action"] = "incomplete", None
        elif kind == "run_cancelled":
            result["status"], result["finished_at"] = "cancelled", event.get("timestamp")
            result["error"] = event.get("error")
            result["outcome"] = event.get("outcome", "cancelled")
            result["reason_code"] = event.get("reason_code")
        elif kind == "run_failed":
            result["status"] = "unverifiable" if result.get("execution_mode") == "skill_fallback" else "failed"
            result["finished_at"] = event.get("timestamp")
            result["error"] = event.get("error")
            result["outcome"] = event.get("outcome", "failed")
            result["reason_code"] = event.get("reason_code")
            for key in ("fallback_reason", "generation_job_id"):
                if key in event:
                    result[key] = event[key]
    return result


def list_workflow_runs(skill: Optional[str] = None, *, limit: int = 50) -> list[dict[str, Any]]:
    rows = _read_run_summaries(skill)
    return rows[:max(1, min(limit, 1000))]


def _read_run_summaries(skill: Optional[str] = None) -> list[dict[str, Any]]:
    rows = []
    for path in (get_hermes_home() / "workflow-runs").glob("*/*/run.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if skill and row.get("skill") != skill:
            continue
        rows.append(row)
    rows.sort(key=lambda row: row.get("started_at", ""), reverse=True)
    return rows


def prune_workflow_runs(*, retention_days: int, max_runs: int) -> None:
    rows = _read_run_summaries()
    cutoff = time.time() - max(retention_days, 1) * 86400
    terminal = [row for row in rows if row.get("status") in _TERMINAL]
    for index, row in enumerate(terminal):
        started = datetime.fromisoformat(row["started_at"]).timestamp()
        if started >= cutoff and index < max_runs:
            continue
        path = Path(row["receipt_path"])
        for child in path.iterdir():
            child.unlink()
        path.rmdir()


def _prune_daily_best_effort() -> None:
    home = str(get_hermes_home())
    now = time.time()
    if now - _LAST_PRUNE_BY_HOME.get(home, 0) < 86400:
        return
    _LAST_PRUNE_BY_HOME[home] = now
    try:
        from hermes_cli.config import load_config_readonly
        skills = load_config_readonly().get("skills", {})
        prune_workflow_runs(retention_days=int(skills.get("workflow_receipt_retention_days", 30)),
                            max_runs=int(skills.get("workflow_receipt_max_runs", 1000)))
    except Exception:
        pass
