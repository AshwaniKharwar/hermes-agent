"""Executable, profile-scoped skill workflow graphs.

This module deliberately owns only data, validation, rendering, and orchestration.
The interactive surfaces supply the model-step callable, keeping normal skill use
and a conversation's cached prompt untouched.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import yaml

from hermes_constants import get_hermes_home
from utils import atomic_write_text

SCHEMA = "hermes/workflow@1"
END = "END"
_BINDING = re.compile(r"^\$(?:input(?:\.[A-Za-z_][\w-]*)*|steps\.[A-Za-z_][\w-]*\.output(?:\.[A-Za-z_][\w-]*)*)$")
_SECRET = re.compile(r"(?:api[_-]?key|secret|password|token)\s*$", re.I)


class WorkflowError(ValueError):
    """A workflow is malformed or cannot be executed safely."""


@dataclass(frozen=True)
class WorkflowSource:
    skill: str
    path: Path
    source: str
    stale: bool = False


def _safe_dump(value: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(value), allow_unicode=True, sort_keys=False)


def _fingerprint(path: Path) -> str:
    """A sidecar is invalidated when the source skill changes."""
    digest = hashlib.sha256()
    candidates = [path / "SKILL.md"]
    for directory in ("references", "templates", "scripts"):
        root = path / directory
        if root.is_dir():
            candidates.extend(candidate for candidate in sorted(root.rglob("*")) if candidate.is_file())
    for candidate in candidates:
        try:
            digest.update(str(candidate.relative_to(path)).encode("utf-8"))
            digest.update(candidate.read_bytes())
        except OSError:
            pass
    return digest.hexdigest()


def workflow_source_fingerprint(skill: str) -> str:
    """Current source fingerprint for persisted generation status."""
    from tools.skill_manager_tool import _find_skill
    found = _find_skill(skill)
    if not found:
        raise WorkflowError(f"Skill {skill!r} was not found.")
    return _fingerprint(Path(found["path"]))


def _overlay_dir(skill_path: Path) -> Path:
    key = hashlib.sha256(str(skill_path.resolve()).encode()).hexdigest()[:20]
    return get_hermes_home() / "workflow-overlays" / key


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WorkflowError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkflowError("Workflow must be a YAML mapping.")
    return value


def _references(value: Any) -> Iterable[str]:
    if isinstance(value, str) and value.startswith("$"):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _references(child)


def _validate_schema(schema: Any, label: str) -> None:
    if not isinstance(schema, Mapping):
        raise WorkflowError(f"{label} must be a JSON Schema mapping.")
    if schema.get("type") not in (None, "object"):
        raise WorkflowError(f"{label} must describe an object.")


def validate_workflow(workflow: Mapping[str, Any], *, skill: Optional[str] = None) -> Dict[str, Any]:
    """Validate and return a JSON-safe normalized graph.

    Validation intentionally stays structural. Model availability is resolved at
    execution time because it is profile- and session-specific.
    """
    data = dict(workflow)
    if data.get("schema") != SCHEMA:
        raise WorkflowError(f"schema must be {SCHEMA!r}.")
    if not isinstance(data.get("id"), str) or not data["id"].strip():
        raise WorkflowError("id is required.")
    if skill is not None and data.get("skill") not in (None, skill):
        raise WorkflowError(f"Workflow belongs to skill {data.get('skill')!r}, not {skill!r}.")
    if not isinstance(data.get("skill"), str) or not data["skill"].strip():
        raise WorkflowError("skill is required.")
    _validate_schema(data.get("input_schema", {}), "input_schema")
    steps = data.get("steps")
    if not isinstance(steps, Mapping) or not steps:
        raise WorkflowError("steps must be a non-empty mapping.")
    entry = data.get("entry")
    if entry not in steps:
        raise WorkflowError("entry must name a step.")
    max_transitions = data.get("max_transitions", 1)
    if not isinstance(max_transitions, int) or isinstance(max_transitions, bool) or not 1 <= max_transitions <= 1000:
        raise WorkflowError("max_transitions must be an integer between 1 and 1000.")
    defaults = data.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise WorkflowError("defaults must be a mapping.")
    for key in ("route", "toolsets"):
        if key in defaults and not isinstance(defaults[key], (Mapping, list)):
            raise WorkflowError(f"defaults.{key} has the wrong type.")
    seen: set[str] = set()
    targets: set[str] = set()
    for step_id, step in steps.items():
        if not isinstance(step_id, str) or not re.match(r"^[A-Za-z_][\w-]*$", step_id):
            raise WorkflowError("step ids must be identifiers.")
        if step_id in seen:
            raise WorkflowError(f"Duplicate step id {step_id!r}.")
        seen.add(step_id)
        if not isinstance(step, Mapping):
            raise WorkflowError(f"steps.{step_id} must be a mapping.")
        if not isinstance(step.get("instructions"), str) or not step["instructions"].strip():
            raise WorkflowError(f"steps.{step_id}.instructions is required.")
        bindings = step.get("input", {})
        if not isinstance(bindings, Mapping):
            raise WorkflowError(f"steps.{step_id}.input must be a mapping.")
        for binding in _references(bindings):
            if not _BINDING.match(binding):
                raise WorkflowError(f"Unsafe binding {binding!r} in step {step_id!r}.")
        _validate_schema(step.get("output_schema", {}), f"steps.{step_id}.output_schema")
        for route in (defaults.get("route"), step.get("route")):
            if route is not None:
                if not isinstance(route, Mapping):
                    raise WorkflowError(f"steps.{step_id}.route must be a mapping.")
                if any(_SECRET.search(str(key)) for key in route):
                    raise WorkflowError("Workflow routes cannot contain credentials.")
        toolsets = step.get("toolsets", defaults.get("toolsets", []))
        if not isinstance(toolsets, list) or not all(isinstance(item, str) and item for item in toolsets):
            raise WorkflowError(f"steps.{step_id}.toolsets must be a list of names.")
        nxt = step.get("next", END)
        if isinstance(nxt, str):
            targets.add(nxt)
        elif isinstance(nxt, Mapping) and nxt:
            if "transition" not in (step.get("output_schema", {}).get("properties", {}) or {}):
                raise WorkflowError(f"steps.{step_id} branches but its output schema lacks transition.")
            targets.update(nxt.values())
        else:
            raise WorkflowError(f"steps.{step_id}.next must be a target or transition map.")
    unknown = {target for target in targets if target != END and target not in steps}
    if unknown:
        raise WorkflowError(f"Unknown step target(s): {', '.join(sorted(unknown))}.")
    # Reachability catches dead authored steps without attempting to prove a
    # loop terminates; max_transitions is the explicit safety bound.
    reachable, pending = set(), [entry]
    while pending:
        step_id = pending.pop()
        if step_id in reachable or step_id == END:
            continue
        reachable.add(step_id)
        nxt = steps[step_id].get("next", END)
        pending.extend([nxt] if isinstance(nxt, str) else list(nxt.values()))
    dead = set(steps) - reachable
    if dead:
        raise WorkflowError(f"Unreachable step(s): {', '.join(sorted(dead))}.")
    data["steps"] = {str(key): dict(value) for key, value in steps.items()}
    data["defaults"] = dict(defaults)
    data["max_transitions"] = max_transitions
    return data


def _resolve_binding(expression: Any, state: Mapping[str, Any]) -> Any:
    if not isinstance(expression, str) or not expression.startswith("$"):
        return expression
    if not _BINDING.match(expression):
        raise WorkflowError(f"Unsafe binding {expression!r}.")
    current: Any = state
    for part in expression[1:].split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise WorkflowError(f"Binding {expression!r} is not available yet.")
        current = current[part]
    return current


def _resolve_value(value: Any, state: Mapping[str, Any]) -> Any:
    if isinstance(value, Mapping):
        return {key: _resolve_value(child, state) for key, child in value.items()}
    if isinstance(value, list):
        return [_resolve_value(child, state) for child in value]
    return _resolve_binding(value, state)


def _receipt_value(value: Any) -> Any:
    """Keep receipts useful without persisting obvious credential fields."""
    if isinstance(value, Mapping):
        return {str(key): "[REDACTED]" if _SECRET.search(str(key)) else _receipt_value(child)
                for key, child in value.items()}
    if isinstance(value, list):
        return [_receipt_value(child) for child in value]
    return value


def render_mermaid(workflow: Mapping[str, Any]) -> str:
    graph = validate_workflow(workflow)
    lines = ["flowchart TD"]
    for step_id, step in graph["steps"].items():
        label = step_id.replace('"', "'")
        route = step.get("route", graph["defaults"].get("route", {}))
        model = route.get("model") if isinstance(route, Mapping) else None
        suffix = "\\n" + str(model) if model else ""
        lines.append(f'  {step_id}["{label}{suffix}"]')
        nxt = step.get("next", END)
        if isinstance(nxt, str):
            lines.append(f"  {step_id} --> {'done((END))' if nxt == END else nxt}")
        else:
            for transition, target in nxt.items():
                lines.append(f"  {step_id} -->|{str(transition).replace('|', '/')}| {'done((END))' if target == END else target}")
    return "\n".join(lines)


def workflow_sources(skill: str) -> list[WorkflowSource]:
    from tools.skill_manager_tool import _find_skill
    found = _find_skill(skill)
    if not found:
        raise WorkflowError(f"Skill {skill!r} was not found.")
    skill_dir = Path(found["path"])
    result = [WorkflowSource(skill, path, "packaged") for path in sorted((skill_dir / "workflows").glob("*.yaml"))]
    overlay = _overlay_dir(skill_dir)
    fingerprint = _fingerprint(skill_dir)
    for path in sorted(overlay.glob("*.yaml")):
        stale = False
        try:
            stale = _load_yaml(path).get("skill_fingerprint") not in (None, fingerprint)
        except WorkflowError:
            pass
        result.append(WorkflowSource(skill, path, "overlay", stale))
    return result


def load_workflow(skill: str, name: str = "default") -> tuple[Dict[str, Any], WorkflowSource]:
    sources = workflow_sources(skill)
    matches = [source for source in sources if source.path.stem == name]
    if not matches:
        raise WorkflowError(f"Skill {skill!r} has no workflow named {name!r}.")
    # An overlay is purposefully more specific than the installed package.
    source = next((item for item in matches if item.source == "overlay"), matches[0])
    if source.stale:
        raise WorkflowError(f"Workflow {name!r} is stale and must be regenerated before it can run.")
    return validate_workflow(_load_yaml(source.path), skill=skill), source


def save_overlay(skill: str, workflow: Mapping[str, Any], *, name: str = "default") -> Path:
    from tools.skill_manager_tool import _find_skill
    found = _find_skill(skill)
    if not found:
        raise WorkflowError(f"Skill {skill!r} was not found.")
    normalized = validate_workflow(workflow, skill=skill)
    normalized["skill_fingerprint"] = _fingerprint(Path(found["path"]))
    target = _overlay_dir(Path(found["path"])) / f"{name}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, _safe_dump(normalized), preserve_mode=True, create_mode=0o600)
    return target


def execute_workflow(
    workflow: Mapping[str, Any], inputs: Mapping[str, Any],
    run_step: Callable[[str, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
    *, receipt_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a validated graph with an injected isolated-agent step executor."""
    graph = validate_workflow(workflow)
    state: Dict[str, Any] = {"input": dict(inputs), "steps": {}}
    current, transitions = graph["entry"], 0
    run_id = uuid.uuid4().hex
    events: list[Dict[str, Any]] = []
    if receipt_path is not None:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
    while current != END:
        if transitions >= graph["max_transitions"]:
            raise WorkflowError("Workflow exceeded max_transitions.")
        step = graph["steps"][current]
        resolved_input = _resolve_value(step.get("input", {}), state)
        route = {**dict(graph["defaults"].get("route", {})), **dict(step.get("route", {}))}
        started = time.time()
        output = dict(run_step(current, step, {"input": resolved_input, "route": route, "workflow": graph}))
        execution = output.pop("__workflow_meta__", None)
        state["steps"][current] = {"output": output}
        nxt = step.get("next", END)
        chosen = nxt if isinstance(nxt, str) else nxt.get(output.get("transition"))
        if chosen is None:
            raise WorkflowError(f"Step {current!r} returned unknown transition {output.get('transition')!r}.")
        event = {"run_id": run_id, "skill": graph["skill"], "workflow": graph["id"], "step": current,
                 "input": _receipt_value(resolved_input), "output": _receipt_value(output),
                 "route": _receipt_value(route), "transition": chosen,
                 "started_at": started, "finished_at": time.time()}
        if isinstance(execution, Mapping):
            event["execution"] = dict(execution)
        events.append(event)
        if receipt_path is not None:
            with receipt_path.open("a", encoding="utf-8") as receipt:
                receipt.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        current, transitions = chosen, transitions + 1
    return {"run_id": run_id, "output": _resolve_value(graph.get("output", {}), state), "events": events}
