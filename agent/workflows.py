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
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import yaml

from hermes_constants import get_hermes_home
from utils import atomic_write_text

SCHEMA = "hermes/workflow@2"
END = "END"
NODE_KINDS = frozenset({"task", "decision", "loop"})
TERMINAL_OUTCOMES = frozenset({"success", "partial", "blocked"})
TERMINAL_REASON_CODES = {"partial": frozenset({"goal_unmet"}),
                         "blocked": frozenset({"blocked_condition", "external_unreadable",
                                                "recovery_exhausted", "verification_unreadable"})}
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
    activation: str = "active"


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


def parse_workflow_contract(content: str, *, require: bool = False) -> Optional[Dict[str, Any]]:
    """Validate the authored flow skeleton before a skill is saved or compiled."""
    if not isinstance(content, str):
        raise WorkflowError("SKILL.md content must be text.")
    section = re.search(r"(?im)^## Workflow contract\s*$([\s\S]*?)(?=^## |\Z)", content)
    if not section:
        if require:
            raise WorkflowError("SKILL.md needs a ## Workflow contract fenced YAML block with entry and steps.")
        return None
    fenced = re.search(r"```ya?ml\s*\n([\s\S]*?)\n```", section.group(1), re.I)
    if not fenced:
        raise WorkflowError("Workflow contract must contain a fenced YAML block.")
    try:
        contract = yaml.safe_load(fenced.group(1))
    except yaml.YAMLError as exc:
        raise WorkflowError(f"Invalid workflow contract YAML: {exc}") from exc
    if not isinstance(contract, dict) or not isinstance(contract.get("steps"), dict) or not contract["steps"]:
        raise WorkflowError("Workflow contract needs entry and non-empty steps mappings.")
    if contract.get("entry") not in contract["steps"]:
        raise WorkflowError("Workflow contract entry must name a step.")
    if set(contract) - {"entry", "steps", "goal_gate"}:
        raise WorkflowError(f"Unknown workflow contract fields: {sorted(set(contract) - {'entry', 'steps', 'goal_gate'})}.")
    allowed = {"kind", "next", "branches", "repeat", "max_iterations", "on_exhausted",
               "max_retries", "on_error", "terminal_outcome", "effect", "resume_step"}
    for name, step in contract["steps"].items():
        if not isinstance(step, Mapping):
            raise WorkflowError(f"Workflow contract step {name!r} must be a mapping.")
        unknown = set(step) - allowed
        if unknown:
            raise WorkflowError(f"Unknown fields on workflow contract step {name!r}: {sorted(unknown)}.")
        if "max_retries" in step and (not isinstance(step["max_retries"], int)
                                       or isinstance(step["max_retries"], bool)
                                       or not 1 <= step["max_retries"] <= 5):
            raise WorkflowError(f"Workflow contract step {name!r} max_retries must be between 1 and 5.")
        if step.get("kind") == "task" and step.get("next") == END and step.get("terminal_outcome") not in TERMINAL_OUTCOMES:
            raise WorkflowError(f"Terminal workflow contract step {name!r} needs terminal_outcome.")
        branches = step.get("branches")
        if step.get("kind") in {"decision", "loop"} and (
            (isinstance(branches, Mapping) and END in branches.values()) or step.get("on_exhausted") == END
        ):
            raise WorkflowError(f"Workflow contract step {name!r} must route terminal paths through a task with terminal_outcome.")
    gate = contract.get("goal_gate")
    if gate is not None:
        if not isinstance(gate, Mapping) or gate.get("operator") not in {"gte", "exists"}:
            raise WorkflowError("Workflow contract goal_gate needs operator gte or exists.")
        for field in ("gate_step", "partial_step", "accept_step", "continue_step"):
            if gate.get(field) not in contract["steps"]:
                raise WorkflowError(f"Workflow contract goal_gate.{field} must name a step.")
        decision = contract["steps"][gate["gate_step"]]
        branches = decision.get("branches")
        if (decision.get("kind") != "decision" or not isinstance(branches, Mapping) or
                not {gate["partial_step"], gate["accept_step"]} <= set(branches.values())):
            raise WorkflowError("Workflow contract goal_gate must route to accept and partial steps.")
        if contract["steps"][gate["partial_step"]].get("terminal_outcome") != "partial":
            raise WorkflowError("Workflow contract goal_gate.partial_step must be a partial terminal.")
        for field in ("target", "actual"):
            if not isinstance(gate.get(field), str) or not _BINDING.match(gate[field]):
                raise WorkflowError(f"Workflow contract goal_gate.{field} must be a binding.")
    skeleton = {}
    for name, step in contract["steps"].items():
        schema = {"type": "object"}
        if step.get("kind") in {"decision", "loop"} and isinstance(step.get("branches"), Mapping):
            schema = {"type": "object", "properties": {"transition": {"enum": list(step["branches"])}},
                      "required": ["transition"]}
        skeleton[name] = {**step, "instructions": "Contract node.", "output_schema": schema}
    validate_workflow({"schema": SCHEMA, "id": "contract", "skill": "contract", "entry": contract["entry"],
                       "max_transitions": 1000, "steps": skeleton})
    return contract


def workflow_contract(skill_path: Path) -> Optional[Dict[str, Any]]:
    """Read the flow skeleton authored in SKILL.md, when one is present."""
    return parse_workflow_contract((skill_path / "SKILL.md").read_text(encoding="utf-8"))


_TOPOLOGY_FIELDS = ("kind", "next", "branches", "repeat", "max_iterations", "on_exhausted",
                    "max_retries", "on_error", "terminal_outcome", "effect", "resume_step")


def validate_workflow_contract(workflow: Mapping[str, Any], skill_path: Path, *,
                               require: bool = False) -> None:
    """Reject a graph that merges or reroutes declared skill checkpoints."""
    contract = workflow_contract(skill_path)
    if contract is None:
        if require:
            raise WorkflowError("Agent-created skills need a ## Workflow contract YAML block.")
        return
    if workflow.get("entry") != contract["entry"]:
        raise WorkflowError(f"Workflow entry must be {contract['entry']!r}.")
    expected, actual = contract["steps"], workflow.get("steps", {})
    if not isinstance(actual, Mapping):
        raise WorkflowError("Workflow steps must be a mapping.")
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    if missing or extra:
        raise WorkflowError(f"Workflow contract step mismatch: missing {sorted(missing)}, extra {sorted(extra)}.")
    if workflow.get("goal_gate") != contract.get("goal_gate"):
        raise WorkflowError("Workflow goal_gate differs from skill contract.")
    for name, declared in expected.items():
        if not isinstance(declared, Mapping):
            raise WorkflowError(f"Workflow contract step {name!r} must be a mapping.")
        if not isinstance(actual[name], Mapping):
            raise WorkflowError(f"Workflow step {name!r} must be a mapping.")
        for field in _TOPOLOGY_FIELDS:
            if field in {"max_retries", "effect"} and field not in declared:
                continue  # The generation agent chooses these when the skill leaves them open.
            expected_value = declared.get(field, 0 if field == "max_retries" else None)
            actual_value = actual[name].get(field, 0 if field == "max_retries" else None)
            if expected_value != actual_value:
                raise WorkflowError(f"Workflow step {name!r} {field} differs from skill contract: "
                                    f"expected {expected_value!r}, got {actual_value!r}.")


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
    from tools.delegation_output_schema import coerce_output_schema
    _validated, error = coerce_output_schema(dict(schema))
    if error:
        raise WorkflowError(f"{label} is invalid: {error}")


def _step_reference(binding: str) -> tuple[str, list[str]] | None:
    parts = binding.removeprefix("$").split(".")
    if len(parts) >= 3 and parts[0] == "steps" and parts[2] == "output":
        return parts[1], parts[3:]
    return None


def _schema_requires_path(schema: Mapping[str, Any], path: list[str]) -> bool:
    current: Any = schema
    for name in path:
        if not isinstance(current, Mapping) or name not in (current.get("required") or []):
            return False
        current = (current.get("properties") or {}).get(name)
    return True


def validate_workflow(workflow: Mapping[str, Any], *, skill: Optional[str] = None,
                      reliable: bool = False) -> Dict[str, Any]:
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
    max_transitions = data.get("max_transitions")
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
    edges: dict[str, list[str]] = {}
    for step_id, step in steps.items():
        if not isinstance(step_id, str) or step_id == END or not re.match(r"^[A-Za-z_][\w-]*$", step_id):
            raise WorkflowError("step ids must be identifiers.")
        if step_id in seen:
            raise WorkflowError(f"Duplicate step id {step_id!r}.")
        seen.add(step_id)
        if not isinstance(step, Mapping):
            raise WorkflowError(f"steps.{step_id} must be a mapping.")
        kind = step.get("kind")
        if kind not in NODE_KINDS:
            raise WorkflowError(f"steps.{step_id}.kind must be task, decision, or loop.")
        retries = step.get("max_retries", 0)
        minimum, maximum = (1, 5) if reliable else (0, 5)
        if not isinstance(retries, int) or isinstance(retries, bool) or not minimum <= retries <= maximum:
            raise WorkflowError(f"steps.{step_id}.max_retries must be between {minimum} and {maximum}.")
        error_target = step.get("on_error")
        if error_target is not None:
            if not isinstance(error_target, str) or not error_target or error_target == END:
                raise WorkflowError(f"steps.{step_id}.on_error must name a recovery step.")
            targets.add(error_target)
        outcome = step.get("terminal_outcome")
        if outcome is not None and outcome not in TERMINAL_OUTCOMES:
            raise WorkflowError(f"steps.{step_id}.terminal_outcome must be success, partial, or blocked.")
        if not isinstance(step.get("instructions"), str) or not step["instructions"].strip():
            raise WorkflowError(f"steps.{step_id}.instructions is required.")
        bindings = step.get("input", {})
        if not isinstance(bindings, Mapping):
            raise WorkflowError(f"steps.{step_id}.input must be a mapping.")
        preconditions = step.get("preconditions", {})
        if not isinstance(preconditions, Mapping):
            raise WorkflowError(f"steps.{step_id}.preconditions must be a mapping.")
        for key, expected in preconditions.items():
            if not isinstance(key, str) or key not in bindings:
                raise WorkflowError(f"steps.{step_id}.preconditions field {key!r} is absent from this step's input; "
                                    f"available fields: {sorted(bindings)}.")
            if not isinstance(expected, (str, int, float, bool, type(None))):
                raise WorkflowError(f"steps.{step_id}.preconditions field {key!r} must compare a scalar value.")
            if isinstance(expected, str) and expected.startswith("$"):
                raise WorkflowError(f"steps.{step_id}.preconditions field {key!r} must compare a literal value, "
                                    "not a binding; the step input already resolves bindings.")
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
        if retries and not set(toolsets) <= {"web"} and not reliable:
            raise WorkflowError(f"steps.{step_id} cannot automatically retry toolsets with possible side effects.")
        if kind == "task":
            nxt = step.get("next")
            if not isinstance(nxt, str) or not nxt:
                raise WorkflowError(f"steps.{step_id}.next must name one target.")
            if any(key in step for key in ("branches", "repeat", "max_iterations", "on_exhausted")):
                raise WorkflowError(f"steps.{step_id} is a task; use only next.")
            edges[step_id] = [nxt]
            if outcome is not None and nxt != END:
                raise WorkflowError(f"steps.{step_id}.terminal_outcome is only valid before END.")
            if error_target is not None:
                edges[step_id].append(error_target)
            targets.add(nxt)
            continue

        if "next" in step:
            raise WorkflowError(f"steps.{step_id} is a {kind}; use branches, not next.")
        branches = step.get("branches")
        if not isinstance(branches, Mapping) or not branches or not all(
            isinstance(label, str) and label and isinstance(target, str) and target
            for label, target in branches.items()
        ):
            raise WorkflowError(f"steps.{step_id}.branches must map non-empty labels to targets.")
        transition = (step.get("output_schema", {}).get("properties", {}) or {}).get("transition")
        enum = transition.get("enum") if isinstance(transition, Mapping) else None
        required = step.get("output_schema", {}).get("required", [])
        if (not isinstance(enum, list) or not all(isinstance(label, str) and label for label in enum)
                or set(enum) != set(branches) or not isinstance(required, list) or "transition" not in required):
            raise WorkflowError(f"steps.{step_id}.output_schema transition enum must exactly match branches.")
        if kind == "decision" and any(key in step for key in ("repeat", "max_iterations", "on_exhausted")):
            raise WorkflowError(f"steps.{step_id} is a decision; loop settings are not allowed.")
        edges[step_id] = list(branches.values())
        if outcome is not None:
            raise WorkflowError(f"steps.{step_id}.terminal_outcome is only valid on terminal tasks.")
        if error_target is not None:
            edges[step_id].append(error_target)
        targets.update(branches.values())
        if kind == "loop":
            limit = step.get("max_iterations")
            repeat = step.get("repeat")
            exhausted = step.get("on_exhausted")
            if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
                raise WorkflowError(f"steps.{step_id}.max_iterations must be an integer between 1 and 1000.")
            if not isinstance(repeat, str) or repeat not in branches:
                raise WorkflowError(f"steps.{step_id}.repeat must name a branch.")
            if not isinstance(exhausted, str) or not exhausted:
                raise WorkflowError(f"steps.{step_id}.on_exhausted must name a target.")
            edges[step_id].append(exhausted)
            targets.add(exhausted)
    unknown = {target for target in targets if target != END and target not in steps}
    if unknown:
        raise WorkflowError(f"Unknown step target(s): {', '.join(sorted(unknown))}.")
    # Reachability catches dead authored steps. Per-loop limits bound authored
    # cycles; max_transitions is the whole-run safety backstop.
    reachable, pending = set(), [entry]
    while pending:
        step_id = pending.pop()
        if step_id in reachable or step_id == END:
            continue
        reachable.add(step_id)
        pending.extend(edges[step_id])
    dead = set(steps) - reachable
    if dead:
        raise WorkflowError(f"Unreachable step(s): {', '.join(sorted(dead))}.")
    conservative_transitions = len(steps) + sum(
        step.get("max_iterations", 1) - 1 for step in steps.values() if step["kind"] == "loop"
    )
    if max_transitions < conservative_transitions:
        raise WorkflowError(
            f"max_transitions must be at least {conservative_transitions} for this graph's steps and loop bounds."
        )
    for step_id, step in steps.items():
        if step["kind"] != "loop":
            continue
        pending, seen_targets = [step["branches"][step["repeat"]]], set()
        while pending:
            target = pending.pop()
            if target == step_id:
                break
            if target in seen_targets or target == END:
                continue
            seen_targets.add(target)
            pending.extend(edges[target])
        else:
            raise WorkflowError(f"steps.{step_id}.repeat must lead back to the loop.")

    visiting, visited = [], set()

    def visit(step_id: str) -> None:
        if step_id == END or step_id in visited:
            return
        if step_id in visiting:
            cycle = visiting[visiting.index(step_id):]
            if not any(steps[item]["kind"] == "loop" for item in cycle):
                raise WorkflowError(f"Cycle {' -> '.join(cycle)} must include a loop node.")
            return
        visiting.append(step_id)
        for target in edges[step_id]:
            visit(target)
        visiting.pop()
        visited.add(step_id)

    visit(entry)

    predecessors: dict[str, list[tuple[str, bool]]] = {step_id: [] for step_id in steps}
    for source, destinations in edges.items():
        for destination in destinations:
            if destination != END:
                predecessors[destination].append((source, destination != steps[source].get("on_error")))
    guaranteed: dict[str, set[str]] = {
        step_id: set() if step_id == entry else set(steps) for step_id in steps
    }
    changed = True
    while changed:
        changed = False
        for step_id in steps:
            if step_id == entry:
                continue
            incoming = [guaranteed[parent] | ({parent} if succeeded else set())
                        for parent, succeeded in predecessors[step_id]]
            value = set.intersection(*incoming) if incoming else set()
            if value != guaranteed[step_id]:
                guaranteed[step_id] = value
                changed = True

    def reference_path(source: str, target: str, *, produced_output: bool) -> str:
        """Show a route to target with or without source's successful output."""
        pending = deque([(entry, False, entry)])
        seen = set()
        while pending:
            current, produced, path = pending.popleft()
            if current == target and produced == produced_output:
                return path
            if (current, produced) in seen:
                continue
            seen.add((current, produced))
            step = steps[current]
            routes = ([("next", step["next"])] if step["kind"] == "task"
                      else list(step["branches"].items()))
            if step["kind"] == "loop":
                routes.append(("on_exhausted", step["on_exhausted"]))
            if step.get("on_error"):
                routes.append(("on_error", step["on_error"]))
            for label, destination in routes:
                if destination != END:
                    succeeded = current == source and label != "on_error"
                    pending.append((destination, produced or succeeded,
                                    f"{path} --{label}--> {destination}"))
        return ""

    def validate_references(value: Any, label: str, *, available: Optional[set[str]] = None) -> None:
        for binding in _references(value):
            if binding.startswith("$input."):
                path = binding.removeprefix("$input.").split(".")
                if path[0] == "previous_output" and label.startswith("steps."):
                    raise WorkflowError(f"{label} binding {binding!r} is unavailable on the first visit; "
                                        "loop previous_output is supplied automatically on repeats.")
                if not _schema_requires_path(data.get("input_schema", {}), path):
                    raise WorkflowError(f"{label} binding {binding!r} is not required by input_schema.")
                continue
            reference = _step_reference(binding)
            if reference is None:
                continue
            source, path = reference
            if source not in steps:
                raise WorkflowError(f"{label} references unknown step {source!r}.")
            if available is not None and source not in available:
                target = label.split(".")[1]
                missing_path = reference_path(source, target, produced_output=False)
                available_path = reference_path(source, target, produced_output=True)
                contrast = f" Path with its output: {available_path}." if available_path else ""
                raise WorkflowError(f"{label} binding {binding!r} references step {source!r}, which is not "
                                    f"guaranteed to run first. Path without its output: {missing_path}." + contrast)
            if path and not _schema_requires_path(steps[source].get("output_schema", {}), path):
                raise WorkflowError(f"{label} references optional or undeclared output {binding!r}.")

    for step_id, step in steps.items():
        validate_references(step.get("input", {}), f"steps.{step_id}.input", available=guaranteed[step_id])
        if reliable:
            effect = step.get("effect")
            recovery = step.get("recovery")
            if effect not in {"read", "write", "external_action"}:
                raise WorkflowError(f"steps.{step_id}.effect must be read, write, or external_action.")
            method = "retry" if effect == "read" else "reconcile"
            if not isinstance(recovery, Mapping) or recovery.get("method") != method:
                raise WorkflowError(f"steps.{step_id}.recovery.method must be {method}.")
            if effect != "read" and (not isinstance(recovery.get("instructions"), str)
                                     or not recovery["instructions"].strip()):
                raise WorkflowError(f"steps.{step_id}.recovery needs reconciliation instructions.")
            if step.get("terminal_outcome") in {"partial", "blocked"}:
                reason = step.get("terminal_reason")
                evidence = reason.get("evidence") if isinstance(reason, Mapping) else None
                if (not isinstance(reason, Mapping) or not isinstance(reason.get("code"), str)
                        or not reason["code"] or not isinstance(evidence, list)
                        or not all(isinstance(key, str) and key in step.get("input", {}) for key in evidence)):
                    raise WorkflowError(f"steps.{step_id}.terminal_reason must use a code and own input fields.")
                if reason["code"] not in TERMINAL_REASON_CODES[step["terminal_outcome"]]:
                    raise WorkflowError(f"steps.{step_id}.terminal_reason.code must be a typed, generic code: "
                                        f"{sorted(TERMINAL_REASON_CODES[step['terminal_outcome']])}.")
            resume_step = step.get("resume_step")
            if resume_step is not None and (step.get("terminal_outcome") != "blocked" or resume_step not in steps):
                raise WorkflowError(f"steps.{step_id}.resume_step must target a step from a blocked terminal.")
    if reliable:
        gate = data.get("goal_gate")
        if gate is None and any(step.get("terminal_outcome") == "partial" for step in steps.values()):
            raise WorkflowError("A partial terminal requires goal_gate and approval routes.")
        if gate is not None:
            if not isinstance(gate, Mapping) or gate.get("operator") not in {"gte", "exists"}:
                raise WorkflowError("goal_gate needs operator gte or exists.")
            for field in ("gate_step", "partial_step", "accept_step", "continue_step"):
                if gate.get(field) not in steps:
                    raise WorkflowError(f"goal_gate.{field} must name a step.")
            if steps[gate["partial_step"]].get("terminal_outcome") != "partial":
                raise WorkflowError("goal_gate.partial_step must be a partial terminal.")
            gate_step = steps[gate["gate_step"]]
            if (gate_step.get("kind") != "decision" or
                    not {gate["partial_step"], gate["accept_step"]} <= set(gate_step.get("branches", {}).values())):
                raise WorkflowError("goal_gate decision must branch to partial_step and accept_step.")
            for field in ("target", "actual"):
                if not isinstance(gate.get(field), str) or not _BINDING.match(gate[field]):
                    raise WorkflowError(f"goal_gate.{field} must be a binding.")
                validate_references(gate[field], f"steps.{gate['gate_step']}.goal_gate",
                                    available=guaranteed[gate["gate_step"]])
            if gate["gate_step"] not in guaranteed[gate["accept_step"]]:
                raise WorkflowError("goal_gate.gate_step must precede every path to accept_step.")
            if gate["gate_step"] not in guaranteed[gate["partial_step"]]:
                raise WorkflowError("goal_gate.gate_step must precede every path to partial_step.")
        elif ("target_count" in (data.get("input_schema", {}).get("properties") or {}) or
              any("target_count" in (step.get("output_schema", {}).get("properties") or {})
                  for step in steps.values())):
            raise WorkflowError("A declared target_count requires goal_gate.")
    validate_references(data.get("output", {}), "output")
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


_UNAVAILABLE = object()


def _resolve_final_value(value: Any, state: Mapping[str, Any]) -> tuple[Any, bool]:
    """Resolve branch-dependent final output, omitting bindings from paths not taken."""
    if isinstance(value, Mapping):
        resolved, unavailable = {}, False
        for key, child in value.items():
            item, missing = _resolve_final_value(child, state)
            unavailable |= missing
            if item is not _UNAVAILABLE:
                resolved[key] = item
        return resolved, unavailable
    if isinstance(value, list):
        resolved, unavailable = [], False
        for child in value:
            item, missing = _resolve_final_value(child, state)
            unavailable |= missing
            if item is not _UNAVAILABLE:
                resolved.append(item)
        return resolved, unavailable
    try:
        return _resolve_binding(value, state), False
    except WorkflowError as exc:
        if isinstance(value, str) and value.startswith("$steps.") and "is not available yet" in str(exc):
            return _UNAVAILABLE, True
        raise


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
    lines = ["flowchart TD", f"  workflow_start([START]) --> {graph['entry']}", "  workflow_end((END))"]
    for step_id, step in graph["steps"].items():
        label = step_id.replace('"', "'")
        route = step.get("route", graph["defaults"].get("route", {}))
        model = route.get("model") if isinstance(route, Mapping) else None
        suffix = "\\n" + str(model) if model else ""
        if step["kind"] == "decision":
            lines.append(f"  {step_id}{{{label}}}")
        elif step["kind"] == "loop":
            lines.append(f"  {step_id}{{{label} loop max {step['max_iterations']}}}")
        else:
            lines.append(f'  {step_id}["{label}{suffix}"]')
        if step["kind"] == "task":
            target = step["next"]
            lines.append(f"  {step_id} --> {'workflow_end' if target == END else target}")
            if step.get("on_error"):
                lines.append(f"  {step_id} -. error .-> {step['on_error']}")
            continue
        for transition, target in step["branches"].items():
            edge_label = re.sub(r"[^A-Za-z0-9_-]+", "_", str(transition)).strip("_") or "branch"
            if step["kind"] == "loop" and transition == step["repeat"]:
                edge_label = f"repeat_max_{step['max_iterations']}"
            lines.append(f"  {step_id} -->|{edge_label}| {'workflow_end' if target == END else target}")
        if step["kind"] == "loop":
            target = step["on_exhausted"]
            lines.append(f"  {step_id} -. exhausted .-> {'workflow_end' if target == END else target}")
        if step.get("on_error"):
            lines.append(f"  {step_id} -. error .-> {step['on_error']}")
    return "\n".join(lines)


def workflow_sources(skill: str) -> list[WorkflowSource]:
    from tools.skill_manager_tool import _find_skill
    found = _find_skill(skill)
    if not found:
        raise WorkflowError(f"Skill {skill!r} was not found.")
    skill_dir = Path(found["path"])
    contract = workflow_contract(skill_dir)
    fingerprint = _fingerprint(skill_dir)
    result = []
    for path in sorted((skill_dir / "workflows").glob("*.yaml")):
        try:
            data = _load_yaml(path)
            stale = contract is not None and data.get("skill_fingerprint") != fingerprint
        except WorkflowError:
            stale = False
        result.append(WorkflowSource(skill, path, "packaged", stale))
    overlay = _overlay_dir(skill_dir)
    for path in sorted(overlay.glob("*.yaml")):
        stale = False
        activation = "active"
        try:
            data = _load_yaml(path)
            stale = data.get("skill_fingerprint") not in (None, fingerprint)
            activation = data.get("activation", "active")
            if activation not in {"active", "candidate"}:
                activation = "candidate"
        except WorkflowError:
            pass
        result.append(WorkflowSource(skill, path, "overlay", stale, activation))
    return result


def load_workflow(skill: str, name: str = "default") -> tuple[Dict[str, Any], WorkflowSource]:
    sources = workflow_sources(skill)
    from tools.skill_manager_tool import _find_skill
    skill_dir = Path(_find_skill(skill)["path"])
    matches = [source for source in sources if source.path.stem == name]
    if not matches:
        raise WorkflowError(f"Skill {skill!r} has no workflow named {name!r}.")
    # A fresh active overlay is more specific than the installed package. A stale,
    # invalid, or candidate overlay never hides an authored packaged workflow.
    ordered = sorted(matches, key=lambda item: item.source != "overlay")
    errors = []
    for source in ordered:
        if source.stale or source.activation != "active":
            continue
        try:
            graph = validate_workflow(_load_yaml(source.path), skill=skill, reliable=True)
            validate_workflow_contract(graph, skill_dir)
            return graph, source
        except WorkflowError as exc:
            errors.append(str(exc))
    if any(source.activation == "candidate" for source in matches):
        raise WorkflowError(f"Workflow {name!r} exists only as an inactive candidate.")
    if any(source.stale for source in matches):
        raise WorkflowError(f"Workflow {name!r} is stale and must be regenerated before it can run.")
    raise WorkflowError(errors[0] if errors else f"Workflow {name!r} has no active usable source.")


def save_overlay(
    skill: str, workflow: Mapping[str, Any], *, name: str = "default", activation: str = "active",
) -> Path:
    from tools.skill_manager_tool import _find_skill
    found = _find_skill(skill)
    if not found:
        raise WorkflowError(f"Skill {skill!r} was not found.")
    normalized = validate_workflow(workflow, skill=skill, reliable=True)
    validate_workflow_contract(normalized, Path(found["path"]))
    if activation not in {"active", "candidate"}:
        raise WorkflowError("activation must be 'active' or 'candidate'.")
    normalized["skill_fingerprint"] = _fingerprint(Path(found["path"]))
    normalized["activation"] = activation
    target = _overlay_dir(Path(found["path"])) / f"{name}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, _safe_dump(normalized), preserve_mode=True, create_mode=0o600)
    return target


def activate_workflow(skill: str, name: str = "default") -> Path:
    """Promote an overlay candidate without mutating packaged skill content."""
    matches = [source for source in workflow_sources(skill)
               if source.path.stem == name and source.source == "overlay"]
    if not matches:
        raise WorkflowError(f"Skill {skill!r} has no overlay workflow named {name!r}.")
    source = matches[0]
    if source.stale:
        raise WorkflowError(f"Workflow {name!r} is stale and must be regenerated before activation.")
    data = validate_workflow(_load_yaml(source.path), skill=skill, reliable=True)
    from tools.skill_manager_tool import _find_skill
    validate_workflow_contract(data, Path(_find_skill(skill)["path"]))
    data["skill_fingerprint"] = workflow_source_fingerprint(skill)
    data["activation"] = "active"
    atomic_write_text(source.path, _safe_dump(data), preserve_mode=True, create_mode=0o600)
    return source.path


def has_valid_candidate(skill: str, name: str = "default") -> bool:
    """Whether explicit activation, rather than another compilation, is the next action."""
    for source in workflow_sources(skill):
        if source.path.stem != name or source.activation != "candidate" or source.stale:
            continue
        try:
            validate_workflow(_load_yaml(source.path), skill=skill, reliable=True)
            from tools.skill_manager_tool import _find_skill
            validate_workflow_contract(_load_yaml(source.path), Path(_find_skill(skill)["path"]))
            return True
        except WorkflowError:
            pass
    return False


def execute_workflow(
    workflow: Mapping[str, Any], inputs: Mapping[str, Any],
    run_step: Callable[[str, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
    *, receipt_path: Optional[Path] = None,
    event_callback: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
    run_id: Optional[str] = None,
    resume_state: Optional[Mapping[str, Any]] = None,
    resume_choice: str = "",
    checkpoint_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run a validated graph with an injected isolated-agent step executor."""
    declared_steps = workflow.get("steps")
    reliable = isinstance(declared_steps, Mapping) and any(
        isinstance(step, Mapping) and "effect" in step for step in declared_steps.values())
    graph = validate_workflow(workflow, reliable=reliable)
    state: Dict[str, Any] = dict(resume_state["state"]) if resume_state else {"input": dict(inputs), "steps": {}}
    current = str(resume_state["current"]) if resume_state else graph["entry"]
    last_step = resume_state.get("last_step") if resume_state else None
    transitions = int(resume_state.get("transitions", 0)) if resume_state else 0
    loop_repeats = dict(resume_state.get("loop_repeats", {})) if resume_state else {}
    visit_counts = dict(resume_state.get("visit_counts", {})) if resume_state else {}
    accepted_partial = bool(resume_state.get("accepted_partial", False)) if resume_state else False
    pending = resume_state.get("pending") if resume_state else None
    reconcile_first = False
    outcome = "success"
    run_id = run_id or uuid.uuid4().hex
    events: list[Dict[str, Any]] = []
    if receipt_path is not None:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
    def checkpoint(pending_action: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        value = {"state": state, "current": current, "last_step": last_step,
                 "transitions": transitions, "loop_repeats": loop_repeats,
                 "visit_counts": visit_counts,
                 "accepted_partial": accepted_partial,
                 "pending": pending_action}
        if checkpoint_callback:
            checkpoint_callback(value)
        return value

    if pending:
        choices = pending.get("choices", {})
        if resume_choice not in choices:
            raise WorkflowError(f"Resume requires one of {sorted(choices)}.")
        if resume_choice == "stop":
            return {"run_id": run_id, "output": pending["output"], "events": [],
                    "outcome": pending["outcome"]}
        current = choices[resume_choice]
        reconcile_first = pending.get("code") == "side_effect_unknown" and resume_choice == "retry"
        if resume_choice == "accept":
            accepted_partial = True
        transitions = 0  # Each user-authorized continuation gets a fresh bounded cycle.
        if resume_choice == "continue":
            loop_repeats[current] = 0
        checkpoint()
    while current != END:
        if transitions >= graph["max_transitions"]:
            raise WorkflowError("Workflow exceeded max_transitions.")
        step = graph["steps"][current]
        gate = graph.get("goal_gate")
        if gate and current == gate["gate_step"]:
            target = _resolve_binding(gate["target"], state)
            actual = _resolve_binding(gate["actual"], state)
            if gate["operator"] == "gte":
                if (not isinstance(actual, (int, float)) or isinstance(actual, bool)
                        or not isinstance(target, (int, float)) or isinstance(target, bool)):
                    raise WorkflowError("goal_gate gte values must be numbers.")
                met = actual >= target
            else:
                met = actual is not None and actual is not False and actual != "" and actual != []
            chosen = gate["accept_step"] if met else gate["partial_step"]
            branch = next(label for label, destination in step["branches"].items() if destination == chosen)
            output = {"transition": branch, "target": target, "actual": actual}
            state["steps"][current] = {"output": output}
            event = {"run_id": run_id, "skill": graph["skill"], "workflow": graph["id"],
                     "step": current, "attempt": 1, "input": {}, "output": _receipt_value(output),
                     "transition": chosen, "started_at": time.time(), "finished_at": time.time(),
                     "route": {}, "allowed_toolsets": []}
            if event_callback:
                event_callback("step_started", event)
                event_callback("step_completed", event)
            events.append(event)
            if receipt_path is not None:
                with receipt_path.open("a", encoding="utf-8") as receipt:
                    receipt.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            last_step, current, transitions = current, chosen, transitions + 1
            checkpoint()
            if transitions >= graph["max_transitions"] and current != END:
                raise WorkflowError("Workflow exceeded max_transitions.")
            step = graph["steps"][current]
        resolved_input = _resolve_value(step.get("input", {}), state)
        if step["kind"] == "loop" and current in state["steps"]:
            resolved_input = {**resolved_input, "previous_output": state["steps"][current]["output"]}
        for key, expected in step.get("preconditions", {}).items():
            if resolved_input[key] != expected:
                raise WorkflowError(f"Step {current!r} precondition {key!r} was not met.")
        route = {**dict(graph["defaults"].get("route", {})), **dict(step.get("route", {}))}
        allowed_toolsets = list(step.get("toolsets", graph["defaults"].get("toolsets", [])))
        retry_error = ""
        if not reconcile_first:
            visit_counts[current] = visit_counts.get(current, 0) + 1
        operation_id = f"{run_id}:{current}:{visit_counts[current]}"
        for attempt in range(1, step.get("max_retries", 0) + 2):
            started = time.time()
            base_event = {"run_id": run_id, "skill": graph["skill"], "workflow": graph["id"],
                          "step": current, "attempt": attempt, "route": route,
                          "allowed_toolsets": allowed_toolsets, "started_at": started}
            if event_callback:
                event_callback("step_started", {**base_event, "input": resolved_input})
            try:
                context = {"input": resolved_input, "route": route, "workflow": graph,
                           "attempt": attempt, "previous_error": retry_error,
                           "operation_id": operation_id}
                if (attempt > 1 or reconcile_first) and step.get("effect") in {"write", "external_action"}:
                    try:
                        checked = dict(run_step(current, step, {**context, "reconcile": True}))
                    except Exception:
                        checked = {"state": "unknown"}
                    observed = checked.get("state")
                    if observed == "done" and not isinstance(checked.get("output"), Mapping):
                        observed = "unknown"
                    if observed not in {"done", "not_done"}:
                        if event_callback:
                            event_callback("step_failed", {**base_event, "error": "side_effect_unknown",
                                                           "finished_at": time.time(), "retry": False})
                        pending_action = {"outcome": "blocked", "code": "side_effect_unknown",
                                          "step": current, "output": {"reason_code": "side_effect_unknown",
                                          "operation_id": context["operation_id"],
                                          "action": "Verify the operation's observed state, repair access if needed, then retry."},
                                          "choices": {"retry": current}}
                        snapshot = checkpoint(pending_action)
                        return {"run_id": run_id, "output": pending_action["output"],
                                "events": events, "outcome": "blocked", "paused": True,
                                "checkpoint": snapshot, "pending_action": pending_action}
                    output = dict(checked["output"]) if observed == "done" else dict(run_step(current, step, context))
                elif step.get("terminal_reason"):
                    reason = step["terminal_reason"]
                    if reason["code"] in {"external_unreadable", "verification_unreadable"} and not any(
                            value == "unreadable" for value in
                            (resolved_input[key] for key in reason["evidence"])):
                        raise WorkflowError(f"Step {current!r} cannot claim unreadable without unreadable evidence.")
                    action = ("Repair access to the unreadable external state and retry the check."
                              if reason["code"] in {"external_unreadable", "verification_unreadable"} else
                              "Review the observed evidence and repair the blocking condition."
                              if reason["code"] in {"blocked_condition", "recovery_exhausted"}
                              else reason.get("action", ""))
                    output = {"reason_code": reason["code"],
                              "evidence": {key: resolved_input[key] for key in reason["evidence"]},
                              "action": action}
                else:
                    output = dict(run_step(current, step, context))
                execution = output.pop("__workflow_meta__", None)
                exhausted = False
                if step["kind"] == "task":
                    chosen = step["next"]
                else:
                    transition = output.get("transition")
                    chosen = step["branches"].get(transition)
                    if chosen is None:
                        raise WorkflowError(f"Step {current!r} returned unknown transition {transition!r}.")
                    if step["kind"] == "loop" and transition == step["repeat"]:
                        loop_repeats[current] = loop_repeats.get(current, 0) + 1
                        if loop_repeats[current] >= step["max_iterations"]:
                            chosen, exhausted = step["on_exhausted"], True
            except BaseException as exc:
                cancelled = isinstance(exc, (KeyboardInterrupt, SystemExit))
                retry = not cancelled and attempt <= step.get("max_retries", 0)
                error_target = step.get("on_error") if not retry and not cancelled else None
                if event_callback:
                    event_callback("step_cancelled" if cancelled else "step_failed", {
                        **base_event, "error": str(exc), "finished_at": time.time(),
                        "retry": retry, **({"transition": error_target} if error_target else {}),
                    })
                if cancelled:
                    raise
                if retry:
                    retry_error = str(exc)
                    continue
                if error_target:
                    current, transitions = error_target, transitions + 1
                    reconcile_first = False
                    break
                raise
            state["steps"][current] = {"output": output}
            event = {**base_event, "input": _receipt_value(resolved_input),
                     "output": _receipt_value(output), "route": _receipt_value(route),
                     "transition": chosen, "exhausted": exhausted, "finished_at": time.time()}
            if isinstance(execution, Mapping):
                event["execution"] = dict(execution)
            events.append(event)
            if event_callback:
                event_callback("step_completed", event)
            if receipt_path is not None:
                with receipt_path.open("a", encoding="utf-8") as receipt:
                    receipt.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            last_step, current, transitions = current, chosen, transitions + 1
            reconcile_first = False
            outcome = step.get("terminal_outcome", "success") if chosen == END else outcome
            if chosen == END and outcome == "partial" and gate:
                pending_action = {"outcome": "partial", "code": output.get("reason_code", "goal_unmet"),
                                  "output": output, "choices": {"accept": gate["accept_step"],
                                  "continue": gate["continue_step"], "stop": END}}
                snapshot = checkpoint(pending_action)
                return {"run_id": run_id, "output": output, "events": events,
                        "outcome": "partial", "paused": True, "checkpoint": snapshot,
                        "pending_action": pending_action}
            if chosen == END and outcome == "blocked" and step.get("resume_step"):
                pending_action = {"outcome": "blocked", "code": output.get("reason_code", "blocked"),
                                  "output": output, "choices": {"retry": step["resume_step"]}}
                snapshot = checkpoint(pending_action)
                return {"run_id": run_id, "output": output, "events": events,
                        "outcome": "blocked", "paused": True, "checkpoint": snapshot,
                        "pending_action": pending_action}
            checkpoint()
            break
    declared_output = graph.get("output", {})
    output, branch_binding_missing = _resolve_final_value(declared_output, state)
    if last_step is not None and (branch_binding_missing or declared_output == {}):
        terminal_output = state["steps"][last_step]["output"]
        if declared_output == {}:
            output = terminal_output
        elif isinstance(output, Mapping) and isinstance(terminal_output, Mapping):
            output = {**terminal_output, **output}
        elif output is _UNAVAILABLE:
            output = terminal_output
    return {"run_id": run_id, "output": output, "events": events,
            "outcome": "partial" if accepted_partial and outcome == "success" else outcome}
