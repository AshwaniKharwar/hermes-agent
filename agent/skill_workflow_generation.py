"""Build and approve skill graphs in isolated Hermes subagents."""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from agent.skill_workflow_store import (
    current_skill, read_record, required_records, save_record, skill_digest, workflow_dir,
)
from tools.daemon_pool import DaemonThreadPoolExecutor
from utils import atomic_json_write

logger = logging.getLogger(__name__)
_EXECUTOR = DaemonThreadPoolExecutor(max_workers=2, thread_name_prefix="skill-workflow")
_inflight: set[tuple[str, str]] = set()
_lock = threading.Lock()
_MAX_COMPLETED_FAILURES = 3


class InterruptedGeneration(RuntimeError):
    pass


def _event(directory: Path, event: str, **details) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "generation.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "event": event,
                                 **details}, ensure_ascii=False) + "\n")


def _checklist_items(raw: list, skill: str, existing: list[dict] | None = None) -> list[dict]:
    from agent.skill_workflow_runtime import validate_checklist

    if not isinstance(raw, list):
        raise ValueError("reviewer did not return a requirements list")
    items = list(existing or [])
    seen = {item["quote"] for item in items}
    for item in raw:
        if (not isinstance(item, dict) or set(item) != {"quote", "description"}
                or not isinstance(item["quote"], str) or not isinstance(item["description"], str)):
            raise ValueError("reviewer returned an invalid requirement")
        if item["quote"] not in seen:
            items.append({"id": f"r{len(items) + 1}", **item})
            seen.add(item["quote"])
    validate_checklist(items, skill)
    return items


def _json_object(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("subagent did not return a JSON object")
    return parsed


def _ask_child(parent, goal: str, context: str) -> dict:
    from agent.subagent_lifecycle import (
        SubagentLaunchRequest, SubagentLifecycleService, SubagentState,
    )

    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(SubagentLaunchRequest(
        goal=goal, context=context, role="leaf", allowed_toolsets=(),
        include_workspace_context=False,
    ))
    terminal = service.wait(handle)
    result = service.result(handle)
    if getattr(terminal, "timed_out", False):
        raise RuntimeError(result.error_message or "workflow subagent timed out")
    if not terminal.completed or result.terminal_state in {
        SubagentState.INTERRUPTED, SubagentState.CANCELLED, SubagentState.CANCEL_REQUESTED,
    }:
        raise InterruptedGeneration(result.error_message or "workflow subagent interrupted")
    if result.terminal_state is not SubagentState.SUCCEEDED:
        raise RuntimeError(result.error_message or "workflow subagent failed")
    return _json_object(result.summary or "")


def _build(directory: Path, source_sha256: str, parent) -> None:
    record = read_record(directory)
    if not record or record.get("source_sha256") != source_sha256:
        return
    skill_dir = current_skill(record)
    if skill_dir is None or skill_digest(skill_dir) != source_sha256:
        return
    skill = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    if len(skill) > 27_000:
        _event(directory, "generation_stage_failed", stage="checklist",
               reason="SKILL.md exceeds workflow writer context budget")
        raise ValueError("SKILL.md exceeds workflow writer context budget")
    from agent.skill_workflow_runtime import checklist_digest, validate_graph

    try:
        raw_checklist = _ask_child(
            parent, "Extract the required workflow checklist from this skill.",
            "Return only JSON {requirements:[{quote:string, description:string}]}. Include every "
            "required step, constraint, conditional step, and completion check. Each quote must be "
            "an exact short passage from SKILL.md. Use each passage once; combine its constraints "
            "in one description. Do not add requirements from elsewhere. "
            "Never write files or call tools.\n\nSKILL.md:\n" + skill,
        )
        checklist = _checklist_items(raw_checklist.get("requirements"), skill)
    except Exception as exc:
        _event(directory, "generation_stage_failed", stage="checklist", reason=str(exc))
        raise
    atomic_json_write(directory / "requirements_draft.json", checklist)
    _event(directory, "checklist_created", requirements=len(checklist), source_sha256=source_sha256)
    feedback = ""
    for attempt in range(1, 4):
        contract = (
            "Return only JSON: {version:4, description:string, entry:string, max_steps:int, "
            "nodes:[{id:string, goal:string, covers:[requirement_id], "
            "inputs:[{key:lowercase_identifier, description:string, source:request|node_id.output_key}], "
            "outputs:[{key:lowercase_identifier, description:string}], "
            "effects:[lowercase_identifier], "
            "next:{success:id|END, blocked:id|END, error:id|END}}]}. "
            "Create a fresh graph from SKILL.md and the complete checklist; do not rely on any earlier draft. "
            "Each node is one Hermes agent task. Covers must name every checklist requirement. "
            "If a checklist passage names a tool or command, name it exactly in the covering node goal; do not paraphrase it. "
            "Request inputs are extracted as short quotes for that node only; node output inputs "
            "come from named earlier nodes. Never request the whole user prompt or unrelated inputs. "
            "Outputs are named string facts or artifact paths needed by later nodes; use [] when "
            "none are needed. Keep large results in an artifact and output its path. Effects name "
            "externally visible actions only; read-only checks may have outputs but need no effects. "
            "A node with effects must not be reachable again through any success, blocked, or error "
            "edges, including a path through another node. Keep read-only checks out of effects. "
            "Use an acyclic graph unless SKILL.md requires a retry; a retry cycle may contain only "
            "nodes with effects=[]. "
            "A node must not perform a later node's effect. Bound retries with max_steps <= 20. "
            "Execution nodes inherit all tools available to the parent agent; do not declare toolsets. "
            "Never write files or call tools.\n\nSKILL.md:\n"
            + skill + "\n\nREQUIREMENTS:\n" + json.dumps(checklist, ensure_ascii=False)
            + ("\n\nPREVIOUS FEEDBACK (no earlier graph is provided): " + feedback if feedback else "")
        )
        if len(contract) > 32_000:
            _event(directory, "generation_stage_failed", stage="writer", attempt=attempt,
                   reason="skill and checklist exceed workflow writer context budget")
            raise ValueError("skill and checklist exceed workflow writer context budget")
        try:
            graph = _ask_child(parent, "Design a LangGraph workflow from this skill.", contract)
        except Exception as exc:
            _event(directory, "generation_stage_failed", stage="writer", attempt=attempt, reason=str(exc))
            raise
        graph["source_sha256"] = source_sha256
        graph["requirements_sha256"] = checklist_digest(checklist)
        try:
            validate_graph(graph, checklist, require_coverage=False)
        except (TypeError, ValueError) as exc:
            feedback = f"Graph validation failed: {exc}. Return a fresh graph from SKILL.md and the checklist that fixes this error."
            _event(directory, "draft_rejected", attempt=attempt, stage="validation", reason=str(exc))
            continue
        uncovered = {item["id"] for item in checklist} - {
            identifier for node in graph["nodes"] for identifier in node["covers"]
        }
        graph_text = json.dumps(graph, ensure_ascii=False)
        if len(graph_text) > 14_000:
            feedback = "Graph draft exceeds the reviewer size budget; return a more concise graph."
            _event(directory, "draft_rejected", attempt=attempt, stage="size", reason=feedback)
            continue
        review_context = (
            "Return only JSON {aligned:boolean, checklist_complete:boolean, missing_ids:[string], "
            "missing_requirements:[{quote:string,description:string}], "
            "coverage_assignments:[{id:string,node:string,evidence:string}], reason:string}. Re-read SKILL.md "
            "to find any omitted checklist requirement, then check every checklist ID against the "
            "actual node goals, inputs, outputs, and edges; a claimed covers ID is not proof. "
            "Add missing_requirements only for SKILL.md passages not already quoted in the checklist; "
            "use missing_ids for incomplete implementation of an existing requirement. "
            "For an ID absent from covers but fulfilled by a node goal, assign it to that node "
            "with an exact substring of the node goal as evidence. Return coverage_assignments "
            "only for IDs listed under IDS ABSENT FROM COVERS; use [] when none are absent. "
            "If no node fulfills it, "
            "list it in missing_ids and reject the graph. "
            "Treat named tools and commands in SKILL.md as exact requirements, not paraphrases. "
            "Execution nodes inherit all tools available to the parent agent. Do not require or review "
            "per-node toolsets. Reject invented steps, missing completion checks, broad node inputs, "
            "and external actions assigned to the wrong node. Effects mean externally visible "
            "actions only; read-only checks need no effects. Outputs are needed only when later nodes "
            "consume their facts. "
            "Report missing checklist items with exact SKILL.md quotes.\n\nSKILL.md:\n" + skill
            + "\n\nREQUIREMENTS:\n" + json.dumps(checklist, ensure_ascii=False)
            + "\n\nIDS ABSENT FROM COVERS:\n" + json.dumps(sorted(uncovered))
        )
        if len(review_context) > 32_000:
            _event(directory, "generation_stage_failed", stage="review", attempt=attempt,
                   reason="skill and checklist exceed workflow reviewer context budget")
            raise ValueError("skill and checklist exceed workflow reviewer context budget")
        try:
            review = _ask_child(
                parent, "Independently review whether this graph faithfully implements the skill.\n\nGRAPH:\n" + graph_text,
                review_context,
            )
        except Exception as exc:
            _event(directory, "generation_stage_failed", stage="review", attempt=attempt, reason=str(exc))
            raise
        missing_ids = review.get("missing_ids")
        additions = review.get("missing_requirements")
        if (type(review.get("aligned")) is not bool or type(review.get("checklist_complete")) is not bool
                or not isinstance(missing_ids, list) or any(
                    not isinstance(item, str) or item not in {entry["id"] for entry in checklist}
                    for item in missing_ids)
                or not isinstance(additions, list) or not isinstance(review.get("reason"), str)):
            _event(directory, "generation_stage_failed", stage="review", attempt=attempt,
                   reason="workflow reviewer returned an invalid verdict")
            raise ValueError("workflow reviewer returned an invalid verdict")
        if review["aligned"] and review["checklist_complete"] and not missing_ids and not additions:
            reported_assignments = review.get("coverage_assignments", [])
            assignments = ([item for item in reported_assignments
                            if isinstance(item, dict) and item.get("id") in uncovered]
                           if isinstance(reported_assignments, list) else reported_assignments)
            by_id = {node["id"]: node for node in graph["nodes"]}
            if (not isinstance(assignments, list) or len(assignments) != len(uncovered)
                    or any(not isinstance(item, dict) or set(item) != {"id", "node", "evidence"}
                           or not isinstance(item["id"], str) or not isinstance(item["node"], str)
                           or item["node"] not in by_id
                           or not isinstance(item["evidence"], str) or not item["evidence"].strip()
                           or item["evidence"] not in by_id[item["node"]]["goal"]
                           for item in assignments)
                    or {item["id"] for item in assignments} != uncovered):
                feedback = f"Review must assign uncovered requirement IDs to goals with exact evidence: {', '.join(sorted(uncovered))}."
                _event(directory, "draft_rejected", attempt=attempt, stage="coverage_review", reason=feedback)
                continue
            for item in assignments:
                by_id[item["node"]]["covers"].append(item["id"])
            validate_graph(graph, checklist)
            _event(directory, "draft_accepted", attempt=attempt, requirements=len(checklist))
            break
        try:
            previous_count = len(checklist)
            checklist = _checklist_items(additions, skill, checklist)
        except ValueError as exc:
            _event(directory, "generation_stage_failed", stage="review", attempt=attempt, reason=str(exc))
            raise
        atomic_json_write(directory / "requirements_draft.json", checklist)
        reason = review["reason"][:1000]
        feedback = (f"IDs absent from covers: {', '.join(sorted(uncovered)) or 'none'}. "
                    f"Unmet checklist IDs: {', '.join(missing_ids) or 'none reported'}. "
                    f"Reviewer reason: {reason}. Return a fresh graph from SKILL.md and the complete checklist.")
        _event(directory, "draft_rejected", attempt=attempt, stage="review", reason=reason,
               missing_ids=missing_ids, added_requirements=len(checklist) - previous_count)
    else:
        raise ValueError(f"workflow generation exhausted three drafts: {feedback[:1000]}")
    latest = read_record(directory)
    if not latest or latest.get("source_sha256") != source_sha256 or skill_digest(skill_dir) != source_sha256:
        _event(directory, "generation_superseded", source_sha256=source_sha256)
        return
    record = latest
    payload = {
        "action": "workflow_publish", "name": record["name"],
        "skill_path": record["skill_path"], "source_sha256": source_sha256,
        "graph": graph, "checklist": checklist,
    }
    from tools import write_approval as wa
    if wa.write_approval_enabled(wa.SKILLS):
        existing = _pending_graph_id(record)
        pending_id = existing or wa.stage_write(
            wa.SKILLS, payload, summary=f"approve workflow for skill '{record['name']}'",
            origin="background_review",
        )["id"]
        record["status"] = "awaiting_approval"
        record["pending_id"] = pending_id
        save_record(directory, record)
    else:
        result = publish_graph(payload)
        if not result["success"]:
            raise ValueError(result["error"])


def publish_graph(payload: dict) -> dict:
    """CAS publication, also used by /skills approve."""
    skill_dir = Path(str(payload.get("skill_path", ""))).resolve()
    directory = workflow_dir(skill_dir)
    record = read_record(directory)
    if not record or current_skill(record) != skill_dir:
        return {"success": False, "error": "workflow skill is no longer available"}
    digest = skill_digest(skill_dir)
    if digest != record.get("source_sha256") or digest != payload.get("source_sha256"):
        return {"success": False, "error": "skill changed while workflow was being prepared"}
    graph = payload.get("graph")
    checklist = payload.get("checklist")
    from agent.skill_workflow_runtime import checklist_digest, validate_checklist, validate_graph
    try:
        validate_checklist(checklist, (skill_dir / "SKILL.md").read_text(encoding="utf-8"))
        validate_graph(graph, checklist)
    except (TypeError, ValueError) as exc:
        return {"success": False, "error": str(exc)}
    if graph.get("source_sha256") != digest:
        return {"success": False, "error": "workflow source hash mismatch"}
    atomic_json_write(directory / "requirements.json", checklist)
    atomic_json_write(directory / "graph.json", graph)
    record["status"] = "ready"
    record["requirements_sha256"] = checklist_digest(checklist)
    record["completed_failures"] = 0
    record.pop("error", None)
    record["graph_sha256"] = hashlib.sha256(
        json.dumps(graph, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    record.pop("pending_id", None)
    save_record(directory, record)
    return {"success": True, "message": f"Workflow for '{record['name']}' activated."}


def reject_graph(payload: dict) -> None:
    directory = workflow_dir(Path(str(payload.get("skill_path", ""))).resolve())
    record = read_record(directory)
    if record and record.get("source_sha256") == payload.get("source_sha256"):
        record["status"] = "rejected"
        record.pop("pending_id", None)
        save_record(directory, record)


def _pending_graph_id(record: dict) -> str | None:
    from tools import write_approval as wa

    for pending in wa.list_pending(wa.SKILLS):
        payload = pending.get("payload", {})
        if (payload.get("action") == "workflow_publish"
                and payload.get("skill_path") == record.get("skill_path")
                and payload.get("source_sha256") == record.get("source_sha256")):
            graph = payload.get("graph")
            if not isinstance(graph, dict) or graph.get("version") != 4 or not isinstance(payload.get("checklist"), list):
                wa.discard_pending(wa.SKILLS, pending["id"])
                continue
            return pending.get("id")
    return None


def _worker(directory: Path, digest: str, parent, key: tuple[str, str]) -> None:
    try:
        from tools.skill_usage import skill_file_lock
        from agent.skill_workflow_store import graph_state

        with skill_file_lock(directory / "build.lock"):
            record = read_record(directory)
            if (record and record.get("source_sha256") == digest
                    and record.get("completed_failures", 0) < _MAX_COMPLETED_FAILURES):
                status, _graph = graph_state(directory, record)
                if status not in {"ready", "awaiting_approval", "rejected"}:
                    _build(directory, digest, parent)
    except InterruptedGeneration as exc:
        _event(directory, "generation_interrupted", reason=str(exc))
        record = read_record(directory)
        if record and record.get("source_sha256") == digest:
            record["status"] = "pending"
            save_record(directory, record)
    except Exception as exc:
        logger.warning("Skill workflow generation failed for %s: %s", directory, exc, exc_info=True)
        _event(directory, "generation_failed", reason=str(exc))
        record = read_record(directory)
        if record and record.get("source_sha256") == digest:
            record["status"] = "failed"
            record["error"] = str(exc)[:500]
            record["completed_failures"] = record.get("completed_failures", 0) + 1
            save_record(directory, record)
    finally:
        with _lock:
            _inflight.discard(key)


def schedule_repair(record: dict, parent=None) -> bool:
    """Single flight by profile/skill/version. Pending state survives process exit."""
    if parent is None:
        from agent.subagent_lifecycle import get_active_subagent_parent
        parent = get_active_subagent_parent()
    if parent is None:
        return False
    skill_dir = current_skill(record)
    if skill_dir is None:
        return False
    directory = workflow_dir(skill_dir)
    digest = skill_digest(skill_dir)
    key = (str(directory), digest)
    with _lock:
        if (key in _inflight or record.get("status") in {"awaiting_approval", "rejected"}
                or record.get("completed_failures", 0) >= _MAX_COMPLETED_FAILURES):
            return False
        _inflight.add(key)
    try:
        _EXECUTOR.submit(contextvars.copy_context().run, _worker, directory, digest, parent, key)
    except Exception:
        with _lock:
            _inflight.discard(key)
        raise
    return True


def resume_pending(parent) -> None:
    from agent.skill_workflow_store import refresh_source
    from tools import write_approval as wa

    for directory, record in required_records():
        record = refresh_source(directory, record)
        pending = wa.get_pending(wa.SKILLS, record.get("pending_id", "")) if record.get("pending_id") else None
        if (pending and pending.get("payload", {}).get("action") == "workflow_publish"
                and (not isinstance(pending["payload"].get("graph"), dict)
                     or pending["payload"]["graph"].get("version") != 4
                     or not isinstance(pending["payload"].get("checklist"), list))):
            wa.discard_pending(wa.SKILLS, pending["id"])
            record["status"] = "pending"
            record.pop("pending_id", None)
            save_record(directory, record)
        if record.get("status") == "pending" and (pending_id := _pending_graph_id(record)):
            record["status"] = "awaiting_approval"
            record["pending_id"] = pending_id
            save_record(directory, record)
        if record.get("status") == "awaiting_approval" and not wa.get_pending(wa.SKILLS, record.get("pending_id", "")):
            record["status"] = "rejected"
            record.pop("pending_id", None)
            save_record(directory, record)
        if record.get("status") in {"pending", "failed"}:
            schedule_repair(record, parent)
