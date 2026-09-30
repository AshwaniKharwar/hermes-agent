"""Profile-owned binding between an agent-created skill and its approved graph."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hermes_constants import get_hermes_home
from utils import atomic_json_write


def skill_digest(skill_dir: Path) -> str:
    """Hash the instructions and supporting files, including edits outside skill_manage."""
    digest = hashlib.sha256()
    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file() or path.is_symlink() or ".git" in path.parts:
            continue
        relative = path.relative_to(skill_dir).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def workflow_dir(skill_dir: Path, home: Path | None = None) -> Path:
    home = home or get_hermes_home()
    key = hashlib.sha256(str(skill_dir.resolve()).encode("utf-8")).hexdigest()[:24]
    return home / "workflows" / key


def read_record(directory: Path) -> dict | None:
    try:
        record = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        return record if isinstance(record, dict) and record.get("required") is True else None
    except (OSError, ValueError):
        return None


def save_record(directory: Path, record: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json_write(directory / "state.json", record)


def record_committed_skill(name: str, action: str) -> dict | None:
    """Called only after an approved, committed mutation (never per batch op)."""
    from tools.skill_manager_tool import _find_skill

    found = _find_skill(name)
    if not found:
        return None
    skill_dir = found["path"].resolve()
    directory = workflow_dir(skill_dir)
    existing = read_record(directory)
    if action != "create" and existing is None:
        return None
    from agent.skill_utils import parse_frontmatter

    frontmatter, _ = parse_frontmatter((skill_dir / "SKILL.md").read_text(encoding="utf-8"))
    record = dict(existing or {})
    record.update(
        required=True, name=name, skill_path=str(skill_dir),
        description=str(frontmatter.get("description", "")),
        source_sha256=skill_digest(skill_dir), status="pending", completed_failures=0,
    )
    record.pop("error", None)
    save_record(directory, record)
    return record


def required_records(home: Path | None = None) -> list[tuple[Path, dict]]:
    root = (home or get_hermes_home()) / "workflows"
    if not root.exists():
        return []
    return [(directory, record) for directory in root.iterdir()
            if directory.is_dir() and (record := read_record(directory)) is not None]


def current_skill(record: dict) -> Path | None:
    """A manifest cannot make an arbitrary file into a routable skill."""
    from tools.skill_manager_tool import _find_skill

    found = _find_skill(str(record.get("name", "")))
    if not found:
        return None
    path = found["path"].resolve()
    return path if str(path) == record.get("skill_path") else None


def refresh_source(directory: Path, record: dict) -> dict:
    skill_dir = current_skill(record)
    if skill_dir is None:
        return record
    digest = skill_digest(skill_dir)
    if digest != record.get("source_sha256"):
        from agent.skill_utils import parse_frontmatter

        frontmatter, _ = parse_frontmatter((skill_dir / "SKILL.md").read_text(encoding="utf-8"))
        record = dict(record, source_sha256=digest,
                      description=str(frontmatter.get("description", "")), status="pending",
                      completed_failures=0)
        record.pop("pending_id", None)
        record.pop("error", None)
        save_record(directory, record)
    return record


def graph_state(directory: Path, record: dict) -> tuple[str, dict | None]:
    skill_dir = current_skill(record)
    if skill_dir is None:
        return "missing_skill", None
    if skill_digest(skill_dir) != record.get("source_sha256"):
        return "stale", None
    if record.get("status") in {"awaiting_approval", "rejected"}:
        return record["status"], None
    if record.get("status") != "ready":
        return "missing", None
    try:
        graph = json.loads((directory / "graph.json").read_text(encoding="utf-8"))
        checklist = json.loads((directory / "requirements.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "missing", None
    except (OSError, ValueError):
        return "invalid", None
    if not isinstance(graph, dict):
        return "invalid", None
    if graph.get("source_sha256") != record.get("source_sha256"):
        return "stale", None
    canonical = json.dumps(graph, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != record.get("graph_sha256"):
        return "invalid", None
    try:
        from agent.skill_workflow_runtime import checklist_digest, validate_checklist, validate_graph
        validate_checklist(checklist, (skill_dir / "SKILL.md").read_text(encoding="utf-8"))
        if checklist_digest(checklist) != record.get("requirements_sha256"):
            return "invalid", None
        validate_graph(graph, checklist)
    except (ValueError, TypeError, ImportError):
        return "invalid", None
    return "ready", graph
