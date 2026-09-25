"""Bounded, profile-scoped background compilation for missing skill workflows."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home
from tools.daemon_pool import DaemonThreadPoolExecutor

_ABANDONED_SECONDS = 600
_FAILURE_BACKOFF_SECONDS = 3600
_POOL = DaemonThreadPoolExecutor(max_workers=2, thread_name_prefix="workflow-generation")
_ACTIVE: dict[tuple[str, str, str], tuple[float, str]] = {}
_ACTIVE_LOCK = threading.Lock()


def _lock_path(skill: str, fingerprint: str) -> Path:
    key = hashlib.sha256(f"{skill}:{fingerprint}".encode()).hexdigest()
    path = get_hermes_home() / "workflow-overlays" / ".generation-locks" / f"{key}.lock"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def _claim(path: Path, job_id: str) -> bool:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        try:
            if time.time() - path.stat().st_mtime <= _ABANDONED_SECONDS:
                return False
            path.unlink()
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except (OSError, FileExistsError):
            return False
    with os.fdopen(fd, "w") as handle:
        json.dump({"pid": os.getpid(), "created_at": time.time(), "job_id": job_id}, handle)
    return True


def _in_failure_backoff(skill: str, fingerprint: str) -> bool:
    from tools.skill_usage import load_usage
    row = load_usage().get(skill, {}).get("workflow_generation", {})
    if row.get("status") != "failed" or row.get("source_fingerprint") != fingerprint:
        return False
    try:
        return time.time() - datetime.fromisoformat(row["updated_at"]).timestamp() < _FAILURE_BACKOFF_SECONDS
    except (KeyError, TypeError, ValueError):
        return False


def enqueue_workflow_generation(skill: str, name: str = "default") -> Optional[str]:
    from agent.workflows import workflow_source_fingerprint
    from tools.skill_usage import record_workflow_generation

    fingerprint = workflow_source_fingerprint(skill)
    key = (str(get_hermes_home()), skill, fingerprint)
    if _in_failure_backoff(skill, fingerprint):
        return None
    job_id = uuid.uuid4().hex
    with _ACTIVE_LOCK:
        active = _ACTIVE.get(key)
        if active is not None and time.time() - active[0] <= _ABANDONED_SECONDS:
            return active[1]
        _ACTIVE[key] = (time.time(), job_id)
    lock = _lock_path(skill, fingerprint)
    if not _claim(lock, job_id):
        with _ACTIVE_LOCK:
            if _ACTIVE.get(key, (None, None))[1] == job_id:
                _ACTIVE.pop(key, None)
        from tools.skill_usage import load_usage
        try:
            owner = json.loads(lock.read_text(encoding="utf-8")).get("job_id")
        except (OSError, ValueError):
            owner = None
        return owner or load_usage().get(skill, {}).get("workflow_generation", {}).get("job_id")
    record_workflow_generation(skill, status="queued", fingerprint=fingerprint, job_id=job_id)

    def generate() -> None:
        try:
            from hermes_cli.workflow_cmd import _generate
            from tools.skill_usage import record_workflow_generation, workflow_eligibility
            activation = "active" if workflow_eligibility(skill) == "eligible" else "candidate"
            record_workflow_generation(skill, status="generating", fingerprint=fingerprint,
                                       job_id=job_id, activation=activation)
            result = _generate(skill, name, activation=activation, expected_fingerprint=fingerprint,
                               expected_job_id=job_id)
            status = "generated" if activation == "active" else "candidate"
            record_workflow_generation(skill, status=status, fingerprint=fingerprint, path=result["path"],
                                       job_id=job_id, activation=activation)
        except Exception as exc:
            from tools.skill_usage import load_usage, record_workflow_generation
            current_job = load_usage().get(skill, {}).get("workflow_generation", {}).get("job_id")
            if current_job == job_id:
                record_workflow_generation(skill, status="failed", fingerprint=fingerprint,
                                           error=str(exc), job_id=job_id)
        finally:
            try:
                if json.loads(lock.read_text(encoding="utf-8")).get("job_id") == job_id:
                    lock.unlink()
            except (OSError, ValueError):
                pass
            with _ACTIVE_LOCK:
                if _ACTIVE.get(key, (None, None))[1] == job_id:
                    _ACTIVE.pop(key, None)

    try:
        _POOL.submit(generate)
    except Exception:
        try:
            lock.unlink()
        except OSError:
            pass
        with _ACTIVE_LOCK:
            _ACTIVE.pop(key, None)
        raise
    return job_id
