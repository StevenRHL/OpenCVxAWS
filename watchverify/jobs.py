"""Durable, single-worker local jobs. No inference state lives in the UI."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import uuid

from .alerts import ACTION_BRANCH, ALERT_BRANCH

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = {"queued", "running", "cancelling"}
REVIEW_LABELS = {"relevant", "false_alarm", "unclear", "unreviewed"}
# What a person decided when an alert asked them to escalate. A review label is a judgement
# about the observation; an escalation is an action taken in the world, so the two are kept
# on separate axes and stored separately.
ESCALATION_ACTIONS = {"police_called", "police_not_called",
                      "ambulance_called", "ambulance_not_called", "deferred"}
_PROCESSES: dict[str, subprocess.Popen] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_dir(run_id: str) -> Path:
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", run_id):
        raise ValueError("Invalid analysis identifier")
    return ROOT / "outputs" / run_id


@contextmanager
def _lock():
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    with (ROOT / "outputs" / ".jobs.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _read(run_id: str) -> dict:
    return json.loads((run_dir(run_id) / "manifest.json").read_text())


def _write(job: dict) -> None:
    folder = run_dir(job["run_id"])
    temp = folder / f".manifest-{uuid.uuid4().hex}.tmp"
    with temp.open("w") as handle:
        json.dump(job, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, folder / "manifest.json")


def update_job(run_id: str, **changes) -> dict:
    """Worker/UI metadata update serialized across processes; preserves other keys."""
    if "run_id" in changes and changes["run_id"] != run_id:
        raise ValueError("An analysis identifier cannot change")
    with _lock():
        job = _read(run_id)
        job.update(changes)
        job["updated_at_utc"] = _now()
        _write(job)
        return job


def _alive(job: dict) -> bool:
    process = _PROCESSES.get(job["run_id"])
    if process is not None:
        return process.poll() is None
    pid = job.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _reconcile(job: dict) -> dict:
    if job["status"] in {"running", "cancelling"} and not _alive(job):
        job.update(status="cancelled" if job["status"] == "cancelling" else "interrupted",
                   finished_at_utc=_now(),
                   error=None if job["status"] == "cancelling" else "The analysis worker stopped before it saved a final result.")
        _write(job)
    return job


def create_job(source_path: str | Path, config: dict | None = None, *, original_name: str | None = None) -> str:
    """Copy a selected file once; future analysis never edits this source copy."""
    source = Path(source_path)
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("Select a non-empty video file")
    if source.stat().st_size > 500 * 1024**2:
        raise ValueError("The initial video limit is 500 MB")
    name = Path(original_name or source.name).name
    suffix = Path(name).suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise ValueError("Select an MP4, MOV, AVI, MKV or WebM video")
    settings = {"analysis_fps": 10, "max_people": 4, "pose_variant": "full", "recording_start": None}
    settings.update(config or {})
    if settings.get("recording_start"):
        date = datetime.fromisoformat(settings["recording_start"].replace("Z", "+00:00"))
        if date.tzinfo is None:
            raise ValueError("Recording date must include a timezone offset")
        settings["recording_start"] = date.isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    with _lock():
        if shutil.disk_usage(ROOT).free - source.stat().st_size < 15 * 1024**3:
            raise ValueError("Not enough disk space: preserve at least 15 GB free")
        folder = run_dir(run_id)
        folder.mkdir()
        destination = folder / ("source" + suffix)
        try:
            digest = hashlib.sha256()
            with source.open("rb") as incoming, destination.open("xb") as outgoing:
                for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                    digest.update(chunk)
                    outgoing.write(chunk)
            destination.chmod(0o444)
            _write({"run_id": run_id, "status": "queued", "progress": 0.0,
                    "source_path": str(destination), "source_sha256": digest.hexdigest(),
                    "source_bytes": destination.stat().st_size, "original_name": name,
                    "created_at_utc": _now(), "config": settings, "error": None, "summary": {}})
        except Exception:
            shutil.rmtree(folder)
            raise
    return run_id


def launch_job(run_id: str) -> dict:
    """Idempotently launch one process; reject another active local analysis."""
    with _lock():
        job = _reconcile(_read(run_id))
        if job["status"] in {"running", "cancelling", "completed"}:
            return job
        if job["status"] != "queued":
            raise ValueError("Start a new analysis to retry this video")
        for manifest in (ROOT / "outputs").glob("*/manifest.json"):
            other = _reconcile(json.loads(manifest.read_text()))
            if other["run_id"] != run_id and other["status"] in {"running", "cancelling"}:
                raise RuntimeError("Another video is being analysed. Wait for it to finish or cancel it first.")
        with (run_dir(run_id) / "worker.log").open("ab") as log:
            process = subprocess.Popen([sys.executable, "-m", "watchverify.worker", "--run-id", run_id],
                                       cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        _PROCESSES[run_id] = process
        job.update(status="running", pid=process.pid, started_at_utc=_now(), error=None)
        _write(job)
        return job


def cancel_job(run_id: str) -> dict:
    with _lock():
        job = _reconcile(_read(run_id))
        if job["status"] in {"running", "cancelling"}:
            (run_dir(run_id) / "cancel.request").touch(exist_ok=True)
            job.update(status="cancelling")
        elif job["status"] == "queued":
            job.update(status="cancelled", finished_at_utc=_now())
        _write(job)
        return job


def get_job(run_id: str) -> dict:
    with _lock():
        return _reconcile(_read(run_id))


def list_jobs() -> list[dict]:
    with _lock():
        jobs = []
        for manifest in (ROOT / "outputs").glob("*/manifest.json"):
            try:
                jobs.append(_reconcile(json.loads(manifest.read_text())))
            except (OSError, ValueError, KeyError):
                continue
        return sorted(jobs, key=lambda job: job.get("created_at_utc", ""), reverse=True)


def _database(run_id: str) -> sqlite3.Connection:
    if not (run_dir(run_id) / "manifest.json").exists():
        raise FileNotFoundError("Analysis not found")
    connection = sqlite3.connect(run_dir(run_id) / "events.db", timeout=10)
    connection.execute("PRAGMA busy_timeout=10000")
    connection.execute("CREATE TABLE IF NOT EXISTS reviews (event_id TEXT PRIMARY KEY, label TEXT NOT NULL, note TEXT NOT NULL, reviewed_at_utc TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS escalations (sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_ids TEXT NOT NULL, action TEXT NOT NULL, note TEXT NOT NULL, shown_rate TEXT NOT NULL, decided_at_utc TEXT NOT NULL)")
    connection.commit()
    return connection


def save_review(run_id: str, event_id: str, label: str, note: str = "") -> None:
    if label not in REVIEW_LABELS:
        raise ValueError("Unknown review label")
    if not any(event.get("event_id") == event_id for event in load_events(run_id)):
        raise ValueError("This event is not in the selected analysis")
    connection = _database(run_id)
    try:
        with connection:
            connection.execute("INSERT INTO reviews VALUES (?, ?, ?, ?) ON CONFLICT(event_id) DO UPDATE SET label=excluded.label, note=excluded.note, reviewed_at_utc=excluded.reviewed_at_utc",
                               (event_id, label, note, _now()))
    finally:
        connection.close()


def save_escalation(run_id: str, event_ids: list[str], action: str, note: str = "",
                    shown_rate: str = "") -> None:
    """Append one escalation decision. Never updates: a changed mind is a new row.

    The alert a person was shown, and what they decided at that moment, is a record of an
    action taken in the world. Rewriting it would destroy the only evidence of what the
    system actually claimed when the decision was made. `shown_rate` stores the accuracy
    caveat that was on screen, so a later reader can see what the person was told.
    """
    if action not in ESCALATION_ACTIONS:
        raise ValueError("Unknown escalation action")
    if not event_ids:
        raise ValueError("An escalation must reference at least one event")
    known = {event.get("event_id"): event for event in load_events(run_id)}
    unknown = [identifier for identifier in event_ids if identifier not in known]
    if unknown:
        raise ValueError(f"These events are not in the selected analysis: {unknown}")
    if action != "deferred":
        branch = ACTION_BRANCH[action]
        if any(ALERT_BRANCH.get(known[identifier].get("category")) != branch
               for identifier in event_ids):
            raise ValueError("Escalation action does not match the observation type")
    connection = _database(run_id)
    try:
        with connection:
            connection.execute(
                "INSERT INTO escalations (event_ids, action, note, shown_rate, decided_at_utc) VALUES (?, ?, ?, ?, ?)",
                (json.dumps(list(event_ids)), action, note, shown_rate, _now()))
    finally:
        connection.close()


def get_escalations(run_id: str) -> list[dict]:
    """Every escalation decision for this run, oldest first."""
    connection = _database(run_id)
    try:
        return [{"sequence": row[0], "event_ids": json.loads(row[1]), "action": row[2],
                 "note": row[3], "shown_rate": row[4], "decided_at_utc": row[5]}
                for row in connection.execute(
                    "SELECT sequence,event_ids,action,note,shown_rate,decided_at_utc FROM escalations ORDER BY sequence")]
    finally:
        connection.close()


def get_reviews(run_id: str) -> dict:
    connection = _database(run_id)
    try:
        return {row[0]: {"label": row[1], "note": row[2], "reviewed_at_utc": row[3]}
                for row in connection.execute("SELECT event_id,label,note,reviewed_at_utc FROM reviews")}
    finally:
        connection.close()


def load_events(run_id: str) -> list[dict]:
    folder = run_dir(run_id)
    # The append-only database remains useful while the worker is running.
    if (folder / "events.db").exists():
        connection = sqlite3.connect(folder / "events.db", timeout=5)
        try:
            rows = connection.execute("SELECT payload FROM revisions r WHERE revision=(SELECT MAX(revision) FROM revisions WHERE event_id=r.event_id)").fetchall()
            return sorted([json.loads(row[0]) for row in rows], key=lambda event: event.get("source_start_s", 0))
        except sqlite3.OperationalError:
            pass
        finally:
            connection.close()
    file = folder / "events.json"
    if file.exists():
        data = json.loads(file.read_text())
        return data if isinstance(data, list) else data.get("events", [])
    return []


def delete_job(run_id: str) -> None:
    """Delete only this completed/inactive run, including its owned source copy."""
    with _lock():
        job = _reconcile(_read(run_id))
        if job["status"] in {"running", "cancelling"}:
            raise RuntimeError("Wait for analysis to stop before deleting it")
        shutil.rmtree(run_dir(run_id))
        _PROCESSES.pop(run_id, None)
