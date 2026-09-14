"""Durable, single-worker local jobs. No inference state lives in the UI."""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import io
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
    manifest = folder / "manifest.json"
    previous = None
    if manifest.exists():
        try:
            previous = json.loads(manifest.read_text())
        except (OSError, ValueError):
            previous = None
    if not job.get("status_history"):
        history = []
        if previous and previous.get("status_history"):
            history = list(previous["status_history"])
        elif previous:
            if previous.get("created_at_utc"):
                history.append({"status": "queued", "at": previous["created_at_utc"]})
            if previous.get("started_at_utc"):
                history.append({"status": "running", "at": previous["started_at_utc"]})
            if previous.get("status") not in ACTIVE:
                history.append({"status": previous.get("status"), "at": previous.get("finished_at_utc")})
            if history:
                job["status_history_partial"] = True
        elif job.get("created_at_utc"):
            history.append({"status": job.get("status", "queued"), "at": job["created_at_utc"]})
        if history:
            job["status_history"] = history
    if previous is not None and previous.get("status") != job.get("status"):
        history = list(job.get("status_history") or previous.get("status_history") or [])
        transition_at = job.get("updated_at_utc") or _now()
        if job.get("status") == "running":
            transition_at = job.get("started_at_utc") or transition_at
        elif job.get("status") not in ACTIVE:
            transition_at = job.get("finished_at_utc") or transition_at
        history.append({"status": job["status"], "at": transition_at})
        job["status_history"] = history
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
        if (changes.get("status") not in (None, job.get("status"))
                and changes.get("status") not in ACTIVE
                and "finished_at_utc" not in changes):
            changes["finished_at_utc"] = _now()
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


def list_jobs(*, errors: list[str] | None = None) -> list[dict]:
    with _lock():
        jobs = []
        for manifest in (ROOT / "outputs").glob("*/manifest.json"):
            try:
                jobs.append(_reconcile(json.loads(manifest.read_text())))
            except (OSError, ValueError, KeyError) as error:
                if errors is not None:
                    errors.append(f"{manifest.parent.name}: analysis unavailable ({error})")
                continue
        return sorted(jobs, key=lambda job: job.get("created_at_utc", ""), reverse=True)


def _database(run_id: str) -> sqlite3.Connection:
    if not (run_dir(run_id) / "manifest.json").exists():
        raise FileNotFoundError("Analysis not found")
    connection = sqlite3.connect(run_dir(run_id) / "events.db", timeout=10)
    connection.execute("PRAGMA busy_timeout=10000")
    connection.execute("CREATE TABLE IF NOT EXISTS reviews (event_id TEXT PRIMARY KEY, label TEXT NOT NULL, note TEXT NOT NULL, reviewed_at_utc TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS escalations (sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_ids TEXT NOT NULL, action TEXT NOT NULL, note TEXT NOT NULL, shown_rate TEXT NOT NULL, decided_at_utc TEXT NOT NULL)")
    connection.execute("""CREATE TABLE IF NOT EXISTS review_actions (
        action_id TEXT PRIMARY KEY, event_id TEXT NOT NULL, action TEXT NOT NULL,
        note TEXT NOT NULL DEFAULT '', previous_action_id TEXT,
        event_revision_at_action INTEGER NOT NULL, idempotency_key TEXT NOT NULL,
        created_at_utc TEXT NOT NULL)""")
    connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS review_actions_idempotency ON review_actions(idempotency_key)")
    connection.execute("CREATE INDEX IF NOT EXISTS review_actions_event_idx ON review_actions(event_id, created_at_utc)")
    connection.execute("""CREATE TABLE IF NOT EXISTS learning_candidates (
        candidate_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, event_id TEXT,
        source_start_s REAL NOT NULL, source_end_s REAL NOT NULL,
        proposed_label TEXT NOT NULL, visible_action_label TEXT NOT NULL DEFAULT '',
        person_if_identifiable TEXT NOT NULL DEFAULT '', note TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'needs_annotation',
        source_sha256 TEXT NOT NULL, recording_group TEXT NOT NULL,
        model_provenance TEXT NOT NULL DEFAULT '{}', source_restrictions TEXT NOT NULL DEFAULT '{}',
        created_at_utc TEXT NOT NULL, updated_at_utc TEXT NOT NULL)""")
    connection.execute("CREATE INDEX IF NOT EXISTS learning_candidates_run ON learning_candidates(run_id, source_start_s)")
    connection.execute("""CREATE TABLE IF NOT EXISTS learning_actions (
        action_id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL, action TEXT NOT NULL,
        changes TEXT NOT NULL, note TEXT NOT NULL DEFAULT '', idempotency_key TEXT NOT NULL,
        created_at_utc TEXT NOT NULL)""")
    connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS learning_actions_idempotency ON learning_actions(idempotency_key)")
    connection.execute("CREATE INDEX IF NOT EXISTS learning_actions_candidate ON learning_actions(candidate_id, created_at_utc)")
    connection.commit()
    return connection


def save_review(run_id: str, event_id: str, label: str, note: str = "") -> None:
    """Save a review label. `relevant`/`false_alarm`/`unclear` also append to the durable
    review-action history (`watchverify.review`); `unreviewed` explicitly un-sets a prior
    judgement rather than recording a new one, so it stays a direct upsert with no history row.
    """
    if label not in REVIEW_LABELS:
        raise ValueError("Unknown review label")
    if not any(event.get("event_id") == event_id for event in load_events(run_id)):
        raise ValueError("This event is not in the selected analysis")
    if label == "unreviewed":
        connection = _database(run_id)
        try:
            with connection:
                connection.execute("INSERT INTO reviews VALUES (?, ?, ?, ?) ON CONFLICT(event_id) DO UPDATE SET label=excluded.label, note=excluded.note, reviewed_at_utc=excluded.reviewed_at_utc",
                                   (event_id, label, note, _now()))
        finally:
            connection.close()
        return
    from . import review  # local import: avoids a jobs<->review import cycle at module load
    review.record_action(run_id, event_id, label, note)


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


EVENT_CSV_FIELDS = ("event_id", "category", "track_id", "source_start_s", "source_end_s",
                    "emitted_source_s", "occurred_at_utc", "created_at_utc", "emitted_at_utc",
                    "status", "reason", "score", "score_type", "observations",
                    "review_label", "review_note", "reviewed_at_utc")


def events_csv(run_id: str) -> str:
    """One exported row per event, carrying the observations and the reviewer's judgement.

    The worker cannot write this file: a review is added after analysis ends, so the export
    is built when it is requested. Unreviewed events say so explicitly rather than leaving
    the column blank, which would read as an assessment nobody made.
    """
    reviews = get_reviews(run_id)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(EVENT_CSV_FIELDS), extrasaction="ignore")
    writer.writeheader()
    for event in load_events(run_id):
        review = reviews.get(event.get("event_id"), {})
        row = dict(event)
        row["observations"] = "; ".join(str(v) for v in event.get("observations", []))
        row["review_label"] = review.get("label", "unreviewed")
        row["review_note"] = review.get("note", "")
        row["reviewed_at_utc"] = review.get("reviewed_at_utc", "")
        writer.writerow(row)
    return buffer.getvalue()


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


def load_event_revisions(run_id: str, event_id: str) -> list[dict]:
    """Every revision of one event, oldest first. Index 0 is the immutable first alert.

    `revisions` is INSERT OR IGNORE keyed on (event_id, revision), so an earlier row is
    never rewritten by a later one — this reads the full history, not just the latest.
    """
    folder = run_dir(run_id)
    if not (folder / "events.db").exists():
        return []
    connection = sqlite3.connect(folder / "events.db", timeout=5)
    try:
        rows = connection.execute(
            "SELECT payload FROM revisions WHERE event_id=? ORDER BY revision", (event_id,)
        ).fetchall()
        return [json.loads(row[0]) for row in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        connection.close()


_EVENT_FRAME_COLUMNS = ("event_id", "t", "track_id", "source", "fall_score", "fall_threshold",
                       "fall_positive", "fall_version", "activity_score", "activity_threshold",
                       "activity_positive", "activity_version", "quality", "angle", "down",
                       "gate_state", "old_track_id", "handover_reason")


def _event_frames_rows(run_id: str, event_id: str) -> list[dict]:
    if not (run_dir(run_id) / "events.db").exists():
        return []
    connection = sqlite3.connect(run_dir(run_id) / "events.db", timeout=5)
    try:
        rows = connection.execute(
            "SELECT " + ", ".join(_EVENT_FRAME_COLUMNS) + " FROM event_frames WHERE event_id=? ORDER BY t",
            (event_id,)).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        connection.close()
    result = []
    for row in rows:
        data = dict(zip(_EVENT_FRAME_COLUMNS, row))
        data["gate_state"] = json.loads(data["gate_state"]) if data["gate_state"] else None
        result.append(data)
    return result


def _reconstruct_from_predictions(run_id: str, event_id: str, predictions_file: Path) -> tuple[list[dict], int]:
    """Best-effort trace built by matching track_id + time-range overlap.

    Only kept where exactly one event's window on this track contains the frame; a frame
    inside more than one event's window is genuinely ambiguous and is dropped, not guessed.
    """
    events = load_events(run_id)
    event = next((e for e in events if e.get("event_id") == event_id), None)
    if event is None:
        return [], 0
    track_id = event.get("track_id")
    start = float(event.get("source_start_s") or 0)
    end = float(event.get("source_end_s") or start)
    other_windows = [(float(e.get("source_start_s") or 0), float(e.get("source_end_s") or 0))
                     for e in events if e.get("event_id") != event_id and e.get("track_id") == track_id]
    rows, ambiguous = [], 0
    with predictions_file.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("track_id") != track_id:
                continue
            t = record.get("t")
            if t is None or not (start <= t <= end):
                continue
            if any(other_start <= t <= other_end for other_start, other_end in other_windows):
                ambiguous += 1
                continue
            fall, activity = record.get("fall_model"), record.get("activity_model")
            rows.append({
                "event_id": event_id, "t": t, "track_id": track_id, "source": "reconstructed",
                "fall_score": fall.get("score") if fall else None,
                "fall_threshold": fall.get("threshold") if fall else None,
                "fall_positive": fall.get("positive") if fall else None,
                "fall_version": fall.get("version") if fall else None,
                "activity_score": activity.get("score") if activity else None,
                "activity_threshold": activity.get("threshold") if activity else None,
                "activity_positive": activity.get("positive") if activity else None,
                "activity_version": activity.get("version") if activity else None,
                "quality": record.get("quality"), "angle": record.get("angle"),
                "down": record.get("down"), "gate_state": None,
                "old_track_id": None, "handover_reason": None})
    return rows, ambiguous


def load_event_trace(run_id: str, event_id: str) -> dict:
    """Structured per-frame trace for one event: real if recorded, best-effort otherwise.

    `kind='recorded'`: this run stamped `trace_schema_version` and the worker wrote
    `event_frames` rows linked to this event by the incident manager at the moment each
    frame was processed — a real causal record. `kind='reconstructed'`: no linkage was
    recorded (an older run), but `predictions.jsonl` exists; rows are matched by track_id
    and time-range overlap, which is an approximation, never claimed as causal.
    `kind='unavailable'`: neither source exists for this run.
    """
    job = get_job(run_id)
    if int(job.get("trace_schema_version", 0) or 0) >= 1:
        rows = _event_frames_rows(run_id, event_id)
        if rows:
            return {"kind": "recorded", "rows": rows}
    predictions_file = run_dir(run_id) / "predictions.jsonl"
    if predictions_file.exists():
        rows, ambiguous = _reconstruct_from_predictions(run_id, event_id, predictions_file)
        if rows:
            return {"kind": "reconstructed", "rows": rows, "ambiguous_frames_dropped": ambiguous}
    return {"kind": "unavailable", "rows": []}


def load_event_handovers(run_id: str, event_id: str) -> list[dict]:
    """Per-timestamp evidence handovers recorded for this event, if any were."""
    return [{"t": row["t"], "old_track_id": row["old_track_id"], "reason": row["handover_reason"]}
            for row in _event_frames_rows(run_id, event_id) if row.get("old_track_id") is not None]


def current_event_revision(run_id: str, event_id: str) -> int:
    """The latest revision number recorded for one event, or 0 if none exists."""
    revisions = load_event_revisions(run_id, event_id)
    return max((int(r.get("revision", 0)) for r in revisions), default=0)


def _promoted_candidates(run_id: str) -> list[dict]:
    """Learning candidates already promoted past curation: never silently dropped."""
    if not (run_dir(run_id) / "events.db").exists():
        return []
    connection = _database(run_id)
    try:
        return [{"candidate_id": row[0], "status": row[1]} for row in connection.execute(
            "SELECT candidate_id, status FROM learning_candidates "
            "WHERE status IN ('ready_for_dataset_review', 'exported')")]
    finally:
        connection.close()


def _archive_promoted_candidates(run_id: str, promoted: list[dict]) -> None:
    """Preserve promoted learning examples (rows + clips) outside the run before it is deleted."""
    from . import review  # local import: avoids a jobs<->review import cycle at module load
    destination = ROOT / "outputs" / "_retained_learning" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    candidates = [review.get_candidate(run_id, item["candidate_id"]) for item in promoted]
    (destination / "candidates.json").write_text(json.dumps(candidates, indent=2))
    clips = run_dir(run_id) / "clips"
    if clips.exists():
        shutil.copytree(clips, destination / "clips", dirs_exist_ok=True)


def delete_job(run_id: str, retain_learning_examples: bool | None = None) -> None:
    """Delete only this completed/inactive run, including its owned source copy.

    A run may have learning-queue candidates already promoted past curation
    (`ready_for_dataset_review`/`exported`). Deleting the run out from under those requires an
    explicit choice: preserve them (archived under `outputs/_retained_learning/`) or accept
    they become unavailable. Candidates that never left `needs_annotation`/`excluded` were
    never promoted into the curated pipeline, so they never block deletion.
    """
    with _lock():
        job = _reconcile(_read(run_id))
        if job["status"] in {"running", "cancelling"}:
            raise RuntimeError("Wait for analysis to stop before deleting it")
        promoted = _promoted_candidates(run_id)
        if promoted and retain_learning_examples is None:
            raise ValueError(
                f"{len(promoted)} learning example(s) from this run are ready or exported; "
                "choose whether to preserve them before deleting.")
        if promoted and retain_learning_examples:
            _archive_promoted_candidates(run_id, promoted)
        shutil.rmtree(run_dir(run_id))
        _PROCESSES.pop(run_id, None)
