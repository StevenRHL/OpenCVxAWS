"""Durable review-action history, Ignore/Undo, and the learning-candidate queue.

Both logs are append-only, mirroring the pattern `jobs.escalations` already uses: a changed
mind is a new row, never a rewrite of an earlier one. `jobs.reviews` keeps its existing
single-row-per-event shape for backward compatibility (exports/UI read it directly); this
module is the source of truth for *history* and for Ignore/Undo/learning-candidate state,
which that table cannot represent.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from . import incidents, jobs

REVIEW_ACTIONS = {"relevant", "false_alarm", "unclear", "ignore", "undo"}
CANDIDATE_STATUSES = {"needs_annotation", "ready_for_dataset_review", "exported", "excluded"}
PROPOSED_LABELS = {"positive", "negative", "uncertain"}
MAX_EXPORT_BATCH = 200

_CANDIDATE_COLUMNS = ("candidate_id", "run_id", "event_id", "source_start_s", "source_end_s",
                     "proposed_label", "visible_action_label", "person_if_identifiable", "note",
                     "status", "source_sha256", "recording_group", "model_provenance",
                     "source_restrictions", "created_at_utc", "updated_at_utc")
_ACTION_COLUMNS = ("action_id", "event_id", "action", "note", "previous_action_id",
                   "event_revision_at_action", "idempotency_key", "created_at_utc")


class DuplicateCandidateWarning(ValueError):
    """Raised when a new learning candidate overlaps an existing one and was not confirmed."""

    def __init__(self, overlaps: list[dict]):
        super().__init__(f"{len(overlaps)} existing candidate(s) overlap this interval")
        self.overlaps = overlaps


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


def _require_known_event(run_id: str, event_id: str) -> None:
    if not any(event.get("event_id") == event_id for event in jobs.load_events(run_id)):
        raise ValueError("This event is not in the selected analysis")


# --- review actions (relevant / false_alarm / unclear / ignore / undo) ----------------------

def record_action(run_id: str, event_id: str, action: str, note: str = "",
                  previous_action_id: str | None = None, idempotency_key: str | None = None) -> dict:
    if action not in REVIEW_ACTIONS:
        raise ValueError("Unknown review action")
    if action == "undo" and not previous_action_id:
        raise ValueError("An undo action must reference the action it reverses")
    _require_known_event(run_id, event_id)
    idempotency_key = idempotency_key or _new_id()
    revision = jobs.current_event_revision(run_id, event_id)
    connection = jobs._database(run_id)
    try:
        with connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO review_actions (action_id, event_id, action, note, "
                "previous_action_id, event_revision_at_action, idempotency_key, created_at_utc) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (_new_id(), event_id, action, note, previous_action_id, revision,
                 idempotency_key, _now()))
            if action in ("relevant", "false_alarm", "unclear") and cursor.rowcount:
                connection.execute(
                    "INSERT INTO reviews VALUES (?, ?, ?, ?) ON CONFLICT(event_id) DO UPDATE "
                    "SET label=excluded.label, note=excluded.note, "
                    "reviewed_at_utc=excluded.reviewed_at_utc", (event_id, action, note, _now()))
    finally:
        connection.close()
    return current_state(run_id, event_id)


def list_actions(run_id: str, event_id: str | None = None) -> list[dict]:
    if not (jobs.run_dir(run_id) / "events.db").exists():
        return []
    connection = jobs._database(run_id)
    try:
        query = "SELECT " + ", ".join(_ACTION_COLUMNS) + " FROM review_actions"
        params: tuple = ()
        if event_id is not None:
            query += " WHERE event_id=?"
            params = (event_id,)
        query += " ORDER BY created_at_utc, action_id"
        return [dict(zip(_ACTION_COLUMNS, row)) for row in connection.execute(query, params)]
    finally:
        connection.close()


def current_state(run_id: str, event_id: str) -> dict:
    """The effective review state after replaying history: label, and whether it is ignored.

    A later meaningful revision reopens an ignored event; an ordinary refresh does not. The
    event's revision only ever bumps alongside a real snapshot change (see `core.py`), so
    "the event has revised since the ignore action" is a correct, cheap proxy for that.
    """
    rows = list_actions(run_id, event_id)
    undone_ids = {row["previous_action_id"] for row in rows
                 if row["action"] == "undo" and row["previous_action_id"]}
    label, ignored, ignore_row, last_action = "unreviewed", False, None, None
    for row in rows:
        if row["action_id"] in undone_ids:
            continue
        if row["action"] in ("relevant", "false_alarm", "unclear"):
            label, ignored, ignore_row, last_action = row["action"], False, None, row
        elif row["action"] == "ignore":
            ignored, ignore_row, last_action = True, row, row
        elif row["action"] == "undo":
            last_action = row
    if ignored and ignore_row:
        ignored = jobs.current_event_revision(run_id, event_id) <= ignore_row["event_revision_at_action"]
    return {"label": label, "ignored": ignored, "last_action": last_action}


# --- learning-candidate queue ----------------------------------------------------------------

def _row_to_candidate(row) -> dict:
    data = dict(zip(_CANDIDATE_COLUMNS, row))
    data["model_provenance"] = json.loads(data["model_provenance"] or "{}")
    data["source_restrictions"] = json.loads(data["source_restrictions"] or "{}")
    return data


def find_overlaps(run_id: str, event_id: str | None, source_start_s: float,
                  source_end_s: float) -> list[dict]:
    """Existing, non-excluded candidates whose interval overlaps this one.

    Scoped to the same target: the same event when `event_id` is given, otherwise every
    candidate in the run (a missed-interval candidate can overlap an event-attached one).
    """
    overlaps = []
    for candidate in list_candidates(run_id):
        if candidate["status"] == "excluded":
            continue
        if event_id is not None and candidate.get("event_id") != event_id:
            continue
        if source_start_s < candidate["source_end_s"] and candidate["source_start_s"] < source_end_s:
            overlaps.append(candidate)
    return overlaps


def create_candidate(run_id: str, event_id: str | None, source_start_s: float, source_end_s: float,
                     proposed_label: str, visible_action_label: str = "",
                     person_if_identifiable: str = "", note: str = "",
                     idempotency_key: str | None = None, confirmed_duplicate: bool = False) -> dict:
    if proposed_label not in PROPOSED_LABELS:
        raise ValueError("Unknown proposed label")
    if not (source_end_s > source_start_s):
        raise ValueError("A candidate interval must have a positive duration")
    if event_id is not None:
        _require_known_event(run_id, event_id)
    if not confirmed_duplicate:
        overlaps = find_overlaps(run_id, event_id, source_start_s, source_end_s)
        if overlaps:
            raise DuplicateCandidateWarning(overlaps)
    job = jobs.get_job(run_id)
    candidate_id = _new_id()
    idempotency_key = idempotency_key or _new_id()
    now = _now()
    connection = jobs._database(run_id)
    try:
        with connection:
            connection.execute(
                "INSERT INTO learning_candidates (candidate_id, run_id, event_id, source_start_s, "
                "source_end_s, proposed_label, visible_action_label, person_if_identifiable, note, "
                "status, source_sha256, recording_group, model_provenance, source_restrictions, "
                "created_at_utc, updated_at_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (candidate_id, run_id, event_id, source_start_s, source_end_s, proposed_label,
                 visible_action_label, person_if_identifiable, note, "needs_annotation",
                 job.get("source_sha256", ""), job.get("source_sha256", ""), "{}", "{}", now, now))
            connection.execute(
                "INSERT OR IGNORE INTO learning_actions (action_id, candidate_id, action, changes, "
                "note, idempotency_key, created_at_utc) VALUES (?,?,?,?,?,?,?)",
                (_new_id(), candidate_id, "created", "{}", note, idempotency_key, now))
    finally:
        connection.close()
    return get_candidate(run_id, candidate_id)


def get_candidate(run_id: str, candidate_id: str) -> dict:
    connection = jobs._database(run_id)
    try:
        row = connection.execute(
            "SELECT " + ", ".join(_CANDIDATE_COLUMNS) +
            " FROM learning_candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ValueError("Unknown learning candidate")
    return _row_to_candidate(row)


def list_candidates(run_id: str | None = None) -> list[dict]:
    run_ids = [run_id] if run_id else [job["run_id"] for job in jobs.list_jobs()]
    results = []
    for identifier in run_ids:
        if not (jobs.run_dir(identifier) / "events.db").exists():
            continue
        connection = jobs._database(identifier)
        try:
            rows = connection.execute(
                "SELECT " + ", ".join(_CANDIDATE_COLUMNS) +
                " FROM learning_candidates ORDER BY source_start_s").fetchall()
        finally:
            connection.close()
        results.extend(_row_to_candidate(row) for row in rows)
    return results


def correct_candidate(run_id: str, candidate_id: str, **fields) -> dict:
    allowed = {"proposed_label", "visible_action_label", "person_if_identifiable", "note",
              "source_start_s", "source_end_s"}
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Cannot correct fields: {sorted(unknown)}")
    if "proposed_label" in fields and fields["proposed_label"] not in PROPOSED_LABELS:
        raise ValueError("Unknown proposed label")
    existing = get_candidate(run_id, candidate_id)
    changes = {name: [existing.get(name), value] for name, value in fields.items()
              if existing.get(name) != value}
    if not changes:
        return existing
    connection = jobs._database(run_id)
    try:
        with connection:
            assignments = ", ".join(f"{name}=?" for name in fields)
            connection.execute(
                f"UPDATE learning_candidates SET {assignments}, updated_at_utc=? WHERE candidate_id=?",
                (*fields.values(), _now(), candidate_id))
            connection.execute(
                "INSERT INTO learning_actions (action_id, candidate_id, action, changes, note, "
                "idempotency_key, created_at_utc) VALUES (?,?,?,?,?,?,?)",
                (_new_id(), candidate_id, "label_corrected", json.dumps(changes), "", _new_id(), _now()))
    finally:
        connection.close()
    return get_candidate(run_id, candidate_id)


def eligibility(candidate: dict) -> tuple[bool, str]:
    """Whether a candidate may move to `ready_for_dataset_review`, and why not if it can't."""
    if candidate.get("proposed_label") == "uncertain":
        return False, "Uncertain examples are not ready: choose positive or negative."
    if not candidate.get("visible_action_label"):
        return False, "Describe the visible action before marking ready."
    if not (candidate.get("source_end_s", 0) > candidate.get("source_start_s", 0)):
        return False, "The interval must have a positive duration."
    return True, "Ready for dataset review."


def set_candidate_status(run_id: str, candidate_id: str, status: str, note: str = "",
                         idempotency_key: str | None = None) -> dict:
    if status not in CANDIDATE_STATUSES:
        raise ValueError("Unknown candidate status")
    candidate = get_candidate(run_id, candidate_id)
    if status == "ready_for_dataset_review":
        ok, reason = eligibility(candidate)
        if not ok:
            raise ValueError(reason)
    idempotency_key = idempotency_key or _new_id()
    connection = jobs._database(run_id)
    try:
        with connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO learning_actions (action_id, candidate_id, action, changes, "
                "note, idempotency_key, created_at_utc) VALUES (?,?,?,?,?,?,?)",
                (_new_id(), candidate_id, "status_changed",
                 json.dumps({"status": [candidate["status"], status]}), note, idempotency_key, _now()))
            if cursor.rowcount:
                connection.execute(
                    "UPDATE learning_candidates SET status=?, updated_at_utc=? WHERE candidate_id=?",
                    (status, _now(), candidate_id))
    finally:
        connection.close()
    return get_candidate(run_id, candidate_id)


def remove_candidate(run_id: str, candidate_id: str, note: str = "",
                     idempotency_key: str | None = None) -> dict:
    """Exclude a candidate. The row is retained with history, never deleted."""
    candidate = get_candidate(run_id, candidate_id)
    idempotency_key = idempotency_key or _new_id()
    connection = jobs._database(run_id)
    try:
        with connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO learning_actions (action_id, candidate_id, action, changes, "
                "note, idempotency_key, created_at_utc) VALUES (?,?,?,?,?,?,?)",
                (_new_id(), candidate_id, "removed",
                 json.dumps({"status": [candidate["status"], "excluded"]}), note, idempotency_key, _now()))
            if cursor.rowcount:
                connection.execute(
                    "UPDATE learning_candidates SET status='excluded', updated_at_utc=? WHERE candidate_id=?",
                    (_now(), candidate_id))
    finally:
        connection.close()
    return get_candidate(run_id, candidate_id)


def restore_candidate(run_id: str, candidate_id: str, note: str = "",
                      idempotency_key: str | None = None) -> dict:
    connection = jobs._database(run_id)
    try:
        with connection:
            candidate = get_candidate(run_id, candidate_id)
            idempotency_key = idempotency_key or _new_id()
            cursor = connection.execute(
                "INSERT OR IGNORE INTO learning_actions (action_id, candidate_id, action, changes, "
                "note, idempotency_key, created_at_utc) VALUES (?,?,?,?,?,?,?)",
                (_new_id(), candidate_id, "restored",
                 json.dumps({"status": [candidate["status"], "needs_annotation"]}), note,
                 idempotency_key, _now()))
            if cursor.rowcount:
                connection.execute(
                    "UPDATE learning_candidates SET status='needs_annotation', updated_at_utc=? "
                    "WHERE candidate_id=?", (_now(), candidate_id))
    finally:
        connection.close()
    return get_candidate(run_id, candidate_id)


def candidate_clip(run_id: str, candidate: dict) -> Path | None:
    key = candidate.get("event_id") or f"candidate-{candidate['candidate_id']}"
    return incidents.build_clip_interval(
        run_id, (candidate["source_start_s"], candidate["source_end_s"]), key)


# --- export (dataset review, not training) ----------------------------------------------------

def export_candidates(candidate_ids: list[str], destination: Path | str | None = None) -> dict:
    """Copy ready candidates' clips + provenance into a durable, labeled export bundle.

    Marking a candidate exported is the end of this module's responsibility. Deciding
    train/held-out split membership from the preserved `recording_group`/`source_restrictions`
    fields, and any actual training, is explicitly out of scope here.
    """
    if len(candidate_ids) > MAX_EXPORT_BATCH:
        raise ValueError(f"Export at most {MAX_EXPORT_BATCH} candidates at a time")
    destination = Path(destination) if destination else jobs.ROOT / "outputs" / "_exports" / _new_id()
    clips_dir = destination / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    exported, skipped = [], []
    manifest_rows = []
    by_id = {c["candidate_id"]: c for c in list_candidates()}
    for candidate_id in candidate_ids:
        candidate = by_id.get(candidate_id)
        if candidate is None:
            skipped.append({"candidate_id": candidate_id, "reason": "not found"})
            continue
        if candidate["status"] != "ready_for_dataset_review":
            skipped.append({"candidate_id": candidate_id, "reason": "not ready"})
            continue
        if candidate["source_restrictions"].get("exclude_from_export"):
            skipped.append({"candidate_id": candidate_id, "reason": "marked excluded from export"})
            continue
        clip = candidate_clip(candidate["run_id"], candidate)
        if clip is None:
            skipped.append({"candidate_id": candidate_id, "reason": "clip unavailable"})
            continue
        destination_clip = clips_dir / f"{candidate_id}.mp4"
        destination_clip.write_bytes(clip.read_bytes())
        manifest_rows.append(dict(candidate, clip_file=str(destination_clip.relative_to(destination))))
        set_candidate_status(candidate["run_id"], candidate_id, "exported",
                             note="Exported for dataset review")
        exported.append(candidate_id)
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_rows, indent=2))
    return {"exported": exported, "skipped": skipped, "manifest_path": str(manifest_path)}
