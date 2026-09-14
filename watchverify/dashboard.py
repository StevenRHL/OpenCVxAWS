"""Read-only dashboard projections. No video decoding or model loading."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from . import jobs

CATEGORIES = {"possible_fall": "Possible fall", "person_down": "Person remains down",
              "unusual_activity": "Unusual movement", "activity": "Unusual movement",
              "activity_candidate": "Unusual movement"}
REVIEW_LABELS = {"unreviewed": "Not reviewed", "relevant": "Relevant",
                 "false_alarm": "False alarm", "unclear": "Unclear"}
PRIORITIES = {"urgent": "Urgent review", "review": "Not urgent · Review",
              "system": "System attention", "info": "Info"}


def parsed_time(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.astimezone(timezone.utc) if result.tzinfo else None
    except (TypeError, ValueError):
        return None


def date_text(value, tz):
    date = parsed_time(value)
    return date.astimezone(tz).strftime("%d %b %Y, %H:%M:%S %Z (UTC%z)") if date else "Time not recorded"


def priority(event):
    return "urgent" if event.get("category") in {"possible_fall", "person_down"} else "review"


def sort_key(row):
    return (parsed_time(row.get("at")) or datetime.min.replace(tzinfo=timezone.utc), row["id"])


def _ignored_by_event(connection, current_revision):
    """Replay `review_actions` to the same effective-ignore state `review.current_state` computes.

    Duplicated here (rather than imported) because this connection is opened read-only and
    must never create tables on a dashboard refresh, while `review.py`'s helpers assume a
    writable `jobs._database()` connection.
    """
    rows = connection.execute(
        "SELECT event_id, action, previous_action_id, event_revision_at_action, action_id "
        "FROM review_actions ORDER BY created_at_utc, action_id").fetchall()
    by_event = {}
    for event_id, action, previous_action_id, revision_at, action_id in rows:
        by_event.setdefault(event_id, []).append(dict(
            action=action, previous_action_id=previous_action_id,
            revision_at=revision_at, action_id=action_id))
    ignored = {}
    for event_id, actions in by_event.items():
        undone_ids = {a["previous_action_id"] for a in actions
                     if a["action"] == "undo" and a["previous_action_id"]}
        state, ignore_revision = False, None
        for action in actions:
            if action["action_id"] in undone_ids:
                continue
            if action["action"] in ("relevant", "false_alarm", "unclear"):
                state, ignore_revision = False, None
            elif action["action"] == "ignore":
                state, ignore_revision = True, action["revision_at"]
        if state and ignore_revision is not None:
            state = current_revision.get(event_id, 0) <= ignore_revision
        ignored[event_id] = state
    return ignored


def read_records(run_id):
    """Read a consistent SQLite snapshot without creating tables on dashboard refresh."""
    folder = jobs.run_dir(run_id)
    events, reviews, ignored = [], {}, {}
    has_revisions = False
    database = folder / "events.db"
    if database.exists():
        connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=1)
        try:
            connection.execute("BEGIN")
            tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "revisions" in tables:
                has_revisions = True
                events = [json.loads(r[0]) for r in connection.execute(
                    "SELECT payload FROM revisions r WHERE revision=(SELECT MAX(revision) "
                    "FROM revisions WHERE event_id=r.event_id)")]
            if "reviews" in tables:
                reviews = {r[0]: dict(label=r[1], reviewed_at_utc=r[2]) for r in connection.execute(
                    "SELECT event_id,label,reviewed_at_utc FROM reviews")}
            if "review_actions" in tables:
                current_revision = {event["event_id"]: event.get("revision", 0) for event in events}
                ignored = _ignored_by_event(connection, current_revision)
        finally:
            connection.close()
    if not has_revisions and (folder / "events.json").exists():
        data = json.loads((folder / "events.json").read_text())
        events = data if isinstance(data, list) else data.get("events", [])
    if not isinstance(events, list) or any(not isinstance(e, dict) or not e.get("event_id") for e in events):
        raise ValueError("Invalid observation records")
    return events, reviews, ignored


def build_snapshot(records=None):
    """Use only recorded wall times; revised events keep their FIRST alert date."""
    updates, observations, attention, errors = [], [], [], []
    records = jobs.list_jobs(errors=errors) if records is None else records
    legacy_runs = set()
    for job in records:
        run_id = job["run_id"]
        base = {"run_id": run_id, "source": job.get("original_name", run_id),
                "event_id": None, "review": None, "job_status": job["status"]}
        if job.get("status_history_partial"):
            legacy_runs.add(run_id)
        transitions = job.get("status_history")
        if transitions is None:
            legacy_runs.add(run_id)
            transitions = [{"status": "queued", "at": job.get("created_at_utc") }]
            if job.get("started_at_utc"):
                transitions.append({"status": "running", "at": job["started_at_utc"]})
            if job["status"] not in jobs.ACTIVE:
                transitions.append({"status": job["status"], "at": job.get("finished_at_utc")})
        for index, transition in enumerate(transitions):
            state = transition["status"]
            updates.append(dict(base, id=f"{run_id}:job:{index}", at=transition.get("at"),
                                title=f"Analysis {state}", kind="Analysis",
                                priority="system" if state in {"failed", "interrupted", "cancelled"} else "info"))
        summary = job.get("summary") or {}
        reasons = []
        if job["status"] in {"failed", "interrupted", "cancelled"}:
            reasons.append(f"Analysis {job['status']}")
        if summary.get("unobserved_source_s", 0):
            reasons.append(f"{summary['unobserved_source_s']:.1f}s unobserved in analysed footage")
        if summary.get("unprocessed_source_s", 0) > 0.05:
            reasons.append(f"{summary['unprocessed_source_s']:.1f}s of source not analysed")
        if not job.get("source_path") or not Path(job["source_path"]).is_file():
            reasons.append("Source video unavailable")
        if reasons:
            attention.append(dict(base, reasons=reasons))
        try:
            events, reviews, ignored = read_records(run_id)
        except (OSError, ValueError, sqlite3.Error) as error:
            errors.append(f"{base['source']}: records unavailable ({error})")
            continue
        for event in events:
            review = reviews.get(event["event_id"], {})
            row = dict(base, id=f"{run_id}:event:{event['event_id']}", event_id=event["event_id"],
                       at=event.get("emitted_at_utc"), title=CATEGORIES.get(event.get("category"), "Observation"),
                       kind="Observation", priority=priority(event),
                       review=review.get("label", "unreviewed"), event=event,
                       ignored=ignored.get(event["event_id"], False))
            observations.append(row)
            updates.append(row)
            if event.get("revision", 0):
                legacy_runs.add(run_id)  # Core revisions do not record their own wall-clock time.
            if review:
                updates.append(dict(row, id=f"{run_id}:review:{event['event_id']}",
                                    at=review.get("reviewed_at_utc"), kind="Review",
                                    title="Review saved · " + REVIEW_LABELS.get(row['review'], row['review'])))
    # Ignored is a queue-visibility choice, not a judgement about the observation: it removes
    # an item from the default urgent/pending counts and feed, never from the record itself.
    urgent = sorted([r for r in observations if r["priority"] == "urgent" and not r["ignored"] and
                     r["review"] in {"unreviewed", "unclear"}], key=sort_key, reverse=True)
    return dict(updates=sorted(updates, key=sort_key, reverse=True), urgent=urgent,
                other_pending=sum(r["priority"] != "urgent" and not r["ignored"] and
                                  r["review"] in {"unreviewed", "unclear"} for r in observations),
                active=[j for j in records if j["status"] in jobs.ACTIVE],
                attention=attention, errors=errors, legacy_runs=len(legacy_runs), total_runs=len(records))


def filter_updates(rows, *, urgency="All", kind="All", review="All", dates=None,
                   tz=timezone.utc, show_ignored=False):
    result = []
    for row in rows:
        if row.get("ignored") and not show_ignored:
            continue
        if urgency != "All" and row["priority"] != urgency:
            continue
        if kind != "All" and row["kind"] != kind:
            continue
        if review != "All" and row.get("review") != review:
            continue
        if dates:
            at = parsed_time(row.get("at"))
            if at is None or not dates[0] <= at.astimezone(tz).date() <= dates[1]:
                continue
        result.append(row)
    return result
