"""One incident, gathered from wherever its evidence lives, and cut to its own clip.

An analysis run is a unit of processing; an incident is the unit a person actually acts on.
This module is the translation between them. It reads across every run in `outputs/` so an
administrator sees a queue of things that happened rather than a list of files that were
processed, and it cuts each incident out of its recording so the clip opens on the event
instead of on minute zero of the footage.

Nothing here decides anything or contacts anyone. It selects, cuts and describes; every
decision still belongs to the person reading it, and is recorded through the existing
append-only escalation log in `jobs.py` rather than a second one invented here.
"""
from __future__ import annotations

from pathlib import Path

import av

from . import jobs
from .alerts import ALERT_BRANCH, pending_alerts
from .perception import VideoExport, frames

ROOT = Path(__file__).resolve().parents[1]

# How much footage to keep either side of the recorded interval. An incident clip that
# starts exactly on the alert shows the consequence and not the cause, which is the one
# thing a reviewer needs in order to judge it. The tail is shorter because what follows an
# incident is usually the response to it, not evidence about it.
LEAD_S = 5.0
TAIL_S = 3.0
# Cutting is re-encoding, so a runaway interval would cost real time. Ten minutes is the
# app's own upload ceiling; an incident longer than that is a whole recording, not a clip.
MAX_CLIP_S = 600.0

# What kind of attention an observation asks for. This is the branch it already belongs to,
# named for a reader rather than for the rule layer: the fall branch asks a medical
# question, the activity branch a security one. It is not a ranking of seriousness, and
# must not be read as one — an unusual-movement observation is not evidence of theft.
SEVERITY = {
    "fall": ("Medical", "Someone may be hurt or unable to get up."),
    "activity": ("Security", "Movement resembling clips labelled shoplifting. Not evidence of theft."),
}

CATEGORY_TEXT = {
    "possible_fall": "Possible fall",
    "person_down": "Person remains down",
    "unusual_activity": "Unusual movement",
    "activity": "Unusual movement",
    "activity_candidate": "Unusual movement",
}


def clip_bounds(event: dict, duration_s: float | None,
                lead_s: float = LEAD_S, tail_s: float = TAIL_S) -> tuple[float, float]:
    """The source interval to cut, padded and clamped to the recording that exists.

    `source_end_s` is the last revision's timestamp, so an incident that ended because the
    track was lost or the video stopped is already bounded by its own evidence. Padding is
    added outside that, never inside it.
    """
    start = float(event.get("source_start_s") or 0.0)
    end = float(event.get("source_end_s") or start)
    if end < start:
        end = start
    start = max(0.0, start - lead_s)
    end = end + tail_s
    if duration_s:
        end = min(end, float(duration_s))
    if end <= start:
        end = start + 1.0
    return start, min(end, start + MAX_CLIP_S)


def clip_path(run_id: str, event_id: str, lead_s: float = LEAD_S, tail_s: float = TAIL_S) -> Path:
    """Where this incident's clip lives, at this padding.

    The padding is part of the name on purpose. A clip re-cut with more lead is a different
    file, so changing the padding cannot serve a stale one from a player that is caching the
    path it was handed — and each padding a reviewer tries stays cached in its own right.
    """
    return jobs.run_dir(run_id) / "clips" / f"{event_id}-l{lead_s:g}-t{tail_s:g}.mp4"


def source_for_clip(run_id: str) -> Path | None:
    """Prefer the annotated export: it carries the overlay that shows what was detected.

    The original recording is the fallback while a run is still writing its export, or when
    the export failed. Both are the same footage; only the overlay differs.
    """
    folder = jobs.run_dir(run_id)
    annotated = folder / "annotated.mp4"
    if annotated.exists() and annotated.stat().st_size:
        return annotated
    try:
        source = Path(jobs.get_job(run_id)["source_path"])
    except (OSError, ValueError, KeyError, FileNotFoundError):
        return None
    return source if source.exists() else None


def build_clip(run_id: str, event: dict, lead_s: float = LEAD_S, tail_s: float = TAIL_S,
               force: bool = False) -> Path | None:
    """Cut this incident out of its recording. Returns the clip, or None if it cannot.

    Cached: an incident's interval does not change once the run has finished, so a clip that
    already exists at this padding is reused. Written to a partial name and renamed on
    success, so an interrupted cut can never be served as a complete one.
    """
    event_id = event.get("event_id")
    if not event_id:
        return None
    target = clip_path(run_id, event_id, lead_s, tail_s)
    if target.exists() and target.stat().st_size and not force:
        return target
    origin = source_for_clip(run_id)
    if origin is None:
        return None
    duration = (jobs.get_job(run_id).get("media") or {}).get("duration_s")
    start, end = clip_bounds(event, duration, lead_s, tail_s)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.stem}.partial.mp4")
    encoder = None
    written = 0
    try:
        with av.open(str(origin)) as probe:
            stream = probe.streams.video[0]
            width, height = stream.width, stream.height
            rate = float(stream.average_rate or 25)
        for _index, t, bgr in frames(origin):
            if t < start:
                continue
            if t > end:
                break
            if encoder is None:
                encoder = VideoExport(partial, width, height, rate)
            # Clip time restarts at zero so the player opens on the incident itself.
            encoder.write(bgr, t - start)
            written += 1
    except (OSError, ValueError, RuntimeError, av.FFmpegError):
        if encoder is not None:
            try:
                encoder.close()
            except (OSError, ValueError, RuntimeError, av.FFmpegError):
                pass
        partial.unlink(missing_ok=True)
        return None
    if encoder is None or not written:
        partial.unlink(missing_ok=True)
        return None
    try:
        encoder.close()
    except (OSError, ValueError, RuntimeError, av.FFmpegError):
        partial.unlink(missing_ok=True)
        return None
    partial.replace(target)
    return target


def describe(run_id: str, job: dict, event: dict, decided: bool) -> dict:
    """One incident as a queue row: what, when, whose recording, and whether it is settled."""
    branch = ALERT_BRANCH.get(event.get("category"))
    kind, meaning = SEVERITY.get(branch, ("Review", "An observation awaiting human review."))
    return dict(
        run_id=run_id,
        event_id=event.get("event_id"),
        category=event.get("category"),
        label=CATEGORY_TEXT.get(event.get("category"), event.get("category", "Observation")),
        branch=branch,
        kind=kind,
        meaning=meaning,
        track_id=event.get("track_id"),
        source_start_s=event.get("source_start_s"),
        source_end_s=event.get("source_end_s"),
        occurred_at_utc=event.get("occurred_at_utc"),
        created_at_utc=event.get("created_at_utc"),
        status=event.get("status"),
        reason=event.get("reason"),
        score=event.get("score"),
        score_type=event.get("score_type"),
        observations=list(event.get("observations", [])),
        recording=job.get("original_name", run_id),
        recording_start=(job.get("config") or {}).get("recording_start"),
        analysed_at_utc=job.get("started_at_utc") or job.get("created_at_utc"),
        decided=decided,
        event=event,
    )


def collect(include_decided: bool = True, run_ids: list[str] | None = None) -> list[dict]:
    """Every incident across every analysis, newest recording first.

    An observation counts as an incident when it is one a reviewer is asked to decide on —
    the same set `alerts.py` defines, so the dashboard and the per-analysis prompt can never
    disagree about what needs attention. Recovery and activity-clear are state changes, not
    incidents, and are excluded there rather than filtered again here.
    """
    rows = []
    for job in jobs.list_jobs():
        run_id = job["run_id"]
        if run_ids is not None and run_id not in run_ids:
            continue
        try:
            events = jobs.load_events(run_id)
            escalations = jobs.get_escalations(run_id)
        except (OSError, ValueError, KeyError, FileNotFoundError):
            continue
        waiting = {event.get("event_id") for event in pending_alerts(events, escalations)}
        for event in events:
            if ALERT_BRANCH.get(event.get("category")) is None:
                continue
            decided = event.get("event_id") not in waiting
            if decided and not include_decided:
                continue
            rows.append(describe(run_id, job, event, decided))
    rows.sort(key=lambda row: (row.get("analysed_at_utc") or "",
                               row.get("source_start_s") or 0), reverse=True)
    return rows


def counts(rows: list[dict]) -> dict:
    """Headline numbers for the queue: how much is waiting, and of what kind."""
    waiting = [row for row in rows if not row["decided"]]
    return dict(total=len(rows), waiting=len(waiting),
                medical=sum(row["branch"] == "fall" for row in waiting),
                security=sum(row["branch"] == "activity" for row in waiting),
                recordings=len({row["run_id"] for row in rows}))
