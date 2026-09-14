import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from watchverify import jobs


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    monkeypatch.setattr(jobs.shutil, "disk_usage", lambda path: SimpleNamespace(free=50 * 1024**3))
    source = tmp_path / "original.mp4"
    source.write_bytes(b"example media bytes for trace tests")
    return source


def _write_event(run_id, **overrides):
    event = {"event_id": "incident-1", "revision": 0, "category": "possible_fall",
             "track_id": 1, "source_start_s": 1.0, "source_end_s": 2.0, "status": "active"}
    event.update(overrides)
    with sqlite3.connect(jobs.run_dir(run_id) / "events.db") as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS revisions(event_id TEXT, revision INTEGER, "
                           "payload TEXT, PRIMARY KEY(event_id,revision))")
        connection.execute("INSERT INTO revisions VALUES(?,?,?)",
                           (event["event_id"], event["revision"], json.dumps(event)))
    return event


def test_recorded_trace_when_schema_version_and_frames_exist(workspace):
    run_id = jobs.create_job(workspace)
    _write_event(run_id)
    jobs.update_job(run_id, trace_schema_version=1)
    with sqlite3.connect(jobs.run_dir(run_id) / "events.db") as connection:
        connection.execute("CREATE TABLE event_frames(event_id TEXT, t REAL, track_id INTEGER, "
                           "source TEXT, fall_score REAL, fall_threshold REAL, fall_positive INTEGER, "
                           "fall_version TEXT, activity_score REAL, activity_threshold REAL, "
                           "activity_positive INTEGER, activity_version TEXT, quality REAL, angle REAL, "
                           "down INTEGER, gate_state TEXT, old_track_id INTEGER, handover_reason TEXT, "
                           "PRIMARY KEY(event_id,t,track_id))")
        connection.execute("INSERT INTO event_frames VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           ("incident-1", 1.5, 1, "live", 0.9, 0.3, 1, "v1", None, None, None, None,
                            0.8, 45.0, 1, json.dumps({"low": 1.5}), None, None))
    trace = jobs.load_event_trace(run_id, "incident-1")
    assert trace["kind"] == "recorded"
    assert len(trace["rows"]) == 1
    assert trace["rows"][0]["fall_score"] == 0.9
    assert trace["rows"][0]["gate_state"] == {"low": 1.5}


def test_reconstructed_trace_from_predictions_jsonl_without_schema_version(workspace):
    run_id = jobs.create_job(workspace)
    _write_event(run_id)
    predictions = jobs.run_dir(run_id) / "predictions.jsonl"
    with predictions.open("w") as handle:
        handle.write(json.dumps({"t": 1.5, "track_id": 1,
                                 "fall_model": {"score": 0.7, "threshold": 0.3, "positive": True},
                                 "activity_model": None, "quality": 0.9, "angle": 50.0, "down": True}) + "\n")
        handle.write(json.dumps({"t": 5.0, "track_id": 1,
                                 "fall_model": {"score": 0.1, "threshold": 0.3, "positive": False},
                                 "activity_model": None}) + "\n")  # outside the event window
    trace = jobs.load_event_trace(run_id, "incident-1")
    assert trace["kind"] == "reconstructed"
    assert len(trace["rows"]) == 1
    assert trace["rows"][0]["t"] == 1.5
    assert trace["ambiguous_frames_dropped"] == 0


def test_ambiguous_frames_are_dropped_not_guessed(workspace):
    run_id = jobs.create_job(workspace)
    _write_event(run_id, event_id="incident-1", source_start_s=1.0, source_end_s=3.0)
    _write_event(run_id, event_id="incident-2", revision=0, source_start_s=1.5, source_end_s=2.5)
    predictions = jobs.run_dir(run_id) / "predictions.jsonl"
    with predictions.open("w") as handle:
        handle.write(json.dumps({"t": 2.0, "track_id": 1, "fall_model": None, "activity_model": None}) + "\n")
        handle.write(json.dumps({"t": 1.2, "track_id": 1, "fall_model": None, "activity_model": None}) + "\n")
    trace = jobs.load_event_trace(run_id, "incident-1")
    assert trace["kind"] == "reconstructed"
    assert len(trace["rows"]) == 1  # t=1.2 unambiguous; t=2.0 overlaps incident-2
    assert trace["ambiguous_frames_dropped"] == 1


def test_unavailable_when_neither_source_exists(workspace):
    run_id = jobs.create_job(workspace)
    _write_event(run_id)
    trace = jobs.load_event_trace(run_id, "incident-1")
    assert trace == {"kind": "unavailable", "rows": []}


def test_load_event_handovers_returns_only_frames_with_a_handover(workspace):
    run_id = jobs.create_job(workspace)
    _write_event(run_id)
    jobs.update_job(run_id, trace_schema_version=1)
    with sqlite3.connect(jobs.run_dir(run_id) / "events.db") as connection:
        connection.execute("CREATE TABLE event_frames(event_id TEXT, t REAL, track_id INTEGER, "
                           "source TEXT, fall_score REAL, fall_threshold REAL, fall_positive INTEGER, "
                           "fall_version TEXT, activity_score REAL, activity_threshold REAL, "
                           "activity_positive INTEGER, activity_version TEXT, quality REAL, angle REAL, "
                           "down INTEGER, gate_state TEXT, old_track_id INTEGER, handover_reason TEXT, "
                           "PRIMARY KEY(event_id,t,track_id))")
        connection.execute("INSERT INTO event_frames VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           ("incident-1", 1.0, 1, "live", None, None, None, None, None, None, None,
                            None, None, None, None, None, None, None))
        connection.execute("INSERT INTO event_frames VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           ("incident-1", 1.5, 1, "live", None, None, None, None, None, None, None,
                            None, None, None, None, None, 7, "evidence_relay"))
    handovers = jobs.load_event_handovers(run_id, "incident-1")
    assert len(handovers) == 1
    assert handovers[0] == {"t": 1.5, "old_track_id": 7, "reason": "evidence_relay"}
