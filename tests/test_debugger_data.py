import json
import sqlite3

import pytest

from watchverify import jobs


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    source = tmp_path / "original.mp4"
    source.write_bytes(b"example media bytes for debugger data tests")
    return source


def _write_revisions(run_id, revisions):
    with sqlite3.connect(jobs.run_dir(run_id) / "events.db") as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS revisions(event_id TEXT, revision INTEGER, "
                           "payload TEXT, PRIMARY KEY(event_id,revision))")
        for revision in revisions:
            connection.execute("INSERT INTO revisions VALUES(?,?,?)",
                               (revision["event_id"], revision["revision"], json.dumps(revision)))


def test_revisions_are_returned_oldest_first(workspace):
    run_id = jobs.create_job(workspace)
    first = {"event_id": "incident-1", "revision": 0, "status": "active", "reason": "detected"}
    second = {"event_id": "incident-1", "revision": 1, "status": "resolved", "reason": "observed_recovery"}
    _write_revisions(run_id, [second, first])  # insert out of order
    revisions = jobs.load_event_revisions(run_id, "incident-1")
    assert [r["revision"] for r in revisions] == [0, 1]
    assert revisions[0] == first


def test_first_revision_is_immutable_after_a_later_one_is_appended(workspace):
    run_id = jobs.create_job(workspace)
    first = {"event_id": "incident-1", "revision": 0, "status": "active", "reason": "detected",
             "observations": ["rapid_posture_change"]}
    _write_revisions(run_id, [first])
    assert jobs.load_event_revisions(run_id, "incident-1")[0] == first
    second = {"event_id": "incident-1", "revision": 1, "status": "resolved",
              "reason": "observed_recovery", "observations": ["sustained_normal_activity"]}
    _write_revisions(run_id, [second])
    revisions = jobs.load_event_revisions(run_id, "incident-1")
    assert len(revisions) == 2
    assert revisions[0] == first, "an existing revision row must never be rewritten"
    assert revisions[1] == second


def test_missing_run_or_event_returns_empty(workspace):
    run_id = jobs.create_job(workspace)
    assert jobs.load_event_revisions(run_id, "no-such-event") == []
    assert jobs.load_event_revisions("no-such-run", "no-such-event") == []
