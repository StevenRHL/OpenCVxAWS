"""Escalation log and alert-prompt selection.

These test that a decision is recorded faithfully and that the right prompt is raised —
not that either branch detects anything. Whether an alert deserves a call is a human
judgement the application only ever asks for.
"""
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from watchverify import jobs, alerts


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    monkeypatch.setattr(jobs, "_PROCESSES", {})
    monkeypatch.setattr(jobs.shutil, "disk_usage", lambda path: SimpleNamespace(free=50 * 1024**3))
    source = tmp_path / "original.mp4"
    source.write_bytes(b"example media bytes for escalation tests")
    return source


def seed(run_id, events):
    with sqlite3.connect(jobs.run_dir(run_id) / "events.db") as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS revisions(event_id TEXT, revision INTEGER, payload TEXT, PRIMARY KEY(event_id,revision))")
        for event in events:
            connection.execute("INSERT INTO revisions VALUES(?,?,?)",
                               (event["event_id"], 1, json.dumps(event)))


def event(identifier, category="possible_fall", status="active", start=1.0):
    return {"event_id": identifier, "revision": 1, "category": category,
            "status": status, "source_start_s": start, "track_id": 1}


def test_an_escalation_is_appended_never_rewritten(workspace):
    run_id = jobs.create_job(workspace)
    seed(run_id, [event("incident-1")])
    jobs.save_escalation(run_id, ["incident-1"], "ambulance_not_called", "Looked like sitting down", "caveat text")
    jobs.save_escalation(run_id, ["incident-1"], "ambulance_called", "Changed my mind", "caveat text")
    records = jobs.get_escalations(run_id)
    assert [record["action"] for record in records] == ["ambulance_not_called", "ambulance_called"]
    assert records[0]["sequence"] < records[1]["sequence"]
    assert records[0]["note"] == "Looked like sitting down", "an earlier decision must survive a later one"


def test_the_caveat_shown_at_decision_time_is_stored_with_it(workspace):
    """What the person was told is part of the record. Without it nobody can later tell
    whether a decision was made on an informed basis."""
    run_id = jobs.create_job(workspace)
    seed(run_id, [event("incident-1", "unusual_activity")])
    jobs.save_escalation(run_id, ["incident-1"], "police_not_called", "", "29% of ordinary clips also raise this")
    assert jobs.get_escalations(run_id)[0]["shown_rate"] == "29% of ordinary clips also raise this"


def test_unknown_actions_and_unknown_events_are_refused(workspace):
    run_id = jobs.create_job(workspace)
    seed(run_id, [event("incident-1")])
    with pytest.raises(ValueError, match="Unknown escalation action"):
        jobs.save_escalation(run_id, ["incident-1"], "call_everyone")
    with pytest.raises(ValueError, match="not in the selected analysis"):
        jobs.save_escalation(run_id, ["somebody-elses-event"], "police_called")
    with pytest.raises(ValueError, match="at least one event"):
        jobs.save_escalation(run_id, [], "police_called")


def test_reviews_and_escalations_are_separate_records(workspace):
    """A judgement about the observation and an action taken in the world are different
    things; recording one must not imply the other."""
    run_id = jobs.create_job(workspace)
    seed(run_id, [event("incident-1")])
    jobs.save_review(run_id, "incident-1", "false_alarm", "Bending to a low shelf")
    jobs.save_escalation(run_id, ["incident-1"], "ambulance_called", "Called anyway to be safe")
    assert jobs.get_reviews(run_id)["incident-1"]["label"] == "false_alarm"
    assert jobs.get_escalations(run_id)[0]["action"] == "ambulance_called"


# ---------------------------------------------------------------- prompt selection

def test_pending_alerts_ignore_decided_events_but_not_deferred_ones():
    events = [event("a", "possible_fall"), event("b", "unusual_activity")]
    decided = [{"event_ids": ["a"], "action": "ambulance_called"}]
    assert [e["event_id"] for e in alerts.pending_alerts(events, decided)] == ["b"]
    snoozed = [{"event_ids": ["a"], "action": "deferred"}]
    assert len(alerts.pending_alerts(events, snoozed)) == 2, "'decide later' is a snooze, not a decision"


def test_non_alertable_categories_never_prompt():
    """`recovery` and `activity_clear` are state changes, not alerts."""
    assert alerts.pending_alerts([event("a", "activity_clear")], []) == []
    assert alerts.pending_alerts([event("a", "recovery")], []) == []


def test_police_prompt_takes_precedence_when_both_branches_wait():
    fall_only = [event("a", "person_down")]
    activity_only = [event("b", "unusual_activity")]
    assert alerts.primary_branch(fall_only) == "fall"
    assert alerts.primary_branch(activity_only) == "activity"
    assert alerts.primary_branch(fall_only + activity_only) == "activity"
    assert alerts.primary_branch([]) is None


def test_a_person_down_is_still_offered_behind_a_police_prompt():
    """The precedence rule chooses which question is asked first. It must not make a
    possible medical emergency unreachable."""
    both = [event("a", "person_down"), event("b", "unusual_activity")]
    secondary = alerts.secondary_fall_alerts(both, "activity")
    assert [e["event_id"] for e in secondary] == ["a"]
    assert alerts.secondary_fall_alerts(both, "fall") == []


def test_every_branch_prompt_is_completely_specified():
    for branch, prompt in alerts.BRANCH_PROMPT.items():
        assert set(prompt) == {"heading", "model", "call", "decline", "fallback"}
        for action, _label in (prompt["call"], prompt["decline"]):
            assert action in jobs.ESCALATION_ACTIONS
    assert set(alerts.ALERT_BRANCH.values()) <= set(alerts.BRANCH_PROMPT)


@pytest.mark.parametrize("action", ["police_called", "police_not_called"])
def test_police_decision_leaves_medical_alert_pending(workspace, action):
    run_id = jobs.create_job(workspace)
    events = [event("medical", "person_down"), event("retail", "unusual_activity")]
    seed(run_id, events)
    selected = alerts.alerts_for_branch(events, "activity")
    jobs.save_escalation(run_id, [e["event_id"] for e in selected], action)
    assert [e["event_id"] for e in alerts.pending_alerts(events, jobs.get_escalations(run_id))] == ["medical"]
    jobs.save_escalation(run_id, ["medical"], "ambulance_not_called")
    assert alerts.pending_alerts(events, jobs.get_escalations(run_id)) == []


@pytest.mark.parametrize("action", ["ambulance_called", "ambulance_not_called"])
def test_medical_decision_leaves_activity_pending(workspace, action):
    run_id = jobs.create_job(workspace)
    events = [event("medical"), event("retail", "unusual_activity")]
    seed(run_id, events)
    jobs.save_escalation(run_id, ["medical"], action)
    assert [e["event_id"] for e in alerts.pending_alerts(events, jobs.get_escalations(run_id))] == ["retail"]


@pytest.mark.parametrize("action", ["police_called", "police_not_called", "ambulance_called", "ambulance_not_called"])
def test_mixed_decision_is_rejected_without_writing(workspace, action):
    run_id = jobs.create_job(workspace)
    seed(run_id, [event("medical"), event("retail", "unusual_activity")])
    with pytest.raises(ValueError, match="does not match"):
        jobs.save_escalation(run_id, ["medical", "retail"], action)
    assert jobs.get_escalations(run_id) == []


def test_legacy_mixed_log_does_not_hide_medical_alert():
    events = [event("medical"), event("retail", "unusual_activity")]
    legacy = [{"event_ids": ["medical", "retail"], "action": "police_not_called"}]
    before = json.dumps(legacy)
    assert [e["event_id"] for e in alerts.pending_alerts(events, legacy)] == ["medical"]
    assert json.dumps(legacy) == before
