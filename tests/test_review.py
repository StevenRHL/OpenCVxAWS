import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from watchverify import jobs, review


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    monkeypatch.setattr(jobs.shutil, "disk_usage", lambda path: SimpleNamespace(free=50 * 1024**3))
    source = tmp_path / "original.mp4"
    source.write_bytes(b"example media bytes for review queue tests")
    return source


def _write_revisions(run_id, revisions):
    with sqlite3.connect(jobs.run_dir(run_id) / "events.db") as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS revisions(event_id TEXT, revision INTEGER, "
                           "payload TEXT, PRIMARY KEY(event_id,revision))")
        for revision in revisions:
            connection.execute("INSERT INTO revisions VALUES(?,?,?)",
                               (revision["event_id"], revision["revision"], json.dumps(revision)))


def _event(run_id, revision=0, **overrides):
    base = {"event_id": "incident-1", "revision": revision, "category": "possible_fall",
            "source_start_s": 2.0, "source_end_s": 3.0, "status": "active"}
    base.update(overrides)
    _write_revisions(run_id, [base])
    return base


def test_duplicate_idempotency_key_is_a_no_op(workspace):
    run_id = jobs.create_job(workspace)
    _event(run_id)
    key = "fixed-key"
    review.record_action(run_id, "incident-1", "ignore", idempotency_key=key)
    review.record_action(run_id, "incident-1", "ignore", idempotency_key=key)
    actions = review.list_actions(run_id, "incident-1")
    assert len(actions) == 1
    assert review.current_state(run_id, "incident-1")["ignored"] is True


def test_ignore_then_undo_clears_ignored_and_both_rows_are_kept(workspace):
    run_id = jobs.create_job(workspace)
    _event(run_id)
    ignored = review.record_action(run_id, "incident-1", "ignore", note="looks like a shelf")
    assert ignored["ignored"] is True
    ignore_action = review.list_actions(run_id, "incident-1")[0]
    undone = review.record_action(run_id, "incident-1", "undo",
                                  previous_action_id=ignore_action["action_id"])
    assert undone["ignored"] is False
    actions = review.list_actions(run_id, "incident-1")
    assert [a["action"] for a in actions] == ["ignore", "undo"]


def test_a_meaningful_revision_reopens_an_ignored_event_without_an_undo(workspace):
    run_id = jobs.create_job(workspace)
    _event(run_id, revision=0)
    state = review.record_action(run_id, "incident-1", "ignore")
    assert state["ignored"] is True
    _event(run_id, revision=1, status="resolved", reason="observed_recovery")
    assert review.current_state(run_id, "incident-1")["ignored"] is False
    assert not any(a["action"] == "undo" for a in review.list_actions(run_id, "incident-1"))


def test_false_alarm_label_and_a_learning_candidate_persist_independently(workspace):
    run_id = jobs.create_job(workspace)
    _event(run_id)
    review.record_action(run_id, "incident-1", "false_alarm", note="shelf, not a person")
    candidate = review.create_candidate(run_id, "incident-1", 2.0, 3.0, "negative",
                                        visible_action_label="person reaching for a shelf")
    assert jobs.get_reviews(run_id)["incident-1"]["label"] == "false_alarm"
    assert review.get_candidate(run_id, candidate["candidate_id"])["proposed_label"] == "negative"


def test_missed_interval_candidate_has_no_event_id(workspace):
    run_id = jobs.create_job(workspace)
    candidate = review.create_candidate(run_id, None, 40.0, 45.0, "positive",
                                        visible_action_label="person falls near the counter")
    assert candidate["event_id"] is None
    assert candidate["candidate_id"] in {c["candidate_id"] for c in review.list_candidates(run_id)}


def test_overlap_warns_unless_confirmed(workspace):
    run_id = jobs.create_job(workspace)
    _event(run_id)
    review.create_candidate(run_id, "incident-1", 2.0, 3.0, "positive",
                            visible_action_label="falls")
    with pytest.raises(review.DuplicateCandidateWarning) as excinfo:
        review.create_candidate(run_id, "incident-1", 2.5, 3.5, "positive",
                                visible_action_label="falls again")
    assert len(excinfo.value.overlaps) == 1
    confirmed = review.create_candidate(run_id, "incident-1", 2.5, 3.5, "positive",
                                        visible_action_label="falls again", confirmed_duplicate=True)
    assert confirmed["candidate_id"]


def test_remove_candidate_retains_the_row_with_history(workspace):
    run_id = jobs.create_job(workspace)
    candidate = review.create_candidate(run_id, None, 5.0, 6.0, "uncertain")
    removed = review.remove_candidate(run_id, candidate["candidate_id"], note="duplicate upload")
    assert removed["status"] == "excluded"
    assert removed["candidate_id"] in {c["candidate_id"] for c in review.list_candidates(run_id)}
    with sqlite3.connect(jobs.run_dir(run_id) / "events.db") as connection:
        rows = connection.execute("SELECT action FROM learning_actions WHERE candidate_id=?",
                                  (candidate["candidate_id"],)).fetchall()
    assert ("removed",) in rows


def test_delete_job_blocks_on_promoted_candidates_unless_a_choice_is_made(workspace):
    run_id = jobs.create_job(workspace)
    candidate = review.create_candidate(run_id, None, 5.0, 6.0, "positive",
                                        visible_action_label="clear fall")
    review.set_candidate_status(run_id, candidate["candidate_id"], "ready_for_dataset_review")
    with pytest.raises(ValueError, match="learning example"):
        jobs.delete_job(run_id)
    jobs.delete_job(run_id, retain_learning_examples=True)
    archive = jobs.ROOT / "outputs" / "_retained_learning" / run_id / "candidates.json"
    assert archive.exists()
    assert not jobs.run_dir(run_id).exists()


def test_delete_job_can_drop_promoted_candidates_when_chosen(workspace):
    run_id = jobs.create_job(workspace)
    candidate = review.create_candidate(run_id, None, 5.0, 6.0, "positive",
                                        visible_action_label="clear fall")
    review.set_candidate_status(run_id, candidate["candidate_id"], "ready_for_dataset_review")
    jobs.delete_job(run_id, retain_learning_examples=False)
    assert not jobs.run_dir(run_id).exists()
    assert not (jobs.ROOT / "outputs" / "_retained_learning" / run_id).exists()


def test_needs_annotation_and_excluded_candidates_never_block_deletion(workspace):
    run_id = jobs.create_job(workspace)
    candidate = review.create_candidate(run_id, None, 5.0, 6.0, "uncertain")
    review.remove_candidate(run_id, candidate["candidate_id"])
    jobs.delete_job(run_id)  # must not raise
    assert not jobs.run_dir(run_id).exists()


def test_no_model_file_changes_across_review_and_queue_actions(workspace):
    models_dir = jobs.ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    model_file = models_dir / "fall.json"
    model_file.write_text('{"threshold": 0.6}')
    before = hashlib.sha256(model_file.read_bytes()).hexdigest()
    run_id = jobs.create_job(workspace)
    _event(run_id)
    review.record_action(run_id, "incident-1", "ignore")
    candidate = review.create_candidate(run_id, "incident-1", 2.0, 3.0, "positive",
                                        visible_action_label="falls")
    review.set_candidate_status(run_id, candidate["candidate_id"], "ready_for_dataset_review")
    review.remove_candidate(run_id, candidate["candidate_id"])
    after = hashlib.sha256(model_file.read_bytes()).hexdigest()
    assert before == after
