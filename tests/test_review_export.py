import json
from types import SimpleNamespace

import pytest

from watchverify import jobs, review


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    monkeypatch.setattr(jobs.shutil, "disk_usage", lambda path: SimpleNamespace(free=50 * 1024**3))
    source = tmp_path / "original.mp4"
    source.write_bytes(b"example media bytes for export tests")
    return source


def _ready_candidate(run_id, **restriction_overrides):
    candidate = review.create_candidate(run_id, None, 1.0, 2.0, "positive",
                                        visible_action_label="a clear fall")
    review.set_candidate_status(run_id, candidate["candidate_id"], "ready_for_dataset_review")
    if restriction_overrides:
        with __import__("sqlite3").connect(jobs.run_dir(run_id) / "events.db") as connection:
            connection.execute("UPDATE learning_candidates SET source_restrictions=? WHERE candidate_id=?",
                               (json.dumps(restriction_overrides), candidate["candidate_id"]))
    return review.get_candidate(run_id, candidate["candidate_id"])


def test_export_copies_clip_and_preserves_provenance_fields(workspace, tmp_path, monkeypatch):
    run_id = jobs.create_job(workspace)
    candidate = _ready_candidate(run_id)
    # candidate_clip needs a real source video to cut from; stub it so export logic is
    # exercised without depending on the video pipeline.
    fake_clip = tmp_path / "fake_clip.mp4"
    fake_clip.write_bytes(b"clip bytes")
    monkeypatch.setattr(review, "candidate_clip", lambda run_id, candidate: fake_clip)
    destination = tmp_path / "export"
    result = review.export_candidates([candidate["candidate_id"]], destination=destination)
    assert result["exported"] == [candidate["candidate_id"]]
    assert result["skipped"] == []
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest[0]["recording_group"] == candidate["recording_group"]
    assert manifest[0]["source_restrictions"] == candidate["source_restrictions"]
    assert (destination / manifest[0]["clip_file"]).read_bytes() == b"clip bytes"
    assert review.get_candidate(run_id, candidate["candidate_id"])["status"] == "exported"


def test_not_ready_candidates_are_skipped_and_reported(workspace, tmp_path):
    run_id = jobs.create_job(workspace)
    candidate = review.create_candidate(run_id, None, 1.0, 2.0, "uncertain")
    result = review.export_candidates([candidate["candidate_id"]], destination=tmp_path / "export")
    assert result["exported"] == []
    assert result["skipped"] == [{"candidate_id": candidate["candidate_id"], "reason": "not ready"}]


def test_exclude_from_export_restriction_is_honored(workspace, tmp_path, monkeypatch):
    run_id = jobs.create_job(workspace)
    candidate = _ready_candidate(run_id, exclude_from_export=True)
    fake_clip = tmp_path / "fake_clip.mp4"
    fake_clip.write_bytes(b"clip bytes")
    monkeypatch.setattr(review, "candidate_clip", lambda run_id, candidate: fake_clip)
    result = review.export_candidates([candidate["candidate_id"]], destination=tmp_path / "export")
    assert result["exported"] == []
    assert result["skipped"][0]["reason"] == "marked excluded from export"
    assert review.get_candidate(run_id, candidate["candidate_id"])["status"] == "ready_for_dataset_review"


def test_export_batch_size_is_capped(workspace):
    run_id = jobs.create_job(workspace)
    too_many = [f"candidate-{i}" for i in range(review.MAX_EXPORT_BATCH + 1)]
    with pytest.raises(ValueError, match="at most"):
        review.export_candidates(too_many)


def test_no_model_file_changes_across_export(workspace, tmp_path, monkeypatch):
    import hashlib
    models_dir = jobs.ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    model_file = models_dir / "fall.json"
    model_file.write_text('{"threshold": 0.6}')
    before = hashlib.sha256(model_file.read_bytes()).hexdigest()
    run_id = jobs.create_job(workspace)
    candidate = _ready_candidate(run_id)
    fake_clip = tmp_path / "fake_clip.mp4"
    fake_clip.write_bytes(b"clip bytes")
    monkeypatch.setattr(review, "candidate_clip", lambda run_id, candidate: fake_clip)
    review.export_candidates([candidate["candidate_id"]], destination=tmp_path / "export")
    after = hashlib.sha256(model_file.read_bytes()).hexdigest()
    assert before == after
