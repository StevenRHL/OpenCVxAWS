"""Deterministic cancellation/failure checks with real local video encoding.

Pose and input timing are controlled fixtures. These test lifecycle and evidence
preservation, not detection accuracy or operating-system process termination.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from watchverify import jobs, worker
from watchverify.perception import frames as decoded_frames
from test_visibility import standing_pose


@pytest.mark.parametrize("stop", ["cancel", "failure", "before_first_frame"])
def test_interruption_preserves_evidence_and_retry_uses_new_run(tmp_path, monkeypatch, stop):
    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(jobs, "_PROCESSES", {})
    monkeypatch.setattr(jobs.shutil, "disk_usage", lambda _: SimpleNamespace(free=50*1024**3))
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source bytes preserved; decoder replaced by timed fixture")
    monkeypatch.setattr(worker, "video_info", lambda _: dict(
        duration_s=4.0, width=64, height=64, fps=10))
    class Models:
        loaded = {}
        status = {}
        def disclosures(self): return {}
        def score(self, *args): return None
    monkeypatch.setattr(worker, "Models", Models)
    class Pose:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def detect(self, frame, t): return [standing_pose()]
    monkeypatch.setattr(worker, "PoseEstimator", Pose)
    class Rules:
        # `transition_window` and `release` are part of the detector contract the worker
        # relies on to carry fall evidence across a renumbering; this fixture never ends an
        # identity, so it releases nothing.
        transition_window = 2.
        def __init__(self, **kwargs): pass
        def reset(self, track): pass
        def release(self, track): return None
        def adopt(self, track, evidence, t): return False
        def update(self, track, features, t):
            return [dict(category="person_down", score=1., observations=["fixture_down"])]
    monkeypatch.setattr(worker, "RuleDetector", Rules)
    run_id = jobs.create_job(source)
    def input_frames(_):
        for i, t in enumerate([0., .1, .2, 3.]):
            if (stop == "before_first_frame" and i == 0) or (stop == "cancel" and i == 3):
                jobs.cancel_job(run_id)
            if stop == "failure" and i == 3:
                raise ValueError("fixture decoder failure")
            yield i, t, np.zeros((64, 64, 3), dtype=np.uint8)
    monkeypatch.setattr(worker, "frames", input_frames)
    # Direct worker invocation has no subprocess handle; keep cancellation reconciliation
    # from treating this deliberately in-process fixture as an exited child.
    monkeypatch.setattr(jobs, "_alive", lambda _: True)
    assert worker.run(run_id) == ("failed" if stop == "failure" else "cancelled")
    folder = jobs.run_dir(run_id)
    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics["source_duration_processed_s"] == (0. if stop == "before_first_frame" else .2)
    assert metrics["unprocessed_source_s"] == (4. if stop == "before_first_frame" else 3.8)
    assert jobs.get_job(run_id)["summary"]["source_time_s"] == metrics["source_duration_processed_s"]
    events = jobs.load_events(run_id)
    if stop == "before_first_frame":
        assert not events and metrics["analysed_frames"] == 0
    else:
        assert events and all(e["status"] == "incomplete" for e in events)
        assert all(e["reason"] == ("processing_failed" if stop == "failure" else "cancelled") for e in events)
        assert list(decoded_frames(folder / "annotated.mp4"))[-1][1] == pytest.approx(.2)
        jobs.save_review(run_id, events[0]["event_id"], "unclear", "Preserve on retry")
    with pytest.raises(ValueError, match="new analysis"):
        jobs.launch_job(run_id)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in folder.iterdir() if p.is_file()}
    retry = jobs.create_job(jobs.get_job(run_id)["source_path"])
    monkeypatch.setattr(worker, "frames", lambda _: iter([
        (0, 0., np.zeros((64, 64, 3), dtype=np.uint8)),
        (1, .1, np.zeros((64, 64, 3), dtype=np.uint8))]))
    assert retry != run_id and worker.run(retry) == "completed"
    assert not (jobs.run_dir(retry) / "cancel.request").exists()
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.iterdir() if p.is_file()} == before
