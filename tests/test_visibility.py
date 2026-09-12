"""What the run could not see, reported as time rather than silence.

A run that emits no events tells a reviewer nothing on its own: it could mean nobody fell,
or it could mean nobody was ever visible. These tests pin the difference. The pose stub
replaces MediaPipe so the visible and invisible stretches are chosen by the test rather
than by a model's behaviour on real footage; no detection accuracy is measured here.
"""
import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from watchverify import jobs, worker
from watchverify.core import Tracker

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "data/processed/urfall/fall-04.mp4"
# Pose is stubbed here, but the worker still decodes a real file. Prepared footage is not
# in the repository, so on a fresh clone this says what is missing rather than failing
# inside create_job with "Select a non-empty video file", which names the wrong cause.
needs_clip = pytest.mark.skipif(not CLIP.exists(),
                                reason="needs prepared UR Fall footage: run "
                                       "scripts/acquire_data.py then scripts/prepare_urfall.py")


def standing_pose(cx=300.0, cy=200.0):
    p = np.zeros((33, 4), dtype=float)
    values = {11: (-10, -40), 12: (10, -40), 13: (-15, -20), 14: (15, -20),
              15: (-18, 10), 16: (18, 10), 23: (-8, 0), 24: (8, 0),
              25: (-8, 30), 26: (8, 30), 27: (-8, 60), 28: (8, 60)}
    for joint, (dx, dy) in values.items():
        p[joint, :2] = (cx + dx, cy + dy)
        p[joint, 3] = 1.0
    return p


def pose_stub(visible_until):
    class Stub:
        def __init__(self, *args, **kwargs):
            pass

        def detect(self, bgr, t):
            return [standing_pose()] if t < visible_until else []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False
    return Stub


def run_with(visible_until, folder):
    target = Path(folder)
    with patch.object(jobs, "ROOT", target), patch.object(worker, "ROOT", target), \
         patch.object(worker, "PoseEstimator", pose_stub(visible_until)):
        run_id = jobs.create_job(CLIP)
        state = worker.run(run_id)
        metrics = json.loads((target / "outputs" / run_id / "metrics.json").read_text())
    return run_id, state, metrics


@needs_clip
def test_time_with_no_usable_person_is_reported_as_an_interval_not_as_quiet():
    with tempfile.TemporaryDirectory() as folder:
        _run_id, state, metrics = run_with(1.0, folder)
    assert state == "completed"
    assert metrics["frames_without_usable_person"] > 0
    intervals = metrics["unobserved_intervals"]
    assert intervals, "the blind stretch must be named, not left implied by an empty event list"
    start, end = intervals[-1]
    assert start >= 1.0 and end == pytest.approx(metrics["source_duration_processed_s"], abs=0.2)
    assert metrics["unobserved_source_s"] == pytest.approx(sum(b - a for a, b in intervals), abs=1e-3)
    assert any("unknown, not clear" in warning for warning in metrics["warnings"])


@needs_clip
def test_a_fully_observed_run_claims_no_blind_time():
    with tempfile.TemporaryDirectory() as folder:
        _run_id, state, metrics = run_with(math.inf, folder)
    assert state == "completed"
    assert metrics["unobserved_intervals"] == []
    assert metrics["unobserved_source_s"] == 0.0
    assert metrics["frames_without_usable_person"] == 0


@needs_clip
def test_decoded_and_total_source_duration_are_both_recorded():
    with tempfile.TemporaryDirectory() as folder:
        _run_id, _state, metrics = run_with(math.inf, folder)
    assert metrics["source_duration_s"] == pytest.approx(3.2, abs=0.1)
    assert metrics["source_duration_processed_s"] <= metrics["source_duration_s"] + 0.1
    assert metrics["unprocessed_source_s"] == pytest.approx(
        max(metrics["source_duration_s"] - metrics["source_duration_processed_s"], 0), abs=1e-3)


def test_tracker_separates_a_crossing_from_an_absence():
    """Both end an identity, but only one means two people may have been confused."""
    tracker = Tracker()
    tracker.update([standing_pose(100)], 0.0)
    tracker.update([standing_pose(101)], 0.1)
    assert tracker.gap_retired_ids == [] and tracker.ambiguous_ids == []
    tracker.update([], 2.0)  # Longer than max_gap: the person simply stopped being visible.
    assert tracker.gap_retired_ids and not tracker.ambiguous_ids

    crossing = Tracker()
    crossing.update([standing_pose(100), standing_pose(140)], 0.0)
    crossing.update([standing_pose(120), standing_pose(121)], 0.1)
    assert crossing.ambiguous_ids, "an ambiguous crossing must be distinguishable in the counts"
    assert set(crossing.ambiguous_ids) <= set(crossing.retired_ids)
