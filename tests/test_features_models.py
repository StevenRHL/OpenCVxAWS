"""Windowed descriptor and card-driven model loading. Software behaviour, not accuracy."""
import hashlib
import json

import numpy as np
import pytest

from watchverify.features import (CLIP_COLUMNS, WINDOW_S, WindowAggregator, clip_descriptor,
                                  windows)


def frame(n=1, value=0.5):
    return np.full((n, 24), value)


def test_descriptor_length_matches_its_column_names():
    d = clip_descriptor(frame(5), np.zeros(5), np.zeros(5), np.zeros(5, bool), np.ones(5, bool), np.ones(5))
    assert d.shape == (len(CLIP_COLUMNS),) == (127,)


def test_descriptor_rejects_the_wrong_frame_width():
    with pytest.raises(ValueError, match="frame features"):
        clip_descriptor(np.zeros((5, 12)), np.zeros(5), np.zeros(5), np.zeros(5, bool),
                        np.zeros(5, bool), np.ones(5))


def test_windows_are_fixed_length_and_stride_forward():
    t = np.arange(0, 12, 0.1)
    x = frame(len(t))
    built = windows(t, x, np.zeros(len(t)), np.zeros(len(t)),
                    np.zeros(len(t), bool), np.ones(len(t), bool), np.ones(len(t)))
    assert built, "a 12 s sequence must yield windows"
    assert all(abs((end - start) - WINDOW_S) < 1e-9 for start, end, _ in built)
    starts = [s for s, _, _ in built]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)


def test_a_sequence_shorter_than_one_window_yields_nothing():
    t = np.arange(0, 2, 0.1)
    assert windows(t, frame(len(t)), np.zeros(len(t)), np.zeros(len(t)),
                   np.zeros(len(t), bool), np.zeros(len(t), bool), np.ones(len(t))) == []


# --- the aggregator abstains rather than guessing ---------------------------

def _feed(aggregator, times, track=1):
    """Observation times at which the aggregator closed at least one interval."""
    emitted = []
    for t in times:
        for _ in aggregator.update(track, t, np.full(12, 0.5), np.ones(12, bool), 0.1, 1.0, False, True, 0.9):
            emitted.append(t)
    return emitted


def _windows(aggregator, times, track=1):
    emitted = []
    for t in times:
        emitted.extend(aggregator.update(track, t, np.full(12, 0.5), np.ones(12, bool), 0.1, 1.0, False, True, 0.9))
    return emitted


def test_no_score_before_a_full_window_exists():
    """Silence through 4.9 s, not merely through 3 s.

    The earlier version of this test stopped at 3 s and so could not see the runtime
    emitting its first descriptor at 4.0 s — a window the training builder never produces.
    """
    aggregator = WindowAggregator()
    assert _feed(aggregator, np.arange(0, 5.0, 0.1)) == [], "must abstain on a partial window"
    assert _feed(aggregator, [5.0]) == [5.0], "a complete window must score immediately"


def test_a_score_appears_once_the_window_fills_and_then_respects_the_stride():
    aggregator = WindowAggregator()
    emitted = _feed(aggregator, np.arange(0, 12, 0.1))
    assert emitted, "a full window must produce a descriptor"
    gaps = np.diff(emitted)
    assert (gaps >= 0.99).all(), "emissions must be at least one stride apart"


def test_a_gap_clears_the_window_so_time_is_never_averaged_across_it():
    aggregator = WindowAggregator()
    _feed(aggregator, np.arange(0, 12, 0.1))
    # Jump forward past max_gap: the window must rebuild from scratch.
    assert _feed(aggregator, [30.0, 30.1, 30.2]) == []


def test_reset_discards_a_persons_history():
    aggregator = WindowAggregator()
    _feed(aggregator, np.arange(0, 12, 0.1))
    aggregator.reset(1)
    assert _feed(aggregator, [12.1, 12.2, 12.3]) == []


# --- the app and the trainer build the same object ---------------------------
#
# These assert descriptor equality, not merely similar behaviour. The defect they close
# was a runtime window that no training window could equal: it closed a second early and
# included its own endpoint, so the shipped app fed the model an input shape that never
# appeared in the corpus the model was fitted on.


def _sequence(times, seed=0):
    rng = np.random.default_rng(seed)
    t = np.asarray(times, dtype=float)
    return (t, rng.normal(size=(len(t), 24)), rng.normal(size=len(t)), rng.normal(size=len(t)),
            rng.integers(0, 2, len(t)).astype(bool), rng.integers(0, 2, len(t)).astype(bool),
            rng.random(len(t)))


def _runtime(sequence, **kwargs):
    t, x, hip, angular, down, upright, quality = sequence
    aggregator = WindowAggregator(**kwargs)
    emitted = []
    for i in range(len(t)):
        emitted.extend(aggregator.update(1, t[i], x[i][:12], x[i][12:], hip[i], angular[i],
                                         down[i], upright[i], quality[i]))
    return emitted


def test_the_runtime_descriptor_equals_the_training_descriptor_on_regular_timing():
    sequence = _sequence(np.arange(0, 12, 0.1))
    offline = windows(*sequence)
    live = _runtime(sequence)
    assert len(live) == len(offline) > 0
    for window, (start, end, descriptor) in zip(live, offline):
        assert (window.start, window.end) == (start, end)
        assert np.array_equal(window.descriptor, descriptor), "runtime must feed the trained input"


def test_the_two_paths_agree_on_irregular_timing_within_the_gap_limit():
    rng = np.random.default_rng(7)
    times = np.cumsum(rng.uniform(0.05, 0.45, 120))
    sequence = _sequence(times, seed=3)
    offline = windows(*sequence)
    live = _runtime(sequence)
    assert len(live) == len(offline) > 0
    assert all(np.array_equal(w.descriptor, d) for w, (_, _, d) in zip(live, offline))


def test_on_gapped_input_the_runtime_emits_a_strict_subset_of_training_windows():
    """The one declared asymmetry: the trainer summarises across a hole, the app refuses.

    Every window the app emits must still be one the trainer would have produced, so the
    app is never fed a window shape the corpus does not contain.
    """
    times = list(np.arange(0, 6, 0.1)) + list(np.arange(7.0, 13, 0.1))
    sequence = _sequence(times, seed=5)
    offline = windows(*sequence)
    live = _runtime(sequence)
    assert len(live) < len(offline), "a gap must cost the runtime windows"
    by_start = {round(start, 6): descriptor for start, _, descriptor in offline}
    for window in live:
        assert round(window.start, 6) in by_start, "runtime emitted a window training never would"
        assert np.array_equal(window.descriptor, by_start[round(window.start, 6)])


def test_continuity_is_reported_so_a_run_cannot_be_counted_across_a_gap():
    live = _runtime(_sequence(np.arange(0, 9, 0.1)))
    assert [w.continuous for w in live] == [False] + [True] * (len(live) - 1)

    times = list(np.arange(0, 6, 0.1)) + list(np.arange(7.0, 13, 0.1))
    across = _runtime(_sequence(times, seed=5))
    assert any(not w.continuous for w in across[1:]), "the window after a gap is not a continuation"


# --- card-driven loading ----------------------------------------------------

class _Probability:
    n_features_in_ = 3
    def predict_proba(self, x):
        return np.array([[0.2, 0.8]])


def _install(tmp_path, monkeypatch, card_extra, model=None):
    import watchverify.models as module
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    import joblib
    artifact = models_dir / "activity.joblib"
    joblib.dump(model or _Probability(), artifact)
    card = {"run_id": "test", "artifact": "activity.joblib",
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "threshold": 0.5, **card_extra}
    (models_dir / "activity.json").write_text(json.dumps(card))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module.Models()


def test_a_mismatched_feature_count_is_refused_at_load_not_at_runtime(tmp_path, monkeypatch):
    models = _install(tmp_path, monkeypatch,
                      {"input": "clip_descriptor", "scoring": "probability", "n_features": 99})
    assert "activity" not in models.loaded
    assert "expects 99 features" in models.status["activity"]


def test_a_card_claiming_anomaly_scoring_on_a_classifier_is_refused(tmp_path, monkeypatch):
    models = _install(tmp_path, monkeypatch,
                      {"input": "clip_descriptor", "scoring": "anomaly_score", "n_features": 3})
    assert "activity" not in models.loaded
    assert "anomaly" in models.status["activity"]


def test_a_clip_descriptor_model_is_scored_without_appending_a_mask(tmp_path, monkeypatch):
    models = _install(tmp_path, monkeypatch,
                      {"input": "clip_descriptor", "scoring": "probability", "n_features": 3})
    result = models.score("activity", np.zeros(3))
    assert result["score"] == pytest.approx(0.8) and result["positive"] is True
    assert result["score_type"] == "probability"


def test_a_wrong_length_input_returns_no_score_rather_than_raising(tmp_path, monkeypatch):
    models = _install(tmp_path, monkeypatch,
                      {"input": "clip_descriptor", "scoring": "probability", "n_features": 3})
    assert models.score("activity", np.zeros(7)) is None
