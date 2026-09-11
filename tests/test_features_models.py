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
    emitted = []
    for t in times:
        out = aggregator.update(track, t, np.full(12, 0.5), np.ones(12, bool), 0.1, 1.0, False, True, 0.9)
        if out is not None:
            emitted.append(t)
    return emitted


def test_no_score_before_a_full_window_exists():
    aggregator = WindowAggregator()
    assert _feed(aggregator, np.arange(0, 3, 0.1)) == [], "must abstain on a partial window"


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
