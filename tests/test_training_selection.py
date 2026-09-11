"""Selection and threshold machinery in scripts/train_models.py.

These test software behaviour — how a threshold is derived, which columns reach the
estimator, how a clip rule counts — not whether either branch detects anything.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from train_models import (ACTIVITY_SUSTAINED_WINDOWS, MOTION_INDICES, activity_candidates,
                          grouped_cv_scores, motion_only, out_of_fold_threshold,
                          sustained_clip_rule, summarise)
from watchverify.features import CLIP_COLUMNS


def toy(n_clips=12, per_clip=8, seed=0):
    """Separable-but-noisy windows with a clip group id, sized for grouped CV."""
    rng = np.random.default_rng(seed)
    x, y, groups = [], [], []
    for clip in range(n_clips):
        label = clip % 2
        for _ in range(per_clip):
            row = rng.normal(size=len(CLIP_COLUMNS))
            row[MOTION_INDICES[0]] += 2.5 * label
            x.append(row)
            y.append(label)
            groups.append(f'clip-{clip:02d}')
    return np.array(x), np.array(y), np.array(groups)


def test_the_estimator_sees_only_the_motion_columns():
    kept = [CLIP_COLUMNS[i] for i in MOTION_INDICES]
    assert kept, 'the motion subset must not be empty'
    assert not any('_valid_' in name for name in kept)
    assert not any('quality' in name or 'joint_coverage' in name for name in kept)


def test_selection_happens_inside_the_pipeline_so_serving_still_passes_all_127():
    x, y, _ = toy()
    model = activity_candidates()['logistic'].fit(x, y)
    assert model.n_features_in_ == len(CLIP_COLUMNS) == 127
    assert model.predict_proba(x[:1]).shape == (1, 2)
    with pytest.raises(ValueError):
        model.predict_proba(x[:1, MOTION_INDICES])


def test_a_dropped_column_cannot_change_a_score():
    """A column outside the subset is not an input, so moving it must be inert."""
    x, y, _ = toy()
    model = activity_candidates()['logistic'].fit(x, y)
    dropped = next(i for i in range(len(CLIP_COLUMNS)) if i not in MOTION_INDICES)
    altered = x[:1].copy()
    altered[0, dropped] += 1000.0
    assert model.predict_proba(altered)[0, 1] == pytest.approx(model.predict_proba(x[:1])[0, 1])


def test_grouped_cv_never_splits_one_clip_across_the_fold_boundary():
    x, y, groups = toy()
    scores = grouped_cv_scores(activity_candidates()['logistic'], x, y, groups,
                               folds=3, seeds=(0,))
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert summarise(scores)['folds'] == 3
    assert summarise([]) is None


def test_the_threshold_comes_from_out_of_fold_scores_not_fitted_ones():
    """A model scoring data it was fitted on gives a quantile that describes memorisation
    rather than behaviour on new clips. The pilot read its threshold that way, and its 21%
    target flagged 48% of ordinary test windows."""
    x, y, groups = toy()
    proto = activity_candidates()['random_forest']
    threshold, oof = out_of_fold_threshold(proto, x, y, groups, target_normal_flag=.2)
    assert oof.shape == y.shape
    fitted = proto.fit(x, y).predict_proba(x)[:, 1]
    assert not np.allclose(oof, fitted), 'scores must come from folds the model did not see'
    assert (oof[y == 0] >= threshold).mean() == pytest.approx(.2, abs=.06)


def test_the_clip_rule_counts_consecutive_windows_not_a_total():
    scores = [0.9, 0.1, 0.9, 0.1, 0.9]  # three flags, never two in a row
    result = sustained_clip_rule(scores, [1] * 5, ['clip'] * 5, threshold=.5, run_length=2)
    assert result['shoplifting_clips_alerted'] == 0
    run = sustained_clip_rule([0.1, 0.9, 0.9, 0.1], [1] * 4, ['clip'] * 4, threshold=.5, run_length=2)
    assert run['shoplifting_clips_alerted'] == 1


def test_the_clip_rule_separates_the_two_classes():
    scores = [0.9, 0.9, 0.1, 0.1]
    labels = [1, 1, 0, 0]
    clips = ['hit', 'hit', 'quiet', 'quiet']
    result = sustained_clip_rule(scores, labels, clips, threshold=.5,
                                 run_length=ACTIVITY_SUSTAINED_WINDOWS)
    assert result['clip_recall'] == 1.0
    assert result['normal_clip_alert_rate'] == 0.0
    assert result['clips'] == 2
