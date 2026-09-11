"""Experiment boundaries and compatibility; no corpus accuracy is measured here."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_models import development_rows, load_blocks, candidates


def manifest(tmp_path, rows):
    folder = tmp_path / 'data/manifests'
    folder.mkdir(parents=True)
    (folder / 'splits.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))


def test_test_cache_is_excluded_before_feature_loading(tmp_path):
    rows = []
    for split in ('train', 'validation', 'test'):
        path = tmp_path / f'{split}.npz'
        if split == 'test':
            path.write_bytes(b'not a readable cache; experiment must never open this')
        else:
            np.savez(path, source_sha256=split, x=np.ones((2, 3)))
        rows.append(dict(corpus='mnnit', eligible=True, split=split,
                         source_sha256=split, feature_cache=path.name, recording_id=split))
    manifest(tmp_path, rows)
    training = SimpleNamespace(clip_windows=lambda d: [(0, 0, row) for row in d['x']])
    blocks, opened = load_blocks(tmp_path, 'activity', development_rows(tmp_path), training)
    assert set(blocks) == {'train', 'validation'}
    assert [r['split'] for r in opened] == ['train', 'validation']


def test_frozen_duplicate_content_in_different_splits_is_rejected(tmp_path):
    manifest(tmp_path, [dict(eligible=True, source_sha256='same', split=s)
                        for s in ('train', 'test')])
    with pytest.raises(ValueError, match='multiple frozen splits'):
        development_rows(tmp_path)


@pytest.mark.parametrize('name', ['extra_trees_leaf2', 'extra_trees_leaf8',
                                 'hist_gradient_leaves7', 'hist_gradient_leaves15'])
def test_new_families_fit_and_reload_with_probability_interface(tmp_path, name):
    import joblib
    from threadpoolctl import threadpool_limits
    model = candidates('fall', None)[name]
    rng = np.random.default_rng(42)
    x = rng.normal(size=(80, 24))
    y = (x[:, 0] > 0).astype(int)
    with threadpool_limits(limits=1):
        model.fit(x, y)
        artifact = tmp_path / 'candidate.joblib'
        joblib.dump(model, artifact)
        scores = joblib.load(artifact).predict_proba(x[:5])
    assert scores.shape == (5, 2)
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores.sum(axis=1), 1.)
    if name.startswith('hist'):
        assert model.early_stopping is False
