"""Bounded development comparison; test caches are excluded before loading.

Writes experiment artifacts only. Installed models are never overwritten. Uses the
frozen manifest, existing features and 5-fold/3-seed recording-grouped CV. Validation
is inspected only for the best new family selected by training CV and the incumbent.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def development_rows(root):
    rows = [json.loads(line) for line in
            (root / 'data/manifests/splits.jsonl').read_text().splitlines() if line.strip()]
    seen = {}
    for row in rows:
        if not row['eligible']:
            continue
        group = row['source_sha256']
        if group in seen and seen[group] != row['split']:
            raise ValueError('Content occurs in multiple frozen splits')
        seen[group] = row['split']
    # Never open a test cache, even to build an inventory or check its shape.
    return [r for r in rows if r['eligible'] and r['split'] in ('train', 'validation')]


def candidates(branch, training):
    values = {}
    for leaf in (2, 8):
        values[f'extra_trees_leaf{leaf}'] = ExtraTreesClassifier(
            n_estimators=200, min_samples_leaf=leaf, max_features=.7,
            class_weight='balanced', random_state=42, n_jobs=1)
    for leaves in (7, 15):
        # No automatic random validation split: all selection stays grouped.
        values[f'hist_gradient_leaves{leaves}'] = HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=leaves, learning_rate=.05,
            l2_regularization=2., min_samples_leaf=20, early_stopping=False,
            random_state=42)
    if branch == 'activity':
        values = {k: make_pipeline(training.motion_only(), v) for k, v in values.items()}
    return values


def load_blocks(root, branch, rows, training):
    blocks = {'train': [], 'validation': []}
    opened = []
    # UR label table includes all source IDs, but we only use labels for selected
    # development recordings. Test features, footage and test metrics stay unopened.
    labels = training.load_posture_labels(root=root) if branch == 'fall' else None
    for r in rows:
        corpus, split = r['corpus'], r['split']
        if branch == 'activity' and corpus != 'mnnit': continue
        if branch == 'fall' and corpus != 'urfall':
            if not (corpus == 'mnnit' and split == 'train'
                    and r['recording_id'].startswith('normal_')): continue
        path = root / r['feature_cache']
        with np.load(path, allow_pickle=False) as d:
            if str(d['source_sha256']) != r['source_sha256']:
                raise ValueError(f'Frozen source mismatch: {path}')
            if branch == 'activity':
                x = np.asarray([v for _, _, v in training.clip_windows(d)])
                y = np.full(len(x), int(r['recording_id'].startswith('shoplifting')))
            else:
                x = training.xy(d)
                if corpus == 'urfall':
                    if str(d['frame_id_source']) != 'sidecar_mapping':
                        raise ValueError('Missing source frame mapping')
                    posture = np.asarray([labels[r['recording_id']].get(int(i), 0)
                                          for i in d['source_frame']])
                    x, y = x[posture != 0], (posture[posture != 0] == 1).astype(int)
                else:
                    y = np.zeros(len(x), dtype=int)  # Existing weak retail negatives.
            if len(x):
                blocks[split].append((x, y, np.full(len(x), r['source_sha256']),
                                      np.full(len(x), r['recording_id'])))
        opened.append(dict(path=r['feature_cache'], split=split,
                           cache_sha256=digest(path), source_sha256=r['source_sha256']))
    return {s: tuple(np.concatenate([r[i] for r in records]) for i in range(4))
            for s, records in blocks.items()}, opened


def fall_points(model, root, rows, training, thresholds, card):
    from watchverify.evaluation import frame_times, fall_ground_truth, replay, match_fall_events, aggregate
    from watchverify.core import RuleDetector
    from evaluate_events import unobserved_seconds
    labels = training.load_posture_labels(root=root)
    all_results = {float(t): [] for t in thresholds}
    for row in rows:
        if row['corpus'] != 'urfall' or row['split'] != 'validation': continue
        path = root / row['feature_cache']
        with np.load(path, allow_pickle=False) as d:
            x = training.xy(d)
            attempts = d['attempts'].copy()
            reasons = [str(v) for v in d['attempt_reason']]
        probabilities = model.predict_proba(x)[:, 1]
        lookup = {v.tobytes(): float(p) for v, p in zip(x, probabilities)}
        def score(v, mask): return lookup[np.concatenate([v, mask.astype(float)]).tobytes()]
        times = frame_times(row['recording_id'], root=root)
        truth = fall_ground_truth(row['recording_id'], labels, times)
        for threshold in all_results:
            events = replay(path, detector=RuleDetector(
                down_hold=card.get('down_hold_s', 2.), fall_hold=card.get('fall_hold_s', 0.)),
                score=score, threshold=threshold)
            all_results[threshold].append(match_fall_events(events, truth, max(times.values()),
                unobserved_s=unobserved_seconds(attempts, reasons, max(times.values()))))
    return [dict(threshold=t, **aggregate(values)) for t, values in all_results.items()]


def retail_alert_burden(model, root, rows, threshold, card):
    """Ordinary validation clips only; no person-down truth labels, hence no FP claim."""
    from watchverify.evaluation import replay, FALL_CATEGORIES
    from watchverify.core import RuleDetector
    result = []
    for row in rows:
        if not (row['corpus'] == 'mnnit' and row['split'] == 'validation'
                and row['recording_id'].startswith('normal_')): continue
        path = root / row['feature_cache']
        with np.load(path, allow_pickle=False) as d:
            x = np.concatenate([d['x'], d['mask'].astype(float)], axis=1)
        scores = model.predict_proba(x)[:, 1]
        lookup = {v.tobytes(): float(p) for v, p in zip(x, scores)}
        def score(v, mask): return lookup[np.concatenate([v, mask.astype(float)]).tobytes()]
        events = replay(path, detector=RuleDetector(down_hold=card.get('down_hold_s', 2.),
            fall_hold=card.get('fall_hold_s', 0.)), score=score, threshold=threshold)
        result.append(dict(clip=row['recording_id'],
                           alerts=sum(e['category'] in FALL_CATEGORIES for e in events)))
    return dict(clips=len(result), clips_alerted=sum(r['alerts'] > 0 for r in result),
                per_clip=result, note='Alert burden only; no per-person posture truth labels.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root), str(root / 'scripts')]
    import train_models as training
    from watchverify.models import Models
    rows = development_rows(root)
    installed = Models()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(seed=42, sklearn=sklearn.__version__, python=sys.version,
        split_manifest_sha256=digest(root / 'data/manifests/splits.jsonl'),
        selection='5 folds x seeds 0,1,2; training only; content checksum groups',
        promotion_policy='No automatic promotion. Require CV improvement and a validation operating-point improvement without lower recall.',
        limits='Repeated development/validation comparisons; no independent accuracy claim. Existing dataset/label limitations persist.',
        test_feature_caches_opened=0, branches={})
    with threadpool_limits(limits=1):
        for branch in ('activity', 'fall'):
            tic = time.monotonic()
            blocks, opened = load_blocks(root, branch, rows, training)
            x, y, groups, clips = blocks['train']
            vx, vy, _, vclips = blocks['validation']
            current, card = installed.loaded[branch]
            proposals = candidates(branch, training)
            comparison = {}
            for name, proto in {'incumbent': clone(current), **proposals}.items():
                scores = training.grouped_cv_scores(proto, x, y, groups)
                comparison[name] = dict(training.summarise(scores), fold_scores=scores)
                print(json.dumps(dict(branch=branch, candidate=name,
                                     mean_cv_auc=comparison[name]['mean_roc_auc'])), flush=True)
            chosen = max(proposals, key=lambda n: comparison[n]['mean_roc_auc'])
            proto = proposals[chosen]
            model = clone(proto).fit(x, y)
            result = dict(challenger=chosen, comparison=comparison, opened_caches=opened,
                training_rows=len(y), training_groups=len(set(groups)),
                incumbent_sha256=card['sha256'], incumbent_run_id=card['run_id'])
            if branch == 'activity':
                threshold, _ = training.out_of_fold_threshold(proto, x, y, groups)
                baseline = training.sustained_clip_rule(current.predict_proba(vx)[:, 1], vy, vclips, card['threshold'])
                proposed = training.sustained_clip_rule(model.predict_proba(vx)[:, 1], vy, vclips, threshold)
                gain = (proposed['clip_recall'] >= baseline['clip_recall'] and
                        proposed['normal_clip_alert_rate'] <= baseline['normal_clip_alert_rate'] and
                        (proposed['clip_recall'] > baseline['clip_recall'] or
                         proposed['normal_clip_alert_rate'] < baseline['normal_clip_alert_rate']))
            else:
                baseline = fall_points(current, root, rows, training, [card['threshold']], card)[0]
                curve = fall_points(model, root, rows, training, [.3,.4,.5,.6,.7,.8,.9], card)
                proposed = max(curve, key=lambda p: (p['on_time_recall'], -p['false_alerts'], -p['median_latency_s']))
                threshold = proposed['threshold']
                result['validation_threshold_curve'] = curve
                gain = (proposed['on_time_recall'] >= baseline['on_time_recall'] and
                        proposed['false_alerts'] < baseline['false_alerts'])
                retail_baseline = retail_alert_burden(current, root, rows, card['threshold'], card)
                retail_proposed = retail_alert_burden(model, root, rows, threshold, card)
                result['retail_validation_incumbent'] = retail_baseline
                result['retail_validation_challenger'] = retail_proposed
                gain = gain and retail_proposed['clips_alerted'] <= retail_baseline['clips_alerted']
            cv_gain = comparison[chosen]['mean_roc_auc'] - comparison['incumbent']['mean_roc_auc']
            result.update(validation_incumbent=baseline, validation_challenger=proposed,
                threshold=float(threshold), cv_auc_delta=cv_gain,
                clears_development_gate=bool(gain and cv_gain > 0),
                elapsed_s=round(time.monotonic()-tic,2))
            artifact = args.output / f'{branch}_challenger.joblib'
            joblib.dump(model, artifact)
            # Verify the exact fitted artifact, rather than assuming serialization works.
            np.testing.assert_allclose(joblib.load(artifact).predict_proba(vx[:8]), model.predict_proba(vx[:8]))
            result['artifact_sha256'] = digest(artifact)
            report['branches'][branch] = result
            (args.output / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False))
            print(json.dumps(dict(branch=branch, challenger=chosen, cv_auc_delta=cv_gain,
                 validation_incumbent=baseline, validation_challenger=proposed,
                 clears_development_gate=result['clears_development_gate'])), flush=True)
    for branch, (_, card) in installed.loaded.items():
        assert digest(root / 'models' / card['artifact']) == card['sha256'], 'Installed model changed'


if __name__ == '__main__':
    main()
