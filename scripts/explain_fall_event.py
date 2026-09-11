"""Explain why one cached sequence did or did not produce a fall alert.

A rate tells you how often the detector was wrong; it never tells you why. This replays
a single sequence and prints the three things that decide a `possible_fall`: the model's
per-frame posture score against its threshold, the tracked identity, and the rule state
that the score feeds. When no alert is emitted it names the precondition that failed,
so a miss can be attributed to the model, the operating point, or the tracker rather
than guessed at.

Read-only: it loads an existing feature cache and model artifact and writes nothing.

    .venv/bin/python scripts/explain_fall_event.py --sequence fall-01
    .venv/bin/python scripts/explain_fall_event.py --sequence fall-01 \
        --model runs/<experiment>/fall_challenger.joblib --threshold 0.8
"""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from watchverify.core import RuleDetector
from watchverify.evaluation import (FALL_CATEGORIES, fall_ground_truth, frame_times,
                                    load_posture_labels, replay)


def blank(value):
    return '-' if value is None else '%.3f' % value


def frames(cache):
    """Per-frame rows plus the attempt log, which is the only record of blind frames."""
    with np.load(cache, allow_pickle=False) as d:
        return (dict(t=d['t'].copy(), track=d['track'].copy(), down=d['down'].copy(),
                     x=np.concatenate([d['x'], d['mask'].astype(float)], axis=1)),
                d['attempts'].copy(), [str(v) for v in d['attempt_reason']])


def observation_gaps(attempts, reasons, max_gap):
    """Blind spans long enough for the rule layer to discard a track's history.

    `max_gap` here is the detector's, not the tracker's: the rule layer forgets on its own
    once the interval between two observed frames exceeds it, whichever identity survives.
    """
    observed = [float(t) for (t, *_), reason in zip(attempts, reasons)
                if reason not in ('no_pose', 'pose_geometry_rejected')]
    return [(a, b) for a, b in zip(observed, observed[1:]) if b - a > max_gap]


def identity_changes(rows):
    return [(float(rows['t'][i]), int(rows['track'][i - 1]), int(rows['track'][i]))
            for i in range(1, len(rows['t'])) if rows['track'][i] != rows['track'][i - 1]]


def trace(cache, model, threshold, card):
    """Replay once, recording the rule state at every observed frame."""
    rows, attempts, reasons = frames(cache)
    probabilities = model.predict_proba(rows['x'])[:, 1]
    lookup = {v.tobytes(): float(p) for v, p in zip(rows['x'], probabilities)}
    def score(v, mask): return lookup[np.concatenate([v, mask.astype(float)]).tobytes()]

    detector = RuleDetector(down_hold=card.get('down_hold_s', 2.),
                            fall_hold=card.get('fall_hold_s', 0.))
    steps = []
    update = detector.update
    def record(track_id, features, t):
        result = update(track_id, features, t)
        if features is not None:
            state = detector._states[track_id]
            steps.append(dict(t=float(t), track=int(track_id), down=bool(features['down']),
                              posture_score=features.get('posture_score'),
                              transition=state['transition'], low=state['low'],
                              emitted=[e['category'] for e in result]))
        return result
    detector.update = record
    events = replay(cache, detector=detector, score=score, threshold=threshold)
    return rows, probabilities, steps, events, attempts, reasons, detector


def diagnose(steps, events, detector, duration_s):
    """Name the precondition that blocked a fall alert, or the evidence that carried one."""
    alerted = [e for e in events if e['category'] in FALL_CATEGORIES]
    if alerted:
        first = alerted[0]
        return (f"Fall alert emitted: {first['category']} at {first['t']:.3f}s. The rule "
                f"layer held both the rapid-posture-change marker and a down frame on one "
                f"identity at the same moment.")
    down_steps = [s for s in steps if s['down']]
    if not down_steps:
        return ("No fall alert: the model never called a frame down at this threshold, so "
                "neither the possible_fall nor the person_down path could start.")
    first_down = down_steps[0]
    if first_down['transition'] is None:
        # The marker is set only on a frame that is *not* down. Once a person is already
        # on the ground it can never be rebuilt, so this miss is permanent for the clip.
        rebuildable = any(not s['down'] for s in steps if s['t'] > first_down['t'])
        held = first_down['t'] - (down_steps[0]['low'] or first_down['t'])
        return (f"No fall alert: the first down frame is {first_down['t']:.3f}s on track "
                f"{first_down['track']}, and that track carries no rapid-posture-change "
                f"marker, so possible_fall cannot fire. "
                + ("The marker can still be rebuilt later in this clip."
                   if rebuildable else
                   "The person never returns to a non-down posture afterwards, so the "
                   "marker can never be rebuilt and the possible_fall path stays closed "
                   "for the rest of the clip.")
                + f" The person_down fallback needs {detector.down_hold:.1f}s of continuous "
                f"down posture from {first_down['low']:.3f}s, which the sequence "
                f"({duration_s:.3f}s) ends before reaching."
                if duration_s - (first_down['low'] or 0) < detector.down_hold else "")
    return ("No fall alert: a rapid-posture-change marker was present but expired before a "
            "down frame arrived, or the down posture was not held long enough.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sequence', default='fall-01')
    parser.add_argument('--source', default='urfall')
    parser.add_argument('--model', default='models/fall.joblib')
    parser.add_argument('--card', default='models/fall.json')
    parser.add_argument('--threshold', type=float,
                        help='Override the card threshold, e.g. to score a challenger.')
    parser.add_argument('--json', help='Also write the trace to this path.')
    args = parser.parse_args()

    cache = ROOT / f'data/processed/features/{args.source}/{args.sequence}.npz'
    if not cache.exists():
        raise SystemExit(f'{cache} is missing; run scripts/extract_features.py first')
    card = json.loads((ROOT / args.card).read_text())
    model = joblib.load(ROOT / args.model)
    threshold = args.threshold if args.threshold is not None else card['threshold']

    rows, probabilities, steps, events, attempts, reasons, detector = trace(
        cache, model, threshold, card)
    times = frame_times(args.sequence, root=ROOT, source=args.source)
    duration_s = max(times.values())
    truth = fall_ground_truth(args.sequence, load_posture_labels(root=ROOT, source=args.source),
                              times)

    print(f'sequence      {args.sequence} ({args.source}), {duration_s:.3f}s, '
          f'{len(steps)} observed frames')
    print(f'model         {args.model} at threshold {threshold} '
          f'(card threshold {card["threshold"]})')
    print(f'rule timing   fall_hold {detector.fall_hold}s, down_hold {detector.down_hold}s, '
          f'max_gap {detector.max_gap}s')
    labelled = ('none labelled' if not truth else
                'fall %.3fs to %.3fs' % (truth[0]['onset_s'], truth[0]['end_s']))
    print('ground truth  ' + labelled)

    blind = observation_gaps(attempts, reasons, detector.max_gap)
    print('blind spans   ' + str([(round(a, 3), round(b, 3)) for a, b in blind]
                                 or 'none longer than max_gap'))
    print('identity      ' + str(identity_changes(rows) or 'one track throughout'))
    print()
    print('%7s %5s %6s %5s %7s %6s  emitted'
          % ('t', 'track', 'score', 'down', 'marker', 'low'))
    for step in steps:
        inside = bool(truth) and truth[0]['onset_s'] <= step['t'] <= truth[0]['end_s']
        print('%7.3f %5d %6s %5s %7s %6s  %s%s'
              % (step['t'], step['track'], blank(step['posture_score']), step['down'],
                 blank(step['transition']), blank(step['low']),
                 ','.join(step['emitted']), '  <fall' if inside else ''))
    print()
    print('events        ' + (str([(e['category'], round(e['t'], 3)) for e in events])
                              or 'none'))
    print('diagnosis     ' + diagnose(steps, events, detector, duration_s))

    if args.json:
        Path(args.json).write_text(json.dumps(dict(
            sequence=args.sequence, source=args.source, model=args.model,
            threshold=threshold, duration_s=duration_s, ground_truth=truth,
            blind_spans=blind, identity_changes=identity_changes(rows),
            steps=steps, events=[dict(e, t=round(e['t'], 3)) for e in events],
            diagnosis=diagnose(steps, events, detector, duration_s)), indent=2, default=str))


if __name__ == '__main__':
    main()
