"""Expand hand-written own-camera clip labels into per-frame posture codes.

You write one row per clip in `data/raw/owncam/labels.csv`:

    clip,fall_start_s,fall_end_s,down_end_s,notes
    fall-own-01,3.4,4.1,9.0,forward fall onto mattress
    adl-own-01,,,,sitting down and standing up
    adl-own-07,5.0,6.2,11.5,lying down on purpose then getting up

This writes `data/raw/owncam/owncam-posture.csv` in the same three-column shape the UR
publisher uses, so `watchverify/evaluation.py` reads both corpora through one code path.

Codes follow the publisher's convention: -1 not lying, 0 falling/transition, 1 lying.
Frames between `fall_start_s` and `fall_end_s` are the transition; frames from `fall_end_s`
to `down_end_s` are lying; everything else is not lying.

Naming matters and is enforced. `fall_ground_truth` treats a `fall*` clip as containing a
fall event and any other clip as a fall negative, so a deliberate lie-down must be named
`adl-*`: it carries lying frames the posture model should learn from, while remaining a
negative for the event logic. Mislabel that and the fall/lie distinction becomes untestable.

Times that are missing, out of order or past the end of the clip are rejected. Nothing is
guessed or clipped into range — a wrong label is worse than a refused one (D025).
"""
from pathlib import Path
import argparse
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watchverify.evaluation import LYING, NOT_LYING, TRANSITION

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ('clip', 'fall_start_s', 'fall_end_s', 'down_end_s', 'notes')
FALL_PREFIX = 'fall'
ORDINARY_PREFIX = 'adl'


def paths(corpus):
    """(prepared, raw, labels, posture) directories/files for one named corpus.

    'owncam' is the original hand-recorded corpus; any other name (e.g. 'reviewed', built
    by scripts/import_reviewed_exports.py) reuses this same labelling machinery on its own
    data/raw/<corpus> and data/processed/<corpus> directories.
    """
    prepared = ROOT / 'data/processed' / corpus
    raw = ROOT / 'data/raw' / corpus
    return prepared, raw, raw / 'labels.csv', raw / f'{corpus}-posture.csv'


def _seconds(row, column, clip):
    raw = (row.get(column) or '').strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f'{clip}: {column}={raw!r} is not a number')
    if value < 0:
        raise ValueError(f'{clip}: {column} must not be negative')
    return value


def clip_frames(clip, prepared):
    """[(source_frame, output_t)] from the preparation sidecar."""
    sidecar = prepared / f'{clip}.json'
    if not sidecar.exists():
        raise ValueError(f'{clip}: no sidecar at {sidecar.relative_to(ROOT)}; prepare it first '
                         '(capture_camera.py for a live recording, import_reviewed_exports.py '
                         'for a reviewed clip)')
    card = json.loads(sidecar.read_text())
    rows = card.get('frames') or []
    if not rows:
        raise ValueError(f'{clip}: sidecar records no frames')
    return [(int(r['source_frame']), float(r['output_t'])) for r in rows]


def posture_codes(clip, fall_start, fall_end, down_end, frames):
    """Per-frame codes for one clip, after checking the times are usable."""
    duration = frames[-1][1]
    if not clip.startswith((FALL_PREFIX, ORDINARY_PREFIX)):
        raise ValueError(f'{clip}: name must start with {FALL_PREFIX!r} (contains a fall) '
                         f'or {ORDINARY_PREFIX!r} (ordinary activity, a fall negative)')
    if fall_start is None and fall_end is None and down_end is None:
        if clip.startswith(FALL_PREFIX):
            raise ValueError(f'{clip}: a fall clip needs fall_start_s and fall_end_s')
        return {frame: NOT_LYING for frame, _ in frames}
    if fall_start is None or fall_end is None:
        raise ValueError(f'{clip}: give both fall_start_s and fall_end_s, or leave every time blank')
    if fall_end <= fall_start:
        raise ValueError(f'{clip}: fall_end_s ({fall_end}) must be after fall_start_s ({fall_start})')
    # A person still on the ground when the recording stops is an unfinished observation,
    # not a recovery. Defaulting to the clip end records exactly what was seen.
    if down_end is None:
        down_end = duration
    if down_end < fall_end:
        raise ValueError(f'{clip}: down_end_s ({down_end}) must not be before fall_end_s ({fall_end})')
    for name, value in (('fall_start_s', fall_start), ('fall_end_s', fall_end), ('down_end_s', down_end)):
        if value > duration + 1e-6:
            raise ValueError(f'{clip}: {name}={value} is past the end of the clip ({duration:.3f}s)')
    codes = {}
    for frame, t in frames:
        if fall_start <= t < fall_end:
            codes[frame] = TRANSITION
        elif fall_end <= t <= down_end:
            codes[frame] = LYING
        else:
            codes[frame] = NOT_LYING
    return codes


def build(corpus='owncam'):
    prepared_dir, raw, labels, posture = paths(corpus)
    if not labels.exists():
        raise SystemExit(f'Write your clip labels to {labels.relative_to(ROOT)} first. Columns: {",".join(COLUMNS)}')
    with labels.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f'{labels.relative_to(ROOT)} has no rows')
    missing = [column for column in COLUMNS[:4] if column not in (rows[0].keys() or {})]
    if missing:
        raise SystemExit(f'{labels.relative_to(ROOT)} is missing columns: {missing}')
    seen = set()
    prepared, failed = [], []
    table = []
    for row in rows:
        clip = (row.get('clip') or '').strip()
        if not clip:
            continue
        try:
            if clip in seen:
                raise ValueError(f'{clip}: listed twice')
            seen.add(clip)
            frames = clip_frames(clip, prepared_dir)
            codes = posture_codes(clip, _seconds(row, 'fall_start_s', clip),
                                  _seconds(row, 'fall_end_s', clip),
                                  _seconds(row, 'down_end_s', clip), frames)
            for frame, _ in frames:
                table.append((clip, frame, codes[frame]))
            prepared.append({'clip': clip, 'status': 'labelled', 'frames': len(frames),
                             'lying_frames': sum(1 for v in codes.values() if v == LYING),
                             'transition_frames': sum(1 for v in codes.values() if v == TRANSITION),
                             'kind': 'fall' if clip.startswith(FALL_PREFIX) else 'ordinary',
                             'duration_s': round(frames[-1][1], 3)})
        except ValueError as error:
            failed.append({'clip': clip, 'status': 'rejected', 'error': str(error)})
    unlabelled = sorted(path.stem for path in prepared_dir.glob('*.mp4') if path.stem not in seen)
    if failed:
        for failure in failed:
            print(json.dumps(failure), flush=True)
        raise SystemExit(f'{len(failed)} clip(s) rejected; nothing was written. Fix the rows above and re-run.')
    raw.mkdir(parents=True, exist_ok=True)
    with posture.open('w', newline='') as handle:
        csv.writer(handle).writerows(table)
    report = {'clips': len(prepared), 'rows': len(table), 'prepared': prepared,
              'recorded_but_unlabelled': unlabelled,
              'posture_table': str(posture.relative_to(ROOT)),
              'note': 'Codes follow the UR publisher convention: -1 not lying, 0 transition, 1 lying.'}
    (raw / '_label_report.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='owncam',
                        help="Which data/raw/<corpus> + data/processed/<corpus> pair to label. "
                             "'owncam' (default) is the hand-recorded corpus; "
                             "'reviewed' is admin-curated clips from import_reviewed_exports.py.")
    corpus = parser.parse_args().corpus
    report = build(corpus)
    for entry in report['prepared']:
        print(json.dumps(entry), flush=True)
    print(json.dumps({k: report[k] for k in ('clips', 'rows', 'recorded_but_unlabelled', 'posture_table')}, indent=2))
    if report['recorded_but_unlabelled']:
        print('\nThese clips were recorded but have no label row, so they will not be used:', flush=True)
        print('  ' + ', '.join(report['recorded_but_unlabelled']), flush=True)


if __name__ == '__main__':
    main()
