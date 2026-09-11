"""Freeze the corpus split to a file, so it can be read instead of recomputed.

The splits themselves are already deterministic — `train_models.py` derives them from
hashed recording identifiers every time it runs. That is enough for reproducibility and
not enough for review: nobody can inspect a function to see which recording carries which
licence, which label unit, or which group a piece of footage was kept with. This writes
one row per source recording with exactly that, and fails loudly if the frozen file on
disk disagrees with what the code would assign today.

Writes nothing else. No extraction, no training, no downloads.

    python scripts/write_split_manifest.py            # check the frozen file
    python scripts/write_split_manifest.py --write    # rewrite it
"""
from pathlib import Path
import argparse, json, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
MANIFEST = ROOT / 'data/manifests/splits.jsonl'
# What one label actually describes. A scene label is not a person label and not a frame
# label; recording it here stops a downstream reader from treating them as interchangeable.
FRAME_POSTURE = 'frame_posture_code'
SCENE_ONLY = 'scene_level_only'
# `release_cleared` stays None for this corpus on purpose: attribution terms are known,
# clearance for releasing a model derived from it has not been verified by anyone here,
# and False would assert a refusal that nobody issued.
MNNIT_NOTE = ('CC BY 4.0, attribution required — cite DOI 10.17632/r3yjf35hzr.1. '
              'Release clearance for derived models is unverified.')


def mnnit_licence():
    """Licence text as recorded at extraction time, not restated from memory."""
    path = ROOT / 'data/manifests/mnnit.jsonl'
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get('license'):
            return row['license']
    return None


def rows():
    from train_models import SOURCE_LICENSE, fall_sequences, mnnit_eligible

    result = []
    for source in ('urfall', 'owncam'):
        if not (ROOT / 'data/processed/features' / source).exists():
            continue
        for record in fall_sequences((source,)):
            with np.load(record['path'], allow_pickle=False) as d:
                checksum = str(d['source_sha256'])
                attempts = len(d['attempt_reason'])
                usable = sum(1 for v in d['attempt_reason'] if str(v) == 'ok')
            licence = SOURCE_LICENSE[source]
            result.append(dict(
                corpus=source, branch='fall', recording_id=record['sequence_id'],
                session_id=record['sequence_id'], group_key=checksum,
                source_sha256=checksum, feature_cache=str(record['path'].relative_to(ROOT)),
                licence=licence['license'], licence_note=licence['note'],
                release_cleared=licence['release_cleared'],
                label_unit=FRAME_POSTURE, split=record['split'], eligible=True,
                excluded_reason=None,
                sampled_frames=attempts, usable_frames=usable))

    if (ROOT / 'data/processed/features/mnnit').exists():
        licence = mnnit_licence()
        eligible, excluded, _summary, _assignment = mnnit_eligible()
        for record in eligible:
            result.append(dict(
                corpus='mnnit', branch='activity', recording_id=record['source'],
                session_id=record['source'], group_key=record['sha256'],
                source_sha256=record['sha256'],
                feature_cache=f"data/processed/features/mnnit/{record['source']}.npz",
                licence=licence, licence_note=MNNIT_NOTE,
                release_cleared=None, label_unit=SCENE_ONLY, split=record['split'],
                eligible=True, excluded_reason=None,
                sampled_frames=record['rows'], usable_frames=record['rows']))
        for record in excluded:
            # Excluded clips stay in the manifest. A reader who only sees what survived
            # cannot tell whether a filter removed one clip or a third of a class.
            result.append(dict(
                corpus='mnnit', branch='activity', recording_id=record['source'],
                session_id=record['source'], group_key=record['sha256'],
                source_sha256=record['sha256'],
                feature_cache=f"data/processed/features/mnnit/{record['source']}.npz",
                licence=licence, licence_note=MNNIT_NOTE,
                release_cleared=None, label_unit=SCENE_ONLY, split=None,
                eligible=False, excluded_reason=record['reason'],
                sampled_frames=record['rows'], usable_frames=None))
    if not result:
        raise SystemExit('No feature caches found; nothing to freeze.')
    result.sort(key=lambda row: (row['corpus'], row['recording_id']))
    return result


def check(current):
    """Compare the frozen file against what the code assigns now."""
    if not MANIFEST.exists():
        return [f'{MANIFEST.relative_to(ROOT)} does not exist yet']
    frozen = {(r['corpus'], r['recording_id']): r
              for r in (json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip())}
    live = {(r['corpus'], r['recording_id']): r for r in current}
    problems = [f'no longer present: {key}' for key in sorted(frozen.keys() - live.keys())]
    problems += [f'not in the frozen manifest: {key}' for key in sorted(live.keys() - frozen.keys())]
    for key in sorted(frozen.keys() & live.keys()):
        for field in ('split', 'group_key', 'label_unit', 'licence', 'eligible'):
            if frozen[key].get(field) != live[key].get(field):
                problems.append(f'{key} {field}: frozen {frozen[key].get(field)!r} vs current {live[key].get(field)!r}')
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true', help='rewrite the frozen manifest')
    args = parser.parse_args()
    current = rows()
    problems = check(current)
    if args.write:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(''.join(json.dumps(row) + '\n' for row in current))
        counts = {}
        for row in current:
            counts[f"{row['corpus']}/{row['split']}"] = counts.get(f"{row['corpus']}/{row['split']}", 0) + 1
        print(f'wrote {len(current)} rows to {MANIFEST.relative_to(ROOT)}')
        print(json.dumps(dict(sorted(counts.items())), indent=2))
        if problems:
            print('\nthis rewrite changed the frozen split:')
            print('\n'.join(f'  {p}' for p in problems))
        return 0
    if problems:
        print('frozen split manifest does not match the current assignment:')
        print('\n'.join(f'  {p}' for p in problems))
        return 1
    print(f'{MANIFEST.relative_to(ROOT)} matches the current assignment ({len(current)} rows)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
