"""Turn Learning-queue exports into a 'reviewed' corpus the labelling/training tools already know.

An admin exports ready candidates from the Learning queue UI (watchverify/ui/queue.py),
which writes clips plus a manifest to outputs/_exports/<id>/ (watchverify/review.py
export_candidates). Nothing reads that bundle back on its own — this script is that
reader. It does the same mechanical preparation scripts/capture_camera.py does for a
live recording (decode the clip once, record each frame's real timestamp so
extract_features.py can sample it later) but never invents a label: exact fall/lie
timing is always left for a human to fill in, the same discipline
scripts/prepare_owncam.py already enforces for hand-recorded clips.

Scope: fall/posture branch only. The labelling this script prepares clips for
(scripts/prepare_owncam.py) is specifically "was this person lying down", so only a
candidate linked to a fall-branch event (possible_fall/person_down — see
watchverify.alerts.ALERT_BRANCH) is imported. A candidate from an activity/security alert,
or a "missed interval" candidate with no linked event, has no reliable branch and is
skipped rather than guessed onto the wrong corpus.

Usage:
    python scripts/import_reviewed_exports.py

Then, same as any owncam clip:
    1. Open data/raw/reviewed/labels.csv and replace every "REVIEW NEEDED" row's
       fall_start_s/fall_end_s/down_end_s with the exact times you observe in the clip
       (leave all three blank only for a genuine "adl-*" negative — that is already
       filled in correctly for negatives and will be rejected if you touch a positive
       "fall-*" row without giving times).
    2. python scripts/prepare_owncam.py --corpus reviewed
    3. python scripts/extract_features.py --source reviewed
    4. python scripts/train_models.py --fall-sources urfall,reviewed   (or your own mix)

Re-running this script only adds clips it has not imported before; it never rewrites a
row you have already corrected (INSERT-OR-IGNORE, the same idempotency rule
watchverify/review.py already uses for review actions).
"""
from pathlib import Path
import csv
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watchverify import jobs
from watchverify.alerts import ALERT_BRANCH
from watchverify.perception import frames, sha256, video_info
from watchverify.review import PROPOSED_LABELS

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / 'outputs' / '_exports'
PREPARED = ROOT / 'data/processed/reviewed'
RAW = ROOT / 'data/raw/reviewed'
LABELS = RAW / 'labels.csv'
COLUMNS = ('clip', 'fall_start_s', 'fall_end_s', 'down_end_s', 'notes')
PREPARER_VERSION = 1
LABEL_PREFIX = {'positive': 'fall', 'negative': 'adl'}


def _existing_clips():
    if not LABELS.exists():
        return set()
    with LABELS.open(newline='') as handle:
        return {row['clip'] for row in csv.DictReader(handle) if row.get('clip')}


def _prepare_clip(name, clip_path):
    """Decode once to record real per-frame timestamps; copy the file as-is (already a
    single continuous encode, so no frame remapping is needed — source_frame ==
    decoded_index, exactly like a fresh scripts/capture_camera.py recording)."""
    mapping = [{'source_frame': index, 'decoded_index': index, 'output_t': t}
              for index, t, _ in frames(clip_path)]
    if len(mapping) < 2:
        raise ValueError('clip decodes to fewer than 2 frames')
    PREPARED.mkdir(parents=True, exist_ok=True)
    destination = PREPARED / f'{name}.mp4'
    shutil.copy2(clip_path, destination)
    card = {'source_id': name, 'preparer_version': PREPARER_VERSION,
            'video_sha256': sha256(destination), 'media': video_info(destination),
            'frame_count': len(mapping), 'frames': mapping,
            'timestamp_source': 'decoded from the exported clip, which is already a single '
                                'continuous encode of the source interval',
            'license': 'Admin-curated from reviewed application footage; see SOURCE_LICENSE '
                       "['reviewed'] in scripts/train_models.py",
            'research_only': True,
            'note': 'Imported by scripts/import_reviewed_exports.py from a Learning-queue export.'}
    (PREPARED / f'{name}.json').write_text(json.dumps(card, indent=2))


def _fall_branch_event_ids(run_id, events_by_run):
    if run_id not in events_by_run:
        try:
            events = jobs.load_events(run_id)
        except (OSError, ValueError):
            events = []
        events_by_run[run_id] = {event['event_id'] for event in events
                                 if ALERT_BRANCH.get(event.get('category')) == 'fall'}
    return events_by_run[run_id]


def import_exports():
    manifests = sorted(EXPORTS.glob('*/manifest.json'))
    seen = _existing_clips()
    imported, skipped = [], []
    new_rows = []
    events_by_run = {}
    for manifest_path in manifests:
        try:
            candidates = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as error:
            skipped.append({'manifest': str(manifest_path), 'reason': str(error)})
            continue
        for candidate in candidates:
            candidate_id = candidate.get('candidate_id', '')
            event_id = candidate.get('event_id')
            if not event_id or event_id not in _fall_branch_event_ids(candidate.get('run_id', ''), events_by_run):
                skipped.append({'candidate_id': candidate_id,
                                'reason': 'not linked to a fall-branch event (possible_fall/person_down); '
                                          'this script only prepares fall/posture data'})
                continue
            proposed = candidate.get('proposed_label')
            prefix = LABEL_PREFIX.get(proposed)
            if prefix is None:
                skipped.append({'candidate_id': candidate_id, 'reason': f'label {proposed!r} not usable'})
                continue
            name = f'{prefix}-reviewed-{candidate_id}'
            if name in seen:
                continue
            clip_file = candidate.get('clip_file')
            clip_path = (manifest_path.parent / clip_file) if clip_file else None
            if not clip_path or not clip_path.exists() or not clip_path.stat().st_size:
                skipped.append({'candidate_id': candidate_id, 'reason': 'clip file missing or empty'})
                continue
            try:
                _prepare_clip(name, clip_path)
            except ValueError as error:
                skipped.append({'candidate_id': candidate_id, 'reason': str(error)})
                continue
            note = ('REVIEW NEEDED before training — proposed: ' + proposed
                    + (f' · {candidate["visible_action_label"]}' if candidate.get('visible_action_label') else '')
                    + (f' · {candidate["note"]}' if candidate.get('note') else '')
                    + f' (candidate {candidate_id}, run {candidate.get("run_id", "?")})')
            new_rows.append({'clip': name, 'fall_start_s': '', 'fall_end_s': '', 'down_end_s': '',
                             'notes': note})
            seen.add(name)
            imported.append(name)
    if new_rows:
        RAW.mkdir(parents=True, exist_ok=True)
        is_new = not LABELS.exists()
        with LABELS.open('a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            if is_new:
                writer.writeheader()
            writer.writerows(new_rows)
    return {'imported': imported, 'skipped': skipped, 'labels_csv': str(LABELS.relative_to(ROOT))}


def main():
    assert set(LABEL_PREFIX) <= PROPOSED_LABELS
    report = import_exports()
    print(json.dumps(report, indent=2))
    if report['imported']:
        print(f"\n{len(report['imported'])} new clip(s) imported into {report['labels_csv']}.")
        print('Every new row needs fall_start_s/fall_end_s/down_end_s filled in by hand for '
             '"fall-reviewed-*" clips before scripts/prepare_owncam.py --corpus reviewed will '
             'accept it (an "adl-reviewed-*" negative is already usable as-is).')
    else:
        print('\nNo new clips to import — export more ready candidates from the Learning queue first.')


if __name__ == '__main__':
    main()
