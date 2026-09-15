"""Build activity-branch training descriptors from RetailS's supplied COCO17 poses.

RetailS ships pose only (YOLOv8 + ByteTrack + HRNet, upstream, not MediaPipe) with genuine
per-person, per-frame binary ground truth on its two test splits (`staged`, `realworld`).
That is the fix for the activity branch's MNNIT label ceiling (one label per whole clip);
see docs/DATASETS.md's "RetailS archive inspection" and D050.

Two things this script cannot avoid being explicit about:

1. **No raw video is supplied**, so `scripts/extract_features.py`'s MediaPipe pipeline does
   not apply. Poses are mapped onto this project's 12-joint canonical layout by semantic
   name (docs/MODELS.md's documented COCO17 adapter: MediaPipe 11,12,13,14,15,16,23,24,25,
   26,27,28 <-> COCO17 5,6,7,8,9,10,11,12,13,14,15,16) and fed straight into the existing
   causal `FeatureBuffer`, using RetailS's own ByteTrack identities rather than
   re-associating with this project's `Tracker` — the source tracker is a dedicated,
   purpose-built one; re-deriving identity from re-projected keypoints would only add
   error. MediaPipe and this upstream extractor are not calibrated identically (confidence
   scales, joint precision); this is a cross-extractor training source, not a same-pipeline
   one, and the model card must say so.
2. **Frame rate is not stated in the dataset's own README.** The paper (arXiv:2603.04723,
   "camera specifications") states six indoor cameras at 1080x720, 15 FPS; used here for
   frame_id -> seconds. If that figure is wrong, every velocity-derived feature (hip speed,
   joint motion, angular speed) for this source is scaled off by the same factor. Recorded
   here, not silently assumed.

RetailS's clips are short — median tracked-person duration is well under this project's
5-second activity window (WINDOW_S in watchverify/features.py), which assumes the video
sampling rate seen everywhere else in this project (10 fps offline, matching the app's
`extract_features.py` default). Requiring a full 5s span here would discard most of the
already-small labelled corpus (53 anomalous frames total in `realworld`). Instead this
script builds ONE whole-track descriptor per (clip, track) from whatever span is actually
available (>=3 usable feature rows), via the same `clip_descriptor()` the runtime window
builder uses — same fixed-length output regardless of input duration — labelled by the
majority frame-level ground truth over the frames that track was actually visible in. This
is a documented deviation from the sliding 5s window used elsewhere, not a runtime change:
it only supplies extra *training* examples with the same feature shape (`CLIP_COLUMNS`).

Frame-level labels are scene-level across everyone visible in that frame, not per-person:
a bystander standing next to someone concealing an item inherits that frame's positive
label under this scheme. This is the same class of weak-label assumption already recorded
for MNNIT (docs/DATASETS.md), one level more granular (temporal, not per-clip), not
resolved outright.
"""
from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from watchverify.core import FeatureBuffer, FEATURE_NAMES
from watchverify.features import clip_descriptor, CLIP_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT / 'data/raw/retails/extracted/RetailS'
OUT = ROOT / 'data/processed/retails_activity.npz'
REPORT = ROOT / 'data/processed/retails_activity_report.json'
FPS = 15.0  # arXiv:2603.04723 camera specifications; not stated in the dataset README itself.
LICENSE_NOTE = ('No LICENSE file in the repository or archive as of the 2026-09-11 '
                 'inspection; the README carries only a citation request. The project '
                 'owner (Steven) reported on 2026-09-16 having contacted the paper\'s '
                 'authors and received clearance to use the dataset for training; this '
                 'script and the resulting artifact take that report at face value — no '
                 'written license text or correspondence has been independently reviewed '
                 'by this codebase, so it should not be treated as a citable license grant. '
                 'Confirm the actual correspondence exists before any public distribution '
                 'of a trained artifact (see docs/DECISIONS.md D050).')

MP_JOINTS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
COCO_JOINTS = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
SPLITS = {
    'staged': ARCHIVE_ROOT / 'RetailS_test_staged',
    'realworld': ARCHIVE_ROOT / 'RetailS_test_realworld',
}


def pose33_from_coco17(keypoints):
    """17*[x,y,conf] COCO list -> this project's (33,4) canonical pose array.

    Only the 12 joints this project actually uses (JOINTS in watchverify/core.py) are
    filled; everything else is marked invalid (NaN position, zero confidence) exactly as
    an occluded MediaPipe joint would be, so the existing validity-threshold logic in
    `FeatureBuffer` needs no special-casing for this source.
    """
    kp = np.asarray(keypoints, dtype=float).reshape(17, 3)
    p = np.full((33, 4), np.nan)
    p[:, 3] = 0.0
    for mp_index, coco_index in zip(MP_JOINTS, COCO_JOINTS):
        x, y, c = kp[coco_index]
        p[mp_index] = (x, y, 0.0, c)
    return p


def track_rows(track_frames, gt):
    """One track's frame dict -> per-frame feature rows via the causal FeatureBuffer.

    A fresh buffer per track: RetailS track ids repeat across clips and are not globally
    unique, and different tracks within one clip must not share temporal history.
    """
    buffer = FeatureBuffer()
    rows = []
    for frame_id in sorted(int(f) for f in track_frames):
        entry = track_frames[str(frame_id)]
        pose = pose33_from_coco17(entry['keypoints'])
        t = frame_id / FPS
        feature = buffer.update(0, pose, t)
        if feature is None:
            continue
        label = int(gt[frame_id]) if 0 <= frame_id < len(gt) else None
        rows.append((t, frame_id, feature['vector'], feature['feature_mask'],
                     feature['down'], feature['upright'], feature['hip_speed'],
                     feature['angular_speed'], feature['quality'], label))
    return rows


def descriptor_for_track(rows):
    x = np.array([np.concatenate([r[2], r[3].astype(float)]) for r in rows])
    hip_speed = np.array([r[6] for r in rows])
    angular_speed = np.array([r[7] for r in rows])
    down = np.array([r[4] for r in rows])
    upright = np.array([r[5] for r in rows])
    quality = np.array([r[8] for r in rows])
    return clip_descriptor(x, hip_speed, angular_speed, down, upright, quality)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--min-rows', type=int, default=3,
                   help='Minimum usable feature rows for a track to yield one descriptor '
                        '(same floor as WindowAggregator.min_samples default).')
    a = p.parse_args()

    descriptors, labels, groups, split_names, meta = [], [], [], [], []
    skipped = {'no_gt_file': 0, 'frame_id_out_of_range': 0, 'too_few_rows': 0, 'empty_track': 0}
    tracks_seen = 0

    for split_name, split_root in SPLITS.items():
        pose_dir = split_root / 'pose/test'
        gt_dir = split_root / 'gt/test_frame_mask'
        pose_files = sorted(pose_dir.glob('*.json'))
        if not pose_files:
            raise SystemExit(f'No pose files found under {pose_dir}; check the extraction.')
        for pose_path in pose_files:
            clip = pose_path.stem
            gt_path = gt_dir / f'{clip}.npy'
            if not gt_path.exists():
                skipped['no_gt_file'] += 1
                continue
            gt = np.load(gt_path)
            tracks = json.loads(pose_path.read_text())
            for track_id, frames in tracks.items():
                tracks_seen += 1
                if not frames:
                    skipped['empty_track'] += 1
                    continue
                rows = track_rows(frames, gt)
                rows = [r for r in rows if r[9] is not None]
                if any(int(f) >= len(gt) for f in frames):
                    skipped['frame_id_out_of_range'] += 1
                if len(rows) < a.min_rows:
                    skipped['too_few_rows'] += 1
                    continue
                descriptor = descriptor_for_track(rows)
                if not np.isfinite(descriptor).all():
                    skipped['too_few_rows'] += 1
                    continue
                frame_labels = [r[9] for r in rows]
                label = int(round(float(np.mean(frame_labels))))
                descriptors.append(descriptor)
                labels.append(label)
                groups.append(f'retails_{split_name}_{clip}_{track_id}')
                split_names.append(split_name)
                meta.append({'split': split_name, 'clip': clip, 'track': track_id,
                             'rows': len(rows), 'span_s': float(rows[-1][0] - rows[0][0]),
                             'label': label, 'positive_frame_fraction': float(np.mean(frame_labels))})

    if not descriptors:
        raise SystemExit('No eligible tracks produced a descriptor; check the archive layout.')

    X = np.array(descriptors)
    y = np.array(labels)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, X=X, y=y, groups=np.array(groups), split_source=np.array(split_names),
                         clip_columns=np.array(CLIP_COLUMNS), fps_assumed=FPS,
                         feature_names=np.array(FEATURE_NAMES), license_note=LICENSE_NOTE)

    report = {'tracks_seen': tracks_seen, 'tracks_eligible': len(descriptors),
              'skipped': skipped, 'fps_assumed': FPS,
              'label_counts': {'normal': int((y == 0).sum()), 'shoplifting': int((y == 1).sum())},
              'by_split': {name: {'tracks': int((np.array(split_names) == name).sum()),
                                   'shoplifting': int(((np.array(split_names) == name) & (y == 1)).sum())}
                           for name in SPLITS},
              'span_s_summary': {'min': float(min(m['span_s'] for m in meta)),
                                  'median': float(np.median([m['span_s'] for m in meta])),
                                  'max': float(max(m['span_s'] for m in meta))},
              'license_note': LICENSE_NOTE, 'output': str(OUT.relative_to(ROOT))}
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
