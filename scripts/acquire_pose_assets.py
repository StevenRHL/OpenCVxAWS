"""Fetch the pretrained MediaPipe pose landmarker assets this app runs on.

Why this script exists
----------------------
`models/*.task` is deliberately not committed (see .gitignore): the assets are 15 MB of
upstream binary and belong to Google, not to this project. Nothing else in the repository
obtained them, so a fresh clone installed cleanly and then refused to analyse anything,
while the interface claimed the asset was "being prepared". This is what prepares it.
`Setup WatchVerify.command` runs it; it is also safe to run by hand:

    .venv/bin/python scripts/acquire_pose_assets.py            # both variants
    .venv/bin/python scripts/acquire_pose_assets.py --variant full

Re-running is cheap: a file already present is checksum-verified, not re-downloaded.

Why the URLs carry a version number
-----------------------------------
Google publishes each model under both `.../float16/1/` and `.../float16/latest/`. The
`latest` path is mutable and has already drifted: on 12 September 2026 it served a
pose_landmarker_full.task of identical length but a different digest
(4eaa5eb7...) from the one every feature cache in this project was built against
(5134a3aa...). Pinning a hash against a mutable path would break setup for everyone the
next time upstream republishes, so the versioned path is the only correct source here.

Why the checksums are pinned to these values
--------------------------------------------
The `full` digest is the asset the installed models were actually built against: every
cached feature file records `pose_asset_sha256: 5134a3aa...` in its `config_json`.
Changing an asset silently would change extracted features and invalidate the recorded
validation figures in README.md without anything failing. So a mismatch stops setup and
prints the URL and the expected digest rather than installing whatever arrived.

Licence: Apache-2.0 (Google MediaPipe model release). The model card, including its
stated limitations and fairness evaluation, is linked from
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
A downloaded checkpoint is not a validated model.

Every attempt appends a row to data/manifests/downloads.jsonl recording origin, revision,
size, checksum and licence, as the project's provenance rule requires. That file is
tracked, so it shows as modified after setup on a fresh clone: that is the clone recording
its own download, not a problem to fix.
"""
from pathlib import Path
import argparse, sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acquire_data import fetch

BASE = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker'
REVISION = '1'          # immutable published revision; never 'latest'
PRECISION = 'float16'
SOURCE = 'mediapipe-pose-landmarker'
LICENSE = ('Apache-2.0 (Google MediaPipe model release); model card at '
           'https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker')

# variant -> (expected sha256, expected size in bytes)
ASSETS = {
    'full': ('5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1', 9398198),
    'lite': ('59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a', 5777746),
}


def url_for(variant):
    return f'{BASE}/pose_landmarker_{variant}/{PRECISION}/{REVISION}/pose_landmarker_{variant}.task'


def path_for(variant):
    return f'models/pose_landmarker_{variant}.task'


def acquire(variant):
    """Fetch one variant, verifying it against the pinned digest. Returns the manifest row."""
    digest, size = ASSETS[variant]
    url = url_for(variant)
    row = fetch(url, path_for(variant), SOURCE, LICENSE, sha256=digest)
    if row.get('status') in ('downloaded', 'cached') and row.get('bytes') != size:
        row['status'] = 'failed_size'
        row['error'] = f'Expected {size} bytes, received {row.get("bytes")}'
    ok = row.get('status') in ('downloaded', 'cached')
    if ok:
        print(f'{variant}: ok ({row["status"]}, {row["bytes"]} bytes, sha256 {row["sha256"][:12]}...)')
    else:
        print(f'{variant}: FAILED ({row.get("status")})', file=sys.stderr)
        print(f'  url           {url}', file=sys.stderr)
        print(f'  expected sha  {digest}', file=sys.stderr)
        if row.get('sha256'):
            print(f'  received sha  {row["sha256"]}', file=sys.stderr)
        if row.get('error'):
            print(f'  detail        {str(row["error"]).strip()[:500]}', file=sys.stderr)
        print(f'  nothing was installed at {path_for(variant)}', file=sys.stderr)
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--variant', choices=sorted(ASSETS), action='append',
                        help='fetch only this variant; repeatable. Default: all.')
    args = parser.parse_args(argv)
    wanted = args.variant or sorted(ASSETS)
    results = [acquire(variant) for variant in wanted]
    if not all(results):
        print('Pose assets are incomplete; the app cannot analyse video without them.', file=sys.stderr)
        return 1
    print('Pose assets ready.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
