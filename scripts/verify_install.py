"""Smoke check: prove this installation can actually run inference before claiming it works.

Setup used to end after `pip install`, which says nothing about whether the app can
analyse a video. The pose asset is not committed, so an installation could be complete by
pip's account and still be unable to detect a single person — and the test suite could not
see it, because every test monkeypatches pose estimation away. This closes that gap:

  1. imports the runtime dependencies,
  2. constructs a real `PoseEstimator` against the installed pose asset,
  3. runs one synthetic frame through it,
  4. constructs `Models()` and scores one synthetic vector per loaded artifact,
  5. prints PASS or FAIL and exits non-zero on FAIL.

A synthetic frame tests software wiring only. Nothing here measures detection accuracy,
and passing says nothing about whether an alert on real footage is correct.

    .venv/bin/python scripts/verify_install.py [--variant full|lite]
"""
from pathlib import Path
import argparse, sys, traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SETUP_HINT = ('Run "Setup WatchVerify.command", or fetch the asset directly with '
              '.venv/bin/python scripts/acquire_pose_assets.py')


def synthetic_frame(width=640, height=480):
    """A deterministic non-uniform BGR frame. Pose is not expected to find a person in it."""
    import numpy as np
    y, x = np.mgrid[0:height, 0:width]
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = (x % 256).astype(np.uint8)
    frame[..., 1] = (y % 256).astype(np.uint8)
    frame[..., 2] = ((x + y) % 256).astype(np.uint8)
    frame[height // 4:3 * height // 4, width // 3:2 * width // 3] = 200
    return frame


def check_pose(variant, failures):
    asset = ROOT / 'models' / f'pose_landmarker_{variant}.task'
    if not asset.exists():
        failures.append(f'pose asset missing: {asset.relative_to(ROOT)}. {SETUP_HINT}')
        return
    from watchverify.perception import PoseEstimator
    with PoseEstimator(variant=variant, max_people=1) as estimator:
        people = estimator.detect(synthetic_frame(), 0.0)
    print(f'  pose ({variant}): inference ran, {len(people)} people in the synthetic frame, '
          f'asset sha256 {estimator.asset_sha256[:12]}...')


def check_models(failures):
    import numpy as np
    from watchverify.models import Models, CLIP_DESCRIPTOR, FRAME_VECTOR
    models = Models()
    for name, status in sorted(models.status.items()):
        print(f'  model ({name}): {status}')
        if status.startswith('Unavailable'):
            failures.append(f'model {name} failed to load: {status}')
    for name, (_, card) in sorted(models.loaded.items()):
        width = int(card['n_features'])
        if card.get('input', FRAME_VECTOR) == CLIP_DESCRIPTOR:
            result = models.score(name, np.zeros(width))
        elif card.get('with_mask'):
            half = width // 2
            result = models.score(name, np.zeros(half), mask=np.ones(half))
        else:
            result = models.score(name, np.zeros(width))
        if result is None:
            failures.append(f'model {name} returned no score for a {width}-wide synthetic vector')
        else:
            print(f'  model ({name}): scored a synthetic vector -> {result["score_type"]} '
                  f'{result["score"]:.4f} against threshold {result["threshold"]:.4f}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--variant', default='full', choices=('full', 'lite'))
    args = parser.parse_args(argv)
    failures = []
    print(f'Smoke check (python {sys.version.split()[0]}):')
    try:
        check_pose(args.variant, failures)
        check_models(failures)
    except Exception:
        traceback.print_exc()
        failures.append('an exception was raised during the smoke check (traceback above)')
    if failures:
        print('\nFAIL: this installation cannot run an analysis.', file=sys.stderr)
        for problem in failures:
            print(f'  - {problem}', file=sys.stderr)
        return 1
    print('\nPASS: pose estimation and the installed models ran on synthetic input.')
    print('This checks wiring only. It does not measure detection accuracy.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
