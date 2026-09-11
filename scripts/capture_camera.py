"""Record fixed-camera clips into the timing-correct form the pipeline already expects.

Why this exists: UR Fall is CC BY-NC-SA 4.0 research-only and is currently the entire fall
training path, which blocks a competition release (D029). Footage recorded here belongs to
whoever recorded it, so it is licence-clean, it is target-domain, and — being a new corpus
— it restores a held-out split that has not already been read.

Safety: land on a mattress or crash mat. Do not stage a dangerous fall. A worse model is a
better outcome than an injury, and the point of this corpus is ordinary variety, not force.

Timing is captured, not assumed. Frame times come from the monotonic clock at grab time and
are encoded at the 90 kHz timebase (D023), so real inter-frame spacing survives instead of
being quantised onto a nominal frame rate. The finished file is decoded back and compared
against the recorded table before it is published (D024).
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from watchverify.perception import TIME_BASE, VideoExport, frames, sha256, video_info

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/processed/owncam'
PARTIAL = OUT / '_partial'
CAPTURE_VERSION = 1
# The worker refuses anything larger, so recording larger only wastes disk and time.
MAX_WIDTH, MAX_HEIGHT = 1920, 1080
MAX_SECONDS = 600


def capture(clip, device=0, seconds=30, countdown=5, width=1280, height=720, fps=30,
            license_holder='Own recording'):
    if not clip or any(character in clip for character in '/\\. '):
        raise ValueError('Clip name must be a simple identifier, e.g. fall-own-01')
    if not 0 < seconds <= MAX_SECONDS:
        raise ValueError(f'Recording length must be between 0 and {MAX_SECONDS} seconds')
    if width > MAX_WIDTH or height > MAX_HEIGHT:
        raise ValueError(f'Record at {MAX_WIDTH}x{MAX_HEIGHT} or smaller; the analyser refuses larger video')
    destination = OUT / f'{clip}.mp4'
    if destination.exists():
        raise ValueError(f'{destination.name} already exists. Choose another clip name rather than overwriting footage.')
    OUT.mkdir(parents=True, exist_ok=True)
    PARTIAL.mkdir(parents=True, exist_ok=True)
    temporary = PARTIAL / f'{clip}.mp4'
    temporary.unlink(missing_ok=True)

    camera = cv2.VideoCapture(int(device))
    if not camera.isOpened():
        raise RuntimeError(f'Camera {device} could not be opened. Check the device index and camera permissions.')
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    writer = None
    mapping = []
    started_utc = None
    try:
        # Warm-up: the first frames off a webcam are often stale, dark or auto-exposing,
        # and they would land in the recording as motion that never happened.
        for _ in range(10):
            camera.read()
        for remaining in range(int(countdown), 0, -1):
            print(f'Recording {clip} in {remaining}…', flush=True)
            time.sleep(1)
        print(f'Recording {clip} for {seconds:g}s. Press Ctrl+C to stop early.', flush=True)
        started_utc = datetime.now(timezone.utc).isoformat()
        first = None
        while True:
            ok, bgr = camera.read()
            now = time.perf_counter()
            if not ok:
                raise RuntimeError('The camera stopped returning frames before the recording finished.')
            if first is None:
                first = now
            t = now - first
            if t > seconds:
                break
            if writer is None:
                writer = VideoExport(temporary, bgr.shape[1], bgr.shape[0], fps)
            if mapping and round(t / TIME_BASE) <= round(mapping[-1]['output_t'] / TIME_BASE):
                continue  # Two grabs inside one 90 kHz tick; keep the first rather than invent a time.
            writer.write(bgr, t)
            mapping.append({'source_frame': len(mapping), 'decoded_index': len(mapping), 'output_t': t})
    except KeyboardInterrupt:
        print('\nStopped early; keeping what was recorded.', flush=True)
    finally:
        camera.release()
        if writer is not None:
            writer.close()
    if len(mapping) < 10:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f'Only {len(mapping)} frames were captured; that is too short to be useful.')

    # Publish only after the file proves it decodes back to the times we recorded.
    info = video_info(temporary)
    decoded = [(index, t) for index, t, _ in frames(temporary)]
    if len(decoded) != len(mapping):
        temporary.unlink(missing_ok=True)
        raise ValueError(f'{clip}: encoded {len(mapping)} frames but decoded {len(decoded)}')
    drift = max(abs(t - row['output_t']) for (_, t), row in zip(decoded, mapping))
    if drift > 1e-4:
        temporary.unlink(missing_ok=True)
        raise ValueError(f'{clip}: decoded timing drifted by {drift:.6f}s from the capture table')
    temporary.replace(destination)

    card = {'source_id': clip, 'capture_version': CAPTURE_VERSION,
            'capture_start_utc': started_utc, 'device_index': int(device),
            'requested': {'width': width, 'height': height, 'fps': fps, 'seconds': seconds},
            'video_sha256': sha256(destination), 'time_base': str(TIME_BASE),
            'nominal_rate': fps, 'frame_count': len(mapping), 'max_decode_drift_s': drift,
            'media': info, 'frames': mapping,
            'timestamp_source': 'monotonic clock at frame grab',
            'license': license_holder,
            'research_only': False,
            'note': ('Recorded locally for this project. Frame times are capture times, not a '
                     'nominal frame rate. No claim is made about anyone other than the person '
                     'who recorded and consented to this clip.')}
    (OUT / f'{clip}.json').write_text(json.dumps(card, indent=2))
    return {'clip': clip, 'status': 'captured', 'frames': len(mapping),
            'duration_s': round(mapping[-1]['output_t'], 3), 'max_decode_drift_s': drift,
            'video': str(destination.relative_to(ROOT))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--clip', required=True, help='Identifier, e.g. fall-own-01 or adl-own-01')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--countdown', type=float, default=5)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--license-holder', default='Own recording')
    arguments = parser.parse_args()
    result = capture(arguments.clip, arguments.device, arguments.seconds, arguments.countdown,
                     arguments.width, arguments.height, arguments.fps, arguments.license_holder)
    print(json.dumps(result, indent=2))
    print('\nNext: add a row for this clip to data/raw/owncam/labels.csv, then run '
          'scripts/prepare_owncam.py', flush=True)


if __name__ == '__main__':
    main()
