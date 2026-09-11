"""Media timing round-trips. These check encoder/decoder behaviour, not detection."""
import numpy as np
import pytest

from watchverify.perception import VideoExport, frames, video_info


def _frame(value, width=320, height=240):
    return np.full((height, width, 3), value, dtype=np.uint8)


def _write(path, times, width=320, height=240, rate=30):
    export = VideoExport(path, width, height, rate)
    try:
        for index, t in enumerate(times):
            export.write(_frame(index * 20 % 256, width, height), t)
    finally:
        export.close()
    return path


# UR Fall adl-04..adl-10 are sampled 15 ms apart, closer than one 30 fps interval.
SUB_INTERVAL = [0.0, 0.015, 0.047, 0.093, 0.125, 0.157, 0.190, 0.222, 0.255, 0.287]


def test_sub_frame_interval_spacing_survives_the_round_trip(tmp_path):
    path = _write(tmp_path / "sub_interval.mp4", SUB_INTERVAL)
    decoded = [t for _, t, _ in frames(path)]
    assert len(decoded) == len(SUB_INTERVAL)
    assert decoded == pytest.approx(SUB_INTERVAL, abs=1e-5)
    assert all(b > a for a, b in zip(decoded, decoded[1:]))


def test_irregular_source_spacing_is_preserved_not_resampled(tmp_path):
    times = [0.0, 0.016, 0.5, 0.516, 1.0]
    path = _write(tmp_path / "irregular.mp4", times)
    decoded = [t for _, t, _ in frames(path)]
    assert decoded == pytest.approx(times, abs=1e-5)
    assert video_info(path)["duration_s"] > 0.9


def test_repeated_timestamp_is_refused_before_it_reaches_the_muxer(tmp_path):
    export = VideoExport(tmp_path / "repeat.mp4", 320, 240, 30)
    try:
        export.write(_frame(0), 0.25)
        with pytest.raises(ValueError, match="not after the previous frame"):
            export.write(_frame(1), 0.25)
        with pytest.raises(ValueError, match="not after the previous frame"):
            export.write(_frame(2), 0.10)
    finally:
        export.close()


def test_timestamps_within_one_encoder_tick_are_refused(tmp_path):
    """Two frames closer than 1/90000 s still collide; that must be explicit."""
    export = VideoExport(tmp_path / "tick.mp4", 320, 240, 30)
    try:
        export.write(_frame(0), 1.0)
        with pytest.raises(ValueError, match="not after the previous frame"):
            export.write(_frame(1), 1.0 + 1e-6)
    finally:
        export.close()
