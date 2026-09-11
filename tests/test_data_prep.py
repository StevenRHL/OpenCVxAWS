"""Data-preparation invariants: timing validation, atomic publish, cache identity, safe extraction.

These check software behaviour on synthetic fixtures. They do not measure pose or
detection quality on real footage.
"""
import importlib.util
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = _load("prepare_urfall")
mnnit = _load("prepare_mnnit")
features = _load("extract_features")


@pytest.fixture
def urfall(tmp_path, monkeypatch):
    """Redirect the preparer at a temporary tree so nothing touches real data."""
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    monkeypatch.setattr(prep, "OUT", tmp_path / "data/processed/urfall")
    monkeypatch.setattr(prep, "PARTIAL", tmp_path / "data/processed/urfall/_partial")
    (tmp_path / "data/raw/urfall").mkdir(parents=True)
    return tmp_path


def _sync(root, name, stamps_ms):
    path = root / "data/raw/urfall" / f"{name}-data.csv"
    path.write_text("".join(f"{i},{ms}\n" for i, ms in stamps_ms))
    return path


def _archive(root, name, frame_ids, width=64, height=48):
    path = root / "data/raw/urfall" / f"{name}-cam0-rgb.zip"
    with zipfile.ZipFile(path, "w") as z:
        for i in frame_ids:
            image = np.full((height, width, 3), (i * 7) % 256, dtype=np.uint8)
            z.writestr(f"{name}-cam0-rgb-{i:03d}.png", cv2.imencode(".png", image)[1].tobytes())
    return path


def test_prepare_publishes_only_after_a_successful_decode(urfall):
    ids = list(range(1, 11))
    _sync(urfall, "adl-99", [(i, (i - 1) * 15) for i in ids])  # 15 ms spacing
    archive = _archive(urfall, "adl-99", ids)
    result = prep.prepare(archive)

    assert result["status"] == "prepared" and result["frames"] == 10
    card = json.loads((prep.OUT / "adl-99.json").read_text())
    assert [row["source_frame"] for row in card["frames"]] == ids
    assert card["frames"][0]["decoded_index"] == 0
    assert card["max_decode_drift_s"] == pytest.approx(0, abs=1e-4)
    assert not list(prep.PARTIAL.glob("*"))  # partial cleaned up on success


def test_second_run_is_a_no_op_but_a_missing_sidecar_forces_repreparation(urfall):
    ids = list(range(1, 6))
    _sync(urfall, "adl-98", [(i, (i - 1) * 33) for i in ids])
    archive = _archive(urfall, "adl-98", ids)
    prep.prepare(archive)
    assert prep.prepare(archive)["status"] == "current"

    (prep.OUT / "adl-98.json").unlink()
    assert prep.prepare(archive)["status"] == "prepared"


def test_a_stale_video_from_an_older_preparer_is_not_accepted(urfall):
    ids = list(range(1, 6))
    _sync(urfall, "adl-97", [(i, (i - 1) * 33) for i in ids])
    archive = _archive(urfall, "adl-97", ids)
    prep.prepare(archive)
    card = json.loads((prep.OUT / "adl-97.json").read_text())
    card["preparer_version"] = prep.PREPARER_VERSION - 1
    (prep.OUT / "adl-97.json").write_text(json.dumps(card))
    assert prep.prepare(archive)["status"] == "prepared"


def test_a_failed_preparation_publishes_nothing_and_leaves_no_partial(urfall):
    """An interior frame with no published timing means the timing table is unusable."""
    ids = [1, 2, 3, 4, 5, 6]
    _sync(urfall, "adl-96", [(i, (i - 1) * 33) for i in ids if i != 4])  # frame 4 untimed
    archive = _archive(urfall, "adl-96", ids)

    with pytest.raises(ValueError, match="not a trailing block"):
        prep.prepare(archive)
    assert not (prep.OUT / "adl-96.mp4").exists()
    assert not (prep.OUT / "adl-96.json").exists()
    assert not list(prep.PARTIAL.glob("*"))


def test_trailing_untimed_frames_are_dropped_and_recorded(urfall):
    """UR adl-37 ships 350 frames but only 330 timing rows. Discard the untimed tail
    explicitly rather than inventing timestamps for it."""
    ids = list(range(1, 9))
    _sync(urfall, "adl-95", [(i, (i - 1) * 33) for i in ids[:6]])  # frames 7, 8 untimed
    archive = _archive(urfall, "adl-95", ids)

    result = prep.prepare(archive)
    assert result["frames"] == 6 and result["dropped"] == 2
    card = json.loads((prep.OUT / "adl-95.json").read_text())
    assert [row["source_frame"] for row in card["dropped"]] == [7, 8]
    assert all(row["reason"] == "no_published_timing" for row in card["dropped"])
    assert [row["source_frame"] for row in card["frames"]] == ids[:6]


def test_unusable_source_timing_is_reported_never_replaced(urfall):
    _sync(urfall, "adl-92", [(1, 0), (2, 33), (2, 66)])
    with pytest.raises(ValueError, match="duplicate source frame id 2"):
        prep.read_timing("adl-92")

    _sync(urfall, "adl-94", [(1, 0), (2, 66), (3, 33)])
    with pytest.raises(ValueError, match="source times not increasing"):
        prep.read_timing("adl-94")

    _sync(urfall, "adl-93", [(1, 0), (3, 33), (2, 66)])
    with pytest.raises(ValueError, match="ids not increasing"):
        prep.read_timing("adl-93")


def test_times_sharing_one_output_tick_are_refused():
    with pytest.raises(ValueError, match="share output tick"):
        prep.check_encodable("x", [(1, 0.0), (2, 0.000001)])
    prep.check_encodable("x", [(1, 0.0), (2, 0.015), (3, 0.047)])


# --- feature cache identity -------------------------------------------------

class _Estimator:
    asset_sha256 = "a" * 64
    detection_confidence = presence_confidence = tracking_confidence = 0.5


def _key(**overrides):
    from watchverify.core import FeatureBuffer, Tracker
    arguments = dict(path=ROOT / "data/processed/urfall/adl-01.mp4", card={"archive_sha256": "b" * 64,
                     "preparer_version": 2}, estimator=_Estimator(), tracker=Tracker(),
                     buffer=FeatureBuffer(), max_people=1, interval=0.1, width=640, variant="full")
    arguments.update(overrides)
    return features.build_config(**arguments)[1]


@pytest.mark.skipif(not (ROOT / "data/processed/urfall/adl-01.mp4").exists(),
                    reason="requires a prepared video")
@pytest.mark.parametrize("change", [
    {"width": 480}, {"max_people": 4}, {"interval": 0.2}, {"variant": "lite"},
])
def test_every_setting_that_changes_features_changes_the_cache_key(change):
    assert _key() == _key(), "cache key must be stable for an unchanged configuration"
    assert _key(**change) != _key(), f"{change} must invalidate the cache"


@pytest.mark.skipif(not (ROOT / "data/processed/urfall/adl-01.mp4").exists(),
                    reason="requires a prepared video")
def test_feature_schema_and_pose_asset_participate_in_the_cache_key():
    import watchverify.core as core

    baseline = _key()
    other = _Estimator()
    other.asset_sha256 = "c" * 64
    assert _key(estimator=other) != baseline

    original = core.FEATURE_SCHEMA_VERSION
    try:
        core.FEATURE_SCHEMA_VERSION = original + 1
        features.FEATURE_SCHEMA_VERSION = original + 1
        assert _key() != baseline
    finally:
        core.FEATURE_SCHEMA_VERSION = original
        features.FEATURE_SCHEMA_VERSION = original


# --- archive extraction safety ----------------------------------------------

def _member(filename, external_attr=0):
    info = zipfile.ZipInfo(filename)
    info.external_attr = external_attr
    return info


def test_archive_members_cannot_escape_the_destination(tmp_path):
    destination = tmp_path / "out"
    destination.mkdir()
    for hostile in ["../escape.mp4", "Dataset/../../escape.mp4", "/etc/passwd"]:
        with pytest.raises(ValueError):
            mnnit.safe_target(_member(hostile), destination)


def test_symlink_members_are_refused(tmp_path):
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(ValueError, match="symlink"):
        mnnit.safe_target(_member("Dataset/link.mp4", (0o120777 << 16)), destination)


def test_ordinary_members_resolve_inside_the_destination(tmp_path):
    destination = tmp_path / "out"
    destination.mkdir()
    target = mnnit.safe_target(_member("Dataset/Normal/Normal (1).mp4"), destination)
    assert str(target).startswith(str(destination.resolve()) + "/")
    assert mnnit.classify("Dataset/Shoplifting/Shoplifting (3).mp4") == "shoplifting"
    assert mnnit.classify("Dataset/Other/x.mp4") is None
