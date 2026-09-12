"""A fresh clone must be able to reach a working install, and must say so honestly.

These are the checks the suite was missing. Every other test monkeypatches pose
estimation away, so 177 tests passed on a clone that could not analyse a single video:
`models/*.task` is not committed and nothing obtained it. The rule these tests encode is
that anything the runtime needs is either committed or provisioned by a recorded,
checksum-pinned acquisition step that setup actually runs.

No network and no pose asset is required here. The end-to-end proof that setup works is
`scripts/verify_install.py`, which runs real inference and is deliberately not a test:
it needs the asset these tests only check the provenance of.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("full", "lite")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pose_assets = _load("acquire_pose_assets")
acquire = _load("acquire_data")
SETUP = (ROOT / "Setup WatchVerify.command").read_text()


def manifest_rows():
    path = ROOT / "data/manifests/downloads.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_pose_variant_the_app_offers_is_pinned(variant):
    """app.py offers both variants, so both need a pinned origin, not just the default."""
    assert variant in pose_assets.ASSETS
    digest, size = pose_assets.ASSETS[variant]
    assert len(digest) == 64 and int(digest, 16) >= 0
    assert size > 0
    assert pose_assets.path_for(variant) == f"models/pose_landmarker_{variant}.task"


@pytest.mark.parametrize("variant", VARIANTS)
def test_pinned_url_is_an_immutable_revision(variant):
    """A pinned checksum against a mutable path breaks setup the next time upstream
    republishes. Upstream's own 'latest' alias for the full model already serves different
    bytes from the revision this project's feature caches were built against."""
    url = pose_assets.url_for(variant)
    assert url.startswith("https://")
    assert "/latest/" not in url
    assert f"/{pose_assets.REVISION}/" in url
    assert pose_assets.REVISION.isdigit()


@pytest.mark.parametrize("variant", VARIANTS)
def test_download_manifest_records_the_asset_origin(variant):
    """The provenance rule: origin, revision, size, checksum and terms are recorded for
    every upstream download. Before this work the two .task files had none of that."""
    digest, size = pose_assets.ASSETS[variant]
    path = pose_assets.path_for(variant)
    rows = [row for row in manifest_rows()
            if row.get("path") == path and row.get("status") in ("downloaded", "cached")]
    assert rows, f"no successful acquisition of {path} recorded in data/manifests/downloads.jsonl"
    row = rows[-1]
    assert row["sha256"] == digest
    assert row["bytes"] == size
    assert row["url"] == pose_assets.url_for(variant)
    assert "Apache-2.0" in row["license_note"]
    assert row["retrieved_at_utc"]


def test_setup_installs_the_lock_file_not_the_loose_requirements():
    assert "requirements-lock.txt" in SETUP
    assert "-r requirements.txt" not in SETUP


def test_setup_provisions_the_pose_asset_and_proves_inference_runs():
    """The absence of a smoke check is what let a non-working install report success."""
    assert "scripts/acquire_pose_assets.py" in SETUP
    assert "scripts/verify_install.py" in SETUP
    assert (ROOT / "scripts/verify_install.py").exists()


def test_setup_enforces_the_interpreter_range_the_pins_require():
    assert "MIN_MINOR=10" in SETUP, "floor must match the >=3.10 pins in requirements-lock.txt"
    assert "MAX_MINOR=12" in SETUP, "ceiling must match mediapipe/numpy wheel availability"
    assert "no supported Python was found" in SETUP


def test_fetch_refuses_bytes_that_do_not_match_a_pinned_checksum(tmp_path, monkeypatch):
    """Fail closed, offline: a file already at the target path is verified, not trusted."""
    monkeypatch.setattr(acquire, "ROOT", tmp_path)
    monkeypatch.setattr(acquire, "MANIFEST", tmp_path / "downloads.jsonl")
    target = tmp_path / "models/pose_landmarker_full.task"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not the pose model")
    row = acquire.fetch("https://example.invalid/x.task", "models/pose_landmarker_full.task",
                        "test", "test", sha256=pose_assets.ASSETS["full"][0])
    assert row["status"] == "failed_checksum"
    assert row["expected_sha256"] == pose_assets.ASSETS["full"][0]
    assert row["sha256"] != row["expected_sha256"]
    recorded = json.loads((tmp_path / "downloads.jsonl").read_text().splitlines()[-1])
    assert recorded["status"] == "failed_checksum"


def test_missing_pose_asset_names_the_step_that_installs_it(tmp_path, monkeypatch):
    """The old message said "Run setup first", which was unreachable and, if reached,
    pointed at a setup that did not obtain the asset."""
    pytest.importorskip("mediapipe")
    from watchverify import perception
    monkeypatch.setattr(perception, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError) as error:
        perception.PoseEstimator(variant="full")
    message = str(error.value)
    assert "pose_landmarker_full.task" in message
    assert "Setup WatchVerify.command" in message
    assert "acquire_pose_assets.py" in message


def test_the_interface_does_not_claim_work_nothing_is_doing():
    """Both sidebar strings promised a download that no code performed."""
    app = (ROOT / "app.py").read_text()
    assert "being prepared" not in app
    assert "must finish downloading" not in app
    assert app.count("Setup WatchVerify.command") >= 2


def test_the_printed_instructions_do_not_open_by_asserting_the_install_is_done():
    how_to_run = (ROOT / "HOW TO RUN.txt").read_text()
    assert "Everything is installed" not in how_to_run
    assert "Setup WatchVerify.command" in how_to_run.split("2. START IT")[0]
