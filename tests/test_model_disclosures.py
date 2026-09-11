"""Disclosure provenance and historical-run isolation, without re-evaluating models."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from watchverify.models import Models
from watchverify.alerts import BRANCH_PROMPT

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["fall", "activity"])
def test_installed_disclosure_matches_artifact_and_validation(name):
    card = json.loads((ROOT / "models" / f"{name}.json").read_text())
    evidence = card["disclosure_evidence"]
    assert evidence["artifact_sha256"] == card["sha256"]
    assert hashlib.sha256((ROOT / "models" / card["artifact"]).read_bytes()).hexdigest() == card["sha256"]
    metrics_path = ROOT / evidence["metrics_path"]
    assert hashlib.sha256(metrics_path.read_bytes()).hexdigest() == evidence["metrics_sha256"]
    point = json.loads(metrics_path.read_text())["selected_operating_point"]
    assert point == evidence["operating_point"]
    assert evidence["model_run_id"] == card["run_id"]
    if name == "activity":
        rule = point["sustained_clip_rule"]
        assert f"{rule['normal_clips_alerted']} of {rule['normal_clips']}" in card["alert_caveat"]
        assert card["threshold"] == point["threshold"]
        assert card["release_cleared"] is None
    else:
        assert f"{point['false_alerts']} of {point['alerts']}" in card["alert_caveat"]
        assert card["threshold"] == point["posture_threshold"]
        assert card["release_cleared"] is False


def get_caveat_function(job):
    tree = ast.parse((ROOT / "app.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "alert_caveat")
    namespace = {"BRANCH_PROMPT": BRANCH_PROMPT,
                 "jobs": SimpleNamespace(get_job=lambda _: job),
                 "model_card": lambda _: {"alert_caveat": "WRONG replacement-model figures"}}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "app.py", "exec"), namespace)
    return namespace["alert_caveat"]


def test_saved_run_disclosure_wins_over_replacement_card():
    job = {"model_disclosures": {"activity": {"alert_caveat": "Original run disclosure"}}}
    assert get_caveat_function(job)("activity", "old-run") == "Original run disclosure"


@pytest.mark.parametrize("job", [{}, {"model_disclosures": {}}])
def test_old_or_unavailable_model_never_borrows_current_figures(job):
    text = get_caveat_function(job)("activity", "old-run")
    assert "No model-specific evaluation disclosure was saved" in text
    assert "WRONG" not in text


def test_only_successfully_loaded_models_get_snapshots():
    models = Models.__new__(Models)
    models.loaded = {}
    models.status = {"fall": "Unavailable: checksum mismatch"}
    assert models.disclosures() == {}


def test_worker_persists_loaded_disclosures_before_processing():
    # Exercise the real worker's startup and manifest write; stop at pose initialization.
    import tempfile
    from unittest.mock import patch
    from watchverify import jobs, worker
    class StopPose:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("intentional stop after metadata persistence")
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder)
        with patch.object(jobs, "ROOT", target), patch.object(worker, "ROOT", target), patch.object(worker, "PoseEstimator", StopPose):
            run_id = jobs.create_job(ROOT / "data/processed/urfall/fall-04.mp4")
            assert worker.run(run_id) == "failed"
            saved = jobs.get_job(run_id)
            assert "intentional stop" in saved["error"]
            assert set(saved["model_disclosures"]) == {"fall", "activity"}
            assert saved["model_disclosures"]["activity"]["alert_caveat"]
