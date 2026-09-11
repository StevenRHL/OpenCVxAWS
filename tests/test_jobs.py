from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from watchverify import jobs


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    monkeypatch.setattr(jobs, "_PROCESSES", {})
    monkeypatch.setattr(jobs.shutil, "disk_usage", lambda path: SimpleNamespace(free=50 * 1024**3))
    source = tmp_path / "original.mp4"
    source.write_bytes(b"example media bytes for job ownership tests")
    return source


@pytest.fixture
def processes(monkeypatch):
    instances = []

    class Process:
        def __init__(self, args, **kwargs):
            self.args = args
            self.options = kwargs
            self.pid = 987654 + len(instances)
            self.returncode = None
            instances.append(self)

        def poll(self):
            return self.returncode

    monkeypatch.setattr(jobs.subprocess, "Popen", Process)
    return instances


def test_source_is_owned_immutable_copy(workspace):
    run_id = jobs.create_job(workspace, original_name="holiday.mp4")
    job = jobs.get_job(run_id)
    copy = Path(job["source_path"])
    assert copy != workspace and copy.read_bytes() == workspace.read_bytes()
    assert copy.stat().st_mode & 0o222 == 0
    assert job["original_name"] == "holiday.mp4"
    workspace.write_bytes(b"changed original")
    assert copy.read_bytes() != workspace.read_bytes()
    jobs.delete_job(run_id)
    assert workspace.exists()
    assert not copy.exists()


def test_launch_is_idempotent_and_blocks_second_worker(workspace, processes):
    first = jobs.create_job(workspace)
    second = jobs.create_job(workspace)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: jobs.launch_job(first), range(4)))
    assert len(processes) == 1
    assert all(result["pid"] == processes[0].pid for result in results)
    assert processes[0].args[-2:] == ["--run-id", first]
    with pytest.raises(RuntimeError, match="Another video"):
        jobs.launch_job(second)
    assert len(processes) == 1


def test_cancellation_is_durable_idempotent_and_prevents_delete(workspace, processes):
    run_id = jobs.create_job(workspace)
    jobs.launch_job(run_id)
    assert jobs.cancel_job(run_id)["status"] == "cancelling"
    assert jobs.cancel_job(run_id)["status"] == "cancelling"
    assert (jobs.run_dir(run_id) / "cancel.request").exists()
    with pytest.raises(RuntimeError, match="stop"):
        jobs.delete_job(run_id)
    processes[0].returncode = 0
    assert jobs.get_job(run_id)["status"] == "cancelled"
    jobs.delete_job(run_id)


def test_queued_cancel_does_not_launch(workspace, processes):
    run_id = jobs.create_job(workspace)
    assert jobs.cancel_job(run_id)["status"] == "cancelled"
    with pytest.raises(ValueError, match="new analysis"):
        jobs.launch_job(run_id)
    assert not processes


def test_worker_failure_is_visible_and_frees_slot(workspace, processes):
    first = jobs.create_job(workspace)
    jobs.launch_job(first)
    processes[0].returncode = 1
    assert jobs.get_job(first)["status"] == "interrupted"
    second = jobs.create_job(workspace)
    assert jobs.launch_job(second)["status"] == "running"


def test_atomic_updates_keep_other_fields(workspace):
    run_id = jobs.create_job(workspace)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: jobs.update_job(run_id, **{f"field_{i}": i}), range(20)))
    job = jobs.get_job(run_id)
    assert all(job[f"field_{i}"] == i for i in range(20))
    assert job["source_sha256"]


def test_reviews_persist_and_do_not_rewrite_revisions(workspace):
    run_id = jobs.create_job(workspace)
    event = {"event_id": "incident-1", "revision": 1, "source_start_s": 2, "category": "possible_fall"}
    with sqlite3.connect(jobs.run_dir(run_id) / "events.db") as connection:
        connection.execute("CREATE TABLE revisions(event_id TEXT, revision INTEGER, payload TEXT, PRIMARY KEY(event_id,revision))")
        connection.execute("INSERT INTO revisions VALUES(?,?,?)", ("incident-1", 1, json.dumps(event)))
    jobs.save_review(run_id, "incident-1", "relevant", "Check this moment")
    assert jobs.get_reviews(run_id)["incident-1"]["note"] == "Check this moment"
    jobs.save_review(run_id, "incident-1", "unclear", "Partly occluded")
    assert jobs.get_reviews(run_id)["incident-1"]["label"] == "unclear"
    assert jobs.load_events(run_id) == [event]
    with pytest.raises(ValueError, match="not in"):
        jobs.save_review(run_id, "someone-elses-event", "relevant")


def test_invalid_dates_and_paths_rejected(workspace):
    with pytest.raises(ValueError, match="timezone"):
        jobs.create_job(workspace, {"recording_start": "2026-09-11T12:00:00"})
    with pytest.raises(ValueError, match="identifier"):
        jobs.get_job("../original.mp4")
    run_id = jobs.create_job(workspace, {"recording_start": "2026-09-11T12:00:00+10:00"})
    assert jobs.get_job(run_id)["config"]["recording_start"].endswith("+10:00")
