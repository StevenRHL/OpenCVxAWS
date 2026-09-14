"""The worker's own per-frame diagnostic trace: written going forward, never faked for old runs.

Reuses the synthetic renumbering fixture from `test_worker_evidence_relay.py` (same scripted
fall-then-blind-span-then-reappear sequence) so the handover this file checks is the same one
that test already proves fires a `possible_fall`, not a second hand-built scenario.
"""
import json
import sqlite3

import numpy as np
from types import SimpleNamespace

from watchverify import jobs, worker
from watchverify.core import Tracker

from test_worker_evidence_relay import run_worker


def test_event_frames_are_recorded_and_joined_to_the_right_event(monkeypatch, tmp_path):
    events, _metrics = run_worker(monkeypatch, tmp_path, relay_enabled=True)
    run_id = jobs.list_jobs()[0]['run_id']
    assert jobs.get_job(run_id)['trace_schema_version'] == 1
    trace = jobs.load_event_trace(run_id, events[0]['event_id'])
    assert trace['kind'] == 'recorded'
    assert trace['rows'], 'the fall event should have at least one recorded frame'
    assert all(row['event_id'] == events[0]['event_id'] for row in trace['rows'])


def test_gate_state_seconds_in_fall_hold_is_recorded(monkeypatch, tmp_path):
    """`down_hold=2.0, fall_hold=0.35` defaults; the geometric rule fires quickly once down."""
    events, _metrics = run_worker(monkeypatch, tmp_path, relay_enabled=True)
    run_id = jobs.list_jobs()[0]['run_id']
    trace = jobs.load_event_trace(run_id, events[0]['event_id'])
    gated = [row for row in trace['rows'] if row['gate_state']]
    assert gated, 'at least one recorded frame should carry a gate snapshot'
    assert all(row['gate_state']['seconds_in_fall_hold'] is not None for row in gated
              if row['gate_state']['low'] is not None)


def test_a_handover_records_the_old_track_id_on_the_accepting_frame(monkeypatch, tmp_path):
    events, metrics = run_worker(monkeypatch, tmp_path, relay_enabled=True)
    assert metrics['carried_fall_evidence'] == 1
    run_id = jobs.list_jobs()[0]['run_id']
    handovers = jobs.load_event_handovers(run_id, events[0]['event_id'])
    assert len(handovers) == 1
    assert handovers[0]['old_track_id'] == 1  # the identity that fell, before the blind span
    assert events[0]['track_id'] == 2  # the identity the alert is attributed to
    assert handovers[0]['reason'] == 'evidence_relay'
