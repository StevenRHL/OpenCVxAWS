"""Exercise the actual decision-card functions with Streamlit and isolated storage."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest
from watchverify import jobs


@pytest.mark.parametrize("label,action,remaining", [
    ("Call police", "police_called", "medical"),
    ("Do not call", "police_not_called", "medical"),
    ("Call ambulance for the person down", "ambulance_called", "retail"),
])
def test_mixed_card_buttons_keep_other_decision_pending(tmp_path, monkeypatch, label, action, remaining):
    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    monkeypatch.setattr(jobs.shutil, "disk_usage", lambda path: SimpleNamespace(free=50 * 1024**3))
    source = tmp_path / "original.mp4"
    source.write_bytes(b"fixture; inference is outside this decision-card test")
    run_id = jobs.create_job(source)
    events = [
        {"event_id": "medical", "category": "person_down", "track_id": 1, "source_start_s": 1},
        {"event_id": "retail", "category": "unusual_activity", "track_id": 2, "source_start_s": 2},
    ]
    (jobs.run_dir(run_id) / "events.json").write_text(json.dumps(events))
    # Use the actual app functions, without launching workers or touching saved user runs.
    tree = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text())
    names = {"alert_caveat", "record_escalation", "escalation_body"}
    functions = "\n\n".join(ast.unparse(n) for n in tree.body
                            if isinstance(n, ast.FunctionDef) and n.name in names)
    script = '''import streamlit as st
from watchverify import jobs
from watchverify.alerts import *
CATEGORY = {}
timestamp = str
def model_card(name):
    return {}
''' + functions + f'''
run_id = {run_id!r}
events = jobs.load_events(run_id)
pending = pending_alerts(events, jobs.get_escalations(run_id))
branch = primary_branch(pending)
if branch:
    escalation_body(run_id, pending, branch)
st.write("Pending: " + ",".join(e["event_id"] for e in pending))
'''
    import copy
    from watchverify.models import Models
    snapshot = Models().disclosures()
    assert set(snapshot) == {"fall", "activity"}
    jobs.update_job(run_id, model_disclosures=copy.deepcopy(snapshot))
    expected = snapshot["fall" if action.startswith("ambulance") else "activity"]["alert_caveat"]
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert any(w.value == snapshot["activity"]["alert_caveat"] for w in app.warning)
    assert any(w.value == snapshot["fall"]["alert_caveat"] for w in app.warning)
    next(b for b in app.button if b.label == label).click().run()
    assert not app.exception
    records = jobs.get_escalations(run_id)
    assert len(records) == 1
    assert records[0]["action"] == action
    assert records[0]["shown_rate"] == expected
    assert records[0]["event_ids"] == (["retail"] if remaining == "medical" else ["medical"])
    assert any(m.value == f"Pending: {remaining}" for m in app.markdown)
