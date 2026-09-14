"""Local video review interface. Start with: streamlit run app.py."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import tempfile

import streamlit as st

from watchverify import jobs, ui
from watchverify.alerts import (ALERT_BRANCH, BRANCH_PROMPT, pending_alerts, primary_branch,
                                secondary_fall_alerts, alerts_for_branch)

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="WatchVerify · Video review", page_icon="◉", layout="wide")
ui.style()
st.markdown("""<style>
.intro {font-size:18px;color:#587164;max-width:780px;line-height:1.6}
.quiet {color:#687e71;font-size:14px}
.step {background:white;border:1px solid #dde7df;border-radius:16px;padding:22px;min-height:175px}
.step b {display:block;margin:12px 0 8px;font-size:17px}
.step span {color:#687e71;font-size:14px;line-height:1.6}
@media (max-width:860px) {.step {min-height:0}}
</style>""", unsafe_allow_html=True)

CATEGORY = {"possible_fall": "Possible fall", "person_down": "Person remains down",
            "unusual_activity": "Unusual movement", "activity": "Unusual movement",
            "activity_candidate": "Unusual movement"}
LABELS = {"unreviewed": "Not reviewed", "relevant": "Relevant", "false_alarm": "False alarm", "unclear": "Unclear"}



def timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "Unknown"
    seconds = max(0, float(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, rest = divmod(rest, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{rest:04.1f}"


def date_text(value) -> str:
    if not value:
        return "Recording date unknown"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat(sep=" ", timespec="seconds")
    except (TypeError, ValueError):
        return str(value)


def model_card(name: str) -> dict:
    """The saved model card, for display text only.

    Deliberately not the checksum-verified `Models` loader: this is used to quote the
    artifact's own measured alert rate on screen, and reading the card avoids loading a
    joblib artifact into every UI rerun. Nothing here decides anything.
    """
    file = ROOT / "models" / f"{name}.json"
    if not file.exists():
        return {}
    try:
        return json.loads(file.read_text())
    except (OSError, ValueError):
        return {}


def show_models():
    pose_ready = any((ROOT / "models" / f"pose_landmarker_{variant}.task").exists() for variant in ("lite", "full"))
    st.caption("POSE ESTIMATION")
    st.write("Ready" if pose_ready else "Not installed · run Setup WatchVerify.command")
    for name, label in (("fall", "Fall classifier"), ("activity", "Activity model")):
        file = ROOT / "models" / f"{name}.json"
        st.caption(label.upper())
        st.write("Model artifact available · experimental" if file.exists() else "No trained model available yet")
        card = model_card(name)
        st.caption(card.get("release_status") or "Release status not recorded; clearance unknown.")
    st.caption("An available model is not evidence of reliable detection. Review all alerts against the video.")


with st.sidebar:
    st.markdown("### ◉ WatchVerify")
    st.caption("LOCAL VIDEO REVIEW")
    st.divider()
    st.markdown("#### Analyse a video")
    uploaded = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv", "webm"], help="Your file stays on this computer. The decoder checks compatibility before analysis.")
    with st.expander("Analysis settings"):
        fps = st.select_slider("Analysis frames per second", options=[5, 10, 15, 20], value=10)
        people = st.number_input("Maximum visible people", min_value=1, max_value=4, value=4)
        variant = st.selectbox("Pose model", ["full", "lite"], format_func=lambda value: "Full · more detail" if value == "full" else "Lite · faster")
        recording_start = st.text_input("Recording start date and time (optional)", placeholder="2026-09-11T14:30:00+10:00", help="Include the UTC offset. Leave blank if the recording date is unknown; analysis time will be labelled separately.")
    all_jobs = jobs.list_jobs()
    busy = any(job["status"] in {"running", "cancelling"} for job in all_jobs)
    asset = ROOT / "models" / f"pose_landmarker_{variant}.task"
    if uploaded is not None:
        st.caption(f"{uploaded.size / 1024**2:.1f} MB · stored locally when analysis starts")
    if not asset.exists():
        st.caption(f"The {variant} pose asset is not installed, so analysis cannot start. "
                   "Quit the app, double-click Setup WatchVerify.command (it downloads the asset "
                   "and verifies it), then launch again.")
    if st.button("Start analysis", type="primary", width="stretch", disabled=uploaded is None or busy or not asset.exists()):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(uploaded.getbuffer())
            identifier = jobs.create_job(temporary, {"analysis_fps": fps, "max_people": int(people),
                                         "pose_variant": variant, "recording_start": recording_start.strip() or None},
                                         original_name=uploaded.name)
            jobs.launch_job(identifier)
            st.session_state["selected_run"] = identifier
            st.rerun()
        except (OSError, ValueError, RuntimeError) as error:
            st.error(str(error))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    examples = sorted((ROOT / "data" / "processed" / "urfall").glob("*.mp4"))
    if examples:
        with st.expander("Try a research sample"):
            sample = st.selectbox("Sample recording", examples, format_func=lambda p: p.stem)
            st.caption("Public research footage. Some samples may be used in model development; this is a demonstration, not an independent accuracy test.")
            if st.button("Analyse sample", disabled=busy or not asset.exists()):
                try:
                    identifier = jobs.create_job(sample, {"analysis_fps": fps, "max_people": int(people), "pose_variant": variant}, original_name=f"Research sample · {sample.name}")
                    jobs.launch_job(identifier)
                    st.session_state["selected_run"] = identifier
                    st.rerun()
                except (OSError, ValueError, RuntimeError) as error:
                    st.error(str(error))
    if busy:
        st.caption("One video is being analysed. You can review previous results while it runs.")
    st.divider()
    st.markdown("#### Your analyses")
    if all_jobs:
        choices = [job["run_id"] for job in all_jobs]
        names = {job["run_id"]: f"{job.get('original_name', job['run_id'])} · {job['status']}" for job in all_jobs}
        if st.session_state.get("selected_run") not in choices:
            st.session_state["selected_run"] = choices[0]
        selected = st.selectbox("Select an analysis", choices, key="selected_run", format_func=lambda identifier: names[identifier], label_visibility="collapsed")
    else:
        selected = None
        st.caption("Completed analyses will appear here.")
    st.divider()
    with st.expander("Model readiness"):
        show_models()
    st.caption("Experimental research prototype. Observations can be missed or mistaken. Posture does not establish injury or criminal intent.")


def downloads(job: dict, events: list[dict]):
    folder = jobs.run_dir(job["run_id"])
    st.markdown("#### Take the results with you")
    available = [("Annotated video", "annotated.mp4", "video/mp4"),
                 ("Event list · JSON", "events.json", "application/json")]
    first, second, third = st.columns(3)
    for column, (label, name, mime) in zip((first, second), available):
        file = folder / name
        with column:
            if file.exists() and file.stat().st_size:
                with file.open("rb") as handle:
                    st.download_button(label, handle, file_name=f"{job['run_id']}-{name}", mime=mime, width="stretch", key=f"download-{job['run_id']}-{name}")
            else:
                st.button(label, disabled=True, width="stretch", key=f"missing-{job['run_id']}-{name}")
    with third:
        # Built on request, not read from disk: it carries the review labels, which are
        # saved after the worker has already written its own copy of the event table.
        if events:
            st.download_button("Event table · CSV", jobs.events_csv(job["run_id"]), file_name=f"{job['run_id']}-events.csv", mime="text/csv", width="stretch", key=f"download-{job['run_id']}-events.csv")
        else:
            st.button("Event table · CSV", disabled=True, width="stretch", key=f"missing-{job['run_id']}-events.csv")
    if events:
        review_data = {"run_id": job["run_id"], "events": events,
                       "reviews": jobs.get_reviews(job["run_id"]),
                       "escalations": jobs.get_escalations(job["run_id"])}
        st.download_button("Events with my review notes", json.dumps(review_data, indent=2), file_name=f"{job['run_id']}-review.json", mime="application/json", key=f"review-download-{job['run_id']}")
    st.caption("Source position is the event’s location in the video. Analysis date/time records when this computer processed it. Annotated exports may be silent.")


def visibility_note(job: dict, status: str) -> None:
    """Say how much of the video could not be observed, beside the events themselves.

    Without this line an empty stretch of timeline reads as "nothing happened". It is the
    difference between a period that was watched and a period that was not.
    """
    summary = job.get("summary") or {}
    blind = summary.get("unobserved_source_s")
    if status in jobs.ACTIVE or blind is None:
        return
    total = float(summary.get("source_duration_s") or 0)
    processed_value = summary.get("source_time_s")
    processed = float(total if processed_value is None else processed_value)
    # A completed encode commonly ends a fraction of a frame before the container's
    # stated duration. Name only a material tail, or one left by an incomplete run.
    unprocessed = summary.get("unprocessed_source_s")
    if unprocessed is None:
        unprocessed = max(total - processed, 0) if total else 0
    unprocessed = float(unprocessed)
    partial = unprocessed > 1.0 or (status != "completed" and unprocessed > 0.05)
    denominator = processed if partial else (total or processed)
    tail = (f" A further {unprocessed:.1f}s of the source was not analysed."
            if partial else "")
    if summary.get("analysed_frames") == 0 or processed == 0:
        st.warning("Visibility: no analysed time coverage is available for this run."
                   f"{tail} An empty timeline does not establish that the footage was clear.")
        return
    if not blind:
        st.caption(f"Visibility: a person was observable throughout the {timestamp(processed)} that was analysed."
                   f"{tail} This says nothing about whether an event was correctly judged.")
        return
    share = f" ({blind / denominator:.0%} of the analysed portion)" if denominator else ""
    st.warning(f"Visibility: no usable body position was available for {blind:.1f}s{share}, across {summary.get('unobserved_intervals', 1)} interval(s). Those periods are unknown, not clear — nothing could have been detected in them.{tail}")


def alert_caveat(branch: str, run_id: str) -> str:
    """Quote this run's loaded model, never a replacement card from another run."""
    prompt = BRANCH_PROMPT[branch]
    card = jobs.get_job(run_id).get("model_disclosures", {}).get(prompt["model"], {})
    return card.get("alert_caveat") or (
        "No model-specific evaluation disclosure was saved for this analysis. "
        + prompt["fallback"])


def record_escalation(run_id: str, alerts: list[dict], action: str, note: str, caveat: str):
    try:
        jobs.save_escalation(run_id, [event["event_id"] for event in alerts], action, note, caveat)
    except (ValueError, OSError) as error:
        st.error(str(error))
        return
    st.session_state.pop(f"alert-open-{run_id}", None)
    st.rerun()


def escalation_body(run_id: str, alerts: list[dict], branch: str):
    """Shared body for the dialog and its inline fallback."""
    prompt = BRANCH_PROMPT[branch]
    caveat = alert_caveat(branch, run_id)
    st.markdown(f"**{prompt['heading']}**")
    for event in alerts:
        label = CATEGORY.get(event.get("category"), event.get("category", "Observation"))
        st.write(f"• **{label}** at {timestamp(event.get('source_start_s'))} · Person {event.get('track_id', '?')}"
                 + (f" · score {float(event['score']):.2f}" if isinstance(event.get("score"), (int, float)) else ""))
        for observation in event.get("observations", []):
            st.caption("　" + str(observation).replace("_", " ").capitalize())
    st.warning(caveat)
    st.caption("This application does not contact anyone. It asks you, records your answer, and keeps the original alert unchanged.")
    note = st.text_area("Notes (optional)", key=f"alert-note-{run_id}", max_chars=2000)
    primary_alerts = alerts_for_branch(alerts, branch)
    call_action, call_label = prompt["call"]
    decline_action, decline_label = prompt["decline"]
    first, second, third = st.columns(3)
    if first.button(call_label, type="primary", width="stretch", key=f"alert-call-{run_id}"):
        record_escalation(run_id, primary_alerts, call_action, note, caveat)
    if second.button(decline_label, width="stretch", key=f"alert-decline-{run_id}"):
        record_escalation(run_id, primary_alerts, decline_action, note, caveat)
    if third.button("Decide later", width="stretch", key=f"alert-defer-{run_id}"):
        record_escalation(run_id, alerts, "deferred", note, caveat)
    # Both branches waiting: the police prompt is primary, but the medical action must
    # still be reachable from the same card without dismissing this one.
    down = secondary_fall_alerts(alerts, branch)
    if down:
        st.divider()
        st.warning(alert_caveat("fall", run_id))
        st.caption(f"A possible person-down observation is also waiting ({timestamp(down[0].get('source_start_s'))}).")
        if st.button("Call ambulance for the person down", key=f"alert-ambulance-{run_id}"):
            record_escalation(run_id, down, "ambulance_called", note, alert_caveat("fall", run_id))


@st.dialog("Alert · your decision")
def escalation_dialog(run_id: str, alerts: list[dict], branch: str):
    escalation_body(run_id, alerts, branch)


def escalation_prompt(run_id: str, events: list[dict], escalations: list[dict]):
    """Banner plus a one-time pop-up for observations awaiting a decision."""
    alerts = pending_alerts(events, escalations)
    branch = primary_branch(alerts)
    if branch is None:
        return
    heading = BRANCH_PROMPT[branch]["heading"]
    st.error(f"**{heading}** — {len(alerts)} observation{'s' if len(alerts) > 1 else ''} awaiting your decision.")
    signature = tuple(sorted(event["event_id"] for event in alerts))
    shown = st.session_state.get(f"alert-shown-{run_id}")
    # Pop up once per new set of alerts. Re-opening it on every two-second refresh would
    # fight whoever is trying to read the footage.
    if shown != signature:
        st.session_state[f"alert-shown-{run_id}"] = signature
        st.session_state[f"alert-open-{run_id}"] = True
    if st.session_state.get(f"alert-open-{run_id}"):
        st.session_state.pop(f"alert-open-{run_id}", None)
        escalation_dialog(run_id, alerts, branch)
    elif st.button("Review this alert", type="primary", key=f"alert-reopen-{run_id}"):
        st.session_state[f"alert-open-{run_id}"] = True
        st.rerun()


def escalation_history(run_id: str, escalations: list[dict]):
    if not escalations:
        return
    st.markdown("#### Escalation decisions")
    for record in escalations:
        st.write(f"**{record['action'].replace('_', ' ').capitalize()}** · {date_text(record['decided_at_utc'])}")
        st.caption(f"{len(record['event_ids'])} observation(s)" + (f" · {record['note']}" if record["note"] else ""))
    st.caption("This log is append-only. A later decision is added; it does not replace an earlier one.")


@st.fragment(run_every="2s" if any(job["run_id"] == selected and job["status"] in jobs.ACTIVE for job in all_jobs) else None)
def analysis_view(run_id: str):
    job = jobs.get_job(run_id)
    folder = jobs.run_dir(run_id)
    status = job["status"]
    previous_status = next((item["status"] for item in all_jobs if item["run_id"] == run_id), None)
    if previous_status != status:
        # Stop polling completed results and refresh sidebar availability once.
        st.rerun()
    st.markdown('<div class="eyebrow">YOUR ANALYSIS</div>', unsafe_allow_html=True)
    st.title(job.get("original_name", "Video review"))
    st.caption(f"{date_text(job.get('config', {}).get('recording_start'))}  ·  Analysis started {date_text(job.get('started_at_utc', job.get('created_at_utc')))}")
    if status in {"queued", "running", "cancelling"}:
        message = "Stopping safely…" if status == "cancelling" else "Analysing movement and building the event timeline…"
        progress = min(1.0, max(0.0, float(job.get("progress") or 0)))
        st.progress(progress, text=f"{message} {progress:.0%}")
        if status == "queued":
            if st.button("Resume queued analysis", key=f"launch-{run_id}"):
                try:
                    jobs.launch_job(run_id)
                    st.rerun()
                except (RuntimeError, ValueError) as error:
                    st.error(str(error))
        if st.button("Cancel analysis", disabled=status == "cancelling", key=f"cancel-{run_id}"):
            jobs.cancel_job(run_id)
            st.rerun()
        st.caption("Results below are provisional while processing continues. Ending the video does not mean a person recovered.")
    elif status == "completed":
        st.success("Analysis complete. Review the footage to check each observation.")
    elif status in {"failed", "interrupted"}:
        st.error(job.get("error") or "The analysis could not finish.")
    elif status == "cancelled":
        st.info("Analysis cancelled. Any saved results cover only the processed part of the video.")
    model_status = job.get("model_status", {})
    missing_branches = [f"{'Fall classifier' if name == 'fall' else 'Activity model'}: {description}"
                        for name, description in model_status.items()
                        if "not trained" in str(description).lower() or "unavailable" in str(description).lower()]
    if missing_branches:
        st.warning(" · ".join(missing_branches))
    if status not in jobs.ACTIVE and job.get("summary", {}).get("pose_coverage") == 0:
        st.warning("No usable body positions were recorded in this analysis. An empty event list cannot be interpreted as an absence of falls or unusual activity.")
    visibility_note(job, status)
    events = jobs.load_events(run_id)
    reviews = jobs.get_reviews(run_id)
    escalations = jobs.get_escalations(run_id)
    # Raised before the metrics so an alert cannot be scrolled past, and inside the polling
    # fragment so it appears while the analysis is still running.
    escalation_prompt(run_id, events, escalations)
    columns = st.columns(4)
    columns[0].metric("Observations to review", len(events))
    columns[1].metric("Reviewed", sum(review["label"] != "unreviewed" for review in reviews.values()))
    columns[2].metric("Escalation decisions", len(escalations))
    columns[3].metric("Analysis status", status.capitalize())
    st.write("")
    left, right = st.columns([1.7, 1], gap="large")
    selected_event = None
    with right:
        st.markdown("#### Event timeline")
        if events:
            event_map = {event["event_id"]: event for event in events}
            event_id = st.selectbox("Jump to an observation", list(event_map),
                                    format_func=lambda identifier: f"{timestamp(event_map[identifier].get('source_start_s'))} · {CATEGORY.get(event_map[identifier].get('category'), event_map[identifier].get('category', 'Observation'))} · Person {event_map[identifier].get('track_id', '?')}",
                                    key=f"event-{run_id}")
            selected_event = event_map[event_id]
            st.markdown(f"**{CATEGORY.get(selected_event.get('category'), selected_event.get('category', 'Observation'))}**")
            for observation in selected_event.get("observations", []):
                st.write(f"• {str(observation).replace(chr(95), chr(32)).capitalize()}")
            st.caption(f"First alert: {timestamp(selected_event.get('emitted_source_s'))} in video")
            st.caption(f"Occurrence: {date_text(selected_event.get('occurred_at_utc'))}")
            outcome = selected_event.get("status", "unknown")
            reason = str(selected_event.get("reason") or "").replace("_", " ")
            st.caption(f"Outcome: {outcome}{' · ' + reason if reason else ''}")
            existing = reviews.get(event_id, {"label": "unreviewed", "note": ""})
            with st.form(f"review-{run_id}-{event_id}"):
                label = st.selectbox("Your review", list(LABELS), index=list(LABELS).index(existing.get("label", "unreviewed")), format_func=lambda value: LABELS[value])
                note = st.text_area("Notes (optional)", value=existing.get("note", ""), max_chars=2000)
                if st.form_submit_button("Save review", width="stretch"):
                    jobs.save_review(run_id, event_id, label, note)
                    st.rerun()  # Refresh counts and exports from the saved review.
            with st.expander("Observation details"):
                st.json(selected_event)
        else:
            st.info("No observations have been recorded." if status in jobs.ACTIVE else "No observations were emitted. This does not establish that the video is safe — check the visibility note above for the time that could not be observed at all.")
        st.caption("These are prompts for human review, not diagnoses or findings of wrongdoing.")
        escalation_history(run_id, escalations)
    with left:
        st.markdown("#### Review the footage")
        annotated = folder / "annotated.mp4"
        video = annotated if status == "completed" and annotated.exists() and annotated.stat().st_size else Path(job["source_path"])
        start = max(0, float((selected_event or {}).get("source_start_s", 0)) - 3)
        # Streamlit receives a local path; selecting another event seeks to its evidence.
        if video.exists():
            st.video(str(video), start_time=start)
            st.caption(("Annotated result" if video == annotated else "Original recording") + (f" · starting three seconds before the observation ({timestamp(start)})" if selected_event else ""))
        else:
            st.warning("The source video is no longer available on disk.")
        if selected_event:
            st.caption(f"Observation interval: {timestamp(selected_event.get('source_start_s'))} – {timestamp(selected_event.get('source_end_s'))}. An earlier estimated onset does not mean the alert was emitted then.")
        if job.get("summary"):
            with st.expander("Processing summary and visibility"):
                st.json(job["summary"])
        if (folder / "metrics.json").exists():
            with st.expander("Run measurements"):
                try:
                    st.json(json.loads((folder / "metrics.json").read_text()))
                except (ValueError, OSError):
                    st.caption("Measurements are still being saved.")
    st.divider()
    if status not in jobs.ACTIVE:
        downloads(job, events)
        with st.expander("Manage this analysis"):
            st.caption("Deleting removes this analysis, its uploaded copy, exports and review notes. Your original file is unaffected.")
            confirm = st.checkbox("Delete this analysis and its local files", key=f"confirm-delete-{run_id}")
            if st.button("Delete analysis", disabled=not confirm, key=f"delete-{run_id}"):
                jobs.delete_job(run_id)
                st.session_state.pop("selected_run", None)
                st.rerun()
    if (folder / "worker.log").exists() and status in {"failed", "interrupted"}:
        with st.expander("What went wrong"):
            st.code((folder / "worker.log").read_text(errors="replace")[-6000:])


if selected:
    analysis_view(selected)
else:
    st.markdown('<div class="eyebrow">SEE THE MOMENT. REVIEW THE CONTEXT.</div>', unsafe_allow_html=True)
    st.title("A clearer view of\nwhat needs attention.")
    st.markdown('<p class="intro">Turn a video into a focused review of possible falls, people remaining down, and unusual movement—with the footage beside every observation.</p>', unsafe_allow_html=True)
    st.write("")
    for column, number, title, description in zip(st.columns(3), ["01", "02", "03"],
            ["Choose a recording", "Follow the movement", "Review the evidence"],
            ["Select a video in the sidebar. It stays on your computer, with its recording time kept separate from the analysis time.",
             "Pose estimates and past-only movement features produce observations. Unreliable visibility remains a limitation.",
             "Seek to an event, add your judgement, and download the annotated footage and event list."]):
        with column:
            st.markdown(f'<div class="step"><span>{number}</span><b>{title}</b><span>{description}</span></div>', unsafe_allow_html=True)
    st.write("")
    st.info("Start with a short, fixed-camera recording where people are clearly visible. This is an experimental prototype; its accuracy on your camera has not been established.")
    with st.expander("What the first version looks for"):
        st.write("**Possible fall:** a rapid posture change supported by a low or horizontal position.")
        st.write("**Person remains down:** a sustained low or horizontal posture, including when the fall itself is outside the recording.")
        st.write("**Unusual movement:** an experimental activity score when a trained model is available. Movement alone cannot establish theft or injury.")
        st.write("**Escalation prompts:** when an observation is raised, the app asks you whether to call police or an ambulance, shows how often that alert fires on ordinary footage, and records your answer. It never contacts anyone itself.")
        st.write("Camera input is a later stage. This interface currently analyses uploaded recordings.")
