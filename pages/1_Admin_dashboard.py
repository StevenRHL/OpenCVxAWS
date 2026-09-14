"""One queue of everything awaiting a decision, across every analysis.

The per-analysis view answers "what did this recording contain". This answers the question
an administrator actually has, which is "what needs me, and can I see it". Each row opens
on a clip cut to the incident itself rather than the start of the footage, so judging an
observation does not begin with scrubbing.

The screen is deliberately short. An administrator deciding whether to call an ambulance
needs the footage, what was seen, and how often this alert is wrong — everything else is
provenance, and provenance belongs behind a disclosure rather than between them and the
decision.

It contacts nobody. Deciding here writes to the same append-only escalation log the
per-analysis prompt writes to, so the two views can never disagree about what was decided,
and a decision recorded in either is recorded once.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from watchverify import incidents, jobs, ui
from watchverify.alerts import BRANCH_PROMPT
from watchverify.ui import date_text, facts, timestamp

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="WatchVerify · Admin dashboard", page_icon="◉", layout="wide")
ui.style()


def row_label(row: dict) -> str:
    mark = "●" if not row["decided"] else "○"
    when = date_text(row.get("occurred_at_utc")) if row.get("occurred_at_utc") else timestamp(row.get("source_start_s"))
    return f"{mark} {row['kind']} · {row['label']} · {when} · {row['recording'][:34]}"


st.markdown('<div class="eyebrow">ADMIN DASHBOARD</div>', unsafe_allow_html=True)
st.title("What needs attention")

rows = incidents.collect()
summary = incidents.counts(rows)

with st.container(key="metricrow"):
    columns = st.columns(4)
    columns[0].metric("Awaiting a decision", summary["waiting"])
    columns[1].metric("Medical", summary["medical"])
    columns[2].metric("Security", summary["security"])
    columns[3].metric("Recordings", summary["recordings"])

if not rows:
    st.info("No observations have been raised yet. Analyse a recording on the main page, or "
            "open Live camera to check whether a camera produces usable body pose at all.")
    st.caption("An empty queue does not establish that nothing happened.")
    st.stop()

with st.sidebar:
    # No heading here: the page nav directly above already says which page this is.
    st.caption("SHOW")
    show = st.radio("Show", ["Awaiting a decision", "Everything"], index=0,
                    label_visibility="collapsed")
    kinds = st.multiselect("Kind", ["Medical", "Security"], default=["Medical", "Security"],
                           label_visibility="collapsed",
                           placeholder="No kinds selected")
    st.divider()
    st.caption("CLIP LENGTH")
    lead = st.slider("Seconds before the event", 0, 20, int(incidents.LEAD_S))
    tail = st.slider("Seconds after the event", 0, 20, int(incidents.TAIL_S))
    st.caption("A clip that starts on the alert shows the consequence, not the cause.")

visible = [row for row in rows
           if (show == "Everything" or not row["decided"]) and row["kind"] in kinds]
if not visible:
    st.success("Nothing is waiting under this filter.")
    st.stop()

keys = {f"{row['run_id']}::{row['event_id']}": row for row in visible}
chosen = st.selectbox("Incident", list(keys), format_func=lambda key: row_label(keys[key]),
                      label_visibility="collapsed")
row = keys[chosen]
job = jobs.get_job(row["run_id"])
duration = (job.get("media") or {}).get("duration_s")

branch = row["branch"]
prompt = BRANCH_PROMPT.get(branch)
disclosure = job.get("model_disclosures", {}).get(prompt["model"], {}) if prompt else {}
caveat = disclosure.get("alert_caveat") or (
    "No model-specific evaluation disclosure was saved for this analysis. "
    + (prompt["fallback"] if prompt else ""))


def record(action: str):
    try:
        jobs.save_escalation(row["run_id"], [row["event_id"]], action, note, caveat)
    except (ValueError, OSError) as error:
        st.error(str(error))
        return
    st.rerun()


# The decision sits under the footage it is about, not beside it. That is both the right
# reading order — watch, then decide — and what keeps the two columns roughly the same
# height instead of leaving a column of nothing under a short clip.
main = st.container(key="mainsplit")
left, right = main.columns([1.5, 1], gap="large")

with left:
    clip = incidents.build_clip(row["run_id"], row["event"], lead_s=float(lead), tail_s=float(tail))
    if clip is not None and clip.exists() and clip.stat().st_size:
        st.video(str(clip))
        start, end = incidents.clip_bounds(row["event"], duration, float(lead), float(tail))
        st.caption(f"{timestamp(start)} – {timestamp(end)} of “{row['recording']}”. "
                   f"The observation itself runs {timestamp(row.get('source_start_s'))} – "
                   f"{timestamp(row.get('source_end_s'))}. Clips are silent; the overlay shows "
                   f"what the analysis saw, not what happened.")
        with clip.open("rb") as handle:
            st.download_button("Download this clip", handle,
                               file_name=f"{row['run_id']}-{row['event_id']}.mp4",
                               mime="video/mp4", width="stretch", key=f"clip-{chosen}")
    else:
        st.warning("This incident could not be cut from its recording. The analysis may still "
                   "be running, or its source file may no longer be on disk.")

    st.divider()
    if row["decided"]:
        st.success("A decision is already recorded. Deciding again adds a row; the log is "
                   "append-only and never replaces an earlier decision.")
    st.markdown(f"**{prompt['heading']}**" if prompt else "**Your decision**")
    note = st.text_area("Notes (optional)", key=f"note-{chosen}", max_chars=2000,
                        height=68, placeholder="What you saw, or why you decided this")
    if prompt:
        call_action, call_label = prompt["call"]
        decline_action, decline_label = prompt["decline"]
        buttons = st.container(key="decisionrow")
        first, second, third = buttons.columns(3)
        with first:
            if st.button(call_label, type="primary", width="stretch", key=f"call-{chosen}"):
                record(call_action)
        with second:
            if st.button(decline_label, width="stretch", key=f"decline-{chosen}"):
                record(decline_action)
        with third:
            if st.button("Decide later", width="stretch", key=f"defer-{chosen}"):
                record("deferred")
    st.caption("This application contacts nobody. It records your answer and leaves the "
               "observation unchanged.")

with right:
    tag = "tag-med" if row["branch"] == "fall" else "tag-sec"
    st.markdown(f'<span class="tag {tag}">{row["kind"].upper()}</span>', unsafe_allow_html=True)
    st.markdown(f"#### {row['label']}")
    facts([("In the video", timestamp(row.get("source_start_s"))),
           ("Occurred", date_text(row["occurred_at_utc"]) if row.get("occurred_at_utc") else "Time not supplied"),
           ("Recording", row["recording"]),
           ("Person", row.get("track_id", "?"))])

    st.markdown("**What was observed**")
    for observation in row["observations"]:
        st.markdown(f'<div class="seen">• {str(observation).replace("_", " ").capitalize()}</div>',
                    unsafe_allow_html=True)
    if "evidence_carried_across_identity_change" in row["observations"]:
        st.caption("The two halves of this evidence were seen under different tracked "
                   "identities, either side of a moment when body pose was lost. Expect a "
                   "visible break in the clip between the movement and the landing.")

    with st.expander("How often this alert is wrong, and other detail"):
        st.warning(caveat)
        facts([("Outcome", f"{row.get('status', 'unknown')} · "
                           f"{str(row.get('reason') or '').replace('_', ' ') or '—'}"),
               ("Analysed", date_text(row.get("analysed_at_utc"))),
               ("Ends at", timestamp(row.get("source_end_s"))),
               ("Score", f"{float(row['score']):.2f}" if isinstance(row.get("score"), (int, float)) else "—"),
               ("Event", row["event_id"])])
        st.caption(row["meaning"])
        st.json(row["event"], expanded=False)

    history = [entry for entry in jobs.get_escalations(row["run_id"])
               if row["event_id"] in entry["event_ids"]]
    if history:
        with st.expander(f"Decisions recorded ({len(history)})"):
            for entry in history:
                st.write(f"**{entry['action'].replace('_', ' ').capitalize()}** · "
                         f"{date_text(entry['decided_at_utc'])}")
                if entry["note"]:
                    st.caption(entry["note"])

with st.expander(f"Everything in the queue ({len(rows)})"):
    st.dataframe([{"Kind": item["kind"], "Observation": item["label"],
                   "In video": timestamp(item.get("source_start_s")),
                   "Occurred": date_text(item.get("occurred_at_utc")) if item.get("occurred_at_utc") else "—",
                   "Recording": item["recording"],
                   "Decided": "yes" if item["decided"] else "no"} for item in rows],
                 width="stretch", hide_index=True)

st.caption("These are prompts for human review, not diagnoses or findings of wrongdoing. An "
           "empty queue does not establish that nothing happened — check each analysis for "
           "the time its camera could not observe at all.")
