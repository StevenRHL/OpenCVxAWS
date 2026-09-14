"""Point a camera at the room and see whether the analysis can use what it sees.

A preview, deliberately and only. Nothing is written to disk, no analysis is created, and
nothing reaches the administrator's queue — the observations here disappear when the
session stops. What it is for is the question no uploaded file can answer: whether this
camera, at this height and angle and in this light, produces usable body pose at all.
An empty timeline from a camera that never resolved a body is not evidence of a quiet room.

The detection stack is the one a recorded analysis uses, so what fires here is what would
fire there.
"""
from __future__ import annotations

import time

import streamlit as st

from watchverify import ui
from watchverify.live import LiveSession, available_cameras

st.set_page_config(page_title="WatchVerify · Live camera", page_icon="◉", layout="wide")
ui.style()

CATEGORY = {"possible_fall": "Possible fall", "person_down": "Person remains down",
            "unusual_activity": "Unusual movement"}

st.markdown('<div class="eyebrow">LIVE CAMERA</div>', unsafe_allow_html=True)
st.title("Check what the camera can see")

with st.sidebar:
    # No heading: the page nav directly above already says which page this is.
    st.caption("PREVIEW · NOTHING IS RECORDED")
    if st.button("Find cameras", width="stretch"):
        st.session_state["cameras"] = available_cameras()
    cameras = st.session_state.get("cameras")
    if cameras is None:
        st.caption("Select **Find cameras** to look for an attached camera. macOS asks for "
                   "camera permission the first time.")
        camera = st.number_input("Camera index", min_value=0, max_value=8, value=0)
    elif not cameras:
        st.error("No usable camera was found. It may be in use by another application, or "
                 "camera permission may have been declined in System Settings → Privacy.")
        camera = st.number_input("Camera index", min_value=0, max_value=8, value=0)
    else:
        camera = st.selectbox("Camera", cameras, format_func=lambda index: f"Camera {index}")
    fps = st.select_slider("Analysis frames per second", options=[5, 10, 15, 20], value=10)
    people = st.number_input("Maximum visible people", min_value=1, max_value=4, value=4)
    variant = st.selectbox("Pose model", ["lite", "full"],
                           format_func=lambda v: "Lite · faster" if v == "lite" else "Full · more detail")
    mirror = st.checkbox("Mirror the picture", value=True,
                         help="Mirrors the display only. It does not change the analysis.")
    minutes = st.slider("Stop automatically after (minutes)", 1, 30, 5)
    st.divider()
    st.caption("A preview cannot establish that this camera is reliable. It shows whether "
               "body pose resolves at all, which is the precondition for everything else.")

running = st.session_state.get("live_running", False)
controls = st.container(key="decisionrow")
start, stop = controls.columns(2)
if start.button("Start camera", type="primary", width="stretch", disabled=running):
    st.session_state["live_running"] = True
    st.rerun()
if stop.button("Stop camera", width="stretch", disabled=not running):
    st.session_state["live_running"] = False
    st.rerun()

st.info("Preview only. Nothing here is saved, and nothing here reaches the admin dashboard. "
        "To raise an incident for review, analyse a recording on the main page.")

if not st.session_state.get("live_running"):
    st.caption("The camera is off.")
    explain = st.container(key="mainsplit")
    what, limits = explain.columns(2, gap="large")
    with what:
        st.markdown("#### What this is for")
        ui.facts([("Body pose resolves at all", "the question this answers"),
                  ("Detection stack", "the same one a recorded analysis uses"),
                  ("Analysis rate", f"{fps} frames per second"),
                  ("People tracked", str(people))])
        st.caption("Point the camera where you would actually mount it. A model cannot be "
                   "judged on footage it never resolved a body in.")
    with limits:
        st.markdown("#### What it cannot tell you")
        st.markdown('<div class="seen">• Whether this camera is reliable</div>'
                    '<div class="seen">• Whether an alert here would be correct</div>'
                    '<div class="seen">• That a quiet preview means a quiet room</div>',
                    unsafe_allow_html=True)
        st.caption("Nothing is recorded and nothing reaches the admin dashboard. To raise "
                   "an incident for review, analyse a recording on the main page.")
    st.stop()

picture = st.empty()
status = st.empty()
observations = st.empty()
seen: list[str] = []

try:
    with LiveSession(camera=int(camera), analysis_fps=float(fps), max_people=int(people),
                     pose_variant=variant, mirror=bool(mirror)) as session:
        for warning in session.warnings:
            st.warning(warning)
        deadline = time.perf_counter() + minutes * 60
        while st.session_state.get("live_running") and time.perf_counter() < deadline:
            frame = session.read()
            if not frame["ok"]:
                if frame["fatal"]:
                    st.error(frame["message"])
                    break
                continue
            picture.image(frame["frame"], channels="BGR", width="stretch")
            active = frame["active"]
            status.markdown(
                f"**{frame['people']}** person(s) tracked · **{session.frames_analysed}** frames "
                f"analysed · **{len(active)}** observation(s) active · running "
                f"{frame['t']:.0f}s")
            for candidate in frame["candidates"]:
                if candidate["category"] in CATEGORY:
                    seen.append(f"{frame['t']:.1f}s · {CATEGORY[candidate['category']]} · "
                                f"Person {candidate.get('track_id', '?')}")
            if seen:
                observations.warning("Observed in this session (not saved):\n\n"
                                     + "\n\n".join(f"• {line}" for line in seen[-8:]))
        else:
            if st.session_state.get("live_running"):
                st.info(f"The preview stopped automatically after {minutes} minute(s).")
                st.session_state["live_running"] = False
except (RuntimeError, ValueError, OSError, FileNotFoundError) as error:
    st.session_state["live_running"] = False
    st.error(str(error))
