"""Per-event debugger: what actually triggered an alert, not a re-explanation of it.

Explanation hierarchy, poorest to richest evidence: plain-language saved observations
first, then branch/score type, then a structured decision trace (added in a later UI
milestone). Runs analysed before a trace existed say so — "Not recorded for this run" —
rather than reconstructing one from today's model and settings.
"""
from __future__ import annotations

import json
from pathlib import Path
import uuid

import streamlit as st

from watchverify import jobs, review
from watchverify.alerts import ALERT_BRANCH
from watchverify.dashboard import CATEGORIES, REVIEW_LABELS


def open_debugger(run_id=None, event_id=None):
    st.session_state['page'] = 'Debugger'
    st.session_state['upload_view'] = False
    if run_id:
        st.session_state['debugger_run'] = run_id
    if event_id:
        st.session_state['debugger_event'] = event_id
    elif run_id:
        st.session_state.pop('debugger_event', None)


def _label(event):
    return CATEGORIES.get(event.get('category'), event.get('category', 'Observation'))


def _action_key(prefix, run_id, event_id):
    """A fresh key per genuinely new action, reused across reruns of the same in-flight
    submission — the duplicate-click / fragment-rerun guard `review.record_action` checks.
    """
    state_key = f"{prefix}-key-{run_id}-{event_id}"
    return st.session_state.setdefault(state_key, uuid.uuid4().hex), state_key


def _ignore_controls(run_id, event_id):
    state = review.current_state(run_id, event_id)
    if state['ignored']:
        if st.button('Undo ignore', key=f"debugger-undo-{run_id}-{event_id}"):
            key, state_key = _action_key('undo', run_id, event_id)
            review.record_action(run_id, event_id, 'undo',
                                 previous_action_id=state['last_action']['action_id'],
                                 idempotency_key=key)
            st.session_state.pop(state_key, None)
            st.rerun()
        st.caption('Ignored: removed from the default review queue, not judged false.')
    else:
        with st.popover('Ignore', width='stretch'):
            with st.form(f"debugger-ignore-{run_id}-{event_id}"):
                reason = st.text_input('Reason (optional)')
                if st.form_submit_button('Ignore this observation'):
                    key, state_key = _action_key('ignore', run_id, event_id)
                    review.record_action(run_id, event_id, 'ignore', note=reason, idempotency_key=key)
                    st.session_state.pop(state_key, None)
                    st.rerun()


def _add_to_learning(run_id, event):
    event_id = event['event_id']
    pending_key = f"learn-pending-{run_id}-{event_id}"
    with st.popover('Add to learning', width='stretch'):
        default_start = float(event.get('source_start_s') or 0)
        default_end = float(event.get('source_end_s') or default_start + 1)
        with st.form(f"debugger-learn-{run_id}-{event_id}"):
            label = st.selectbox('Proposed label', sorted(review.PROPOSED_LABELS))
            visible = st.text_input('Visible action')
            person = st.text_input('Person, if identifiable (optional)')
            note = st.text_area('Note (optional)')
            start = st.number_input('Interval start (s)', value=default_start, min_value=0.0)
            end = st.number_input('Interval end (s)', value=default_end, min_value=0.0)
            submitted = st.form_submit_button('Save to learning queue')
        if submitted:
            key, state_key = _action_key('learn', run_id, event_id)
            try:
                review.create_candidate(run_id, event_id, start, end, label,
                                        visible_action_label=visible, person_if_identifiable=person,
                                        note=note, idempotency_key=key)
                st.session_state.pop(state_key, None)
                st.success('Saved to learning queue. The current model has not changed.')
            except review.DuplicateCandidateWarning as warning:
                st.session_state[pending_key] = dict(start=start, end=end, label=label,
                                                     visible=visible, person=person, note=note, key=key)
                st.warning(f"{len(warning.overlaps)} existing candidate(s) overlap this interval.")
            except ValueError as error:
                st.error(str(error))
        pending = st.session_state.get(pending_key)
        if pending and st.button('Create anyway', key=f"debugger-learn-confirm-{run_id}-{event_id}"):
            review.create_candidate(run_id, event_id, pending['start'], pending['end'], pending['label'],
                                    visible_action_label=pending['visible'],
                                    person_if_identifiable=pending['person'], note=pending['note'],
                                    idempotency_key=pending['key'], confirmed_duplicate=True)
            st.session_state.pop(pending_key, None)
            st.success('Saved to learning queue. The current model has not changed.')


def _revision_history(run_id, event_id, timestamp, date_text):
    revisions = jobs.load_event_revisions(run_id, event_id)
    if not revisions:
        st.info('No revision history is available for this observation.')
        return
    first, latest = revisions[0], revisions[-1]
    st.markdown('**At first alert** · the immutable initial record')
    st.caption(f"Recorded {date_text(first.get('created_at_utc'))} · "
               f"video {timestamp(first.get('source_start_s'))}–{timestamp(first.get('source_end_s'))} · "
               f"status {first.get('status', 'unknown')}")
    for observation in first.get('observations', []):
        st.caption('　' + str(observation).replace('_', ' ').capitalize())
    if len(revisions) > 1:
        st.markdown('**Latest evidence** · retrospective, may include later revisions')
        st.caption(f"Recorded {date_text(latest.get('created_at_utc'))} · "
                   f"video {timestamp(latest.get('source_start_s'))}–{timestamp(latest.get('source_end_s'))} · "
                   f"status {latest.get('status', 'unknown')}")
        for observation in latest.get('observations', []):
            st.caption('　' + str(observation).replace('_', ' ').capitalize())
        with st.expander(f"All {len(revisions)} revisions"):
            for revision in revisions:
                st.write(f"Revision {revision.get('revision', '?')} · {date_text(revision.get('created_at_utc'))} "
                         f"· {revision.get('reason') or revision.get('status', 'unknown')}")
    else:
        st.caption('No later revision exists: the first alert is also the latest evidence.')


def _evidence_timeline(run_id, event, timestamp, date_text):
    revisions = jobs.load_event_revisions(run_id, event['event_id'])
    if not revisions:
        st.info('No evidence timeline is available for this observation.')
        return
    for revision in revisions:
        st.write(f"**Revision {revision.get('revision', '?')}** · {date_text(revision.get('created_at_utc'))} "
                 f"· video {timestamp(revision.get('source_start_s'))}–{timestamp(revision.get('source_end_s'))}")
        st.caption(f"Status: {revision.get('status', 'unknown')}"
                   + (f" · {str(revision.get('reason')).replace('_', ' ')}" if revision.get('reason') else ''))
    try:
        metrics_file = jobs.run_dir(run_id) / 'metrics.json'
        metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
    except (OSError, ValueError):
        metrics = {}
    gaps = [gap for gap in metrics.get('unobserved_intervals', [])
            if isinstance(gap, dict) and gap.get('end_s', 0) >= event.get('source_start_s', 0)
            and gap.get('start_s', 0) <= event.get('source_end_s', event.get('source_start_s', 0))]
    if gaps:
        st.warning(f"{len(gaps)} visibility gap(s) overlap this observation's window — evidence during "
                   'those intervals was unavailable, not clear.')
        for gap in gaps:
            st.caption(f"Unavailable pose: {timestamp(gap.get('start_s'))}–{timestamp(gap.get('end_s'))}")
    if jobs.load_event_trace(run_id, event['event_id'])['kind'] == 'recorded':
        handovers = jobs.load_event_handovers(run_id, event['event_id'])
        for handover in handovers:
            st.caption(f"Identity change: evidence carried from track {handover['old_track_id']} "
                      f"at {timestamp(handover['t'])}.")


def _decision_trace(run_id, event_id):
    trace = jobs.load_event_trace(run_id, event_id)
    if trace['kind'] == 'unavailable':
        st.info('Not recorded for this run. No structured score/threshold trace or '
               'predictions log was saved for this analysis.')
        return
    if trace['kind'] == 'reconstructed':
        st.warning('Reconstructed by matching track and time to this observation\'s window; '
                  'not a stored causal record. This run predates per-frame event linkage.')
        if trace.get('ambiguous_frames_dropped'):
            st.caption(f"{trace['ambiguous_frames_dropped']} frame(s) overlapped more than one "
                      "observation on this track and were dropped rather than guessed.")
    rows = trace['rows']
    st.caption(f"{len(rows)} analysed frame(s) in this observation's window.")
    fall_rows = [r for r in rows if r.get('fall_score') is not None]
    if fall_rows:
        st.markdown('**Fall score vs. threshold**')
        st.line_chart({'score': {r['t']: r['fall_score'] for r in fall_rows},
                       'threshold': {r['t']: r['fall_threshold'] for r in fall_rows}})
    activity_rows = [r for r in rows if r.get('activity_score') is not None]
    if activity_rows:
        st.markdown('**Activity score vs. threshold**')
        st.line_chart({'score': {r['t']: r['activity_score'] for r in activity_rows},
                       'threshold': {r['t']: r['activity_threshold'] for r in activity_rows}})
    if trace['kind'] == 'recorded':
        gated = [r for r in rows if r.get('gate_state')]
        if gated:
            last = gated[-1]['gate_state']
            st.markdown('**Gate state at the last recorded frame**')
            st.caption(f"Transition marker: {'set' if last.get('transition') is not None else 'none'}"
                      + (f" · seconds in fall_hold: {last['seconds_in_fall_hold']:.2f}"
                         if last.get('seconds_in_fall_hold') is not None else ''))
        handovers = jobs.load_event_handovers(run_id, event_id)
        if handovers:
            st.markdown('**Evidence handovers**')
            for handover in handovers:
                st.caption(f"At {handover['t']:.1f}s, evidence carried from track "
                          f"{handover['old_track_id']} ({handover['reason']}).")


def _raw_details(run_id, event, job):
    revisions = jobs.load_event_revisions(run_id, event['event_id'])
    if revisions:
        labels = [f"Revision {r.get('revision', index)}" for index, r in enumerate(revisions)]
        choice = st.selectbox('Revision', range(len(revisions)), format_func=lambda i: labels[i],
                              index=len(revisions) - 1, key=f"raw-revision-{run_id}-{event['event_id']}")
        st.json(revisions[choice])
    else:
        st.json(event)
    with st.expander('Model disclosures for this run'):
        st.json(job.get('model_disclosures', {}))


def _missed_event(run_id, job, timestamp):
    st.markdown('#### Investigate a moment with no alert')
    duration = (job.get('media') or {}).get('duration_s') or float(job.get('summary', {}).get('source_duration_s') or 0)
    max_value = float(duration) if duration else 3600.0
    moment = st.number_input('Source time (seconds)', min_value=0.0, max_value=max_value, value=0.0, step=1.0,
                             key=f"missed-moment-{run_id}")
    st.caption(f"Selected: {timestamp(moment)}")
    st.info('No observation was recorded at this time. If no trace exists for this moment, the cause of a '
            'possible miss is unavailable — it is not invented from the current model or settings.')
    folder = jobs.run_dir(run_id)
    annotated = folder / 'annotated.mp4'
    video = annotated if annotated.exists() and annotated.stat().st_size else (
        Path(job['source_path']) if job.get('source_path') else None)
    if video and video.exists():
        st.video(str(video), start_time=max(0, moment - 3))
    else:
        st.warning('The source video is no longer available on disk.')
    pending_key = f"missed-pending-{run_id}"
    with st.expander('Mark this interval and describe what is visible',
                     expanded=bool(st.session_state.get(pending_key))):
        with st.form(f"missed-learn-{run_id}"):
            label = st.selectbox('Proposed label', sorted(review.PROPOSED_LABELS))
            visible = st.text_input('Visible action')
            person = st.text_input('Person, if identifiable (optional)')
            note = st.text_area('Note (optional)')
            start = st.number_input('Interval start (s)', value=max(0.0, moment - 2), min_value=0.0)
            end = st.number_input('Interval end (s)', value=moment + 2, min_value=0.0)
            submitted = st.form_submit_button('Save to learning queue')
        if submitted:
            key, state_key = _action_key('missed-learn', run_id, 'none')
            try:
                review.create_candidate(run_id, None, start, end, label, visible_action_label=visible,
                                        person_if_identifiable=person, note=note, idempotency_key=key)
                st.session_state.pop(state_key, None)
                st.success('Saved to learning queue. The current model has not changed.')
            except review.DuplicateCandidateWarning as warning:
                st.session_state[pending_key] = dict(start=start, end=end, label=label,
                                                     visible=visible, person=person, note=note, key=key)
                st.warning(f"{len(warning.overlaps)} existing candidate(s) overlap this interval.")
            except ValueError as error:
                st.error(str(error))
        pending = st.session_state.get(pending_key)
        if pending and st.button('Create anyway', key=f"missed-learn-confirm-{run_id}"):
            review.create_candidate(run_id, None, pending['start'], pending['end'], pending['label'],
                                    visible_action_label=pending['visible'],
                                    person_if_identifiable=pending['person'], note=pending['note'],
                                    idempotency_key=pending['key'], confirmed_duplicate=True)
            st.session_state.pop(pending_key, None)
            st.success('Saved to learning queue. The current model has not changed.')


def debugger_view(timestamp, date_text):
    if st.session_state.get('page', 'Dashboard') != 'Debugger':
        st.rerun(scope='app')  # Fragment/page-less callbacks must also refresh outer navigation.
    all_jobs = jobs.list_jobs()
    header, back = st.columns([4, 1], vertical_alignment='center')
    with back:
        if st.button('Back to dashboard', width='stretch', key='debugger-back'):
            st.session_state['page'] = 'Dashboard'
            st.rerun()
    with header:
        st.markdown('<div class="eyebrow">WHY THIS WAS FLAGGED</div>', unsafe_allow_html=True)
        st.title('Debugger')
    if not all_jobs:
        st.info('No analyses yet. Choose a video in the sidebar to get started.')
        return
    choices = [job['run_id'] for job in all_jobs]
    names = {job['run_id']: f"{job.get('original_name', job['run_id'])} · {job['status']}" for job in all_jobs}
    if st.session_state.get('debugger_run') not in choices:
        st.session_state['debugger_run'] = choices[0]
    run_id = st.selectbox('Analysis', choices, key='debugger_run', format_func=lambda i: names[i])
    job = jobs.get_job(run_id)
    folder = jobs.run_dir(run_id)
    events = jobs.load_events(run_id)
    event_map = {event['event_id']: event for event in events}

    event_id = st.session_state.get('debugger_event')
    if event_id not in event_map:
        event_id = None

    modes = ['Investigate a recorded observation', 'Investigate a moment with no alert']
    mode = st.radio('Mode', modes, key=f"debugger-mode-{run_id}", label_visibility='collapsed',
                    horizontal=True, index=0 if events else 1)

    if mode == modes[1] or not events:
        _missed_event(run_id, job, timestamp)
        return

    if event_id is None:
        event_id = next(iter(event_map))
    event_id = st.selectbox('Observation', list(event_map),
                            format_func=lambda i: f"{timestamp(event_map[i].get('source_start_s'))} · {_label(event_map[i])} · Person {event_map[i].get('track_id', '?')}",
                            index=list(event_map).index(event_id), key=f"debugger-event-{run_id}")
    st.session_state['debugger_event'] = event_id
    event = event_map[event_id]
    reviews = jobs.get_reviews(run_id)
    review = reviews.get(event_id, {'label': 'unreviewed', 'note': ''})
    branch = ALERT_BRANCH.get(event.get('category'))

    st.markdown(f"### {_label(event)} · {job.get('original_name', run_id)}")
    urgency = ':red-badge[Urgent]' if event.get('category') in {'possible_fall', 'person_down'} else ':orange-badge[Review]'
    review_badge = f":gray-badge[{REVIEW_LABELS.get(review['label'], review['label'])}]"
    st.markdown(f"{urgency} · Video {timestamp(event.get('source_start_s'))} · {review_badge}")

    left, right = st.columns([1.7, 1], gap='large')
    with left:
        st.markdown('#### Evidence')
        annotated = folder / 'annotated.mp4'
        use_overlay = st.toggle('Show pose overlay', value=annotated.exists() and bool(annotated.stat().st_size)
                                if annotated.exists() else False, key=f"debugger-overlay-{run_id}-{event_id}",
                                disabled=not (annotated.exists() and annotated.stat().st_size))
        video = annotated if use_overlay and annotated.exists() else (
            Path(job['source_path']) if job.get('source_path') else None)
        revisions = jobs.load_event_revisions(run_id, event_id)
        latest_end = revisions[-1].get('source_end_s') if revisions else event.get('source_end_s')
        seek_options = {
            f"Before ({timestamp(max(0, float(event.get('source_start_s', 0)) - 3))})": max(0, float(event.get('source_start_s', 0)) - 3),
            f"First alert ({timestamp(event.get('emitted_source_s'))})": float(event.get('emitted_source_s') or event.get('source_start_s') or 0),
            f"Latest ({timestamp(latest_end)})": float(latest_end or 0),
        }
        seek_label = st.radio('Seek to', list(seek_options), key=f"debugger-seek-{run_id}-{event_id}",
                              horizontal=True, label_visibility='collapsed')
        start = seek_options[seek_label]
        if video and video.exists():
            st.video(str(video), start_time=start)
            st.caption(('Annotated result' if use_overlay else 'Original recording')
                      + f" · seeking to {timestamp(start)}")
        else:
            st.warning('The source video is no longer available on disk.')
        tabs = st.tabs(['Evidence timeline', 'Decision trace', 'Revision & review history', 'Raw details'])
        with tabs[0]:
            _evidence_timeline(run_id, event, timestamp, date_text)
        with tabs[1]:
            _decision_trace(run_id, event_id)
        with tabs[2]:
            _revision_history(run_id, event_id, timestamp, date_text)
            st.divider()
            st.caption(f"Review: {REVIEW_LABELS.get(review['label'], review['label'])}"
                      + (f" · {review['note']}" if review.get('note') else ''))
            if review.get('reviewed_at_utc'):
                st.caption(f"Reviewed {date_text(review['reviewed_at_utc'])}")
        with tabs[3]:
            _raw_details(run_id, event, job)
    with right:
        st.markdown('#### Why this was flagged')
        st.write(f"**Branch:** {'Security (activity)' if branch == 'activity' else 'Medical (fall)' if branch == 'fall' else 'Unclassified'}")
        st.write(f"**Score:** {event.get('score')} ({event.get('score_type', 'not recorded')})"
                if event.get('score') is not None else '**Score:** Not recorded for this run')
        trace_kind = jobs.load_event_trace(run_id, event_id)['kind']
        trace_note = {'recorded': 'Actual threshold, gate state and pose-quality signals: '
                                  'see the Decision trace tab.',
                     'reconstructed': 'A best-effort score/threshold trace is available (Decision '
                                      'trace tab); gate state was not recorded for this run.',
                     'unavailable': 'Actual threshold, duration/consecutive-window gates, and '
                                    'pose-quality/track-change signals: Not recorded for this run.'
                     }[trace_kind]
        st.caption(trace_note)
        st.markdown('**Saved observations**')
        if event.get('observations'):
            for observation in event['observations']:
                st.write(f"• {str(observation).replace('_', ' ').capitalize()}")
        else:
            st.caption('No observation tags were saved for this event.')
        st.divider()
        review_col, ignore_col, learn_col = st.columns(3)
        with review_col:
            with st.popover('Review', width='stretch'):
                with st.form(f"debugger-review-{run_id}-{event_id}"):
                    label = st.selectbox('Your review', list(REVIEW_LABELS),
                                         index=list(REVIEW_LABELS).index(review.get('label', 'unreviewed')),
                                         format_func=lambda value: REVIEW_LABELS[value])
                    note = st.text_area('Notes (optional)', value=review.get('note', ''), max_chars=2000)
                    if st.form_submit_button('Save review', width='stretch'):
                        jobs.save_review(run_id, event_id, label, note)
                        st.rerun()
        with ignore_col:
            _ignore_controls(run_id, event_id)
        with learn_col:
            _add_to_learning(run_id, event)
