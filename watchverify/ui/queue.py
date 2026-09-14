"""Learning queue: candidate examples awaiting curation, never automatic training.

Saving here never changes the installed model. Training is a later, explicit, separate
batch step (see docs/UI_PLAN.md section 4) — this page only curates labeled evidence.
"""
from __future__ import annotations

import streamlit as st

from watchverify import jobs, review


def queue_view(timestamp):
    st.markdown('<div class="eyebrow">CURATE EXAMPLES, NEVER TRAIN AUTOMATICALLY</div>',
               unsafe_allow_html=True)
    st.title('Learning queue')
    st.caption('The current model has not changed by anything on this page. Training is a '
              'separate, explicit step outside this app.')
    with st.expander('Add a missed interval'):
        missed_interval_form()
    ready = [c for c in review.list_candidates() if c['status'] == 'ready_for_dataset_review']
    if ready:
        with st.expander(f"Export {len(ready)} ready candidate(s) for dataset review"):
            selected = st.multiselect(
                'Candidates to export', [c['candidate_id'] for c in ready],
                default=[c['candidate_id'] for c in ready],
                format_func=lambda cid: next(c['visible_action_label'] or cid for c in ready if c['candidate_id'] == cid))
            if st.button('Export selected', disabled=not selected, key='queue-export'):
                result = review.export_candidates(selected)
                if result['exported']:
                    st.success(f"Exported {len(result['exported'])} example(s) for dataset review. "
                              "No training or model change occurred.")
                if result['skipped']:
                    st.warning(f"{len(result['skipped'])} candidate(s) skipped: "
                              + '; '.join(f"{s['candidate_id']} ({s['reason']})" for s in result['skipped']))
                st.caption(f"Manifest: {result['manifest_path']}")
                st.caption('To fold this export into a retrain (a separate, manual step outside '
                          'this app — nothing here changes the installed model): run '
                          '`python scripts/import_reviewed_exports.py`, fill in the exact fall '
                          'timing it asks for in `data/raw/reviewed/labels.csv`, then '
                          '`python scripts/prepare_owncam.py --corpus reviewed`, '
                          '`python scripts/extract_features.py --source reviewed`, and '
                          '`python scripts/train_models.py --fall-sources urfall,reviewed`.')
    candidates = review.list_candidates()
    if not candidates:
        st.info('No candidates yet. Add one from the Debugger, on a recorded observation or a '
               'moment with no alert.')
        return
    jobs_by_run = {job['run_id']: job for job in jobs.list_jobs()}
    show = st.radio('Show', ['Needs annotation', 'Ready for dataset review', 'Exported',
                             'Excluded', 'Everything'], horizontal=True, key='queue-show')
    status_filter = {'Needs annotation': 'needs_annotation', 'Ready for dataset review': 'ready_for_dataset_review',
                     'Exported': 'exported', 'Excluded': 'excluded'}.get(show)
    visible = candidates if status_filter is None else [c for c in candidates if c['status'] == status_filter]
    st.caption(f"{len(visible)} candidate(s)")
    if not visible:
        st.info('No candidates match this filter.')
        return
    for candidate in visible:
        job = jobs_by_run.get(candidate['run_id'], {})
        with st.container(border=True):
            ok, reason = review.eligibility(candidate)
            left, right = st.columns([3, 1])
            with left:
                st.markdown(f"**{candidate['proposed_label'].capitalize()}** · "
                           f"{job.get('original_name', candidate['run_id'])} · "
                           f"{timestamp(candidate['source_start_s'])}–{timestamp(candidate['source_end_s'])}")
                st.caption(f"Status: {candidate['status'].replace('_', ' ')}"
                          + (f" · {candidate['visible_action_label']}" if candidate['visible_action_label'] else ''))
                if candidate.get('note'):
                    st.caption(candidate['note'])
                if not ok and candidate['status'] == 'needs_annotation':
                    st.caption(f"Not ready: {reason}")
            with right:
                clip = review.candidate_clip(candidate['run_id'], candidate)
                if clip is not None and clip.exists() and clip.stat().st_size:
                    st.video(str(clip))
                else:
                    st.caption('Clip unavailable')
            _row_actions(candidate)


def _row_actions(candidate):
    candidate_id = candidate['candidate_id']
    run_id = candidate['run_id']
    primary, more = st.columns(2)
    if candidate['status'] == 'needs_annotation':
        ok, reason = review.eligibility(candidate)
        with primary:
            if st.button('Mark ready', key=f"queue-ready-{candidate_id}", disabled=not ok,
                        help=None if ok else reason, width='stretch'):
                review.set_candidate_status(run_id, candidate_id, 'ready_for_dataset_review')
                st.rerun()
    elif candidate['status'] == 'excluded':
        with primary:
            if st.button('Restore', key=f"queue-restore-{candidate_id}", width='stretch'):
                review.restore_candidate(run_id, candidate_id)
                st.rerun()
    with more:
        with st.popover('More', width='stretch'):
            if candidate['status'] == 'needs_annotation':
                label = st.selectbox('Proposed label', sorted(review.PROPOSED_LABELS),
                                     index=sorted(review.PROPOSED_LABELS).index(candidate['proposed_label']),
                                     key=f"queue-label-{candidate_id}")
                visible = st.text_input('Visible action', value=candidate['visible_action_label'],
                                        key=f"queue-visible-{candidate_id}")
                if st.button('Save correction', key=f"queue-save-correction-{candidate_id}"):
                    review.correct_candidate(run_id, candidate_id, proposed_label=label,
                                             visible_action_label=visible)
                    st.rerun()
                st.divider()
            if candidate['status'] != 'excluded':
                if st.button('Remove', key=f"queue-remove-{candidate_id}", width='stretch'):
                    review.remove_candidate(run_id, candidate_id)
                    st.rerun()


def missed_interval_form():
    """Add a candidate for a moment with no recorded observation, from any completed run."""
    all_jobs = [job for job in jobs.list_jobs() if job['status'] == 'completed']
    if not all_jobs:
        st.info('Analyse a video first, then add a missed interval here.')
        return
    names = {job['run_id']: job.get('original_name', job['run_id']) for job in all_jobs}
    with st.form('queue-missed-interval'):
        run_id = st.selectbox('Analysis', list(names), format_func=lambda i: names[i])
        duration = (jobs.get_job(run_id).get('media') or {}).get('duration_s') or 3600.0
        start = st.number_input('Interval start (s)', min_value=0.0, max_value=float(duration), value=0.0)
        end = st.number_input('Interval end (s)', min_value=0.0, max_value=float(duration), value=min(5.0, float(duration)))
        label = st.selectbox('Proposed label', sorted(review.PROPOSED_LABELS))
        visible = st.text_input('Visible action')
        person = st.text_input('Person, if identifiable (optional)')
        note = st.text_area('Note (optional)')
        submitted = st.form_submit_button('Save to learning queue')
    if not submitted:
        return
    try:
        review.create_candidate(run_id, None, start, end, label, visible_action_label=visible,
                                person_if_identifiable=person, note=note)
        st.success('Saved to learning queue. The current model has not changed.')
    except review.DuplicateCandidateWarning:
        st.session_state['queue-missed-pending'] = dict(
            run_id=run_id, start=start, end=end, label=label, visible=visible, person=person, note=note)
        st.warning('An existing candidate overlaps this interval.')
    except ValueError as error:
        st.error(str(error))
    pending = st.session_state.get('queue-missed-pending')
    if pending and st.button('Create anyway', key='queue-missed-confirm'):
        review.create_candidate(pending['run_id'], None, pending['start'], pending['end'], pending['label'],
                                visible_action_label=pending['visible'], person_if_identifiable=pending['person'],
                                note=pending['note'], confirmed_duplicate=True)
        st.session_state.pop('queue-missed-pending', None)
        st.success('Saved to learning queue. The current model has not changed.')
