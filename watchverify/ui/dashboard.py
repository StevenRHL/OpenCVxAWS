"""Dashboard view backed by persisted local analysis records."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import streamlit as st

from watchverify.dashboard import (build_snapshot, date_text, filter_updates, PRIORITIES, REVIEW_LABELS)
from watchverify.ui.debugger import open_debugger


def open_analysis(run_id=None, event_id=None):
    st.session_state['page'] = 'Analyses'
    st.session_state['upload_view'] = False
    if run_id:
        st.session_state['selected_run'] = run_id
    if event_id:
        st.session_state[f'event-{run_id}'] = event_id


def open_upload():
    open_analysis()
    st.session_state['upload_view'] = True


def badge(key):
    colours = {'urgent': 'red', 'review': 'orange', 'system': 'orange', 'info': 'gray'}
    return f":{colours[key]}-badge[{PRIORITIES[key]}]"


def event_context(row, tz, timestamp):
    event = row['event']
    st.caption(f"Recorded video · {row['source']} · Video {timestamp(event.get('source_start_s'))}")
    occurred = event.get('occurred_at_utc')
    st.caption(f"Occurred {date_text(occurred, tz)}" if occurred else 'Recording date unknown')
    outcome = event.get('status', 'unknown')
    if outcome == 'incomplete':
        outcome = 'Incomplete · outcome unknown'
    st.caption(f"Review: {REVIEW_LABELS.get(row['review'], row['review'])} · Incident: {outcome}"
              + (' · Ignored (dismissed from the default queue, not a judgement)' if row.get('ignored') else ''))


def row_view(row, tz, timestamp, key, *, stacked=False):
    with st.container(border=True):
        body, action = ((st.container(), st.container()) if stacked else
                        st.columns([4, 1], vertical_alignment='center'))
        with body:
            st.markdown(badge(row['priority']) + ' **' + row['title'] + '**')
            prefix = 'First alert recorded' if row['kind'] == 'Observation' else 'Update recorded'
            st.caption(f"{prefix} · {date_text(row.get('at'), tz)}")
            if row.get('event'):
                event_context(row, tz, timestamp)
            else:
                st.caption(f"Recorded video · {row['source']}")
        with action:
            if row['event_id']:
                st.button('Open debugger', key=key, on_click=open_debugger,
                          args=(row['run_id'], row['event_id']), width='stretch')
            else:
                st.button('View analysis', key=key, on_click=open_analysis,
                          args=(row['run_id'], row['event_id']), width='stretch')


@st.fragment(run_every='5s')
def dashboard_view(timestamp):
    if st.session_state.get('page', 'Dashboard') != 'Dashboard':
        st.rerun(scope='app')  # Fragment callbacks must also refresh the outer navigation.
    st.markdown('<div class="eyebrow">YOUR REVIEW DESK</div>', unsafe_allow_html=True)
    heading, upload = st.columns([4, 1], vertical_alignment='center')
    with heading:
        st.title('Dashboard')
        st.write('See what needs attention, then review the footage behind it.')
    with upload:
        st.button('Upload video', type='primary', width='stretch', on_click=open_upload)
    controls = st.columns([2, 3, 1])
    zone = controls[0].selectbox('Display timezone', ['Australia/Sydney', 'UTC'], key='dashboard-zone')
    tz = ZoneInfo(zone)
    controls[2].button('Refresh', width='stretch', key='dashboard-refresh')
    try:
        snapshot = build_snapshot()
    except (OSError, ValueError) as error:
        st.error(f'Dashboard could not refresh. Saved results may be unavailable: {error}')
        return
    controls[1].caption(f"Refreshed {date_text(datetime.now(timezone.utc).isoformat(), tz)} · refreshes every 5s while connected")
    st.caption('Recorded video · Urgent means priority for human review, not a live emergency notification.')
    if snapshot['errors']:
        st.warning('Dashboard is incomplete. Counts exclude observations whose records could not be read.')
        for error in snapshot['errors']:
            st.caption(error)
    cols = st.columns(4)
    for col, label, count in zip(cols,
            ['Urgent awaiting review', 'Other awaiting review', 'Analyses active', 'Analyses needing attention'],
            [len(snapshot['urgent']), snapshot['other_pending'], len(snapshot['active']), len(snapshot['attention'])]):
        col.metric(label, count)
    if not snapshot['total_runs'] and snapshot['errors']:
        st.warning('No saved analyses could be read. Check the errors above and refresh after the files are available.')
        return
    if not snapshot['total_runs']:
        st.info('No analyses yet. Choose a video in the sidebar, then select Start analysis.')
        st.caption('An empty dashboard does not establish that footage is safe.')
        return
    if snapshot['active']:
        with st.expander(f"Analysis progress · {len(snapshot['active'])} active", expanded=False):
            for job in snapshot['active']:
                st.progress(min(1.0, max(0.0, float(job.get('progress') or 0))),
                            text=f"{job.get('original_name', job['run_id'])} · {job['status']}")
                st.button('View analysis', key=f"active-{job['run_id']}",
                          on_click=open_analysis, args=(job['run_id'],))
    if snapshot['attention']:
        with st.expander(f"System attention · {len(snapshot['attention'])} analyses"):
            for row in snapshot['attention']:
                st.write(row['source'])
                st.caption(' · '.join(row['reasons']))
                st.button('View analysis', key=f"attention-{row['run_id']}",
                          on_click=open_analysis, args=(row['run_id'],))
    queue, feed = st.columns([1, 2], gap='large')
    with queue:
        st.subheader('Urgent review')
        st.caption('Outstanding urgent observations across all dates. Not affected by the latest-updates filters.')
        if not snapshot['urgent']:
            st.info('No urgent observations awaiting review. This does not establish that the footage is safe.')
        else:
            urgent_pages = max(1, (len(snapshot['urgent']) + 2) // 3)
            page = st.selectbox('Urgent page', range(1, urgent_pages + 1), key='urgent-page') if urgent_pages > 1 else 1
            for row in snapshot['urgent'][(page - 1) * 3:page * 3]:
                row_view(row, tz, timestamp, 'urgent-' + row['id'], stacked=True)
    with feed:
        st.subheader('Latest updates')
        st.caption('Newest recorded updates first. Observation entries show the first alert time; their review and outcome reflect the latest saved record.')
        if snapshot['legacy_runs']:
            st.caption('History is limited: older runs may lack status dates; incident revisions have no separate update time. Only the latest saved review is available. Missing times are not estimated.')
        with st.popover('Filters', width='stretch'):
            urgency = st.selectbox('Urgency', ['All', *PRIORITIES],
                                   format_func=lambda x: PRIORITIES.get(x, x), key='feed-urgency')
            kind = st.selectbox('Update type', ['All', 'Observation', 'Analysis', 'Review'], key='feed-kind')
            review = st.selectbox('Review state', ['All', *REVIEW_LABELS],
                                  format_func=lambda x: REVIEW_LABELS.get(x, x), key='feed-review')
            date_mode = st.selectbox('Update date', ['All dates', 'Today', 'Date range'], key='feed-date-mode')
            show_ignored = st.checkbox('Show ignored observations', key='feed-show-ignored')
            dates = None
            today = datetime.now(tz).date()
            if date_mode == 'Today':
                dates = (today, today)
            elif date_mode == 'Date range':
                dates = st.date_input('Update date range', (today, today), key='feed-date-range')
                if len(dates) != 2:
                    st.info('Choose the start and end dates.')
                    return
        rows = filter_updates(snapshot['updates'], urgency=urgency, kind=kind, review=review, dates=dates,
                              tz=tz, show_ignored=show_ignored)
        st.caption(f'{len(rows)} updates match · Dates use {zone}. Undated entries appear last and are excluded by date filters.')
        if not rows:
            st.info('No updates match these filters.')
            return
        pages = max(1, (len(rows) + 9) // 10)
        page = st.selectbox('Updates page', range(1, pages + 1), key='feed-page')
        for row in rows[(page - 1) * 10:page * 10]:
            row_view(row, tz, timestamp, 'feed-' + row['id'])
