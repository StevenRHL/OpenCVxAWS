"""One look, shared by every page, and the small helpers that render it.

Three pages drifting apart in spacing, button size and breakpoints is the same failure as
two copies of a constant: it is not that either is wrong, it is that nothing keeps them
the same. So the stylesheet lives here and each page calls `style()`.

Most of what follows is undoing Streamlit defaults that do not survive contact with a real
screen:

- **Columns never wrap.** `st.columns` is a flex row with no wrapping, so a two-column
  layout does not become one column on a laptop or a phone — it just gets narrower until
  the video is unwatchable. Pages wrap each column row in `st.container(key=...)`, which
  Streamlit renders with a `st-key-<key>` class, and the breakpoints below target those
  keys so each row can stack at the width where *it* runs out of room.
- **Buttons are sized to their text.** Three buttons side by side come out three different
  widths, and a long label wraps mid-word. Equal width comes from passing
  `width="stretch"`, which is Streamlit's own API; equal height and no mid-word wrapping
  come from here.
- **The app does not fill the window.** Below the last element the browser's own
  background shows through, which reads as the page having broken off.
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

# Matches .streamlit/config.toml. Kept in both places because the config is read by the
# server before any Python here runs, and CSS cannot reach it.
BACKGROUND = "#F7F9F7"
SURFACE = "#FFFFFF"
BORDER = "#E0E8E2"
MUTED = "#6A7F72"
INK = "#182B24"

CSS = f"""<style>
/* --- page frame ---------------------------------------------------------- */
/* Fill the window: without this the page ends in a band of browser background. */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
    background:{BACKGROUND}}}
[data-testid="stAppViewContainer"] {{min-height:100vh}}
/* Streamlit reserves a tall empty header strip above the first element. */
[data-testid="stHeader"] {{height:2.5rem; background:transparent}}
.block-container {{padding-top:1rem; padding-bottom:2.5rem; max-width:1240px}}

h1 {{letter-spacing:-.05em; font-weight:650!important; margin-bottom:.2rem}}
h2, h3 {{letter-spacing:-.025em}}
h4 {{letter-spacing:-.02em; margin:0 0 .6rem}}
.eyebrow {{font-size:11px; font-weight:700; letter-spacing:.16em; color:#537261;
    margin-bottom:.3rem}}

/* --- controls ------------------------------------------------------------ */
/* Equal height and no mid-word wrapping ("Call ambula/nce"). Equal *width* comes from
   width="stretch" on the widget itself; CSS cannot win that one against Streamlit. */
div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button {{height:44px; white-space:nowrap; padding:0 10px}}
div[data-testid="stButton"] > button p,
div[data-testid="stDownloadButton"] > button p {{font-size:13.5px; margin:0}}

[data-testid="stMetric"] {{background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px;
    padding:12px 14px}}
[data-testid="stMetricValue"] {{font-size:26px}}
[data-testid="stMetricLabel"] p {{font-size:12px; color:{MUTED}}}

/* --- sidebar ------------------------------------------------------------- */
/* The default block gap plus divider margins left a hand's width of nothing between
   every control. */
[data-testid="stSidebar"] {{border-right:1px solid #DCE5DE}}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{gap:.5rem}}
[data-testid="stSidebar"] hr {{margin:.75rem 0}}
[data-testid="stSidebar"] label p {{font-size:13px}}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{font-size:11px; font-weight:700;
    letter-spacing:.13em; color:{MUTED}; margin-bottom:.1rem}}

/* --- small components ---------------------------------------------------- */
.tag {{display:inline-block; font-size:11px; font-weight:700; letter-spacing:.09em;
    padding:3px 9px; border-radius:999px; margin-bottom:.5rem}}
.tag-med {{background:#FDECEC; color:#A32B2B; border:1px solid #F3C9C9}}
.tag-sec {{background:#EEF2FD; color:#33468F; border:1px solid #CCD6F5}}
.facts {{border:1px solid {BORDER}; border-radius:12px; background:{SURFACE}; padding:4px 14px;
    margin-bottom:.9rem}}
.facts div {{display:flex; justify-content:space-between; gap:16px; padding:7px 0;
    font-size:13.5px; border-bottom:1px solid #F0F4F1}}
.facts div:last-child {{border-bottom:none}}
.facts span:first-child {{color:{MUTED}}}
.facts span:last-child {{color:#1E2A23; font-weight:550; text-align:right}}
.seen {{font-size:13.5px; color:#2C3B33; margin:.1rem 0 .2rem}}

/* --- responsive ---------------------------------------------------------- */
[data-testid="stHorizontalBlock"] {{flex-wrap:wrap}}
/* A row of buttons should stay a row until a phone, whatever its container width. */
.st-key-decisionrow [data-testid="stColumn"] {{min-width:0 !important; flex:1 1 0 !important}}
/* 1120px, not 1024: with the sidebar open a 1024 laptop left the video column 326px wide,
   which is too narrow to judge footage in. */
@media (max-width:1120px) {{
  .st-key-mainsplit [data-testid="stColumn"] {{flex:1 1 100% !important; min-width:100% !important}}
}}
@media (max-width:860px) {{
  .st-key-metricrow [data-testid="stColumn"] {{flex:1 1 44% !important; min-width:44% !important}}
}}
@media (max-width:560px) {{
  .block-container {{padding-left:1rem; padding-right:1rem}}
  h1 {{font-size:2rem}}
  /* Metrics stay two-up even on a phone: four short counters stacked into four tall cards
     pushes the queue itself below the fold for no gain. */
  .st-key-decisionrow [data-testid="stColumn"] {{flex:1 1 100% !important; min-width:100% !important}}
  /* Stack each fact onto two lines rather than squeezing label against value. */
  .facts div {{flex-direction:column; gap:0; padding:6px 0}}
  .facts span:last-child {{text-align:left}}
}}
</style>"""


def style() -> None:
    """Install the shared stylesheet. Call once, immediately after `set_page_config`."""
    st.markdown(CSS, unsafe_allow_html=True)


def timestamp(seconds) -> str:
    """A position inside a recording, as hh:mm:ss.s."""
    if seconds is None:
        return "Unknown"
    seconds = max(0, float(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, rest = divmod(rest, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{rest:04.1f}"


def date_text(value) -> str:
    """A wall-clock time, or a plain statement that there is none."""
    if not value:
        return "Unknown"
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).isoformat(
            sep=" ", timespec="seconds")
    except (TypeError, ValueError):
        return str(value)


def facts(pairs) -> None:
    """A compact label/value block.

    Streamlit's own bold-label paragraphs stack with a lot of air between them, which is
    what turned four short facts into a screenful.
    """
    body = "".join(f"<div><span>{name}</span><span>{value}</span></div>" for name, value in pairs)
    st.markdown(f'<div class="facts">{body}</div>', unsafe_allow_html=True)
