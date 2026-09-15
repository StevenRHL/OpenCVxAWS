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
BACKGROUND = "#F4F6F5"
SURFACE = "#FFFFFF"
BORDER = "#E2E8E5"
MUTED = "#5C6D64"
INK = "#151C1A"

# Dark chrome (sidebar) and accent, custom-applied via CSS below rather than through
# .streamlit/config.toml: the config's secondaryBackgroundColor also colours ordinary
# widgets (inputs, checkboxes, code blocks) across the whole app, not just the sidebar.
ACCENT = "#1FCB6B"
ACCENT_TEXT = "#147A45"  # accent hue dark enough for text/borders on a light surface
CHROME_BG = "#111B21"
CHROME_TEXT = "#E7EFEA"
CHROME_MUTED = "#7E9186"
CHROME_BORDER = "#1E2B31"
MONO = '"SFMono-Regular", Consolas, "Liberation Mono", monospace'

CSS = f"""<style>
/* --- page frame ---------------------------------------------------------- */
/* Fill the window: without this the page ends in a band of browser background. */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
    background:{BACKGROUND}}}
[data-testid="stAppViewContainer"] {{min-height:100vh}}
/* Streamlit reserves a tall empty header strip above the first element. */
[data-testid="stHeader"] {{height:2.5rem; background:transparent}}
.block-container {{padding-top:4rem; padding-bottom:2.5rem; max-width:1240px}}

h1 {{letter-spacing:-.05em; font-weight:650!important; margin-bottom:.2rem}}
h2, h3 {{letter-spacing:-.025em}}
h4 {{letter-spacing:-.02em; margin:0 0 .6rem}}
.eyebrow {{font-size:11px; font-weight:700; letter-spacing:.16em; color:{ACCENT_TEXT};
    margin-bottom:.3rem}}

/* --- controls ------------------------------------------------------------ */
/* Equal height and no mid-word wrapping ("Call ambula/nce"). Equal *width* comes from
   width="stretch" on the widget itself; CSS cannot win that one against Streamlit. */
div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button {{height:44px; white-space:nowrap; padding:0 10px;
    border-radius:8px}}
div[data-testid="stButton"] > button p,
div[data-testid="stDownloadButton"] > button p {{font-size:13.5px; margin:0}}

[data-testid="stMetric"] {{background:{SURFACE}; border:1px solid {BORDER}; border-left:3px solid {BORDER};
    border-radius:14px; padding:12px 14px}}
[data-testid="stMetricValue"] {{font-size:26px; font-family:{MONO}}}
[data-testid="stMetricLabel"] p {{font-size:11px; font-weight:700; letter-spacing:.08em;
    text-transform:uppercase; color:{MUTED}}}

/* --- sidebar: dark chrome, HackerRank-style nav rail -------------------- */
[data-testid="stSidebar"] {{background:{CHROME_BG}; border-right:1px solid {CHROME_BORDER}}}
[data-testid="stSidebar"] * {{color:{CHROME_TEXT}}}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{gap:.5rem}}
[data-testid="stSidebar"] hr {{margin:.75rem 0; border-color:{CHROME_BORDER}}}
[data-testid="stSidebar"] label p {{font-size:13px}}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{font-size:11px; font-weight:700;
    letter-spacing:.13em; color:{CHROME_MUTED}; margin-bottom:.1rem}}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {{background:{CHROME_BORDER}; color:{CHROME_TEXT}}}
[data-testid="stSidebar"] [data-testid="stExpander"] {{border-color:{CHROME_BORDER}}}

/* Large navigation tiles keep native radio keyboard and selection behavior. */
.st-key-review-navigation {{padding-top:8px}}
.st-key-review-navigation [data-testid="stRadio"],
.st-key-review-navigation [role="radiogroup"],
.st-key-review-navigation [role="radiogroup"] > div {{width:100%!important}}
.st-key-review-navigation [role="radiogroup"] {{gap:10px}}
.st-key-review-navigation label[data-baseweb="radio"] {{
    display:flex; align-items:center; box-sizing:border-box; width:100%;
    min-height:58px; margin:0; padding:14px 18px; border:1px solid #34464F;
    border-radius:10px; background:#1B2A32; cursor:pointer;
    transition:background .15s ease, border-color .15s ease}}
.st-key-review-navigation label[data-baseweb="radio"] > div:first-child {{display:none}}
.st-key-review-navigation label[data-baseweb="radio"] > div:last-child {{margin:0; padding:0}}
.st-key-review-navigation label[data-baseweb="radio"] p {{font-size:16px!important; font-weight:600}}
.st-key-review-navigation label[data-baseweb="radio"]:hover {{background:#293E48; border-color:#769188}}
.st-key-review-navigation label[data-baseweb="radio"]:has(input:checked) {{
    background:#17482F; border-color:#46D78A; box-shadow:inset 4px 0 0 #46D78A}}
.st-key-review-navigation label[data-baseweb="radio"]:has(input:focus-visible) {{
    outline:2px solid #A2F0C5; outline-offset:3px}}

.st-key-metricrow [data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] {{border-left:4px solid #C54A4A; background:#FFF5F5}}
.st-key-metricrow [data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] {{border-left:4px solid #C38B28; background:#FFFAF0}}
.st-key-metricrow [data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"] {{border-left:4px solid #39966A}}

/* --- small components ---------------------------------------------------- */
.tag {{display:inline-flex; align-items:center; gap:5px; font-size:11px; font-weight:700;
    letter-spacing:.09em; text-transform:uppercase; padding:3px 9px 3px 8px; border-radius:999px;
    margin-bottom:.5rem}}
.tag-urgent {{background:#FDECEC; color:#A32B2B; border:1px solid #F3C9C9}}
.tag-review {{background:#FDF3E2; color:#8A5A12; border:1px solid #F2DDB3}}
.tag-system {{background:#FDF3E2; color:#8A5A12; border:1px solid #F2DDB3}}
.tag-info {{background:#EEF1F0; color:{MUTED}; border:1px solid {BORDER}}}
.facts {{border:1px solid {BORDER}; border-radius:14px; background:{SURFACE}; padding:4px 14px;
    margin-bottom:.9rem}}
.facts div {{display:flex; justify-content:space-between; gap:16px; padding:7px 0;
    font-size:13.5px; border-bottom:1px solid #F0F4F1}}
.facts div:last-child {{border-bottom:none}}
.facts span:first-child {{color:{MUTED}}}
.facts span:last-child {{color:{INK}; font-weight:550; text-align:right; font-family:{MONO}}}
.seen {{font-size:13.5px; color:{INK}; margin:.1rem 0 .2rem}}

/* --- row cards: hover like a HackerRank submissions/problem list --------- */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
    border-radius:14px; transition:box-shadow .15s ease, border-color .15s ease}}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    box-shadow:0 4px 16px rgba(21,28,26,.08); border-color:#C9D6CE}}

/* Expandable rows are the primary disclosure control throughout the review desk. */
[data-testid="stExpander"] {{background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px}}
[data-testid="stExpander"] summary {{min-height:52px; padding:12px 16px; cursor:pointer}}
[data-testid="stExpander"] summary:hover {{background:#EDF5F0; border-radius:12px}}
[data-testid="stExpander"] summary:focus-visible {{outline:2px solid {ACCENT_TEXT}; outline-offset:2px}}
[data-testid="stExpander"] summary p {{font-size:14px; line-height:1.5}}
[data-testid="stSidebar"] [data-testid="stExpander"] {{background:{CHROME_BG}}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {{background:{CHROME_BORDER}}}

/* --- responsive ---------------------------------------------------------- */
[data-testid="stHorizontalBlock"] {{flex-wrap:wrap}}
[data-testid="stColumn"] {{min-width:0}}
[data-testid="stMainBlockContainer"] {{width:100%; min-width:0}}
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


_TAG_GLYPH = {"urgent": "▲", "review": "●", "system": "■", "info": "○"}


def tag(kind: str, label: str) -> str:
    """An urgency/status pill: `<span class="tag tag-{kind}">glyph label</span>`.

    Each kind keeps a distinct leading glyph, not just a colour — `review` and `system`
    share an amber colour family, so the glyph is what tells them apart without relying
    on colour alone (docs/UI_PLAN.md §1).
    """
    glyph = _TAG_GLYPH.get(kind, "●")
    return f'<span class="tag tag-{kind}">{glyph} {label}</span>'


def facts(pairs) -> None:
    """A compact label/value block.

    Streamlit's own bold-label paragraphs stack with a lot of air between them, which is
    what turned four short facts into a screenful.
    """
    body = "".join(f"<div><span>{name}</span><span>{value}</span></div>" for name, value in pairs)
    st.markdown(f'<div class="facts">{body}</div>', unsafe_allow_html=True)
