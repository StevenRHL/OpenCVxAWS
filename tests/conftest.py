"""Shared test fixtures.

Streamlit's `PagesManager.uses_pages_directory` is a class attribute, computed once from
whichever script an `AppTest` happens to run first and cached for the rest of the process.
Because the project root has a legacy `pages/` directory next to `app.py`, the first
`AppTest.from_file(app.py)` in a session latches that flag True for every later `AppTest`
in the same process — including `AppTest.from_string(...)` calls whose generated temp
script has no such directory, which then crash Streamlit's own multipage-v1 title check on
an unrelated script. Reset it before each test so one test's AppTest choice cannot leak into
another's.
"""
import pytest


@pytest.fixture(autouse=True)
def _reset_streamlit_pages_manager_cache():
    from streamlit.runtime.pages_manager import PagesManager
    PagesManager.uses_pages_directory = None
    yield
    PagesManager.uses_pages_directory = None
