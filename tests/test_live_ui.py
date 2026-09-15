"""Exercise real page controls with deterministic camera failures."""
import pytest
from streamlit.testing.v1 import AppTest
from watchverify import live


@pytest.mark.parametrize('failure', ['open', 'dependency', 'frames'])
def test_failure_shows_error_and_allows_restart(monkeypatch, failure):
    class Session:
        warnings = []
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            if failure == 'open':
                raise RuntimeError('Camera could not be opened')
            if failure == 'dependency':
                raise ImportError('pose dependency unavailable')
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return dict(ok=False, fatal=True, message='Camera stopped returning frames')
    monkeypatch.setattr(live, 'LiveSession', Session)
    page = AppTest.from_file('pages/2_Live_camera.py').run()
    for _ in range(2):
        next(b for b in page.button if b.label == 'Start camera').click().run()
        assert not page.exception
        assert page.error
        assert not page.session_state['live_running']
        assert not next(b for b in page.button if b.label == 'Start camera').disabled
        assert next(b for b in page.button if b.label == 'Stop camera').disabled
