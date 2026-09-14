"""Streamlit views and shared styling; inference remains in the worker."""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_legacy_path = Path(__file__).resolve().parents[1] / "ui.py"
_spec = spec_from_file_location("_watchverify_legacy_ui", _legacy_path)
if _spec is None or _spec.loader is None:  # pragma: no cover - import machinery guard
    raise ImportError("Unable to load shared Streamlit UI helpers")
_legacy = module_from_spec(_spec)
_spec.loader.exec_module(_legacy)

style = _legacy.style
timestamp = _legacy.timestamp
date_text = _legacy.date_text
facts = _legacy.facts
