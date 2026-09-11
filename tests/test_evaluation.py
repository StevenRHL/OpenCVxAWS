"""Event-level evaluation invariants. Synthetic events; no model accuracy is measured here."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from watchverify.evaluation import (aggregate, fall_ground_truth, match_fall_events,
                                    wilson_interval)
from train_models import assert_no_leakage, mnnit_split, urfall_split


def alert(t, category="possible_fall"):
    return dict(category=category, t=t, track_id=1)


def truth(onset=5.0, end=9.0, sequence_id="fall-01"):
    return [dict(sequence_id=sequence_id, category="fall", onset_s=onset, end_s=end,
                 onset_frame=1, end_frame=2)]


# --- ground truth -----------------------------------------------------------

def test_an_adl_sequence_is_never_a_fall_even_when_the_person_lies_down():
    """adl-10 contains 30 transition and 101 lying frames from a deliberate lie-down.
    Counting that as a fall would make the fall/lie distinction untestable."""
    labels = {"adl-10": {1: -1, 2: 0, 3: 1, 4: 1}}
    times = {1: 0.0, 2: 0.1, 3: 0.2, 4: 0.3}
    assert fall_ground_truth("adl-10", labels, times) == []


def test_a_fall_sequence_spans_onset_of_transition_to_end_of_lying():
    labels = {"fall-01": {1: -1, 2: 0, 3: 0, 4: 1, 5: 1}}
    times = {1: 0.0, 2: 1.0, 3: 1.1, 4: 2.0, 5: 3.0}
    event, = fall_ground_truth("fall-01", labels, times)
    assert event["onset_s"] == 1.0 and event["end_s"] == 3.0


def test_a_sequence_with_no_labels_yields_no_event():
    assert fall_ground_truth("fall-99", {}, {1: 0.0}) == []


# --- matching ---------------------------------------------------------------

def test_one_alert_matches_one_event_and_surplus_alerts_are_false():
    result = match_fall_events([alert(5.5), alert(6.0), alert(7.0)], truth(), duration_s=20)
    assert result["matched"] == 1
    assert result["false_alerts"] == 2, "extra alerts inside the window must still count as false"


def test_an_alert_outside_the_event_window_does_not_match():
    result = match_fall_events([alert(30.0)], truth(), duration_s=40)
    assert result["matched"] == 0 and result["missed"] == 1 and result["false_alerts"] == 1


def test_late_detection_counts_as_detected_but_not_on_time():
    result = match_fall_events([alert(8.9)], truth(onset=5.0, end=9.0), duration_s=20, on_time_s=3.0)
    assert result["matched"] == 1 and result["on_time"] == 0
    assert result["latencies_s"] == [3.9]


def test_an_event_with_no_alert_at_all_is_a_miss():
    """A fall whose pose was never extracted produces no alert. It is still a miss."""
    result = match_fall_events([], truth(), duration_s=20)
    assert result["missed"] == 1 and result["matched"] == 0


def test_recovery_candidates_are_not_fall_alerts():
    result = match_fall_events([alert(5.5, "recovery")], truth(), duration_s=20)
    assert result["alerts"] == 0 and result["missed"] == 1


def test_aggregate_reports_rates_and_false_alerts_per_hour():
    negative = match_fall_events([alert(2.0)], [], duration_s=1800)
    positive = match_fall_events([alert(5.5)], truth(), duration_s=1800)
    total = aggregate([negative, positive])
    assert total["true_events"] == 1 and total["matched"] == 1
    assert total["recall"] == 1.0
    assert total["precision"] == 0.5, "the alert on the negative sequence must reduce precision"
    assert total["false_alerts_per_hour"] == pytest.approx(1.0)


# --- splits -----------------------------------------------------------------

def test_split_assignment_is_deterministic_and_stratified_by_kind():
    assert urfall_split("fall-07") == urfall_split("fall-07")
    falls = [urfall_split(f"fall-{i:02d}") for i in range(1, 31)]
    adls = [urfall_split(f"adl-{i:02d}") for i in range(1, 41)]
    assert falls.count("train") == 18 and falls.count("validation") == 6 and falls.count("test") == 6
    assert adls.count("train") == 24 and adls.count("validation") == 8 and adls.count("test") == 8


def test_split_is_not_ordered_by_sequence_number():
    """Numbered splits were rejected: pose availability degrades with sequence number,
    so consecutive blocks are ordered by difficulty rather than exchangeable."""
    first_ten = {urfall_split(f"fall-{i:02d}") for i in range(1, 11)}
    assert len(first_ten) > 1, "an ordered split would put the first ten in one bucket"


def test_every_sequence_lands_in_exactly_one_split():
    assignment = {f"{kind}-{i:02d}": urfall_split(f"{kind}-{i:02d}")
                  for kind, n in (("fall", 30), ("adl", 40)) for i in range(1, n + 1)}
    assert_no_leakage([(k, v) for k, v in assignment.items()])
    assert set(assignment.values()) == {"train", "validation", "test"}
    assert sum(v == "test" for v in assignment.values()) == 14  # 6 falls + 8 ADLs


def test_leakage_guard_catches_identical_content_in_two_splits():
    """Keyed on content hash, so the same footage under two names is caught."""
    same_hash = "a" * 64
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage([(same_hash, "train"), (same_hash, "test")])
    assert_no_leakage([(same_hash, "train"), ("b" * 64, "test")])


def test_mnnit_split_is_sixty_twenty_twenty_by_position():
    assert [mnnit_split("c", i, 100) for i in (0, 59, 60, 79, 80, 99)] == \
        ["train", "train", "validation", "validation", "test", "test"]


# --- uncertainty ------------------------------------------------------------

def test_small_event_counts_produce_wide_intervals():
    low, high = wilson_interval(8, 10)
    assert low < 0.55 and high > 0.90, "10 events cannot support a precise recall claim"
    assert wilson_interval(0, 0) is None
