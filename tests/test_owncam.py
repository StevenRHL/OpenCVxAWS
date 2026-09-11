"""Own-camera label expansion and split assignment.

Software behaviour only. Whether the recorded footage is any good is a question for the
held-out evaluation, not for these tests.
"""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from prepare_owncam import posture_codes
from train_models import hash_split, owncam_split
from watchverify.evaluation import LYING, NOT_LYING, TRANSITION


def frames(duration=10.0, step=0.1):
    count = int(round(duration / step)) + 1
    return [(index, round(index * step, 6)) for index in range(count)]


def test_intervals_expand_to_the_publisher_posture_codes():
    codes = posture_codes('fall-own-01', 3.0, 4.0, 6.0, frames())
    assert codes[0] == NOT_LYING
    assert codes[35] == TRANSITION, 'between fall_start_s and fall_end_s'
    assert codes[50] == LYING, 'between fall_end_s and down_end_s'
    assert codes[80] == NOT_LYING, 'after down_end_s'


def test_a_clip_with_no_times_is_entirely_not_lying():
    codes = posture_codes('adl-own-01', None, None, None, frames())
    assert set(codes.values()) == {NOT_LYING}


def test_a_deliberate_lie_down_keeps_lying_frames_but_is_named_as_ordinary():
    """An ADL clip may contain lying frames — the posture model should learn from them —
    while remaining a fall negative for the event logic, which keys on the name."""
    codes = posture_codes('adl-own-07', 5.0, 6.0, 8.0, frames())
    assert LYING in codes.values() and TRANSITION in codes.values()


def test_a_person_still_down_at_the_end_stays_down():
    """Running out of recording is not a recovery."""
    codes = posture_codes('fall-own-02', 3.0, 4.0, None, frames(duration=10.0))
    assert codes[max(codes)] == LYING


def test_unusable_times_are_refused_rather_than_repaired():
    with pytest.raises(ValueError, match='must be after'):
        posture_codes('fall-own-03', 5.0, 4.0, 8.0, frames())
    with pytest.raises(ValueError, match='past the end'):
        posture_codes('fall-own-04', 3.0, 4.0, 99.0, frames())
    with pytest.raises(ValueError, match='must not be before'):
        posture_codes('fall-own-05', 3.0, 6.0, 4.0, frames())
    with pytest.raises(ValueError, match='both fall_start_s and fall_end_s'):
        posture_codes('fall-own-06', 3.0, None, 8.0, frames())
    with pytest.raises(ValueError, match='needs fall_start_s'):
        posture_codes('fall-own-07', None, None, None, frames())


def test_clip_names_must_declare_which_kind_they_are():
    with pytest.raises(ValueError, match="must start with"):
        posture_codes('kitchen-clip-01', 3.0, 4.0, 6.0, frames())


def test_own_camera_splits_are_grouped_by_clip_and_cover_both_kinds():
    ids = [f'fall-own-{i:02d}' for i in range(1, 11)] + [f'adl-own-{i:02d}' for i in range(1, 11)]
    assignment = owncam_split(ids)
    assert set(assignment) == set(ids)
    for kind in ('fall', 'adl'):
        members = {sid: split for sid, split in assignment.items() if sid.startswith(kind)}
        assert set(members.values()) == {'train', 'validation', 'test'}, f'{kind} must reach every split'


def test_an_unrecognised_clip_name_fails_the_split_rather_than_vanishing():
    with pytest.raises(ValueError, match='must start with'):
        owncam_split(['fall-own-01', 'mystery-01'])


def test_hash_splits_are_stable_and_independent_of_order():
    members = [f'clip-{i:02d}' for i in range(1, 21)]
    first = hash_split(members, 'salt-a')
    assert first == hash_split(list(reversed(members)), 'salt-a')
    assert first != hash_split(members, 'salt-b'), 'a different salt must reassign'
    assert sorted(first.values()).count('train') == 12
