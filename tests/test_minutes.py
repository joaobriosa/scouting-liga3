import pytest

from scouting.minutes import compute_minutes
from scouting.parsing.zerozero import parse_minute


@pytest.mark.parametrize(
    "started, sub_in, sub_out, red, expected",
    [
        (True, None, None, None, (0, 90, 90)),      # starter, full match
        (True, None, 81, None, (0, 81, 81)),        # starter subbed off
        (False, 56, None, None, (56, 90, 34)),      # sub on, stays
        (False, 42, 82, None, (42, 82, 40)),        # sub on then off
        (True, None, None, 46, (0, 46, 46)),        # starter sent off
        (False, 60, None, 75, (60, 75, 15)),        # sub sent off
        (False, None, None, None, (None, None, 0)), # unused sub
        (False, 90, None, None, (90, 90, 0)),       # enters at 90+x
        (True, None, 70, 70, (0, 70, 70)),          # red and sub same minute
    ],
)
def test_compute_minutes(started, sub_in, sub_out, red, expected):
    assert compute_minutes(started, sub_in, sub_out, red) == expected


@pytest.mark.parametrize(
    "text, expected",
    [("81'", (81, None)), ("45+2'", (45, 2)), ("90+6' ", (90, 6)), ("9' ", (9, None)),
     ("n/d", (None, None)), ("n/d'", (None, None)), (" ", (None, None))],
)
def test_parse_minute(text, expected):
    assert parse_minute(text) == expected


@pytest.mark.parametrize(
    "started, sub_in, sub_out, red, expected",
    [
        (True, None, None, None, (0, 120, 120)),   # starter, full match with extra time
        (False, 105, None, None, (105, 120, 15)),  # sub on in extra time
        (True, None, 98, None, (0, 98, 98)),       # off in extra time
        (False, 120, None, None, (120, 120, 0)),   # on at 120+x (e.g. for penalties)
    ],
)
def test_compute_minutes_extra_time(started, sub_in, sub_out, red, expected):
    assert compute_minutes(started, sub_in, sub_out, red, match_length=120) == expected
