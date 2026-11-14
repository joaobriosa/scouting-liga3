"""Minutes-played rules.

- Base match length is 90; added time is ignored ("45+2" -> 45, "90+3" -> 90).
- Matches with extra time (knockout ties) last 120: minutes run to 120 and
  "120+2" counts as 120.
- Starter: enters at 0. Substitute: enters at their sub-in minute.
- A player leaves at their sub-out minute, at their red-card minute, or at the
  end of the match.
- Unused substitutes are kept with 0 minutes and unused_sub=True.

The raw added-time values are kept in the tables (minute_in_added, ...) so
on-pitch metrics such as +/- can order events inside stoppage time later.
"""

from __future__ import annotations

MATCH_LENGTH = 90
EXTRA_TIME_LENGTH = 120


def clamp_minute(minute: int | None, match_length: int = MATCH_LENGTH) -> int | None:
    """Cap to the 0..match_length range (a "90+3" event arrives here as 90)."""
    if minute is None:
        return None
    return max(0, min(match_length, minute))


def compute_minutes(
    started: bool,
    sub_in: int | None,
    sub_out: int | None,
    red_minute: int | None,
    match_length: int = MATCH_LENGTH,
) -> tuple[int | None, int | None, int]:
    """Return (minute_in, minute_out, minutes_played).

    Minutes passed in are base minutes (added time already split off).
    Returns (None, None, 0) for an unused substitute.
    """
    if started:
        minute_in = 0
    elif sub_in is not None:
        minute_in = clamp_minute(sub_in, match_length)
    else:
        return None, None, 0

    exits = [m for m in (sub_out, red_minute) if m is not None]
    minute_out = clamp_minute(min(exits), match_length) if exits else match_length
    minute_out = max(minute_out, minute_in)
    return minute_in, minute_out, minute_out - minute_in
